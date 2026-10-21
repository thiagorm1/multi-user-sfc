"""Trained MAPPO actor proposals plus real Qwen guidance for unscaled MUVR."""

import copy
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env, safe_comp_latency
from muar_sfc.algorithms.mappo.mappo_core import MAPPOTrainer
from muar_sfc.controllers.cloud_llm_planner import CloudLLMPlanner, CognitiveGuidance
from muar_sfc.core.orchestration.placement import node_demand


class MUVRMAPPOEnv(MAPPO_SFC_Env):
    def __init__(self, graph, sfc, n_regions=4, guidance=None):
        # ER has no spatial coordinates. Remove the placeholder positions only
        # in the observation graph so all regions have a stable six-server shape.
        observation_graph = graph.copy()
        for _, data in observation_graph.nodes(data=True):
            data.pop("position", None)
        super().__init__(
            observation_graph,
            n_regions=n_regions,
            cloud_node=-1,
            share=True,
            shareband=True,
            proposal_only=True,
            policy_weight=0,
            destination_weight=1.0,
            load_balance_scale=2.0,
            cache_preference_weight=0.1,
            cognitive_guidance=guidance,
        )
        self.obs_dim += 2 * self.max_nodes_per_region + 6
        self.state_dim = self.n_regions * self.obs_dim + self.global_stat_dim
        self.actor_decisions = []
        self.reset_sfc(sfc)

    def reset_sfc(self, sfc):
        self.destination_distances = nx.single_source_dijkstra_path_length(
            self.base_graph, sfc.dst_node, weight="latency"
        )
        fastest = max(self.base_graph, key=lambda node: self.base_graph.nodes[node].get("ips", 0))
        self.minimum_tail = {"dst": 0.0}
        for name in reversed(list(nx.topological_sort(sfc.graph))):
            if name == "dst":
                continue
            self.minimum_tail[name] = max(
                (
                    self.minimum_tail[next_name]
                    + (
                        0
                        if next_name == "dst"
                        else safe_comp_latency(
                            self.base_graph, fastest, sfc.get_vnf_by_id(next_name)
                        )
                    )
                )
                for next_name in sfc.graph.successors(name)
            )
        self.actor_decisions = []
        return super().reset_sfc(sfc)

    def _candidate(self, node):
        candidate = super()._candidate(node)
        if candidate is None:
            return None
        vnf = self.vnf_sequence[self.current_vnf_idx]
        lower_bound = (
            candidate[4]
            + self.destination_distances.get(node, float("inf"))
            + self.minimum_tail[vnf.id]
        )
        return candidate if lower_bound <= self.lat_max + 1e-9 else None

    def get_observations(self):
        original = super().get_observations()
        max_ips = max(data.get("ips", 0) for _, data in self.graph.nodes(data=True)) or 1
        cloud = self.graph.nodes[self.cloud_node]
        try:
            source_distance = self._latency_distances(self.current_location).get(
                self.cloud_node, self.lat_max
            )
        except nx.NetworkXNoPath:
            source_distance = self.lat_max
        cloud_features = [
            cloud["cpu_used"] / max(1, cloud["cpu_capacity"]),
            cloud["cache_used"] / max(1, cloud["cache_capacity"]),
            source_distance / self.lat_max,
            self.destination_distances.get(self.cloud_node, self.lat_max) / self.lat_max,
            cloud.get("ips", 0) / max_ips,
            float(cloud.get("is_active", True)),
        ]
        for region, nodes in self.region_nodes.items():
            extra = []
            for node in nodes:
                extra.extend(
                    [
                        self.destination_distances.get(node, self.lat_max) / self.lat_max,
                        self.graph.nodes[node].get("ips", 0) / max_ips,
                    ]
                )
            extra.extend([0.0] * (2 * (self.max_nodes_per_region - len(nodes))))
            original[region] = np.asarray(
                [*original[region], *extra, *cloud_features], dtype=np.float32
            )
        return original

    def step(self, actions):
        name = self.vnf_sequence[self.current_vnf_idx].id
        proposals = {
            region: (
                self.cloud_node
                if action == self.max_nodes_per_region
                else self.region_nodes[region][action]
            )
            for region, action in actions.items()
            if action == self.max_nodes_per_region or 0 <= action < len(self.region_nodes[region])
        }
        result = super().step(actions)
        selected = self.allocated_nodes.get(name)
        if selected is not None and selected not in proposals.values():
            raise AssertionError("Selected host was not proposed by a trained actor")
        self.actor_decisions.append(
            {"vnf": name, "actions": actions, "proposals": proposals, "selected_host": selected}
        )
        return result

    def expert_actions(self):
        actions = {}
        masks = self.get_action_masks()
        vnf = self.vnf_sequence[self.current_vnf_idx]
        for region, nodes in self.region_nodes.items():
            ranked = []
            for action, node in enumerate([*nodes, self.cloud_node]):
                actual_action = self.max_nodes_per_region if node == self.cloud_node else action
                if not masks[region][actual_action]:
                    continue
                candidate = self._evaluated_candidate(node)
                reused = node_demand(self.graph, self.sfc, vnf, node, True) == (0, 0)
                score = (
                    candidate[4]
                    + self.destination_distances.get(node, self.lat_max)
                    + self.minimum_tail[vnf.id]
                    - 2.0 * reused
                )
                ranked.append((score, actual_action))
            actions[region] = min(ranked)[1] if ranked else 0
        return actions, masks


def dimensions(env):
    return {
        "n_agents": env.n_regions,
        "obs_dim": env.obs_dim,
        "action_dim": env.action_dim_per_agent,
        "state_dim": env.state_dim,
    }


class MUVRQwenPlanner(CloudLLMPlanner):
    @staticmethod
    def compact_telemetry(telemetry):
        columns = [
            "id",
            "cpu_free",
            "memory_free",
            "processing_speed_giga",
            "distance_to_source_ms",
            "distance_to_destination_ms",
        ]
        return {
            "sla_ms": telemetry["sla_ms"],
            "cpu_global_pct": round(telemetry["net_cpu_util_pct"], 2),
            "session_cpu_memory": [20, 15],
            "players": telemetry["players"],
            "memory_unit": "normalized",
            "columns": columns,
            "nodes": [
                [node[column] for column in columns] for node in telemetry["nodes_telemetry"]
            ],
            "cache_ready_nodes": telemetry["cache_ready_nodes"],
        }

    @staticmethod
    def response_schema(telemetry):
        known = [node["id"] for node in telemetry["nodes_telemetry"]]
        eligible = telemetry["cache_ready_nodes"]
        lists = {}
        for name, ids in [("avoid_nodes", known), ("priority_cache_nodes", eligible)]:
            lists[name] = {
                "type": "array",
                "maxItems": 3 if ids else 0,
                "items": {"type": "integer", **({"enum": ids} if ids else {})},
            }
        properties = {
            **lists,
            "latency_weight": {"type": "number", "minimum": 0, "maximum": 1},
            "load_balance_weight": {"type": "number", "minimum": 0, "maximum": 1},
            "sync_tolerance_factor": {"type": "number", "const": 1.0},
            "reasoning": {"type": "string", "maxLength": 120},
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    def _call_openai_compatible_api(self, telemetry):
        system = (
            "Você orienta atores MAPPO para aceitar sessões MUVR com SLA rígido de 6 ms. "
            "Use apenas IDs da telemetria. Velocidade de processamento maior é melhor; "
            "distâncias menores são melhores. Priorize latência quando a CPU estiver livre. "
            "Evite apenas gargalos evidentes. Não aumente recursos nem relaxe o SLA. "
            "Responda SOMENTE JSON, sem explicação extra. No máximo três IDs em cada lista "
            "e reasoning em português com no máximo 12 palavras. Schema: "
            '{"avoid_nodes":[],"priority_cache_nodes":[],"latency_weight":0.8,'
            '"load_balance_weight":0.2,"sync_tolerance_factor":1.0,"reasoning":"frase curta"}. '
            "Escolha os pesos entre zero e um, somando um; o exemplo não fixa os valores."
        )
        payload = {
            "model": self.model_name,
            "temperature": 0.1,
            "max_tokens": getattr(self.settings, "cognitive_max_tokens", 256),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "muvr_guidance",
                    "strict": True,
                    "schema": self.response_schema(telemetry),
                },
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(self.compact_telemetry(telemetry))},
            ],
        }
        endpoint = self.settings.cognitive_api_base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.cognitive_api_timeout
            ) as reply:
                result = json.loads(reply.read().decode())
            self.last_qwen_usage = result.get("usage", {})
            return CognitiveGuidance(**json.loads(result["choices"][0]["message"]["content"]))
        except (ValueError, KeyError, OSError) as error:
            self.last_qwen_error = str(error)
            return None

    def extract_telemetry_snapshot(self):
        net = self.substrate_network
        batch = self.current_batch
        graph = net.graph
        nodes = [
            {
                "id": node,
                "cpu_free": round(data["cpu_capacity"] - data["cpu_used"], 2),
                "memory_free": round(data["cache_capacity"] - data["cache_used"], 2),
                "processing_speed_giga": round(data["ips"] / 1e9, 2),
                "distance_to_source_ms": round(
                    nx.shortest_path_length(
                        graph, batch[0].src.get_substrate_node(), node, weight="latency"
                    ),
                    3,
                )
                if nx.has_path(graph, batch[0].src.get_substrate_node(), node)
                else None,
                "distance_to_destination_ms": round(
                    nx.shortest_path_length(graph, node, batch[0].dst_node, weight="latency"), 3
                )
                if nx.has_path(graph, node, batch[0].dst_node)
                else None,
            }
            for node, data in graph.nodes(data=True)
        ]
        return {
            "sla_ms": batch[0].latency_request,
            "players": len(batch),
            "cpu_per_distinct_session": 20,
            "memory_per_distinct_session": 15,
            "memory_unit": "normalized capacity units, not MB",
            "net_cpu_util_pct": net.metrics.total_cpu_used / net.total_cpu_capacity * 100,
            "nodes_telemetry": nodes,
            "high_cpu_nodes": [n["id"] for n in nodes if n["cpu_free"] < 25],
            "cache_ready_nodes": [
                n["id"] for n in nodes if n["memory_free"] >= 15 and n["cpu_free"] >= 20
            ],
            "instruction": "SLA 6 ms is strict. Prefer short paths and fast hosts. "
            "Give concise reasoning (one sentence); weights sum to one. "
            "No capacity increases, no waiting, no SLA relaxation.",
        }


class TrainedMUVRMAPPO:
    def __init__(self, settings, qwen=False):
        self.settings = settings
        self.qwen = qwen
        if qwen and (
            not settings.cognitive_planner or "qwen" not in settings.cognitive_llm_model.lower()
        ):
            raise ValueError("Qwen must be explicitly enabled with a Qwen model")
        self.model_path = Path(settings.mappo_model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained MAPPO checkpoint missing: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.checkpoint_dimensions = {
            key: checkpoint.get(key) for key in ("n_agents", "obs_dim", "action_dim", "state_dim")
        }
        if not (
            checkpoint.get("training_steps", 0) > 0 and checkpoint.get("MAPPO_updates", 0) > 0
        ):
            raise ValueError("Checkpoint lacks evidence of completed training")
        self.trainer = MAPPOTrainer(
            **{key: checkpoint[key] for key in ("n_agents", "obs_dim", "action_dim", "state_dim")},
            device="cpu",
        )
        self.trainer.load(self.model_path)
        for actor in self.trainer.actors:
            actor.eval()
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.guidance = None
        self.planning_records = []
        self.request_evidence = []
        self.call_results = []
        self.executor = ThreadPoolExecutor(max_workers=1) if qwen else None
        self.pending = None
        self.last_submission = None
        self.guidance_ready_at = None
        self.guidance_applied_requests = 0
        self.guidance_misses = 0
        self.background_skipped_busy = 0
        self.closed = False

    def update_cognitive_guidance(self, guidance):
        self.guidance = guidance

    def _background_plan(self, net, batch, now):
        # This worker owns snapshots; it never modifies the live network.
        planner = MUVRQwenPlanner(self.settings, net)
        planner.current_batch = batch
        calls = []
        original = planner._call_openai_compatible_api

        def tracked(telemetry):
            result = original(telemetry)
            calls.append(result is not None)
            return result

        planner._call_openai_compatible_api = tracked
        tick = time.perf_counter()
        try:
            guidance = planner.generate_guidance()
            success = bool(calls and calls[-1])
            error = None if success else "Qwen unavailable; generated fallback discarded"
        except Exception as exc:
            guidance, success, error = None, False, f"{type(exc).__name__}: {exc}"
        ready_at = time.perf_counter()
        return (
            guidance if success else None,
            ready_at,
            {
                "virtual_time": now,
                "qwen_success": success,
                "usage": getattr(planner, "last_qwen_usage", {}),
                "planning_seconds": ready_at - tick,
                "guidance": guidance.model_dump() if success else None,
                "error": error,
            },
        )

    def _collect_background(self):
        if self.pending is None or not self.pending.done():
            return
        try:
            guidance, ready_at, record = self.pending.result()
        except Exception as exc:
            guidance, ready_at = None, time.perf_counter()
            record = {
                "qwen_success": False,
                "planning_seconds": ready_at - self.pending_started_at,
                "error": f"{type(exc).__name__}: {exc}",
                "guidance": None,
            }
        self.pending = None
        self.call_results.append(record["qwen_success"])
        self.planning_records.append(record)
        if guidance is not None:
            self.guidance, self.guidance_ready_at = guidance, ready_at

    def prepare(self, net, batch):
        """Schedule Qwen outside timed admission; never wait for its response."""
        if not self.qwen or self.closed:
            return
        self._collect_background()
        now = batch[0].arrival_time
        if self.last_submission is not None and (
            now - self.last_submission < self.settings.cognitive_planner_interval
        ):
            return
        if self.pending is not None:
            self.background_skipped_busy += 1
            return
        self.pending_started_at = time.perf_counter()
        self.pending = self.executor.submit(
            self._background_plan, copy.deepcopy(net), copy.deepcopy(batch), now
        )
        self.last_submission = now

    def close(self):
        """Drain the worker after the simulation, never during admission."""
        if self.executor and not self.closed:
            self.executor.shutdown(wait=True)
            self._collect_background()
        self.closed = True

    def _search_policy(self, graph, sfc):
        exclusions = [{}]
        prefix_exclusions = []
        visited = set()
        decisions = []
        last_env = None
        search_summary = []
        baseline_budget = min(8, self.settings.mappo_search_attempts)
        while len(search_summary) < self.settings.mappo_search_attempts:
            use_prefix = len(search_summary) >= baseline_budget or not exclusions
            queue = prefix_exclusions if use_prefix else exclusions
            if not queue:
                break
            excluded = queue.pop(0)
            signature = repr(sorted((name, sorted(nodes)) for name, nodes in excluded.items()))
            if signature in visited:
                continue
            visited.add(signature)
            attempt = len(search_summary)
            env = MUVRMAPPOEnv(graph, sfc, self.settings.n_mappo_regions, self.guidance)
            if dimensions(env) != self.checkpoint_dimensions:
                raise ValueError("MAPPO checkpoint dimensions do not match this topology")
            obs = env.get_observations()
            env.excluded_nodes = excluded
            done = False
            while not done:
                masks = env.get_action_masks()
                feasible_actions = {region: int(mask.sum()) for region, mask in masks.items()}
                actions, log_probs = self.trainer.select_actions(
                    obs, masks, deterministic=attempt == 0
                )
                obs, _, _, done, _ = env.step(actions)
                env.actor_decisions[-1]["action_log_probabilities"] = log_probs
                env.actor_decisions[-1]["feasible_actions_per_actor"] = feasible_actions
            for record in env.actor_decisions:
                record["search_attempt"] = attempt + 1
                record["deterministic"] = attempt == 0
            decisions.extend(env.actor_decisions)
            last_env = env
            search_summary.append(
                {
                    "attempt": attempt + 1,
                    "strategy": "fixed_prefix" if use_prefix else "original",
                    "success": bool(env.success),
                    "allocated_functions": len(env.allocated_nodes),
                    "failed_vnf": None if env.success else env.actor_decisions[-1]["vnf"],
                    "failure_reason": env.fail_reason,
                    "no_feasible_masked_action": not any(feasible_actions.values()),
                }
            )
            env.search_summary = search_summary
            if env.success:
                return env, decisions
            # Keep the successful prefix fixed, then forbid its next chosen host.
            # Actors still propose every host; no heuristic host is injected.
            allocated = list(env.allocated_nodes.items())
            alternatives = []
            for index in reversed(range(len(allocated))):
                name, node = allocated[index]
                alternative = copy.deepcopy(excluded)
                for prefix_name, prefix_node in allocated[:index]:
                    alternative[prefix_name] = set(graph) - {prefix_node}
                alternative.setdefault(name, set()).add(node)
                if set(graph) - alternative[name]:
                    alternatives.append(alternative)
            if use_prefix:
                prefix_exclusions = alternatives + prefix_exclusions
            else:
                prefix_exclusions.extend(alternatives)
                # Preserve the original proposal sequence before trying new branches.
                for name, node in reversed(allocated):
                    alternative = copy.deepcopy(excluded)
                    alternative.setdefault(name, set()).add(node)
                    exclusions.append(alternative)
        return last_env, decisions

    def __call__(self, net, batch, _strategy):
        qwen_seconds = 0.0
        self._collect_background()
        if self.guidance_ready_at is not None and (
            time.perf_counter() - self.guidance_ready_at > self.settings.cognitive_guidance_max_age
        ):
            self.guidance = None
        if self.qwen:
            self.guidance_applied_requests += self.guidance is not None
            self.guidance_misses += self.guidance is None
        trial = copy.deepcopy(net)
        latencies, decisions, searches = [], [], []
        for player_index, original in enumerate(batch):
            sfc = copy.deepcopy(original)
            torch.manual_seed(
                self.settings.seed * 100000 + int(sfc.session_id) * 100 + player_index
            )
            env, trace = self._search_policy(trial.graph, sfc)
            decisions.extend(trace)
            searches.append({"player_index": player_index, "attempts": env.search_summary})
            if not env.success:
                self.request_evidence.append(
                    {
                        "session": sfc.session_id,
                        "qwen_seconds": qwen_seconds,
                        "guidance_used": self.guidance is not None,
                        "guidance_cycle": len(self.planning_records),
                        "actor_decisions": decisions,
                        "search_diagnostics": searches,
                        "accepted": False,
                    }
                )
                return None, [], "trained_mappo_no_feasible_proposal"
            sfc.edge_routes = env.edge_routes
            sfc.edge_latencies = env.edge_latencies
            sfc.comp_latencies = env.comp_latencies
            trial.deploy_sfc(sfc, env.route_info)
            latencies.append(env.total_latency)
        self.request_evidence.append(
            {
                "session": batch[0].session_id,
                "qwen_seconds": qwen_seconds,
                "guidance_used": self.guidance is not None,
                "guidance_cycle": len(self.planning_records),
                "actor_decisions": decisions,
                "search_diagnostics": searches,
                "accepted": True,
            }
        )
        return trial, latencies, "trained_mappo_qwen" if self.qwen else "trained_mappo"

    def evidence(self):
        return {
            "model_sha256": self.model_sha256,
            "checkpoint_loaded": True,
            "checkpoint_dimensions": self.checkpoint_dimensions,
            "policy_candidates_only": True,
            "max_internal_search_attempts": self.settings.mappo_search_attempts,
            "search_sampling": "first deterministic, alternatives from trained distributions",
            "search_branching": "original_first_8_then_fixed_prefix_host_exclusions",
            "qwen_enabled": self.qwen,
            "planning_mode": "background_nonblocking",
            "guidance_applied_requests": self.guidance_applied_requests,
            "guidance_missing_or_expired_requests": self.guidance_misses,
            "background_skipped_busy": self.background_skipped_busy,
            "background_pending": self.pending is not None,
            "qwen_successful_calls": sum(self.call_results),
            "qwen_failed_calls": len(self.call_results) - sum(self.call_results),
            "qwen_fallback_calls": 0,
            "qwen_planning_seconds": sum(r["planning_seconds"] for r in self.planning_records),
            "planning_records": self.planning_records,
            "request_evidence": self.request_evidence,
        }
