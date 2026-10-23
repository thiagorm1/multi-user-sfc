"""Measure trained MAPPO, with and without real Qwen, on paired MUVR sessions."""

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger

from muar_sfc.algorithms.mappo.muvr import TrainedMUVRMAPPO
from muar_sfc.config import SimulationSettings
from muar_sfc.core.scenarios.muvr_reference import MUVRProfile, simulate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--sla-ms", type=float, default=6.0,
                        help="SLA de comunicação + processamento em ms (referência: 6).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--variants", nargs="+", choices=["mappo", "mappo_qwen"], default=["mappo", "mappo_qwen"]
    )
    parser.add_argument("--model", default="rl_saved_models/MAPPO_muvr_model.pt")
    parser.add_argument("--planner-interval", type=float, default=60)
    parser.add_argument("--output", type=Path, default=Path("results/mappo_qwen_muvr_async"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--qwen-model", default="qwen2.5:1.5b")
    parser.add_argument("--api-base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--guidance-max-age", type=float, default=120)
    parser.add_argument("--search-attempts", type=int, default=16)
    parser.add_argument("--load-balance-scale", type=float, default=2.0)
    parser.add_argument("--sumo-net", type=Path, default=None,
                        help="Enable SUMO mobility using this road network XML.")
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--mobility-step-s", type=float, default=1.0)
    parser.add_argument("--policy-schema", choices=["muvr_static_v1", "muvr_sumo_v1"],
                        default="muvr_static_v1")
    parser.add_argument("--toporag-enabled", action="store_true")
    parser.add_argument("--toporag-top-k", type=int, default=4)
    parser.add_argument("--toporag-kappa", type=float, default=0.0)
    parser.add_argument(
        "--toporag-confidence-mode", choices=["fixed", "lagged_expost"], default="fixed"
    )
    parser.add_argument("--toporag-kappa-max", type=float, default=0.9)
    parser.add_argument("--toporag-prior-scale", type=float, default=1.0)
    parser.add_argument("--toporag-prompt-context", action="store_true")
    args = parser.parse_args(argv)
    if args.policy_schema == "muvr_sumo_v1" and args.sumo_net is None:
        parser.error("Mobile policy schema requires --sumo-net")
    if args.policy_schema == "muvr_sumo_v1" and args.search_attempts != 1:
        parser.error("Mobile policy evaluation requires --search-attempts 1")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    logger.remove()
    logger.add(args.output / "qwen.log", level="INFO")
    profile = MUVRProfile(requests=args.requests, sla=args.sla_ms)
    settings = SimulationSettings(
        _env_file=None,
        application="muvr_reference",
        alg="mappo",
        latency=args.sla_ms,
        n_mappo_regions=4,
        mappo_model_path=args.model,
        share=True,
        shareband=True,
        cognitive_planner=True,
        cognitive_llm_model=args.qwen_model,
        cognitive_api_base_url=args.api_base_url,
        cognitive_api_timeout=args.api_timeout,
        cognitive_max_tokens=args.max_tokens,
        cognitive_planner_interval=args.planner_interval,
        cognitive_guidance_max_age=args.guidance_max_age,
        mappo_search_attempts=args.search_attempts,
        mappo_load_balance_scale=args.load_balance_scale,
        **{key: value for key, value in vars(args).items() if key.startswith("toporag_")},
    )
    for variant in args.variants:
        document = {
            "profile": asdict(profile),
            "variant": variant,
            "execution_parameters": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "rows": [],
            "method": "Trained MAPPO actors propose the only eligible hosts. "
            "Qwen periodically supplies ranking weights/node preferences when enabled. "
            "No heuristic rescue, no random checkpoint fallback, no queue or resource scaling. "
            "Qwen runs in a detached background worker; missing guidance uses MAPPO only. "
            "Submission and admission response times are reported separately. "
            "Network SLA excludes orchestration; AR display latency is not simulated. "
            "Opt-in SUMO keeps the original source fixed and maps mobile destinations "
            "to spatial anchors; make-before-break failure terminates the whole session. "
            "No radio, packet interruption duration or physical migration time is simulated.",
        }
        for seed in args.seeds:
            adapter = TrainedMUVRMAPPO(
                settings.model_copy(update={"seed": seed}), qwen=variant == "mappo_qwen",
                policy_schema=args.policy_schema
            )
            try:
                mobility_context = nullcontext(None)
                if args.sumo_net is not None:
                    from muar_sfc.core.scenarios.sumo_mobility import SumoMobility

                    mobility_context = SumoMobility(
                        args.sumo_net, seed, profile.servers, profile.requests, profile.players,
                        profile.arrival_interval, profile.lifetime, binary=args.sumo_binary,
                        step_s=args.mobility_step_s,
                    )
                with mobility_context as mobility:
                    row = simulate(profile, seed, variant, admitter=adapter, mobility=mobility)
            finally:
                adapter.close()
            row["inference_evidence"] = adapter.evidence()
            document.setdefault("toporag_manifest", {})[str(seed)] = {
                "parameters": adapter.toporag_parameters(),
                "state_hash": adapter.toporag_memory.state_hash(),
                "memory_size": len(adapter.toporag_memory),
            }
            document["rows"].append(row)
            (args.output / f"{variant}.json").write_text(
                json.dumps(document, indent=2, allow_nan=False)
            )
            evidence = row["inference_evidence"]
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in [
                            "seed",
                            "strategy",
                            "accepted",
                            "acceptance_percent",
                            "mean_latency_ms",
                            "max_latency_ms",
                            "mean_decision_s",
                        ]
                    }
                    | {
                        k: evidence[k]
                        for k in [
                            "checkpoint_loaded",
                            "qwen_successful_calls",
                            "qwen_fallback_calls",
                            "qwen_planning_seconds",
                        ]
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
