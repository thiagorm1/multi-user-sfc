import json
import time
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class CognitiveGuidance(BaseModel):
    """
    Diretrizes estruturadas de alto nível emitidas pelo Orquestrador Cognitivo em Nuvem (LLM).
    São enviadas assincronamente aos controladores locais e agentes MAPPO.
    """

    avoid_nodes: list[int | str | float] = Field(
        default_factory=list,
        description="Nós que estão saturados ou com alto risco de gargalo e devem ser evitados.",
    )
    priority_cache_nodes: list[int | str | float] = Field(
        default_factory=list,
        description="Nós recomendados para hospedar funções de cache/agregação (ex: MA_region).",
    )
    latency_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Peso de priorização da latência na função de recompensa (0.0 a 1.0).",
    )
    load_balance_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Peso de priorização do balanceamento de carga e aceitação (0.0 a 1.0).",
    )
    sync_tolerance_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Fator multiplicador da tolerância de sincronização inter-modal.",
    )
    reasoning: str = Field(
        default="",
        description="Racional e justificativa da decisão macro do orquestrador cognitivo.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Timestamp de geração da diretriz.",
    )


class CloudLLMPlanner:
    """
    Orquestrador Cognitivo em Nuvem baseado em LLM (Macro-Planner).
    Atua no plano de gerenciamento lento (janela periódica de T segundos),
    analisando a telemetria global da infraestrutura e emitindo diretrizes
    cognitivas para os agentes de execução em tempo real (MAPPO/CTDE).
    """

    def __init__(
        self,
        settings: Any,
        substrate_network: Any = None,
        algorithm: Any = None,
    ):
        self.settings = settings
        self.substrate_network = substrate_network
        self.algorithm = algorithm

        self.activated = getattr(settings, "cognitive_planner", False)
        self.planner_interval = getattr(settings, "cognitive_planner_interval", 30.0)
        self.model_name = getattr(settings, "cognitive_llm_model", "mock")
        self.api_key = getattr(settings, "cognitive_api_key", "")

        self.last_planning_time = None
        self.planning_counter = 0
        self.current_guidance = CognitiveGuidance()

        logger.info(
            f"[CloudLLMPlanner] Inicializado (Ativo: {self.activated}, "
            f"Intervalo: {self.planner_interval}s, Modelo: {self.model_name})"
        )

    def extract_telemetry_snapshot(self) -> dict[str, Any]:
        """Extrai um snapshot da telemetria e estado de saturação da rede substrata."""
        if self.substrate_network is None:
            return {"status": "no_network"}

        net = self.substrate_network
        graph = getattr(net, "graph", net)

        nodes_telemetry = []
        high_cpu_nodes = []
        cache_ready_nodes = []

        total_cpu_cap = 0.0
        total_cpu_used = 0.0

        for node_id, data in graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            level_server = data.get("level_server", "")
            if node_type in ["mobile_device", "router"]:
                continue

            cpu_cap = float(data.get("cpu_capacity", 1.0))
            cpu_used = float(data.get("cpu_used", 0.0))
            cache_cap = float(data.get("cache_capacity", 0.0))
            cache_used = float(data.get("cache_used", 0.0))

            total_cpu_cap += cpu_cap
            total_cpu_used += cpu_used

            cpu_util = cpu_used / max(1.0, cpu_cap)
            cache_free = max(0.0, cache_cap - cache_used)

            info = {
                "id": node_id,
                "type": node_type,
                "level": level_server,
                "cpu_util_pct": round(cpu_util * 100, 1),
                "cache_free_mb": round(cache_free, 1),
                "cpu_free": max(0.0, cpu_cap - cpu_used),
                "available": cpu_cap > 0,
            }
            nodes_telemetry.append(info)

            # Detecta saturação proativamente a partir de 75% de CPU.
            if cpu_util >= 0.75 or cpu_cap <= 0:
                high_cpu_nodes.append(node_id)

            # Identifica nós de borda aptos a hospedar cache regional
            is_edge = (node_type == "server_edge") or (
                node_type == "server" and level_server != "cloud"
            )
            if is_edge and cpu_cap > 0 and cache_free >= 150.0 and cpu_util < 0.70:
                cache_ready_nodes.append(node_id)

        # Telemetria de Enlaces Críticos (Identifica pontes e gargalos de banda)
        congested_links = []
        for u, v, d in graph.edges(data=True):
            bw_cap = float(d.get("bandwidth_capacity", 100.0))
            bw_used = float(d.get("bandwidth_used", 0.0))
            bw_free = max(0.0, bw_cap - bw_used)
            util = (bw_used / max(1.0, bw_cap)) * 100.0
            if util >= 75.0 or bw_free < 20.0:
                congested_links.append(
                    {"link": [u, v], "util_pct": round(util, 1), "free_mbps": round(bw_free, 1)}
                )

        net_cpu_pct = (total_cpu_used / total_cpu_cap * 100) if total_cpu_cap > 0 else 0.0

        return {
            "net_cpu_util_pct": round(net_cpu_pct, 2),
            "high_cpu_nodes": high_cpu_nodes,
            "cache_ready_nodes": cache_ready_nodes,
            "congested_links": congested_links[:10],
            "total_nodes_monitored": len(nodes_telemetry),
            "nodes_telemetry": nodes_telemetry,
        }

    def _solve_cognitive_heuristic(self, telemetry: dict[str, Any]) -> CognitiveGuidance:
        """
        Motor analítico/SLM local determinístico.
        Gera diretrizes otimizadas com base nas métricas extraídas sem requisições de rede.
        """
        high_cpu_nodes = telemetry.get("high_cpu_nodes", [])
        cache_ready_nodes = telemetry.get("cache_ready_nodes", [])
        net_cpu_pct = telemetry.get("net_cpu_util_pct", 0.0)

        # Regras de Meta-Política Dinâmica Equilibrada
        if high_cpu_nodes or net_cpu_pct > 60.0:
            # Sob alta saturação: equilíbrio estrito (não abandona a latência)
            lat_w = 0.45
            load_w = 0.55
            sync_factor = 1.15
            strategy = (
                f"Saturação detectada em {len(high_cpu_nodes)} nós. "
                "Balanceando carga com conservação da latência de propagação."
            )
        elif net_cpu_pct > 25.0:
            # Carga moderada: equilíbrio com leve prioridade à proximidade
            lat_w = 0.60
            load_w = 0.40
            sync_factor = 1.05
            strategy = "Carga moderada. Operação balanceada com foco em baixa latência."
        else:
            # Baixa carga: foca em ultra-baixa latência e colocação hiper-local
            lat_w = 0.75
            load_w = 0.25
            sync_factor = 0.95
            strategy = "Rede desafogada. Maximizando proximidade e ultra-baixa latência."

        reasoning = (
            f"Ciclo Cognitivo #{self.planning_counter}: {strategy} "
            f"CPU Global={net_cpu_pct}%. Nós com restrição: {high_cpu_nodes}. "
            f"Nós de cache prioritários: {cache_ready_nodes[:4]}."
        )

        return CognitiveGuidance(
            avoid_nodes=list(high_cpu_nodes),
            priority_cache_nodes=list(cache_ready_nodes[:4]),
            latency_weight=lat_w,
            load_balance_weight=load_w,
            sync_tolerance_factor=sync_factor,
            reasoning=reasoning,
            timestamp=time.time(),
        )

    def _call_openai_compatible_api(self, telemetry: dict[str, Any]) -> CognitiveGuidance | None:
        """
        Executa chamada a endpoints padrão OpenAI (Ollama, vLLM, LMStudio, Groq, OpenRouter).
        Compatível diretamente com Qwen 2.5 e Llama 3 locais ou em nuvem.
        """
        try:
            import urllib.request

            base_url = (
                getattr(self.settings, "cognitive_api_base_url", "") or "http://localhost:11434/v1"
            )
            endpoint = f"{base_url.rstrip('/')}/chat/completions"

            system_prompt = (
                "Maximize a aceitação de SFCs preservando CPU, cache, banda e SLA de latência. "
                "Use apenas IDs presentes na telemetria. Evitar nós é uma preferência, "
                "não uma proibição de alocar. Priorize capacidade livre sob congestionamento. "
                "Escreva reasoning em português. "
                "Responda EXCLUSIVAMENTE em formato JSON com o seguinte schema:\n"
                "{\n"
                '  "avoid_nodes": [int],\n'
                '  "priority_cache_nodes": [int],\n'
                '  "latency_weight": float (0.0 a 1.0),\n'
                '  "load_balance_weight": float (0.0 a 1.0),\n'
                '  "sync_tolerance_factor": float (0.5 a 2.0),\n'
                '  "reasoning": string\n'
                "}"
            )
            user_content = f"Telemetria Atual da Infraestrutura:\n{json.dumps(telemetry)}"

            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
            }
            payload["max_tokens"] = getattr(self.settings, "cognitive_max_tokens", 1024)
            payload["response_format"] = {"type": "json_object"}

            data = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(
                req, timeout=getattr(self.settings, "cognitive_api_timeout", 30.0)
            ) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                msg = res_body["choices"][0]["message"]["content"]
                parsed = json.loads(msg)
                return CognitiveGuidance(**parsed)

        except Exception as e:
            logger.warning(
                f"[CloudLLMPlanner] Falha ao consultar endpoint ({self.model_name}): {e}. "
                "Recorrendo ao motor heurístico."
            )
            return None

    def _call_gemini_api(self, telemetry: dict[str, Any]) -> CognitiveGuidance | None:
        """Executa chamada à API Gemini com Structured Outputs para orquestração."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            prompt = (
                "Você é o Orquestrador Cognitivo em Nuvem de uma rede 6G Metaverso/XR. "
                "Com base na telemetria atual da infraestrutura, defina as diretrizes para "
                "os agentes locais MAPPO alocarem SFCs em DAG com sincronização inter-modal.\n\n"
                f"Telemetria Atual:\n{json.dumps(telemetry, indent=2)}\n\n"
                "Emita a resposta rigorosamente no schema CognitiveGuidance."
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CognitiveGuidance,
                    temperature=0.2,
                ),
            )

            if response.text:
                guidance_dict = json.loads(response.text)
                return CognitiveGuidance(**guidance_dict)

        except Exception as e:
            logger.warning(
                f"[CloudLLMPlanner] Falha na chamada da API LLM ({e}). "
                "Recorrendo ao motor analítico cognitivo de fallback."
            )
        return None

    def generate_guidance(self) -> CognitiveGuidance:
        """Gera novas diretrizes cognitivas utilizando o modelo configurado."""
        self.planning_counter += 1
        telemetry = self.extract_telemetry_snapshot()

        guidance = None
        api_base = getattr(self.settings, "cognitive_api_base_url", "")
        if self.model_name != "mock":
            if "gemini" in self.model_name.lower():
                guidance = self._call_gemini_api(telemetry)
            elif (
                "qwen" in self.model_name.lower() or "llama" in self.model_name.lower() or api_base
            ):
                guidance = self._call_openai_compatible_api(telemetry)
            elif self.api_key:
                guidance = self._call_gemini_api(telemetry)

        if guidance is None:
            guidance = self._solve_cognitive_heuristic(telemetry)

        known_nodes = {n["id"] for n in telemetry.get("nodes_telemetry", [])}
        guidance.avoid_nodes = [n for n in guidance.avoid_nodes if n in known_nodes]
        ready_nodes = set(telemetry.get("cache_ready_nodes", []))
        guidance.priority_cache_nodes = [
            n for n in guidance.priority_cache_nodes if n in ready_nodes
        ]
        weight_sum = guidance.latency_weight + guidance.load_balance_weight
        if weight_sum > 0:
            guidance.latency_weight /= weight_sum
            guidance.load_balance_weight /= weight_sum
        else:
            guidance.latency_weight = guidance.load_balance_weight = 0.5
        self.current_guidance = guidance

        logger.info(
            f"🧠 [CloudLLMPlanner] Novas Diretrizes Emitidas (Ciclo #{self.planning_counter}): "
            f"Evitar={guidance.avoid_nodes}, Cache={guidance.priority_cache_nodes}, "
            f"LatWeight={guidance.latency_weight}, LoadWeight={guidance.load_balance_weight}"
        )
        logger.debug(f"[CloudLLMPlanner] Justificativa: {guidance.reasoning}")

        return guidance

    def apply_guidance_to_algorithm(self, algorithm: Any = None) -> None:
        """Aplica as diretrizes cognitivas atuais ao algoritmo de orquestração ativo."""
        alg = algorithm or self.algorithm
        if alg and hasattr(alg, "update_cognitive_guidance"):
            alg.update_cognitive_guidance(self.current_guidance)

    def step_planning(
        self, current_time: float, algorithm: Any = None
    ) -> CognitiveGuidance | None:
        """
        Verifica periodicamente a janela temporal e aciona o ciclo de planejamento cognitivo.
        Pode ser chamado no loop de controle da rede substrata.
        """
        if not self.activated:
            return None

        if (
            self.last_planning_time is None
            or (current_time - self.last_planning_time) >= self.planner_interval
        ):
            guidance = self.generate_guidance()
            self.last_planning_time = current_time
            self.apply_guidance_to_algorithm(algorithm)
            return guidance

        return None
