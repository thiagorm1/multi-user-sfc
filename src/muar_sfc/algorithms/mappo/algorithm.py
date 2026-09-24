from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

from muar_sfc.algorithms.algorithm import Algorithm
from muar_sfc.algorithms.mappo.env_mappo import MAPPO_SFC_Env
from muar_sfc.algorithms.mappo.mappo_core import MAPPOTrainer
from muar_sfc.config import ROOT_DIR
from muar_sfc.core.sfc import SFC


class MAPPOAlgorithm(Algorithm):
    """
    Algoritmo de Orquestração Cloud-Edge Multi-Agente MAPPO com CTDE.
    Executa inferência descentralizada em tempo real (< 5 ms) respeitando
    restrições de CPU, Cache, Banda e Sincronização Inter-Modal (DAG).
    """

    def __init__(
        self,
        name: str = "MAPPO",
        n_regions: int = 4,
        cloud_node: int | str = 0,
        model_path: str | Path | None = None,
    ):
        self.name = name
        self.n_regions = n_regions
        self.cloud_node = cloud_node

        default_model = ROOT_DIR / "rl_saved_models" / "MAPPO_allocation_model.pt"
        self.model_path = Path(model_path) if model_path else default_model

        self.graph: nx.Graph | None = None
        self.sfc: SFC | None = None
        self.route_info: dict[str, list[Any]] = {}
        self.node_info: dict[str, Any] = {}
        self.latency: float | None = None
        self.fail_reason: str | None = None
        self.trainer: MAPPOTrainer | None = None
        self.env: MAPPO_SFC_Env | None = None
        self.sync_differential: float = 0.0
        self.cognitive_guidance: Any = None

    def update_cognitive_guidance(self, guidance: Any) -> None:
        """Recebe e propaga as diretrizes de alto nível do CloudLLMPlanner."""
        self.cognitive_guidance = guidance
        if self.env is not None:
            self.env.set_cognitive_guidance(guidance)

    def clear_all(self):
        """Limpa o estado da simulação."""
        self.graph = None
        self.sfc = None
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.fail_reason = None
        self.sync_differential = 0.0

    def install_substrate_network(self, substrate_network: Any) -> Any:
        """Instala a rede física (ou grafo shadow)."""
        if isinstance(substrate_network, nx.Graph):
            self.graph = substrate_network
        elif hasattr(substrate_network, "graph"):
            self.graph = substrate_network.graph
        else:
            self.graph = substrate_network
        return self.graph

    def install_SFC(self, sfc: SFC) -> SFC:
        """Instala a requisição de SFC ou DAG."""
        self.sfc = sfc
        self.route_info = {}
        self.node_info = {}
        self.latency = None
        self.fail_reason = None
        self.sync_differential = 0.0
        return self.sfc

    def get_latency(self) -> float:
        """Retorna a latência calculada."""
        return self.latency if self.latency is not None else 0.0

    def get_route_info(self) -> dict:
        """Retorna os caminhos de roteamento das VNFs."""
        return self.route_info

    def get_fail_reason(self) -> str | None:
        """Retorna o motivo da falha caso tenha ocorrido."""
        return self.fail_reason

    def handle_failure(self):
        """Reseta rota e latência em caso de falha."""
        self.route_info = {}
        self.latency = None

    def check_solution(self) -> bool:
        """Valida se a solução de roteamento é consistente."""
        if self.latency is None or not self.route_info:
            return False

        # Verifica se o tempo de latência atende ao SLA
        max_lat = self.sfc.get_latency_request() if self.sfc else 50.0
        if max_lat and self.latency > max_lat:
            self.fail_reason = f"Latência ({self.latency:.2f} ms) excedeu SLA ({max_lat} ms)"
            return False

        return True

    def _ensure_trainer_initialized(self, env: MAPPO_SFC_Env):
        """Garante que o modelo MAPPO está instanciado e carregado se os pesos existirem."""
        if self.trainer is None:
            self.trainer = MAPPOTrainer(
                n_agents=env.n_regions,
                obs_dim=env.obs_dim,
                action_dim=env.action_dim_per_agent,
                state_dim=env.state_dim,
                device="cpu",
            )
            if self.model_path.exists():
                try:
                    import torch
                    ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
                    if (
                        ckpt.get("obs_dim") == env.obs_dim
                        and ckpt.get("action_dim") == env.action_dim_per_agent
                        and ckpt.get("n_agents") == env.n_regions
                    ):
                        self.trainer.load(self.model_path)
                        logger.debug(f"Pesos do MAPPO carregados de {self.model_path}")
                    else:
                        logger.debug("Dimensões do checkpoint diferem; reinicializando pesos.")
                except Exception as e:
                    logger.warning(f"Não foi possível carregar pesos de {self.model_path}: {e}.")
            else:
                logger.debug(f"Arquivo {self.model_path} inexistente. Inicializando pesos padrão.")

    def start_algorithm(self, env=None, args=None) -> bool:
        """Ponto de entrada do algoritmo de orquestração."""
        if self.graph is None or self.sfc is None:
            self.fail_reason = "Grafo ou SFC não instalados."
            self.handle_failure()
            return False

        sync_tol = getattr(self.sfc, "sync_tolerance", 5.0)
        if self.cognitive_guidance:
            factor = getattr(self.cognitive_guidance, "sync_tolerance_factor", 1.0)
            sync_tol *= factor

        self.env = MAPPO_SFC_Env(
            graph=self.graph,
            sfc=self.sfc,
            n_regions=self.n_regions,
            cloud_node=self.cloud_node,
            sync_tolerance=sync_tol,
            is_training=False,
            cognitive_guidance=self.cognitive_guidance,
        )

        self._ensure_trainer_initialized(self.env)
        return self.algorithm(self.graph, self.sfc)

    def algorithm(self, substrate_network: Any, sfc: SFC) -> bool:
        """Lógica principal de inferência multi-agente descentralizada."""
        obs, state = self.env.reset_sfc(sfc)
        done = False
        step_count = 0
        max_steps = len(self.env.vnf_sequence) + 5

        while not done and step_count < max_steps:
            action_masks = self.env.get_action_masks()
            # Inferência Descentralizada: atores locais avaliam obs com action mask
            actions, _ = self.trainer.select_actions(obs, action_masks, deterministic=True)
            obs, state, reward, done, info = self.env.step(actions)
            step_count += 1

        if not self.env.success:
            self.fail_reason = self.env.fail_reason or "Alocação multi-agente não convergiu."
            self.handle_failure()
            return False

        self.route_info = self.env.route_info
        self.latency = round(self.env.total_latency, 2)

        if not self.check_solution():
            self.handle_failure()
            return False

        return True
