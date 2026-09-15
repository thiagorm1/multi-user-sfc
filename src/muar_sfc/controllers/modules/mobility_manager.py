import threading

# Imports condicionais ou mocks poderiam ser usados aqui,
# mas manteremos a estrutura original para simplicidade.
from muar_sfc.core.net_v2 import Net2
from muar_sfc.sumo.luxembourg.luxembourg_trace import Sumo_Luxembourg
from muar_sfc.sumo.tracer_instantiator import TracerInstantiator


class MobilityManager:
    def __init__(self, args):
        """
        Inicializa o MobilityManager.
        Se args.mobility != 'y', o tracer será None, mas o manager continuará funcional
        para registro estático de jogadores.
        """
        self.activated = args.mobility == "y"

        # O Tracer é opcional. Se não houver mobilidade, ele é None.
        self.tracer: Sumo_Luxembourg | None = (
            TracerInstantiator().instantiate_tracer(args.topology) if self.activated else None
        )

        self.players_tracker = {}
        self.running_sfcs = []
        self.lock = threading.Lock()

    def start_simulation(self):
        """Inicia a simulação apenas se o tracer estiver ativo."""
        if self.tracer:
            self.tracer.start_simulation()

    def stop_simulation(self):
        """Encerra a simulação apenas se o tracer estiver ativo."""
        if self.tracer:
            self.tracer.stop_simulation()

    def get_md_distance_from_router(self, id: str, router: str) -> float:
        """
        Retorna a distância. Se a mobilidade estiver desligada, retorna 0.0
        (assumindo co-localização ou irrelevância).
        Isso evita que cálculos matemáticos no Controller quebrem.
        """
        if self.tracer is None:
            return 0.0

        return self.tracer.get_server_distance_from_car(id, router)

    def add_player(self, group_id: str, closer_router: str, sfc_id_list: list):
        """
        Adiciona um jogador.

        BOA PRÁTICA: O estado do jogador é registrado no 'players_tracker' INDEPENDENTE
        da existência do SUMO. Isso desacopla a lógica de negócio (gerenciar players)
        da lógica de infraestrutura (simular física).
        """
        if group_id in self.players_tracker:
            self.players_tracker[group_id]["redeploying"] = False
            return

        # 1. Atualiza estado lógico (Sempre acontece)
        self.players_tracker[group_id] = {
            "sfc_list": sfc_id_list,
            "connected_router": closer_router,
            "redeploying": False,
        }

        # 2. Atualiza estado físico (Apenas se houver tracer)
        if self.tracer:
            self.tracer.create_vehicle(group_id, closer_router)

    def remove_player(self, player_id: str):
        """Remove o jogador do registro lógico e, se aplicável, da simulação física."""
        if player_id not in self.players_tracker:
            # Log de aviso seria ideal aqui, mas lançar erro pode ser excessivo em produção
            # raise ValueError(f'Veh do player {player_id} não encontrado na simulação')
            return

        # 1. Remove da simulação física
        if self.tracer:
            self.tracer.disconnect_vehicle(player_id)

        # 2. Remove do registro lógico
        del self.players_tracker[player_id]

    def mark_vehicle_as_redeploying(self, vehicle_id: str):
        if vehicle_id in self.players_tracker:
            self.players_tracker[vehicle_id]["redeploying"] = True

    def check_all_vehicles_position_changes(self) -> tuple[list, list]:
        """
        Verifica mudanças de posição.
        Retorna listas vazias se a mobilidade estiver desligada, mantendo a interface estável.
        """
        if not self.tracer or not self.activated:
            return [], []

        moved_sfcs = []
        new_locations = []

        # Itera sobre uma cópia ou chaves para evitar erro de modificação durante iteração
        for vehicle_id, vehicle_info in list(self.players_tracker.items()):
            if vehicle_info["redeploying"]:
                continue

            # Obtém dados do SUMO
            closest_router, dist_from_router = self.tracer.get_closest_server(vehicle_id)

            connected_router = vehicle_info["connected_router"]
            distance_from_connected_router = self.tracer.get_server_distance_from_car(
                vehicle_id, connected_router
            )

            # Lógica de Handover (Histerese)
            if closest_router != connected_router:
                # Evita divisão por zero se a distância for 0
                if distance_from_connected_router == 0:
                    reduction_percent = 0
                else:
                    reduction_percent = (
                        distance_from_connected_router - dist_from_router
                    ) / distance_from_connected_router

                if reduction_percent >= 0.30:
                    self.players_tracker[vehicle_id]["connected_router"] = closest_router
                    self.mark_vehicle_as_redeploying(vehicle_id)

                    sfcs_ids = self.players_tracker[vehicle_id]["sfc_list"]
                    moved_sfcs.append(sfcs_ids)
                    new_locations.append(closest_router)

        return moved_sfcs, new_locations

    def register_mobile_user_in_network(self, sfc_list: list, substrate_network: Net2) -> str:
        """Registra o usuário móvel na rede substrata e gerencia seus parâmetros."""
        group_id = sfc_list[0].dst_node
        closer_router = sfc_list[0].closer_router
        sfc_id_list = [sfc.id for sfc in sfc_list]

        self.add_player(group_id, closer_router, sfc_id_list)
        distance = self.get_md_distance_from_router(group_id, closer_router)

        # Os "Magic Numbers" devem, idealmente, vir de uma configuração global (Settings)
        substrate_network.add_node(
            group_id,
            "mobile_device",
            cpu_capacity=25.00,
            cache_capacity=10.00,
            ips=0.1,
            position=distance
        )
        return group_id
