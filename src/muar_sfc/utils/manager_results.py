import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger  # <-- NOVO: Motor de observabilidade

from muar_sfc.core.net_v2 import Net2


def format_nodes_to_string(nodes: np.ndarray) -> str:
    nodes_to_string = np.array2string(nodes, suppress_small=True, precision=3, separator=",")
    return re.sub("[ \n]", "", nodes_to_string)


def format_edges_to_string(edges: list) -> str:
    edges_to_string = ";".join(map(str, edges))
    return re.sub("[ \n]", "", edges_to_string)


def create_directory_if_not_exists(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def create_output_dir(args, topology) -> tuple[dict[str, Path], Path, Path, Path]:
    top_info = topology.get_topology_info()
    ec_servers = top_info["ec_servers"]
    edges = top_info["edges"]

    alg_name = args.alg.replace("_", "")
    availability = args.ava

    number_of_fails = args.number_of_fails
    if float(availability) == 1.0:
        number_of_fails = 0

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f") + str(random.randint(0, 10000))

    base_dir = Path("results")
    paths = ["cache", "cpu", "gpu", "bandwidth", "sf"]
    directories = {}

    for path in paths:
        dir_path = base_dir / f"results_{path}"
        dir_path.mkdir(parents=True, exist_ok=True)

        alg_path = dir_path / (
            f"alg_{args.alg}_s_{args.n_sessions}_p_{args.n_players}_"
            f"a_{availability}_c_{number_of_fails}"
        )
        alg_path.mkdir(parents=True, exist_ok=True)
        directories[path] = alg_path

    # REFATORAÇÃO: Mantendo como objetos Path puros. O open() nativo aceita Path!
    file_paths = {path: directories[path] / f"{timestamp}.csv" for path in paths}

    nodes_string = format_nodes_to_string(np.array(sorted(ec_servers)))
    edges_string = format_edges_to_string(edges)

    for key in ["cache", "cpu", "gpu", "sf"]:
        with open(file_paths[key], "a") as file:
            file.write(f"timestamp,{nodes_string[1:-1]}\n")

    with open(file_paths["bandwidth"], "a") as file:
        file.write(f"timestamp,{edges_string}\n")

    flows_dir = base_dir / "results_flows"
    resilient_dir = base_dir / "results_resilient"

    directory_path = (
        flows_dir
        / f"{alg_name}_s_{args.n_sessions}_p_{args.n_players}_a_{availability}_c_{number_of_fails}"
    )
    res_directory_path = (
        resilient_dir
        / f"{args.alg}_s_{args.n_sessions}_p_{args.n_players}_a_{availability}_c_{number_of_fails}"
    )

    directory_path.mkdir(parents=True, exist_ok=True)
    res_directory_path.mkdir(parents=True, exist_ok=True)

    flows_path = directory_path / f"{timestamp}.csv"
    crash_impact_path = res_directory_path / f"crash_impact_{timestamp}.csv"
    resilient_path = res_directory_path / f"resilient_results_{timestamp}.csv"

    # ... (O restante dos headers longos permanece idêntico, apenas mantive os mesmos nomes) ...
    header_fields = [
        "No.", "timestamp", "time_seconds", "users", "cpu_utilization",
        "gpu_utilization", "bandwidth_utilization", "cache_utilization",
        "network_cpu_utilization", "network_gpu_utilization",
        "network_cache_utilization", "mobile_cpu_utilization",
        "mobile_gpu_utilization", "mobile_cache_utilization", "latency",
        "comp_latency", "comm_latency", "latency_diff", "queue_time",
        "decision_time_ms", "success", "fail_reason", "sfc_id",
        "recovery_time", "sfc_recovered", "cpu_saved", "gpu_saved",
        "cache_saved", "shared_vnfs", "running_sfcs", "running_players",
        "running_sessions", "trascode_bw", "crashing", "acceptance_rate",
        "cpu_per_flow", "gpu_per_flow", "cache_per_flow", "jain_cpu",
        "jain_gpu", "jain_cache", "jain_bw", "server_energy_consumption",
        "mobile_energy_consumption", "total_energy_consumption",
        "avg_sfc_reliability", "high_risk_sfcs", "medium_risk_sfcs", "low_risk_sfcs",
    ]

    crash_header_fields = [
        "crash_trial", "timestamp", "nodes_crashed_count",
        "total_affected_sfcs", "affected_by_high_risk_node",
        "affected_by_med_risk_node", "affected_by_low_risk_node",
        "avg_latency_before", "avg_latency_after", "affected_percentage",
        "avg_latency_diff",
    ]

    crash_header = ",".join(crash_header_fields) + "\n"
    with open(crash_impact_path, "a") as f:
        f.write(crash_header)

    header = ",".join(header_fields) + "\n"
    with open(flows_path, "a") as f:
        f.write(header)

    resilient_header_fields = [
        "crash_trial", "sfc_id", "recover_success", "backup_success",
        "backup_efficient", "latency_before", "latency_after",
        "latency_diff", "time_to_recover", "vnf_id", "latency_degrad",
        "resource_degrad", "risk_level", "final_status",
    ]
    with open(resilient_path, "a") as f:
        f.write(",".join(resilient_header_fields) + "\n")

    return file_paths, flows_path, crash_impact_path, resilient_path


class OutputWritter:
    def __init__(self, topology, file_paths: dict[str, Path], flows_file: Path, crash_impact_file: Path, resilient_file: Path):
        self.topology = topology
        self.file_paths = file_paths
        self.flows_file = flows_file
        self.crash_impact_file = crash_impact_file
        self.resilient_file = resilient_file

        topo_info = topology.get_topology_info()
        self.processing_nodes = topo_info["ec_servers"]
        self.edges = topo_info["edges"]

        self.cpu_utilization_file = file_paths["cpu"]
        self.gpu_utilization_file = file_paths["gpu"]
        self.cache_utilization_file = file_paths["cache"]
        self.bw_utilization_file = file_paths["bandwidth"]
        self.sf_utilization_file = file_paths["sf"]

        self.counter_users = 0
        self.first_time = 0

    def resilient_output(self, sfc_id, info_log, crash_trials):
        file_path = self.resilient_file

        data = [
            str(crash_trials),
            str(sfc_id),
            str(info_log.get("recover_success", False)),
            str(info_log.get("backup_success", False)),
            str(info_log.get("backup_efficient", "N/A")),
            str(info_log.get("latency_before", 0.0)),  # <--- NOVA COLUNA
            str(info_log.get("latency_after", 0.0)),  # <--- NOVA COLUNA
            str(info_log.get("latency_diff", 0.0)),
            str(info_log.get("time_to_recover", 0.0)),
            str(info_log.get("vnf_id", "N/A")),
            str(info_log.get("latency_degrad", 0.0)),
            str(info_log.get("resource_degrad", 0.0)),
            str(info_log.get("risk_level", "Medium")),
            str(info_log.get("final_status", "Failed")),
        ]

        with open(file_path, "a") as f:
            f.write(",".join(data) + "\n")

    def output_crash_impact(
        self,
        crash_trial,
        nodes_count,
        affected_count,
        high,
        medium,
        low,
        lat_before,
        lat_after,
        affected_pct,
        avg_lat_diff,
    ):
        current_time = time.time()
        line = (
            f"{crash_trial},"
            f"{current_time},"
            f"{nodes_count},"
            f"{affected_count},"
            f"{high},"
            f"{medium},"
            f"{low},"
            f"{lat_before:.4f},"
            f"{lat_after:.4f},"
            f"{affected_pct:.2f},"
            f"{avg_lat_diff:.4f}\n"
        )

        with open(self.crash_impact_file, "a") as file:
            file.write(line)

    def _calculate_sfc_reliability_with_backups(
        self, network: Net2, sfc_id: str, backups_dict: dict
    ) -> float:
        """Calcula a confiabilidade real (SFC + Backups) usando lógica Paralela/Série."""
        if sfc_id not in network.sfc_route_info:
            return 0.0

        route_info = network.sfc_route_info[sfc_id]

        # 1. Mapear Backups Instanciados para esta SFC
        backup_reliability_map = {}
        if sfc_id in backups_dict:
            for b in backups_dict[sfc_id]:
                vnf_id = b.get("vnf_id")
                bk_route = b.get("route_info", {})
                # Procura nó onde está o backup (ex: vnf1_b) no dicionário da rota
                bk_node_list = next(
                    (v for k, v in bk_route.items() if k.endswith("_b") and v), None
                )
                if vnf_id and bk_node_list:
                    node_id = bk_node_list[0]
                    # Pega confiabilidade do nó de backup
                    backup_reliability_map[vnf_id] = network.get_node_reliability(node_id)

        # 2. Agrupar VNFs por Nó Físico (Domínio de Falha)
        node_groups = defaultdict(list)
        for vnf_id, path in route_info.items():
            if vnf_id in ["src", "dst"] or not path:
                continue
            # path[0] é o servidor físico
            node_groups[path[0]].append(vnf_id)

        # 3. Calcular Confiabilidade Série-Paralelo
        total_reliability = 1.0

        for node_id, vnfs_list in node_groups.items():
            try:
                reliability_primary = network.get_node_reliability(node_id)
            except KeyError:  # <-- CORRIGIDO: Captura Estrita EAFP
                reliability_primary = 1.0

            all_vnfs_protected = True
            prod_failure_backups = 1.0

            for vnf_id in vnfs_list:
                reliability_backup = backup_reliability_map.get(vnf_id, 0.0)
                if reliability_backup > 0.0:
                    prod_failure_backups *= 1.0 - reliability_backup
                else:
                    all_vnfs_protected = False

            if all_vnfs_protected:
                # Sistema Paralelo: Falha apenas se Primário E Backups falharem
                prob_primary_fail = 1.0 - reliability_primary
                group_reliability = 1.0 - (prob_primary_fail * prod_failure_backups)
            else:
                # Sistema Série: Limitado pelo nó primário
                group_reliability = reliability_primary

            total_reliability *= group_reliability

        return total_reliability

    # --------------------------------------------------------------------------
    # MÉTODOS ABSTRAÍDOS DO CONTROLLER PARA O OUTPUT WRITTER (SRP)
    # --------------------------------------------------------------------------

    def write_full_results(
        self,
        substrate_network: Net2,
        energy_calculator,
        sfc_manager,
        fail_manager,
        success_list: list,
        sfc_id: str,
        results_dict: dict,
        is_success: bool,
        current_time: float,
        remaining_time: float,
        alg: str,
        wait_time: float = 0.0,
    ) -> None:
        """
        Método de fachada (Facade) para orquestrar todas as escritas de log de uma vez.
        Desacopla o Controller da responsabilidade de gerenciar as métricas granulares.
        """
        success_list.append(is_success)
        crashed_nodes = getattr(fail_manager, "nodes_crashed", [])

        # 1. Escreve Recursos Físicos
        self.output_cpu_utilization(substrate_network, current_time, crashed_nodes)
        self.output_gpu_utilization(substrate_network, current_time, crashed_nodes)
        self.output_cache_utilization(substrate_network, current_time, crashed_nodes)
        self.output_bandwidth_utilization(substrate_network, current_time)
        self.output_nodes_sf_utilization(substrate_network, current_time)

        # 2. Prepara Cálculos de Energia
        bw_transcode = 0
        server_energy = energy_calculator.calculate_total_server_power(substrate_network)
        mobile_energy = energy_calculator.calculate_total_mobile_device_power(substrate_network)
        total_energy = server_energy + mobile_energy

        # 3. Prepara Dados de Backup/Confiabilidade
        bm = sfc_manager.backup_manager
        backups_dict_ref = bm.sfcs_backups_instatiated if (bm and not isinstance(bm, int)) else {}

        if bm and not isinstance(bm, int):
            substrate_network.calculate_average_system_reliability(backups_dict_ref)
        else:
            substrate_network.calculate_average_system_reliability(0)

        # 4. Escreve os Fluxos (Flows)
        self.output_flows(
            substrate_network=substrate_network,
            wait_time=wait_time,
            running_players_sessions=sfc_manager.get_running_players_sessions(),
            counter=self.counter_users, # Modificado para usar o contador interno
            remaining_time=remaining_time,
            current_time=current_time,
            sfc_id=sfc_id,
            latency=results_dict.get("latency", 0.0),
            comp_latency=results_dict.get("comp_latency", 0.0),
            comm_latency=results_dict.get("comm_latency", 0.0),
            run_duration=results_dict.get("run_duration", 0.0),
            is_success=is_success,
            fail_reason=None,
            bw_transcode=bw_transcode,
            acceptance_rate=substrate_network.get_acceptance_rate(success_list),
            server_energy_consumption=server_energy,
            mobile_energy_consumption=mobile_energy,
            total_energy_consumption=total_energy,
            latency_diff=None,
            crashing=len(crashed_nodes) > 0,
            alg_name=alg,
            backups_dict=backups_dict_ref,
        )

    def output_flows(
        self,
        substrate_network: Net2,
        wait_time,
        running_players_sessions,
        counter,
        remaining_time,
        current_time,
        sfc_id,
        latency,
        comp_latency,
        comm_latency,
        run_duration,
        is_success,
        fail_reason,
        bw_transcode,
        acceptance_rate,
        server_energy_consumption,
        mobile_energy_consumption,
        total_energy_consumption,
        latency_diff=None,
        crashing=False,
        alg_name="ga",
        backups_dict: dict = None,
    ):
        """Escreve as métricas detalhadas de fluxo em CSV."""
        if backups_dict is None:
            backups_dict = {}

        running_sfcs = substrate_network.get_number_active_primary_sfcs()
        unique_players = set()
        unique_sessions = set()

        if substrate_network.sfc_dict:
            for active_sfc_id in substrate_network.sfc_dict:
                try:
                    parts = active_sfc_id.split("_")
                    for part in parts:
                        if part.startswith("p") and part[1:].isdigit():
                            unique_players.add(part)
                    if parts[-1].isdigit():
                        unique_sessions.add(parts[-1])
                except (IndexError, AttributeError):
                    continue

        running_players = len(unique_players)
        running_sessions = len(unique_sessions)

        cpu_utilization = round(substrate_network.get_total_system_utilization_cpu_rate(), 4)
        network_cpu_utilization = round(substrate_network.get_network_cpu_utilization_percentage(), 4)
        mobile_cpu_utilization = round(substrate_network.get_mobile_cpu_utilization_percentage(), 4)

        gpu_utilization = round(substrate_network.get_total_system_utilization_gpu_rate(), 4)
        network_gpu_utilization = round(substrate_network.get_network_gpu_utilization_percentage(), 4)
        mobile_gpu_utilization = round(substrate_network.get_mobile_gpu_utilization_percentage(), 4)

        cache_utilization = round(substrate_network.get_total_system_utilization_cache_rate(), 4)
        bw_utilization = round(substrate_network.get_bandwidth_utilization_rate(), 4)
        network_cache_utilization = round(substrate_network.get_network_cache_utilization_percentage(), 4)
        mobile_cache_utilization = round(substrate_network.get_mobile_cache_utilization_percentage(), 4)

        cpu_used = substrate_network.get_cpu_total_used()
        cpu_per_flow = cpu_used / running_sfcs if running_sfcs else 0

        gpu_used = substrate_network.get_gpu_total_used()
        gpu_per_flow = gpu_used / running_sfcs if running_sfcs else 0

        cache_used = substrate_network.get_cache_total_used()
        cache_per_flow = cache_used / running_sfcs if running_sfcs else 0

        cpu_saved = substrate_network.get_total_cpu_saved()
        gpu_saved = substrate_network.get_total_gpu_saved()
        cache_saved = substrate_network.get_total_cache_saved()

        shared_vnfs_count = substrate_network.metrics.shared_vnfs_count

        jain_cpu = round(substrate_network.get_cpu_jain_fairness(), 4)
        jain_gpu = round(substrate_network.get_gpu_jain_fairness(), 4)
        jain_cache = round(substrate_network.get_cache_jain_fairness(), 4)
        jain_bw = round(substrate_network.get_bandwidth_jain_fairness(), 4)

        sfc_recovery_time = 0
        sfc_recovered = None

        self.update_user_count(sfc_id)
        self.counter_users += 1 # Incrementa o contador interno da classe

        first_loop = self.first_time == 0
        time_value = 0

        if first_loop:
            self.first_time = current_time
            time_value = 0
        else:
            time_value = round(current_time - self.first_time, 1)
        decision_time = str(round(run_duration * 1000, 3))

        avg_sfc_reliability = 0.0
        count_high_risk = 0
        count_medium_risk = 0
        count_low_risk = 0

        total_reliability_sum = 0.0
        active_sfc_count = 0

        if substrate_network.sfc_dict:
            for s_id, _sfc in substrate_network.sfc_dict.items():
                if "backup" in s_id or s_id not in substrate_network.sfc_route_info:
                    continue

                sfc_reliability = self._calculate_sfc_reliability_with_backups(
                    substrate_network, s_id, backups_dict
                )

                if sfc_reliability > 0.0001:
                    total_reliability_sum += sfc_reliability
                    active_sfc_count += 1

                    if sfc_reliability < 0.8666:
                        count_high_risk += 1
                    elif 0.8666 <= sfc_reliability <= 0.9333:
                        count_medium_risk += 1
                    else:
                        count_low_risk += 1

        avg_sfc_reliability = (total_reliability_sum / active_sfc_count if active_sfc_count > 0 else 0.0)

        line = (
            f"{self.counter_users},"
            f"{current_time},"
            f"{time_value},"
            f"{self.counter_users},"
            f"{cpu_utilization},"
            f"{gpu_utilization},"
            f"{bw_utilization},"
            f"{cache_utilization},"
            f"{network_cpu_utilization},"
            f"{network_gpu_utilization},"
            f"{network_cache_utilization},"
            f"{mobile_cpu_utilization},"
            f"{mobile_gpu_utilization},"
            f"{mobile_cache_utilization},"
            f"{latency},"
            f"{comp_latency},"
            f"{comm_latency},"
            f"{latency_diff},"
            f"{wait_time},"
            f"{decision_time},"
            f"{is_success},"
            f"{fail_reason},"
            f"{sfc_id},"
            f"{sfc_recovery_time},"
            f"{sfc_recovered},"
            f"{cpu_saved},"
            f"{gpu_saved},"
            f"{cache_saved},"
            f"{shared_vnfs_count},"
            f"{running_sfcs},"
            f"{running_players},"
            f"{running_sessions},"
            f"{bw_transcode},"
            f"{crashing},"
            f"{acceptance_rate},"
            f"{cpu_per_flow},"
            f"{gpu_per_flow},"
            f"{cache_per_flow},"
            f"{jain_cpu},"
            f"{jain_gpu},"
            f"{jain_cache},"
            f"{jain_bw},"
            f"{total_energy_consumption},"
            f"{server_energy_consumption},"
            f"{mobile_energy_consumption},"
            f"{avg_sfc_reliability},"
            f"{count_high_risk},"
            f"{count_medium_risk},"
            f"{count_low_risk}\n"
        )

        with open(self.flows_file, "a") as file:
            file.write(line)

    def update_user_count(self, sfc_id: str) -> None:
        try:
            player = int(sfc_id.split("_")[-2][1])
            session = int(sfc_id.split("_")[-1])
            users = (session - 1) * 5 + player
            if users > self.counter_users:
                self.counter_users = users
        except (IndexError, ValueError) as e:
            # REFATORAÇÃO: Substituição do perigoso 'pass' pelo registro em log estruturado
            logger.warning(f"Falha ao extrair usuários do sfc_id '{sfc_id}'. Formato inesperado. Erro: {e}")

    def output_cpu_utilization(
        self, substrate_network, deploy_time, crashed_nodes=None
    ) -> None:
        if crashed_nodes is None:
            crashed_nodes = []
        processing_nodes = sorted(self.processing_nodes)
        cpu_nodes_util = []
        for node in processing_nodes:
            if node in crashed_nodes or str(node).endswith(".1"):
                cpu_nodes_util.append(None)
            else:
                cpu_nodes_util.append(round(substrate_network.get_node_cpu_used(node), 2))
        string_cpu_nodes_util = ",".join(
            ["None" if value is None else f"{value:.2f}" for value in cpu_nodes_util]
        )
        with open(self.cpu_utilization_file, "a") as file:
            file.write(f"{deploy_time},{string_cpu_nodes_util}\n")

    def output_cache_utilization(
        self, substrate_network, deploy_time, crashed_nodes=None
    ) -> None:
        if crashed_nodes is None:
            crashed_nodes = []
        processing_nodes = sorted(self.processing_nodes)
        cache_nodes_util = [
            None
            if node in crashed_nodes
            else round(substrate_network.get_node_cache_used(node), 2)
            for node in processing_nodes
        ]
        string_cache_nodes_util = ",".join(
            ["None" if value is None else f"{value:.2f}" for value in cache_nodes_util]
        )
        with open(self.cache_utilization_file, "a") as file:
            file.write(f"{deploy_time},{string_cache_nodes_util}\n")

    def output_gpu_utilization(
        self, substrate_network, deploy_time, crashed_nodes=None
    ) -> None:
        if crashed_nodes is None:
            crashed_nodes = []
        processing_nodes = sorted(self.processing_nodes)
        gpu_nodes_util = []
        for node in processing_nodes:
            if node in crashed_nodes or not str(node).endswith(".1"):
                gpu_nodes_util.append(None)
            else:
                gpu_nodes_util.append(round(substrate_network.get_node_cpu_used(node), 2))
        string_gpu_nodes_util = ",".join(
            ["None" if value is None else f"{value:.2f}" for value in gpu_nodes_util]
        )
        with open(self.gpu_utilization_file, "a") as file:
            file.write(f"{deploy_time},{string_gpu_nodes_util}\n")

    def output_bandwidth_utilization(self, substrate_network, deploy_time: float) -> None:
        nodes_one = [node_one for node_one, node_two in self.edges]
        nodes_two = [node_two for node_one, node_two in self.edges]
        bw_edges_util = np.array(
            list(map(substrate_network.get_link_bandwidth_used, nodes_one, nodes_two))
        )
        string_bw_edges_util = np.array2string(
            bw_edges_util,
            suppress_small=True,
            precision=3,
            separator=";",
            formatter={"float_kind": lambda x: f"{x:.2f}"},
        )
        string_bw_edges_util = re.sub(" ", "", string_bw_edges_util)
        string_bw_edges_util = re.sub("\n", "", string_bw_edges_util)
        with open(self.bw_utilization_file, "a") as file:
            file.write(str(deploy_time) + ";" + string_bw_edges_util[1:-1] + "\n")

    def output_nodes_sf_utilization(self, substrate_network, deploy_time: float) -> None:
        pass

    def print_output_info(self, substrate_network, success):
        # ... (código anterior) ...
        # Agora o próprio OutputWritter (self) imprime o painel, passando a rede como argumento!
        self.print_telemetry_dashboard(substrate_network, success)


    # =========================================================================
    # PAINÉIS DE TELEMETRIA (Movidos da classe Net2 para o OutputWritter - SRP)
    # =========================================================================

    def print_network(self, network) -> None:
        """Imprime a topologia estrita da rede."""
        logger.debug("--- Nós ---")
        for node, data in network.graph.nodes(data=True):
            logger.debug(f"{node} -> {data}")
        logger.debug("--- Arestas ---")
        for u, v, data in network.graph.edges(data=True):
            logger.debug(f"{u} <-> {v} -> {data}")

    def print_telemetry_dashboard(self, network, success_arr: list, failure_band: float | None = None) -> None:
        """Gera um painel canónico e unificado com todas as métricas da infraestrutura."""
        total_cpu_util = network.get_total_system_utilization_cpu_rate() * 100
        total_gpu_util = network.get_total_system_utilization_gpu_rate() * 100
        total_processing_util = network.get_total_system_processing_utilization_rate() * 100

        cpu_save = (network.metrics.total_cpu_saved / network.metrics.total_cpu_requested * 100) if network.metrics.total_cpu_requested > 0 else 0.0
        gpu_save = (network.metrics.total_gpu_saved / network.metrics.total_gpu_requested * 100) if network.metrics.total_gpu_requested > 0 else 0.0

        cache_cap = network.total_cache_capacity if hasattr(network, 'total_cache_capacity') and network.total_cache_capacity > 0 else 1.0
        cache_util = ((network.metrics.total_cache_used + network.metrics.mobile_cache_used) / cache_cap * 100)
        cache_save = (network.metrics.total_cache_saved / network.metrics.total_cache_requested * 100) if network.metrics.total_cache_requested > 0 else 0.0

        cpu_str = f"{cpu_save:.3f}%" if network.metrics.total_cpu_requested > 0 else "N/A"
        gpu_str = f"{gpu_save:.3f}%" if network.metrics.total_gpu_requested > 0 else "N/A"
        cache_save_str = f"{cache_save:.3f}%" if network.metrics.total_cache_requested > 0 else "N/A"

        bw_cap = network.total_bandwidth_capacity if hasattr(network, 'total_bandwidth_capacity') and network.total_bandwidth_capacity > 0 else 1.0
        bw_util = (network.metrics.total_bandwidth_used * 1.0 / bw_cap * 100)

        import numpy as np  # Certifique-se de que o numpy está importado no topo deste ficheiro
        media_porc = np.mean(success_arr) * 100 if len(success_arr) > 0 else 0.0

        band_failure_str = f"│ ◦ Band Failure  : {failure_band}%\n" if failure_band is not None else ""

        # Construção do Painel Simétrico
        panel = (
            "\n╭── [ PAINEL UNIFICADO DE TELEMETRIA ] ──────────────────────\n"
            f"│ [ ORQUESTRAÇÃO ]\n"
            f"│ ◦ Acceptance Rate : {media_porc:.3f}%\n"
            f"├────────────────────────────────────────────────────────────\n"
            f"│ [ NÓS E PROCESSAMENTO ]\n"
            f"│ ◦ Total Proc. Util: {total_processing_util:.3f}%\n"
            f"│ ◦ Sys CPU Util.   : {total_cpu_util:.3f}%\n"
            f"│ ◦ Sys GPU Util.   : {total_gpu_util:.3f}%\n"
            f"│ ◦ Net CPU Util.   : {network.get_network_cpu_utilization_percentage():.3f}%\n"
            f"│ ◦ Net GPU Util.   : {network.get_network_gpu_utilization_percentage():.3f}%\n"
            f"│ ◦ Mobile CPU Util.: {network.get_mobile_cpu_utilization_percentage():.3f}%\n"
            f"├────────────────────────────────────────────────────────────\n"
            f"│ [ REDE E ECONOMIA (SAVINGS) ]\n"
            f"│ ◦ Bandwidth Util. : {bw_util:.3f}%\n"
            f"{band_failure_str}"
            f"│ ◦ CPU Saving Rate : {cpu_str}\n"
            f"│ ◦ GPU Saving Rate : {gpu_str}\n"
            f"│ ◦ Cache Util.     : {cache_util:.3f}%\n"
            f"│ ◦ Cache Saving    : {cache_save_str}\n"
            "╰────────────────────────────────────────────────────────────"
        )
        logger.info(panel)

    def print_out_acceptance_information(self, success_arr: list) -> None:
        """Imprime métricas básicas de sucesso isoladamente."""
        if len(success_arr) != 0:
            import numpy as np
            media_porc = np.mean(success_arr) * 100
            panel = (
                "\n╭── [ ORQUESTRAÇÃO E RESULTADO ]\n"
                f"╰── Acceptance Rate : {media_porc:.3f}%"
            )
            logger.info(panel)
