import random
import sys
import threading
import time
from collections import deque
from typing import Any

from loguru import logger

from muar_sfc.controllers.modules.backup_manager import BackupManager
from muar_sfc.controllers.modules.crasher import Crasher
from muar_sfc.controllers.modules.failure_orchestrator import FailureOrchestrator
from muar_sfc.controllers.modules.mobility_manager import MobilityManager
from muar_sfc.controllers.modules.sfcs_instatiator import SFCInstatiator
from muar_sfc.controllers.modules.sfcs_manager import SFCManager
from muar_sfc.controllers.sfc_queue import SFCQueue
from muar_sfc.core.net_v2 import Net2
from muar_sfc.utils.manager_results import OutputWritter
from muar_sfc.utils.network_utils import EnergyCalculator


class SubstrateNetworkController:
    def __init__(
        self,
        substrate_network: Net2,
        sfc_queue: SFCQueue,
        sfc_manager: SFCManager,
        sfc_instantiator: SFCInstatiator,
        fail_manager: Crasher,
        output_writter: OutputWritter,
        failure_schedule: list[Any],
        alg: Any,
        energy_calculator: EnergyCalculator | None = None,
        backup_manager: BackupManager | None = None,
        mobility_manager: MobilityManager | None = None,
        players: int = 6,
        flows: int = 0,
        sfc_name: bool = True,
        verbose: bool = False
    ):
        self.substrate_network = substrate_network
        self.sfc_queue = sfc_queue
        self.sfc_manager = sfc_manager
        self.sfc_instantiator = sfc_instantiator
        self.fail_manager = fail_manager
        self.output_writter = output_writter
        self.backup_manager = backup_manager
        self.mobility_manager = mobility_manager
        self.energy_calculator = energy_calculator or EnergyCalculator()

        self.alg = alg
        self.players = players
        self.flows = flows
        self.sfc = sfc_name
        self.verbose = verbose
        self.is_stopped = True
        self.start_time = 0.0
        self.iteration_counter = 0
        self.update_interval = 1
        self.remaining_time: float | None = None

        self.failure_schedule = deque(failure_schedule)
        self.active_failures: list[Any] = []
        self.timer: threading.Timer | None = None
        
        self.mobility_interval = 5
        self.backup_interval_creation = 5
        self.last_mobility_time = 0.0
        self.last_backup_time = 0.0

        self.timer_qeue_sfcs: list[Any] = []
        self.max_queue_size = 0
        self.success: list[Any] = []
        self.counter = 0

        self.failure_orchestrator = FailureOrchestrator(
            substrate_network=self.substrate_network,
            sfc_manager=self.sfc_manager,
            fail_manager=self.fail_manager,
            mobility_manager=self.mobility_manager,
            output_writter=self.output_writter,
            alg=self.alg,
            verbose=self.verbose,
            requeue_callback=self.send_back_to_queue
        )

    def start(self) -> None:
        logger.info("Iniciando o controlador da rede substrata...")
        if not self.is_stopped:
            self.is_stopped = True
            time.sleep(2 * self.update_interval)

        self.is_stopped = False
        
        # EAFP: O contrato determina que se MobilityManager foi instanciado, 'activated' deve estar acessível
        if self.mobility_manager:
            try:
                if self.mobility_manager.activated:
                    self.mobility_manager.start_simulation()
            except AttributeError as e:
                raise RuntimeError("MobilityManager não implementa o atributo obrigatório 'activated'") from e

        threading.Thread(target=self.sequential_operation, daemon=True).start()

    def stop(self) -> None:
        logger.warning("Comando STOP recebido. Finalizando processos...")
        self.is_stopped = True
        if self.mobility_manager:
            self.mobility_manager.stop_simulation()
        if self.timer:
            self.timer.cancel()
        logger.info("Simulação encerrada.")
        sys.exit(0)

    def initialize_timers(self) -> None:
        self.mobility_interval = 5
        self.backup_interval_creation = 40 if self.alg in ["msf", "greedyb"] else 5
        self.start_time = time.time()
        self.last_mobility_time = self.start_time
        self.last_backup_time = self.start_time
        self.failure_schedule = deque(self.failure_schedule)
        self.iteration_counter = 0

    def sequential_operation(self) -> None:
        self.initialize_timers()
        while not self.is_stopped:
            self.handle_resources_cleanup()
            self.handle_mobility()
            self.handle_backups()
            self.handle_fails()

            processed_sfcs = self.submit_sfcs()
            if self.check_simulation_end(processed_sfcs):
                self.stop()
            
            self.iteration_counter += 1
            if not processed_sfcs:
                time.sleep(0.01)

    def handle_mobility(self) -> None:
        if self.mobility_manager and self.mobility_manager.activated:
            if time.time() - self.last_mobility_time >= self.mobility_interval:
                self.check_mobility()
                self.last_mobility_time = time.time()

    def handle_backups(self) -> None:
        if not self.sfc_manager.backup_manager or not self.sfc_manager.backup_manager.backup_activated:
            return

        current_time = time.time()
        if current_time - self.last_backup_time < self.backup_interval_creation:
            return

        agent = self.sfc_instantiator.alg
        created_backup_groups, strategy = self.sfc_manager.create_backups(
            self.substrate_network, agent_ref=agent
        )

        if created_backup_groups:
            self._deploy_backups_to_network(created_backup_groups)
            if self.verbose:
                count = sum(len(group) for group in created_backup_groups)
                logger.info(f"[BACKUP] Ciclo concluído. {count} novos backups registrados via {strategy}.")

        self.last_backup_time = current_time

    def _deploy_backups_to_network(self, backup_groups: list[list[Any]]) -> None:
        for group in backup_groups:
            for backup_sfc in group:
                try:
                    route_info = backup_sfc.pre_calculated_route
                except AttributeError:
                    route_info = {}

                # A máscara de Exception genérica foi removida para garantir Fail-Fast. 
                # Falhas de implantação física deverão abortar o run para evitar falso-positivos na simulação.
                self.substrate_network.deploy_sfc(backup_sfc, route_info)

    def handle_fails(self) -> None:
        if not self.fail_manager.activated:
            return

        elapsed_time = time.time() - self.start_time

        # O(1) Remoção via list comprehension, evitando .remove() iterativo
        pending_failures = []
        for failure in self.active_failures:
            if elapsed_time >= failure["recovery_time"]:
                match failure["type"]:
                    case "node":
                        self.failure_orchestrator.server_recovery_operation(failure["target"])
                    case "link":
                        self.failure_orchestrator.link_recovery_operation(failure["target"])
            else:
                pending_failures.append(failure)
        
        self.active_failures = pending_failures

        if self.failure_schedule and elapsed_time >= self.failure_schedule[0]["start"]:
            event = self.failure_schedule.popleft()
            duration = event["duration"]

            match event["type"]:
                case "node":
                    crashed_nodes, _ = self.failure_orchestrator.server_fail_operation()
                    if crashed_nodes and duration > 0:
                        recovery_time = elapsed_time + duration
                        self.active_failures.append({"type": "node", "target": crashed_nodes, "recovery_time": recovery_time})
                        logger.info(f" -> Recuperação (node) agendada para T={recovery_time:.2f}s")
                case "link":
                    link_crashed, _ = self.failure_orchestrator.link_fail_operation()
                    if link_crashed and duration > 0:
                        recovery_time = elapsed_time + duration
                        self.active_failures.append({"type": "link", "target": link_crashed, "recovery_time": recovery_time})
                        logger.info(f" -> Recuperação (link) agendada para T={recovery_time:.2f}s")

    def submit_sfcs(self) -> list[str]:
        self.check_timer_queue()

        if self.max_queue_size < self.sfc_queue.qsize():
            self.max_queue_size = self.sfc_queue.qsize()
        
        processed_sfcs = []
        start_time = time.time()

        while self.sfc_queue.qsize() != 0:
            if time.time() - start_time > 1:
                break

            sfc_list = self.sfc_queue.peek_sfc()
            dequeue_time = time.time()

            # EAFP estrito e em cascata para extração do tempo
            try:
                wait_time = dequeue_time - sfc_list[0].enqueue_time
            except AttributeError:
                try:
                    wait_time = dequeue_time - sfc_list[0].arrival_time
                except AttributeError:
                    wait_time = 0.0

            for sfc in sfc_list:
                if self.sfc_manager.is_session_active(sfc.dst_node):
                    raise ValueError(f"Estado de Rede Corrompido: SFC {sfc.id} já consta como submetida.")

            log_output, is_success = self.deploy_sfc_list(sfc_list)

            for sfc_id, result_dict in log_output.items():
                processed_sfcs.append(sfc_id)
                self.output_results(sfc_id=sfc_id, results_dict=result_dict, is_success=is_success, wait_time=wait_time)

            if self.verbose:
                sfc_names = ", ".join([s.id for s in sfc_list])
                if is_success:
                    logger.success(f"Physical alloc. complete: batch [{sfc_names}] implanted.")
                else:
                    logger.error(f"Physical alloc. failed: batch [{sfc_names}] rejected due to resource unavailability.")

                self.output_writter.print_output_info(self.substrate_network, self.success)

        return processed_sfcs

    def deploy_sfc_list(self, sfc_list) -> tuple[Any, bool]:
        mob_player_id = self.create_mobile_user(sfc_list)
        solution, is_success = self.sfc_instantiator.search_solution(
            sfc_list, self.substrate_network
        )
        if is_success:
            self.sfc_manager.submit_solution(sfc_list, solution, self.substrate_network)
        else:
            self.remove_mobile_user(mob_player_id)
        return solution, is_success

    def send_back_to_queue(
        self, sfc_list: list, changed_location: bool = False, new_location: Any = False
    ) -> None:
        if not sfc_list:
            return

        session_id = sfc_list[0].dst_node
        new_sfcs, remaining_dur = self.sfc_manager.rebuild_sfcs_for_requeue(
            sfc_list, changed_location, new_location
        )

        if remaining_dur <= 1.0:
            if self.verbose:
                logger.info(f"[Queue] Sessão {session_id} expirou durante falha. Cancelando re-deploy.")
            self._force_cleanup_session(session_id)
            return

        ids_to_remove = self.sfc_manager.cleanup_session_state(session_id)
        for old_sfc_id in ids_to_remove:
            self._force_remove_sfc_and_backups(old_sfc_id)

        self.sfc_queue.put_begin(new_sfcs)

    def _force_cleanup_session(self, session_id: str) -> None:
        ids = self.sfc_manager.cleanup_session_state(session_id)
        for sfc_id in ids:
            self._force_remove_sfc_and_backups(sfc_id)

    def check_timer_queue(self) -> None:
        if not self.timer_qeue_sfcs:
            return
            
        final_time = time.time()
        time_elapsed = final_time - self.timer_qeue_sfcs[0]["timer"]
        
        # Limpeza O(1) estrutural. Em cenários reais, um Deque pode operar na ponta sem reconstruir a lista.
        if time_elapsed >= random.uniform(5, 6):
            current_enqueue_time = time.time()
            for entry in self.timer_qeue_sfcs:
                for sfc in entry["new_sfc_list"]:
                    sfc.enqueue_time = current_enqueue_time
                self.sfc_queue.put_begin(entry["new_sfc_list"])
            self.timer_qeue_sfcs.clear()

    def check_mobility(self) -> None:
        if not self.sfc_manager.tracker.sfcs_tracker:
            return
            
        sfcs_moved, new_locations = self.mobility_manager.check_all_vehicles_position_changes()
        for sfc_list, new_location in zip(sfcs_moved, new_locations, strict=False):
            obj_sfc_list = []
            valid_move = True
            for sfc_id in sfc_list:
                try:
                    sfc_obj = self.substrate_network.get_sfc_by_id(sfc_id)
                    obj_sfc_list.append(sfc_obj)
                except KeyError:
                    if self.verbose:
                        logger.warning(f"[MOBILITY] Ignorando movimento da SFC {sfc_id} (SFC offline).")
                    valid_move = False
                    break

            if valid_move and obj_sfc_list:
                logger.info(f"SFCs moved: {sfc_list} | New Location: {new_location}")
                self.send_back_to_queue(obj_sfc_list, changed_location=True, new_location=new_location)

    def create_mobile_user(self, sfc_list) -> str:
        return self.mobility_manager.register_mobile_user_in_network(sfc_list, self.substrate_network)

    def remove_mobile_user(self, sfc_list_id: str) -> None:
        self.mobility_manager.remove_player(sfc_list_id)
        self.substrate_network.remove_node(sfc_list_id)

    def check_simulation_end(self, sfc_list: list[str]) -> bool:
        last_sf_mono = f"sfc_unique_p{self.players}_{self.flows}"
        last_sf_dec = f"sfc_mono_p{self.players}_{self.flows}"
        for sfc_id in sfc_list:
            if sfc_id in (last_sf_mono, last_sf_dec):
                logger.info(f"[INFO] Last SFC released detected: {sfc_id}")
                logger.info(f"Max queue size: {self.max_queue_size}")
                time.sleep(1)
                return True
        return False

    def handle_resources_cleanup(self) -> None:
        current_time = time.time()
        self.sfc_manager.cleanup_network_resources(self.substrate_network, current_time)

        if self.iteration_counter % 10 == 0:
            self.sfc_manager.run_garbage_collection(self.substrate_network)

    def _force_remove_sfc_and_backups(self, sfc_id: str) -> None:
        bm = self.sfc_manager.backup_manager

        if bm and sfc_id in bm.sfcs_backups_instatiated:
            backups_list = list(bm.sfcs_backups_instatiated[sfc_id])
            for backup_entry in backups_list:
                self._safe_undeploy_backup(backup_entry["sfc_backup_id"])

        try:
            self.substrate_network.undeploy_sfc(sfc_id)
        except KeyError as e:
            # Substituímos Exception genérica pela exceção explícita de nó ausente.
            # Se for um erro crítico estrutural, o framework irá interromper para garantir coerência.
            raise RuntimeError(f"Corrupção de grafo ao remover SFC {sfc_id}: O recurso já não existe.") from e

    def _safe_undeploy_backup(self, backup_id: str) -> None:
        try:
            self.substrate_network.undeploy_sfc(backup_id)
        except KeyError as e:
            raise RuntimeError(f"Corrupção de grafo ao remover backup {backup_id}: O recurso já não existe.") from e

        if self.sfc_manager.backup_manager:
            self.sfc_manager.backup_manager.cleanup_internal_state(backup_id)

    def output_results(
        self, results_dict: dict, sfc_id: str, is_success: bool, res_output: bool = False, wait_time: float | None = None
    ) -> None:
        current_time = time.time()

        if not res_output:
            self.output_writter.write_full_results(
                substrate_network=self.substrate_network,
                energy_calculator=self.energy_calculator,
                sfc_manager=self.sfc_manager,
                fail_manager=self.fail_manager,
                success_list=self.success,
                sfc_id=sfc_id,
                results_dict=results_dict,
                is_success=is_success,
                current_time=current_time,
                remaining_time=self.remaining_time or 0.0,
                alg=self.alg,
                wait_time=wait_time,
            )

        # Usando EAFP para garantir que o FailureOrchestrator implemente o protocolo corretamente.
        try:
            self.failure_orchestrator.update_crash_recovery_status(
                sfc_id=sfc_id,
                results_dict=results_dict,
                is_success=is_success,
                current_time=current_time
            )
        except AttributeError:
            pass # Orquestrador mais simples, sem logging avançado.