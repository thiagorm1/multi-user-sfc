from dataclasses import dataclass


@dataclass
class NetworkMetrics:
    """Gerencia as métricas de recursos da rede de forma centralizada.

    Implementa a lógica de contabilidade para CPU, GPU e Cache, garantindo
    que as taxas de economia sejam calculadas de forma consistente.
    """

    total_cpu_requested: float = 0.0
    total_cpu_used: float = 0.0
    total_cpu_saved: float = 0.0

    total_gpu_requested: float = 0.0
    total_gpu_used: float = 0.0
    total_gpu_saved: float = 0.0

    total_cache_requested: float = 0.0
    total_cache_used: float = 0.0
    total_cache_saved: float = 0.0

    mobile_cpu_used: float = 0.0
    mobile_gpu_used: float = 0.0
    mobile_cache_used: float = 0.0

    total_bandwidth_used: float = 0.0

    shared_vnfs_count: int = 0

    def register_allocation(
        self,
        cpu_req: float,
        cache_req: float,
        is_gpu: bool,
        is_shared: bool,
        is_new_instance: bool,
    ) -> None:
        """Registra a alocação de recursos e atualiza os contadores de economia.

        Args:
            cpu_req: Quantidade de CPU requisitada.
            cache_req: Quantidade de Cache requisitada.
            is_gpu: Se o nó é uma GPU.
            is_shared: Se o serviço é passível de compartilhamento.
            is_new_instance: Se uma nova instância física foi criada.
        """
        # Atualiza requisição total (Demanda)
        if is_gpu:
            self.total_gpu_requested = round(self.total_gpu_requested + cpu_req, 2)
        else:
            self.total_cpu_requested = round(self.total_cpu_requested + cpu_req, 2)

        self.total_cache_requested = round(self.total_cache_requested + cache_req, 2)

        # Lógica de Economia vs Uso Real
        if not is_new_instance:
            # Reutilização (Saving)
            if is_gpu:
                self.total_gpu_saved = round(self.total_gpu_saved + cpu_req, 2)
            else:
                self.total_cpu_saved = round(self.total_cpu_saved + cpu_req, 2)

            self.total_cache_saved = round(self.total_cache_saved + cache_req, 2)
            if is_shared:
                self.shared_vnfs_count += 1
        else:
            # Nova Instância Física (Used) é tratada no nó, mas podemos
            # rastrear o total global aqui se desejar.
            # No código original, 'used' é soma dos nós. Mantemos coerência.
            pass

    def register_deallocation(
        self,
        cpu_req: float,
        cache_req: float,
        is_gpu: bool,
        is_shared: bool,
        remove_physical_instance: bool,
    ) -> None:
        """Registra a desalocação garantindo consistência matemática."""

        # Atualiza requisição total
        if is_gpu:
            self.total_gpu_requested = round(self.total_gpu_requested - cpu_req, 2)
        else:
            self.total_cpu_requested = round(self.total_cpu_requested - cpu_req, 2)

        self.total_cache_requested = round(self.total_cache_requested - cache_req, 2)

        # Atualiza Economia
        if not remove_physical_instance:
            # Se não removeu a instância física, removemos a economia gerada
            if is_gpu:
                self.total_gpu_saved = round(self.total_gpu_saved - cpu_req, 2)
            else:
                self.total_cpu_saved = round(self.total_cpu_saved - cpu_req, 2)

            self.total_cache_saved = round(self.total_cache_saved - cache_req, 2)

            if is_shared:
                self.shared_vnfs_count = max(0, self.shared_vnfs_count - 1)

    def get_cache_saving_rate(self) -> float:
        """Calcula a taxa de economia de cache."""
        if self.total_cache_requested <= 0:
            return 0.0
        return (self.total_cache_saved / self.total_cache_requested) * 100
