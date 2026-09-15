from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Definição do ROOT_DIR usando Pathlib (Seguro e Multiplataforma)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

class SimulationSettings(BaseSettings):
    """
    Configurações unificadas e validadas do Simulador SFC.
    O Pydantic fará a conversão automática de tipos e validação estrutural nativa.
    """
    model_config = SettingsConfigDict(
        env_prefix="MUAR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    time: float = 1000.0

    # --- Parâmetros Gerais e de Log ---
    application: str = "muar"
    alg: str = "replic"
    sfc_lifetime: int = 120

    # REFATORAÇÃO: Uso de booleanos absolutos em vez de strings arcaicas como "y"/"n"
    verbose: bool = True

    # --- Parâmetros de Topologia e Rede ---
    topology: str = "luxembourgv2"
    eco_effi_ratio: float = 0.7
    mobility: bool = True

    # --- Parâmetros de Entrada Obrigatórios e Tráfego ---
    n_sessions: int = 50
    n_players: int = 4

    # --- Parâmetros SFC ---
    allow_md_host: bool = True
    sfc: bool = True
    share: bool = True
    shareband: bool = False
    allow_delay: bool = False

    # --- Parâmetros de Confiabilidade e Falhas ---
    backup: bool = True
    ava: float = 0.99
    number_of_fails: int = 3
    min_fail_duration: float = 20.0
    crash_at: list[float] = [125,155,185]  # Exemplo de tempos fixos para falhas (em segundos)

    # --- Configuração MICRO (Confiabilidade Base por Nível) ---
    rel_high: float = 0.999
    rel_normal: float = 0.98
    rel_low: float = 0.95

    # --- Fator de Estresse ---
    stress_high: float = 0.02
    stress_normal: float = 0.08
    stress_low: float = 0.15

    fail_target: str = "all"  # Mantido como str caso represente categorias (enumeral) no futuro
    link_ava: float = 0.95
    number_of_link_fails: int = 0
    min_link_fail_duration: float = 20.0

    # --- Parâmetros Matemáticos da Simulação ---
    ia: float = 0.0
    number_of_nodes: int = 36
    src_node: int = 0
    max_duration: int = 120
    latency: float = 6.0
    fator: float = 0.25
    average_time_session_arrival: float = 10.0
    ia_bw: float = 150.0
    cpb: float = 10e6
    ca_size: float = 240.0
    chr: float = 1.0 / 3.0
    ft_bw: float = 0.144
    det_bw: float = 0.230

    # --- Campos Derivados ---
    ma_bw: float = 0.0
    ma: float = 0.0
    uni_bw: float = 0.0
    uni: float = 0.0
    re_bw: float = 0.0
    re: float = 0.0
    ec_tc_bw: float = 0.0
    ec_tc: float = 0.0
    det: float = 0.0
    ia_det_ft_bw: float = 0.0
    ia_det_ft: float = 0.0
    total: float = 0.0
    mono: float = 0.0
    ft: float = 0.0

    @model_validator(mode="after")
    def calculate_derived_parameters(self) -> "SimulationSettings":
        """Calcula os parâmetros derivados após a injeção dos dados."""
        self.ma_bw = int(self.ca_size * self.chr)
        self.ma = self.ma_bw * self.cpb
        self.uni_bw = int(self.ca_size * (1 - self.chr))
        self.uni = self.uni_bw * self.cpb
        self.re_bw = self.ma_bw + self.uni_bw
        self.re = int(self.re_bw * self.cpb)
        self.ec_tc_bw = int(self.ia_bw * 0.9 * 0.9 * 0.8)
        self.ec_tc = self.ec_tc_bw * self.cpb
        self.det = self.det_bw * self.cpb

        # REFATORAÇÃO LÓGICA: O cálculo de 'ft' foi realocado para o momento correto,
        # impedindo a injeção de 0.0 nas somatórias subsequentes.
        self.ft = self.ft_bw * self.cpb

        self.ia_det_ft_bw = int(self.ia_bw + self.det_bw + self.ft_bw)
        self.ia_det_ft = int(self.ia + self.det + self.ft)
        self.total = self.ia + self.det + self.ft + self.ma + self.uni + self.re + self.ec_tc
        self.mono = int(self.det + self.ft + self.ma + self.uni + self.re + self.ec_tc)

        return self
