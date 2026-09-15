import math
import random

# =========================================================================
# DOMÍNIO FÍSICO (Refatoração SRP: Isolamento de Telecomunicações)
# =========================================================================



class TelecomPhysics:
    """
    Módulo especialista puro para cálculos de camada física e rádio frequência.
    Garante que a classe de grafos não lide com termodinâmica e propagação de ondas.
    """
    BOLTZMANN: float = 1.380649e-23

    @staticmethod
    def calculate_5g_latency(
        data_bits: float,
        distancia_m: float = 750.0,
        potencia_transmissao_dbm: float = 30.0,
        largura_banda_hz: float = 100e6,
        temperatura_kelvin: float = 290.0,
        figura_ruido_db: float = 10.0,
        eficiencia_codec: float = 0.5,
        snr_minimo_db: float = 0.0,
        freq_portadora_hz: float = 3.5e9,
        sigma_shadowing_db: float = 0.0,
    ) -> float:
        """Calcula a latência de transmissão 5G via modelo de Shannon-Hartley."""
        # Evita log de zero caso a distância seja nula
        dist_segura = max(distancia_m, 1.0)

        pl_db = 28.0 + 22 * math.log10(dist_segura) + 20 * math.log10(freq_portadora_hz / 1e9)
        pl_db += random.gauss(0, sigma_shadowing_db)
        ganho = 10 ** (-pl_db / 10)

        potencia_w = (10 ** (potencia_transmissao_dbm / 10)) / 1000.0
        ruido_w_hz = TelecomPhysics.BOLTZMANN * temperatura_kelvin * (10 ** (figura_ruido_db / 10))
        snr_linear = (ganho * potencia_w) / (ruido_w_hz * largura_banda_hz)
        snr_linear = max(snr_linear, 10 ** (snr_minimo_db / 10))

        taxa_bps = largura_banda_hz * math.log2(1 + snr_linear) * eficiencia_codec

        if taxa_bps <= 0:
            return float('inf')
        return (data_bits / taxa_bps) * 1000.0
