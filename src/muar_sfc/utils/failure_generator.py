import logging
import random

# =============================================================================
# CONFIGURAÇÃO DE LOG
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calcular_janelas_falha(
    duracao_simulacao: float,
    num_falhas: int,
    duracao_minima_falha: float,
    confiabilidade: float,
    start_times: list[float] | None = None,
) -> list[tuple[float, float]]:
    """
    Gera janelas temporais de falha em uma simulação, respeitando:
    - o número de falhas,
    - a duração mínima de cada falha,
    - o orçamento total de downtime definido pela confiabilidade,
    - e opcionalmente tempos de início controlados.

    Parâmetros:
        duracao_simulacao (float):
            Tempo total da simulação (em segundos).

        num_falhas (int):
            Número total de eventos de falha a serem gerados.

        duracao_minima_falha (float):
            Duração mínima de cada evento de falha (em segundos).

        confiabilidade (float):
            Índice de confiabilidade do sistema (0.0 ≤ confiabilidade ≤ 1.0).
            A disponibilidade é calculada como:
                downtime_total = duracao_simulacao * (1 - confiabilidade)

        start_times (Optional[List[float]]):
            Lista opcional de tempos de início das falhas.
            Se fornecida, deve ter exatamente 'num_falhas' elementos.
            As durações continuam sendo calculadas pelo modelo.

    Retorno:
        List[Tuple[float, float]]:
            Lista ordenada de tuplas (inicio, duracao), em segundos.

    Exemplo:
        >>> calcular_janelas_falha(1000, 3, 5, 0.95, start_times=[100, 400, 700])
        [(100.0, 6.23), (400.0, 7.11), (700.0, 5.66)]
    """

    # =========================================================================
    # 1. VALIDAÇÕES DE ENTRADA
    # =========================================================================
    if num_falhas <= 0:
        return []

    if duracao_simulacao <= 0:
        raise ValueError("A duração da simulação deve ser positiva.")

    if duracao_minima_falha <= 0:
        raise ValueError("A duração mínima da falha deve ser positiva.")

    if not (0.0 <= confiabilidade <= 1.0):
        raise ValueError(f"Confiabilidade deve estar entre 0 e 1. Recebido: {confiabilidade}")

    if start_times is not None:
        if len(start_times) != num_falhas:
            raise ValueError(
                f"start_times deve conter exatamente {num_falhas} valores. "
                f"Recebido: {len(start_times)}"
            )
        if any(t < 0 or t > duracao_simulacao for t in start_times):
            raise ValueError(
                "Todos os tempos de início devem estar dentro da duração da simulação."
            )

    # =========================================================================
    # 2. CÁLCULO DO ORÇAMENTO TOTAL DE DOWNTIME
    # =========================================================================
    tempo_total_downtime = duracao_simulacao * (1.0 - confiabilidade)
    custo_minimo_necessario = num_falhas * duracao_minima_falha

    if tempo_total_downtime < custo_minimo_necessario:
        # logger.warning(
        #     "Downtime insuficiente para o número de falhas. "
        #     "Ajustando para o mínimo necessário."
        # )
        tempo_total_downtime = custo_minimo_necessario

    excedente = tempo_total_downtime - custo_minimo_necessario

    # =========================================================================
    # 3. DISTRIBUIÇÃO DAS DURAÇÕES DAS FALHAS (stick-breaking)
    # =========================================================================
    pesos = [random.random() for _ in range(num_falhas)]
    soma_pesos = sum(pesos) if sum(pesos) > 0 else 1.0

    duracoes: list[float] = []
    for peso in pesos:
        proporcao = peso / soma_pesos
        duracao = duracao_minima_falha + proporcao * excedente
        duracoes.append(round(duracao, 2))

    # =========================================================================
    # 4. DEFINIÇÃO DOS TEMPOS DE INÍCIO
    # =========================================================================
    cronograma: list[tuple[float, float]] = []

    for i in range(num_falhas):
        duracao_evento = duracoes[i]

        if start_times is not None:
            inicio_evento = start_times[i]
        else:
            janela_maxima_inicio = duracao_simulacao - duracao_evento
            if janela_maxima_inicio < 0:
                inicio_evento = 0.0
                duracao_evento = duracao_simulacao
            else:
                inicio_evento = random.uniform(0, janela_maxima_inicio)

        cronograma.append((round(inicio_evento, 2), duracao_evento))

    # =========================================================================
    # 5. ORDENAÇÃO CRONOLÓGICA
    # =========================================================================
    cronograma.sort(key=lambda evento: evento[0])

    return cronograma
