import re


def subtrair_valor_padrao(texto_original: str, valor_menos: int) -> str:
    """
    Encontra o padrão "p{num1}_{num2}" em uma string e subtrai 6 de num2.

    Args:
        texto_original: A string de entrada.

    Returns:
        A string modificada.
    """
    # O padrão regex para encontrar "p", seguido por dígitos, um underscore, e mais dígitos.
    # Os parênteses ( ) criam "grupos de captura" para os números.
    # r"p(\d+)_(\d+)"
    #   p         -> Encontra o caractere literal 'p'
    #   (\d+)     -> Grupo 1: Encontra e captura um ou mais dígitos (o primeiro número)
    #   _         -> Encontra o caractere literal '_'
    #   (\d+)     -> Grupo 2: Encontra e captura um ou mais dígitos (o segundo número)
    pattern = r"p(\d+)_(\d+)"

    # A função re.sub pode aceitar uma outra função (ou uma expressão lambda)
    # para determinar pelo que substituir o padrão encontrado.
    # 'match' é um objeto que contém as partes capturadas pelo padrão.
    # match.group(1) é o primeiro número (como string)
    # match.group(2) é o segundo número (como string)
    texto_modificado = re.sub(
        pattern,
        lambda match: f"p{match.group(1)}_{int(match.group(2)) - valor_menos}",
        texto_original,
    )

    return texto_modificado


def add_mobile_user_to_graph(graph, sfc_list):
    mobile_device_id = sfc_list[0].dst_node
    closer_router = sfc_list[0].closer_router

    graph.add_node(
        mobile_device_id,
        type="mobile_device",
        cpu_capacity=25,
        cache_capacity=10,
        cpu_used=0,
        cache_used=0,
        position=59.97796454320501,
        services={},
        ips=10000000000.0,
        reuse=[],
    )
    router = graph._node[closer_router]
    wireless_free = router["w_channel_capacity"] - router["w_channel_used"]

    # TODO Permitir que o próprio algoritmo escolha o roteador
    # TODO calcular a latência do sinal
    signal_latency = 1
    graph.add_edge(
        mobile_device_id,
        closer_router,
        bandwidth_capacity=wireless_free,
        bandwidth_used=0.00,
        latency=signal_latency,
        services_in_transit={},
    )
