"""Source-based comparison of MuSFiCO, energy-aware DRL and the audited implementation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from create_muvr_report import Report, num
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/meu_treino_33_sementes_300/analysis"
OUT = ROOT / "output/pdf/relatorio_comparativo_artigos_mappo_qwen.pdf"
SOURCES = {
    "musfico": "https://doi.org/10.1109/GLOBECOM48099.2022.10000827",
    "thesis": "https://webarchive.inf.unibe.ch/cds/publications/files/Hugo_Santos.pdf",
    "energy": "https://sol.sbc.org.br/index.php/sbrc_estendido/article/download/42616/42383",
    "energy_page": "https://sol.sbc.org.br/index.php/sbrc_estendido/article/view/42616",
}


def dag_diagram(path):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    boxes = {
        "src": (0.15, 1.8, 1.1, 0.7, "Origem"),
        "ia": (1.65, 1.8, 1.1, 0.7, "IA"),
        "pred": (3.4, 3.2, 3.0, 0.7, "BE predito\ncomum + exclusivo"),
        "view": (3.4, 1.8, 3.0, 0.7, "BE da visão\ncomum + exclusivo"),
        "fi": (3.4, 0.4, 3.0, 0.7, "FI exclusivo por jogador"),
        "pr": (7.0, 3.2, 2.0, 0.7, "Renderização\npreditiva"),
        "vr": (7.0, 1.5, 2.0, 0.7, "Renderização\nda visão"),
        "codec": (9.5, 1.8, 1.5, 0.7, "Codec"),
        "dst": (11.5, 1.8, 1.1, 0.7, "Destino"),
    }
    edges = [
        ("src", "ia"),
        ("ia", "pred"),
        ("ia", "view"),
        ("ia", "fi"),
        ("pred", "pr"),
        ("view", "vr"),
        ("fi", "vr"),
        ("pr", "codec"),
        ("vr", "codec"),
        ("codec", "dst"),
    ]
    for u, v in edges:
        x, y, w, h, _ = boxes[u]
        xx, yy, _, hh, _ = boxes[v]
        ax.add_patch(
            FancyArrowPatch(
                (x + w, y + h / 2),
                (xx, yy + hh / 2),
                arrowstyle="-|>",
                mutation_scale=13,
                color="#526170",
                linewidth=1.2,
            )
        )
    for key, (x, y, w, h, label) in boxes.items():
        color = "#fbe4d8" if key == "fi" else "#e8e0f1" if key in ("pred", "view") else "#edf3f7"
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06",
                facecolor=color,
                edgecolor="#183249",
                linewidth=1,
            )
        )
        ax.text(
            x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, color="#183249"
        )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def picture(r, path, maximum_height=None):
    with Image.open(path) as im:
        ratio = im.height / im.width
    width = min(r.available, (maximum_height or (r.y - 125)) / ratio)
    height = width * ratio
    r.pdf.drawImage(
        str(path),
        r.margin + (r.available - width) / 2,
        r.y - height,
        width=width,
        height=height,
        mask="auto",
    )
    r.y -= height + 12


def main():
    s = json.loads((DATA / "summary.json").read_text())
    assert s["status"] == "passed"
    root_results = ROOT / "results"
    previous = json.loads((root_results / "mappo_qwen_muvr/plots/summary.json").read_text())
    asynchronous = json.loads(
        (root_results / "mappo_qwen_muvr_async/plots/summary.json").read_text()
    )
    before = next(row for row in previous if row["model"] == "mappo_qwen")
    after = next(row for row in asynchronous if row["model"] == "mappo_qwen")
    reduction = 100 * (1 - after["mean_decision_ms"] / before["mean_decision_ms"])
    temporary = ROOT / "tmp/pdfs/comparativo_artigos_revisado"
    temporary.mkdir(parents=True, exist_ok=True)
    dag_diagram(temporary / "dag.png")
    r = Report(output=OUT, total=24, date="14 de outubro de 2026")
    r.pdf.setTitle("Nossa implementação: DAG, MAPPO e Qwen - alterações e resultados preliminares")
    r.pdf.setSubject("Evolução técnica, comparação a priori e oportunidades de melhoria")
    r.start(
        "Nossa implementação<br/>DAG, MAPPO e Qwen",
        "Alterações realizadas, sua motivação e resultados preliminares. Última avaliação: 33 sementes com 300 requisições cada. A implementação ainda está sujeita a melhorias.",
    )
    r.cards(
        [
            (num(s["acceptance_percent"]) + "%", "Aceitação: 9.029/9.900"),
            (num(s["latency_ms"]["mean"], 3) + " ms", "Latência média da SFC"),
            (num(s["decision_ms"]["mean"]) + " ms", "Decisão média"),
        ]
    )
    r.h("O que construímos")
    r.p(
        "A solução integra três componentes com papéis diferentes: <b>o DAG descreve as dependências do serviço; o MAPPO aprende a propor servidores; o Qwen orienta a classificação dessas propostas a partir da telemetria</b>. A aceitação depende, em todos os casos, de recursos físicos disponíveis e do SLA de 6 ms. O resultado não foi obtido aumentando CPU, memória ou banda."
    )
    r.table(
        [
            ["Alteração", "Como contribuiu"],
            [
                "DAG e reservas completas",
                "Representação explícita de ramos paralelos, caminho crítico, banda por dependência e liberação correta de recursos",
            ],
            [
                "MAPPO regional treinado",
                "Quatro atores com máscaras de viabilidade e escolhas auditáveis; cooperação aprendida no treino",
            ],
            [
                "Qwen em segundo plano",
                "Orientação estratégica sem aguardar a resposta do LLM no caminho de admissão",
            ],
            [
                "Avaliação ampliada",
                "33 redes independentes, 9.900 solicitações e percentis; caracterização mais ampla da variação",
            ],
        ],
        [145, r.available - 145],
        size=9,
        padding=7,
    )
    r.h("Melhoria medida e resultado atual")
    r.p(
        f"No comparativo interno de três sementes com 50 sessões cada, mover o Qwen para segundo plano reduziu a decisão média em <b>{num(reduction)}%</b>, mantendo a aceitação agregada. Na avaliação ampliada, a solução aceitou <b>91,20%</b> das sessões e todas as aceitas atenderam ao SLA modelado. A ampliação do conjunto de sementes não é, por si, um ganho do algoritmo."
    )
    r.h("Resultado preliminar, sujeito a melhoria")
    r.p(
        "Os números descrevem esta configuração e esta amostra. Ainda precisamos aperfeiçoar a política e o custo de decisão, isolar os efeitos de cada componente e validar energia, mobilidade e latência visual de AR. O relatório distingue melhorias de implementação, efeitos medidos e benefícios esperados que ainda precisam de experimento.",
        size=9.5,
        leading=14,
    )

    r.start(
        "0. O problema MEC + IMS + MSC",
        "A adequação de um algoritmo depende das decisões, dependências e restrições da aplicação, não apenas de sua classificação como DRL.",
    )
    r.h("O que cada parte do contexto exige")
    r.table(
        [
            ["Dimensão", "Problema concreto", "Requisito do orquestrador"],
            [
                "MEC: computação de borda",
                "Servidores têm capacidade limitada e distâncias diferentes dos usuários; os enlaces podem ficar ocupados.",
                "Escolher hosts e rotas considerando capacidade residual, atraso e velocidade de processamento.",
            ],
            [
                "IMS: serviço multimídia imersivo",
                "A entrega precisa ser rápida e as partes do conteúdo precisam chegar de forma compatível.",
                "Respeitar SLA de serviço e sincronização; manter a orquestração separada da atualização visual.",
            ],
            [
                "MSC: cadeia de microsserviços",
                "A posição de uma função altera o custo das seguintes; partes comuns podem ser compartilhadas.",
                "Modelar precedência, junções, reuso e aceitação da sessão completa, não só de funções isoladas.",
            ],
        ],
        [115, 195, r.available - 310],
        size=9,
        padding=7,
    )
    r.h("Por que uma escolha local pode ser insuficiente")
    r.p(
        "Exemplo didático: o servidor mais próximo tem CPU para receber o renderizador, mas a rota até o codec não possui banda suficiente. Um segundo servidor é um pouco mais distante, porém mantém a cadeia completa dentro do SLA. Escolher apenas por distância ou CPU livre pode levar a uma rejeição posterior. Precisamos verificar a combinação de processamento, fluxo e destino."
    )
    r.h("Três camadas da nossa solução")
    r.p(
        "<b>Representação:</b> o DAG expressa o que depende de quê.<br/><b>Aprendizado e orientação:</b> MAPPO propõe hosts e Qwen influencia o ranking.<br/><b>Viabilidade:</b> reservas, rotas, máscaras e validação final determinam se a sessão pode ser aceita. Essas camadas se complementam; nenhum modelo aprendido transforma uma solução fisicamente inviável em válida."
    )
    r.h("O que está realmente sendo simulado")
    r.p(
        "A carga atual é MUVR estática, escolhida como base de estudo para o projeto de AR: 25 servidores, quatro jogadores por sessão, chegadas a cada 10 segundos e lifetime até 120 segundos. Custos de processamento são calculados por tráfego, 60 quadros/s e IPS modelado. Não estamos renderizando imagens nem medindo um headset; o SLA de 6 ms cobre a SFC modelada."
    )
    r.h("Critério de escolha dos modelos")
    r.p(
        "A pergunta correta é: como cada alternativa representará hosts discretos, dependências do DAG, reuso e restrições? Um algoritmo centralizado pode coordenar a rede inteira; um multiagente pode organizar regiões. Para ambos, é necessário definir estado, ações, recompensas e validação compatíveis com a aplicação.",
        size=9.5,
        leading=14,
    )

    r.start(
        "1. DAG: o que é, como usamos e por quê",
        "Directed Acyclic Graph: grafo direcionado acíclico, usado para representar tarefas e suas dependências.",
    )
    picture(r, temporary / "dag.png", maximum_height=175)
    r.p(
        "Diagrama agregado da nossa cadeia. BE contém funções comuns e exclusivas; FI é exclusivo por jogador. Uma sessão de quatro jogadores possui 18 SFs distintas; cada jogador participa de um subgrafo de nove funções, com instâncias comuns compartilhadas.",
        size=9,
        leading=13,
    )
    r.h("O que significa")
    r.p(
        "Cada vértice representa uma função de serviço. Uma seta indica que os dados de uma função alimentam outra. “Direcionado” significa que o fluxo tem sentido; “acíclico” significa que não há um caminho que volte à mesma tarefa. O grafo pode se dividir em ramos paralelos e depois reuni-los, sem impor uma sequência única a todo o processamento."
    )
    r.h("Como foi aplicado")
    r.p(
        "Preservamos as bifurcações e junções da cadeia. O processamento de fundo comum e exclusivo e os ramos preditivos/de visão têm dependências próprias. Todas as arestas possuem rotas e reservas de banda; nas junções, somamos os fluxos de entrada. O atraso da SFC é o do <b>caminho crítico</b>: o maior atraso entre os caminhos de origem a destino, incluindo processamento e comunicação."
    )
    r.h("Por que usamos e como isso melhora o modelo")
    r.p(
        "Uma sequência linear pode somar tarefas que poderiam ocorrer em paralelo; olhar apenas um ramo pode esquecer a banda dos demais. O DAG permite calcular corretamente o atraso sem omitir recursos. Exemplo didático: após uma etapa comum de 1 ms, dois ramos de 2 e 3 ms terminam em 1 + max(2,3) = 4 ms, desde que existam recursos para executar ambos. Este exemplo não é uma medição do experimento."
    )
    r.p(
        "O compartilhamento reutiliza apenas instâncias e fluxos compatíveis. Contadores de referência garantem que liberar uma sessão não remova um recurso ainda usado por outra. O DAG melhora a fidelidade e a correção da simulação; seu ganho isolado de aceitação ainda precisa de ablação. Paralelismo e reuso também aparecem nos trabalhos relacionados [2,3].",
        size=9.5,
        leading=14,
    )

    r.start(
        "1.1. Por que um DAG para esta aplicação?",
        "A escolha do tipo de grafo depende do que ele representa: dependência de processamento ou conectividade física.",
    )
    r.h("Como é um DAG")
    r.p(
        "Um DAG é um grafo com setas e sem ciclos direcionados. Ele admite uma ordenação topológica: se A fornece dados para B, A aparece antes de B nessa ordem. Funções independentes podem estar em ramos diferentes. A ordem é parcial: não é obrigatório colocar em sequência tarefas que não dependem uma da outra."
    )
    r.table(
        [
            ["Representação", "Diferença e adequação"],
            [
                "Grafo não direcionado",
                "A-B expressa ligação, sem indicar quem fornece dados a quem. Adequado à conectividade física bidirecional, mas insuficiente sozinho para expressar a ordem do processamento.",
            ],
            [
                "Grafo direcionado geral",
                "Expressa A→B, mas pode conter A→B→A. É mais geral que um DAG; ciclos exigem uma semântica de repetição ou feedback para calcular quando o processamento termina.",
            ],
            [
                "Cadeia linear orientada",
                "É um caso particular de DAG: A→B→C. Representa etapas sequenciais, mas não preserva a bifurcação e a reunião de dois ramos sem transformar a estrutura.",
            ],
            [
                "Arborescência: árvore orientada da raiz",
                "Também é um caso particular de DAG, mas cada vértice não inicial tem um único pai. Modela expansão de tarefas; não representa naturalmente o codec com entradas de renderizadores diferentes.",
            ],
            [
                "DAG geral",
                "Permite bifurcações e junções, sem dependência circular. Preserva múltiplas entradas do codec e processamento de fundo/visão/predição com dependências explícitas.",
            ],
        ],
        [155, r.available - 155],
        size=8.8,
        padding=6,
    )
    r.h("Por que é adequado aqui")
    r.p(
        "Em um quadro da aplicação, dados de interação alimentam ramos de processamento; os resultados de renderização alimentam a codificação, que precede a entrega. Um ramo pode terminar antes do outro, mas a função que depende de ambos deve esperar os dados necessários. O DAG representa essa precedência e permite medir o caminho crítico, preservando todas as reservas de recursos."
    )
    r.h("O DAG da aplicação não é a topologia da rede")
    r.p(
        "O grafo físico Erdős-Rényi conecta servidores e pode conter ciclos. Esses ciclos oferecem rotas alternativas. O DAG conecta funções e determina a ordem dos dados. Mapear a aplicação significa escolher um servidor para cada função e um caminho físico para cada dependência; uma seta do DAG pode percorrer vários enlaces da rede."
    )
    r.h("Limites da escolha")
    r.p(
        "Interações contínuas e feedback entre quadros pertencem ao controle temporal da aplicação, não precisam formar um ciclo dentro do DAG de um quadro. Tarefas iterativas exigiriam outro modelo ou expansão por iteração. O DAG, sozinho, não comprova paralelismo real, filas de CPU ou atraso do display. O cálculo atual enumera caminhos; se o grafo crescer, uma implementação por programação dinâmica em ordem topológica pode reduzir esse custo.",
        size=9.3,
        leading=13.2,
    )

    r.start(
        "1.1.1. Justificativa técnica da representação",
        "O DAG foi escolhido para expressar precedências, calcular atraso e reservar todos os fluxos da sessão.",
    )
    r.h("Critério principal: preservar fork e join")
    r.p(
        "A aplicação tem trabalho comum, ramos de visão/predição e uma etapa que reúne resultados. Uma cadeia linear impõe precedências extras entre ramos independentes; uma árvore orientada de expansão não admite naturalmente uma função com múltiplos predecessores. O DAG geral representa ambos sem duplicar a etapa de reunião. A escolha decorre dessa estrutura de dados e processamento, e não de o DAG ser universalmente superior a outros grafos.",
        size=9.3, leading=13,
    )
    r.h("Latência: caminho crítico em vez de soma de todos os ramos")
    r.p(
        "Para uma junção que precisa de todos os resultados, a conclusão depende do ramo mais lento. Em um modelo sem filas e com processamento paralelo disponível, calcula-se o instante de cada função a partir do máximo dos instantes de chegada de seus predecessores, acrescido do seu processamento. Isso evita que uma serialização artificial rejeite uma sessão que caberia no SLA de 6 ms. A implementação atual obtém o máximo enumerando caminhos; não executa ainda essa recorrência por programação dinâmica.",
        size=9.3, leading=13,
    )
    r.h("Banda: paralelismo não elimina o consumo dos outros ramos")
    r.p(
        "Mesmo que apenas um ramo determine o atraso, cada dependência transporta dados e precisa de uma rota com banda disponível. O DAG separa essas duas contas: máximo dos caminhos para o atraso e reservas para todas as arestas. Considerar apenas o caminho crítico na reserva subestimaria o consumo e poderia produzir aceitações fisicamente inválidas. O reuso exige ainda compatibilidade de instância e fluxo; não basta dois vértices terem o mesmo nome.",
        size=9.3, leading=13,
    )
    r.h("Por que não usar um grafo direcionado com ciclos aqui?")
    r.p(
        "Um ciclo expressaria que uma tarefa depende, direta ou indiretamente, do próprio resultado. Para representar iterações, seria necessário definir condições de parada e número de repetições ou expandir a execução no tempo. Nosso objeto é a cadeia de processamento de um quadro/sessão com precedências finitas. A ausência de ciclos permite ordenar as funções para alocação sem resolver uma dependência circular. Feedback entre quadros pode ser modelado em outra escala temporal.",
        size=9.3, leading=13,
    )
    r.h("Separação entre aplicação e infraestrutura")
    r.p(
        "O DAG determina quem produz dados para quem; o grafo físico não direcionado determina por onde esses dados podem passar. Ciclos físicos são úteis como rotas alternativas e permanecem na rede Erdős-Rényi. Manter as duas representações permite trocar hosts e caminhos sem alterar as dependências multimídia. Não há motivo técnico para exigir que a rede física também seja acíclica.",
        size=9.3, leading=13,
    )
    r.h("Tradeoff e verificação da melhoria")
    r.p(
        "O DAG aumenta a fidelidade da modelagem, mas pressupor paralelismo sem modelar filas pode subestimar o atraso real. O próximo teste deve comparar representações com as mesmas demandas e recursos, verificando reservas e conclusão de todas as funções. Assim, uma eventual melhora de aceitação pode ser atribuída à remoção de serialização artificial, e não à omissão de trabalho ou à ampliação da capacidade.",
        size=9.2, leading=13.2,
    )

    r.start(
        "1.2. DAG: cálculo, reserva e sincronização",
        "A estrutura do grafo define precedência; a alocação física determina o custo de executar cada dependência.",
    )
    r.h("Tempo de conclusão de uma função")
    r.p(
        "Se uma função v depende de predecessores u, seu início ocorre depois da última entrada necessária. No modelo, podemos escrever:<br/><b>T(v) = max entre predecessores de [T(u) + atraso da rota u→v] + processamento de v.</b><br/>A saída final determina a latência da SFC. Essa regra equivale ao caminho crítico do DAG sob as hipóteses de processamento usadas."
    )
    r.h("Exemplo completo de bifurcação e junção")
    r.table(
        [
            ["Etapa didática", "Conclusão"],
            ["Preparação comum", "1 ms"],
            ["Ramo preditivo: comunicação + processamento = 2 ms", "1 + 2 = 3 ms"],
            ["Ramo de visão: comunicação + processamento = 3 ms", "1 + 3 = 4 ms"],
            ["Codec depende dos dois ramos e processa por 0,5 ms", "max(3,4) + 0,5 = 4,5 ms"],
        ],
        [350, r.available - 350],
        size=9,
        padding=7,
    )
    r.p(
        "Os números são ilustrativos, não resultados do experimento. A diferença de chegada na junção é de 1 ms. Somar os dois ramos como uma sequência produziria outro atraso e perderia a semântica de processamento independente."
    )
    r.h("Atraso máximo não significa reserva de apenas um ramo")
    r.p(
        "Mesmo que o ramo de visão determine o atraso, o ramo preditivo também consome recursos. Cada dependência deve reservar banda ao longo da sua rota; funções novas reservam CPU/memória e funções compatíveis podem reutilizar a mesma instância. Na liberação, removemos a referência da sessão e preservamos recursos que ainda possuem usuários."
    )
    r.h("Três perguntas diferentes")
    r.p(
        "<b>Há capacidade?</b> Todas as funções e fluxos cabem?<br/><b>Há tempo?</b> O caminho crítico atende a 6 ms?<br/><b>Há sincronização?</b> A diferença entre chegadas na junção respeita a tolerância modelada? Uma resposta positiva não substitui as outras. O orquestrador precisa verificar as três antes de publicar a alocação."
    )
    r.h("Limite do modelo e oportunidade de melhoria")
    r.p(
        "O cálculo usa atraso de propagação e processamento estimado, sem uma simulação temporal detalhada de filas ou escalonamento de threads. Ele não prova que os ramos executarão em paralelo em hardware. Com DAGs maiores, enumerar todos os caminhos pode custar mais; calcular o caminho crítico por programação dinâmica em ordem topológica é uma oportunidade de otimização. Isso deve preservar as reservas de todos os ramos.",
        size=9.5,
        leading=14,
    )

    r.start(
        "2. MAPPO: o que é, como usamos e por quê",
        "Multi-Agent Proximal Policy Optimization: aprendizado por reforço cooperativo com múltiplos agentes.",
    )
    r.h("O que é")
    r.p(
        "No aprendizado por reforço, um agente observa o ambiente, escolhe uma ação e recebe uma recompensa. O PPO ajusta a política com atualizações limitadas para dar estabilidade ao treino. O MAPPO aplica essa ideia a agentes cooperativos: cada ator propõe ações a partir de sua observação, enquanto um crítico centralizado usa o estado global para orientar o aprendizado."
    )
    r.h("Como foi aplicado")
    r.table(
        [
            ["Componente", "Nossa configuração"],
            [
                "Agentes",
                "Quatro atores regionais, cada um com seis nós locais e acesso a um nó global de borda finito",
            ],
            [
                "Observações / ações",
                "48 valores por ator; sete ações possíveis; estado global de 196 valores",
            ],
            [
                "Estado observado",
                "Recursos livres, distâncias ao destino e velocidade relativa de processamento",
            ],
            [
                "Treinamento",
                "100 demonstrações; 300 episódios MAPPO; 2.534 passos e 60 atualizações",
            ],
            [
                "Inferência",
                "Pesos fixos, máscaras de viabilidade e somente hosts propostos pelos atores",
            ],
            [
                "Alternativas",
                "Primeira tentativa determinística; até oito tentativas internas por jogador com ações amostradas da política",
            ],
        ],
        [135, r.available - 135],
        size=9,
        padding=6,
    )
    r.h("Por que usamos")
    r.p(
        "A decisão envolve múltiplos servidores e interesses concorrentes: reduzir atraso, evitar nós ocupados e aproveitar funções compartilhadas. A regionalização organiza a busca e permite aprender cooperação entre escolhas locais usando uma avaliação global no treino. O crítico não aloca funções; os atores propõem hosts e o validador verifica a viabilidade física."
    )
    r.h("O que mudou na decisão")
    r.p(
        "A seleção passou a usar um checkpoint treinado para o cenário MUVR. As máscaras rejeitam candidatos sem recursos ou cujo limite otimista de atraso restante já excede o SLA. Não há resgate por uma heurística que procura todos os nós quando os atores falham. A execução dos quatro atores continua no processo Python: a divisão é conceitual, não uma medição de controle distribuído em hardware."
    )
    r.p(
        "A taxa no conjunto isolado de 30 casos permaneceu em 86,67% antes e depois do treino. Portanto, o MAPPO está treinado e é efetivamente aplicado, mas ainda não demonstramos um ganho isolado dessa etapa. O próximo objetivo é melhorar sua generalização e reduzir propostas inviáveis.",
        size=9.5,
        leading=14,
    )

    r.start(
        "2.1. Escolha técnica do MAPPO para esta aplicação",
        "Critérios: coordenação regional, decisões discretas, dependências da sessão e disputa por recursos finitos.",
    )
    r.p(
        "Em MEC + IMS + MSC, escolher um host afeta as funções seguintes, os enlaces entre regiões e o reuso entre jogadores. O objetivo é concluir a sessão dentro do SLA, e não apenas encontrar um servidor livre para a função atual. Escolhemos <b>MAPPO com máscaras</b> para combinar propostas regionais com aprendizado orientado ao resultado conjunto."
    )
    r.table(
        [
            ["Alternativa", "Aplicação ao nosso problema", "Razão da escolha relativa do MAPPO"],
            [
                "PPO centralizado [4]",
                "Um único ator pode observar toda a rede e coordenar a alocação sequencial do DAG.",
                "MAPPO mantém atores por região e crítico global no treino. Essa decomposição corresponde aos domínios locais já definidos; PPO centralizado também é viável, com outra organização da política.",
            ],
            [
                "PPO com máscaras / MPPO [6,12]",
                "Remove hosts inviáveis antes da escolha; pode tratar CPU, memória, banda e limites de atraso.",
                "Máscaras tratam viabilidade, mas não definem cooperação. Usamos esse mecanismo junto ao MAPPO para combinar restrições físicas e aprendizado regional cooperativo.",
            ],
            [
                "IPPO [5]",
                "Atores regionais aprendem com críticos locais; a estrutura de ações pode ser a mesma.",
                "O crítico global do MAPPO recebe o estado conjunto para avaliar efeitos entre regiões. Isso é pertinente quando uma reserva local afeta a conclusão de toda a sessão.",
            ],
            [
                "SAC discreto [7]",
                "Pode escolher hosts e reutilizar experiências; é uma alternativa pertinente quando amostras de treino são caras.",
                "Nosso desenho prioriza atores cooperativos com crítico global e atualizações PPO. SAC exigiria formular uma coordenação equivalente; sua natureza off-policy é um tradeoff, não uma incompatibilidade com MEC.",
            ],
            [
                "DQN [8]",
                "Valores Q podem ordenar hosts discretos em cada etapa da alocação.",
                "MAPPO fornece diretamente políticas regionais categóricas e treino cooperativo. DQN centralizado ou multiagente também exigiria representar dependências e coordenar os efeitos das escolhas locais.",
            ],
        ],
        [100, 175, r.available - 275],
        size=8.1,
        padding=4,
    )
    r.h("O diferencial relevante é a avaliação conjunta")
    r.p(
        "Ações discretas e máscaras também existem nas alternativas. O motivo específico para MAPPO é alinhar os atores regionais a um objetivo de sessão, usando informação global no crítico durante o treino [5]. O crítico estima retorno esperado; ele não é um solver nem garante uma alocação ótima."
    )
    r.p(
        "MPPO significa aqui PPO com máscaras [12]. A comparação fundamenta a escolha de arquitetura; as 33 sementes avaliaram MAPPO + Qwen, não um benchmark entre todos esses modelos.",
        size=9,
        leading=13,
    )

    r.start(
        "2.1.1. Como os requisitos sustentam essa escolha",
        "A adequação vem da relação entre estado, ações, objetivo e restrições da orquestração.",
    )
    r.h("Recursos locais, consequências globais")
    r.p(
        "CPU e memória pertencem aos hosts, mas banda e atraso dependem dos caminhos entre funções. Uma decisão regional pode consumir um enlace necessário a outra região ou tornar inviável a etapa final. No treino, o crítico central recebe as observações das quatro regiões e indicadores globais, oferecendo contexto para avaliar o retorno das propostas locais. Esse acesso à informação é a razão técnica para preferir MAPPO a críticos exclusivamente locais neste desenho."
    )
    r.h("Dependências do DAG e objetivo de sessão")
    r.p(
        "A cadeia multimídia contém ramificações e sincronização: um host com baixa carga pode estar longe dos predecessores ou do destino. Reuso compatível também altera o custo incremental de uma escolha. Nosso ambiente fornece atributos da função, conectividade lógica, distâncias e recursos; o aprendizado considera as consequências ao longo da alocação. O DAG e o mecanismo de reuso são modelados pelo orquestrador: não surgem automaticamente por usar MAPPO e podem ser preservados em outros modelos."
    )
    r.h("Decomposição compatível com o controlador implementado")
    r.p(
        "Há quatro atores, cada um com seis hosts locais e uma opção de nó global. Essa organização produz propostas regionais com conjuntos pequenos de ações, enquanto o avaliador físico escolhe entre elas. Um PPO centralizado poderia selecionar um host por etapa sem enumerar todas as alocações possíveis; nossa preferência pela decomposição regional é uma escolha de organização e treino, não uma alegação de que políticas centralizadas sejam inviáveis."
    )
    r.h("SLA rígido exige validação além do aprendizado")
    r.p(
        "O limite de 6 ms, as reservas e a sincronização são verificados por máscaras e validação física. Penalizar violações na recompensa não basta para garantir o SLA. MAPPO aprende propostas dentro desse mecanismo, e Qwen pode orientar sua ordenação. A segurança da reserva é responsabilidade do simulador; esse mesmo controle deve existir ao comparar PPO, IPPO, SAC discreto ou DQN."
    )
    r.h("Tradeoffs assumidos e critério de avaliação")
    r.p(
        "MAPPO reutiliza menos dados entre atualizações do que métodos off-policy, exige treinar atores e crítico e pode deixar de propor um host viável. A execução atual continua centralizada em Python; CTDE descreve a organização do aprendizado, não uma implantação distribuída já realizada. A escolha é tecnicamente coerente com a cooperação regional, mas a superioridade em aceitação precisa ser medida com as mesmas sementes, máscaras, recompensas, recursos e orçamento de treino. O custo das máscaras e da busca também entra no tempo de decisão, que não é determinado apenas pelo modelo DRL.",
        size=9.2,
        leading=13.2,
    )

    r.start(
        "2.2. Análise das features de entrada",
        "Além das funcionalidades do orquestrador, as observações numéricas ajudam os atores a decidir entre propostas viáveis.",
    )
    r.table(
        [
            ["Grupo de features", "Papel e motivo de uso", "Limite / oportunidade"],
            [
                "CPU e memória usadas por nó",
                "Representam ocupação local. Ajudam a escolher servidores com folga e equilibrar carga.",
                "Uso livre não implica menor atraso; velocidade e rota também importam.",
            ],
            [
                "Banda livre média dos enlaces incidentes",
                "Resume pressão de comunicação próxima ao servidor e contextualiza a escolha.",
                "Uma média pode esconder um enlace saturado; a máscara verifica as rotas concretas.",
            ],
            [
                "Distância da posição atual ao host",
                "Estima a comunicação até o candidato e favorece proximidade.",
                "A distância resumida não substitui o conjunto de dependências de uma junção.",
            ],
            [
                "Demanda de CPU, memória e banda da função",
                "Relaciona o custo da próxima função com a capacidade disponível.",
                "As escalas atuais são fixas; recalibrá-las para o perfil exige validar ou retreinar o checkpoint.",
            ],
            [
                "Indicadores fork/join e região de destino",
                "Sinalizam bifurcação, múltiplos predecessores e localização regional da entrega.",
                "São resumos da estrutura, não uma representação completa de todas as rotas do DAG.",
            ],
            [
                "Distância ao destino e IPS relativo",
                "Diferenciam hosts próximos da entrega e hosts rápidos. Evitam avaliar apenas a proximidade atual.",
                "A fórmula de processamento é uma hipótese do simulador, sem medição de hardware real.",
            ],
            [
                "Estado do nó global de borda",
                "Informa CPU/memória, distâncias, velocidade e disponibilidade da opção comum às quatro regiões.",
                "É um nó de capacidade finita; não representa nuvem gratuita ou recursos adicionais.",
            ],
            [
                "Estado global do crítico",
                "Agrega observações, uso médio, desequilíbrio, atraso parcial e progresso para treinar cooperação.",
                "É usado no treino. Não há atualização online dos pesos durante o benchmark.",
            ],
        ],
        [120, 195, r.available - 315],
        size=8.6,
        padding=6,
    )
    r.h("Como os vetores são organizados")
    r.p(
        "Cada ator recebe 24 valores dos seis nós locais (quatro por nó), seis características da função, 12 valores de distância ao destino/IPS e seis do nó global: <b>48 valores</b>. O crítico usa as quatro observações concatenadas e quatro estatísticas globais: <b>196 valores</b>. Normalização reduz diferenças de escala, mas não garante por si só boa generalização."
    )
    r.h("Observação, máscara e orientação são diferentes")
    r.p(
        "A observação informa o ator; a máscara elimina ações inviáveis; a orientação Qwen fornece preferências ao ranking. Uma previsão favorável da rede neural ou do LLM nunca substitui a validação de recursos, rotas e SLA. Ainda precisamos medir a importância de cada feature retirando-a de forma controlada e repetindo treino/avaliação.",
        size=9.3,
        leading=13.2,
    )

    r.start(
        "2.3. MAPPO: da observação à sessão aceita",
        "A inferência dos atores é uma etapa de um processo híbrido de decisão, não a validação final da cadeia.",
    )
    r.h("Passo a passo aplicado a uma função")
    r.p(
        "<b>1.</b> Selecionar a próxima função respeitando a precedência do DAG.<br/><b>2.</b> Atualizar observações da função e dos servidores.<br/><b>3.</b> Calcular máscaras com reservas, rotas, sincronização e limite de atraso restante.<br/><b>4.</b> Cada ator propõe um host elegível da sua região ou o nó global finito.<br/><b>5.</b> Classificar as propostas viáveis com custo de latência, carga, banda e reuso; usar orientação válida do Qwen quando disponível.<br/><b>6.</b> Reservar a escolha no plano temporário e avançar para a próxima função.<br/><b>7.</b> Ao terminar, verificar entrega e SLA de todos os jogadores antes de publicar a rede candidata."
    )
    r.h("Exemplo didático: função viável, cadeia inviável")
    r.p(
        "Um host tem CPU suficiente e uma rota de entrada rápida. Porém, o atraso parcial somado à menor comunicação até o destino e ao processamento mínimo das funções restantes já ultrapassa 6 ms. A máscara elimina essa escolha antes de o ator selecioná-la. Essa antecipação reduz escolhas inviáveis conhecidas, mas um limite inferior favorável não garante que o restante do DAG caiba."
    )
    r.h("Como treinamento cooperativo entra na decisão")
    r.p(
        "O crítico estima o valor do estado global e orienta as atualizações de vantagem no treinamento. Os atores recebem gradientes para melhorar suas políticas regionais sob o objetivo comum. Demonstrações iniciam as políticas; depois PPO/GAE faz atualizações limitadas. Durante o benchmark, os pesos ficam fixos: não estamos treinando o modelo a cada chegada."
    )
    r.h("Por que cooperação tem sentido aqui")
    r.p(
        "Duas regiões podem disputar o mesmo nó global, alterar rotas de funções que convergem ou reservar recursos necessários a funções posteriores. Um crítico global pode usar esses efeitos no aprendizado. Um agente único bem formulado também pode coordená-los; a escolha multiagente é uma organização da solução, não uma condição para que MEC/IMS/MSC seja tratável."
    )
    r.h("Como medir se o aprendizado está contribuindo")
    r.p(
        "Precisamos comparar MAPPO treinado, política aleatória entre ações válidas e métodos alternativos com as mesmas máscaras e ranking. Assim podemos distinguir aprendizado dos ganhos produzidos pelo validador, pelo reuso e pela busca. O sucesso isolado de 86,67% antes/depois indica que essa verificação é importante; não anula o funcionamento, mas impede atribuir toda a aceitação ao treinamento."
    )
    r.h("Custo de execução")
    r.p(
        "Atores regionais, cálculo de máscaras, cópias de planos e caminhos físicos compõem o tempo de decisão. O máximo de oito tentativas pode recuperar algumas escolhas, mas também aumenta custo. Otimizar a execução exige medir essas etapas separadamente, preservando a viabilidade e sem tratar as restrições como simples penalidades.",
        size=9.5,
        leading=14,
    )

    r.start(
        "3. Qwen: o que é, como usamos e por quê",
        "Modelo de linguagem pré-treinado que fornece orientação de alocação a partir do estado da rede.",
    )
    r.h("O que é")
    r.p(
        "Qwen é uma família de modelos de linguagem capazes de interpretar instruções e dados e produzir respostas estruturadas. Usamos <b>Qwen2.5:1.5b</b>, já pré-treinado, pelo Ollama local em CPU. Não treinamos o Qwen neste projeto; usamos inferência para obter orientação. Ele não substitui os atores MAPPO nem a validação de recursos."
    )
    r.h("Como foi aplicado")
    r.p(
        "Enviamos um snapshot compacto com recursos livres, distâncias e velocidade de processamento. O modelo devolve JSON com preferências de servidores e pesos de classificação. Os atores MAPPO produzem os candidatos; a orientação do Qwen participa do ranking entre propostas viáveis. O LLM não pode aumentar capacidades, relaxar SLA ou alterar os pesos do checkpoint."
    )
    r.h("Por que usamos")
    r.p(
        "O objetivo é permitir orientação contextual além da política já treinada: por exemplo, priorizar caminhos curtos, hosts rápidos ou distribuição de carga conforme a telemetria. O papel do Qwen é estratégico; máscaras e validação física continuam sendo as garantias de viabilidade. A hipótese de que essa orientação melhora a aceitação deve ser verificada com MAPPO sozinho e MAPPO + Qwen nas mesmas sementes."
    )
    r.h("A melhoria do fluxo assíncrono")
    r.p(
        "Há no máximo uma consulta em andamento, sobre cópias do estado. A admissão usa a última orientação válida; se ela não existe ou expirou, o MAPPO segue sozinho. Tentamos renovar a orientação a cada 60 segundos virtuais, com validade de 120 segundos reais. Falhas são registradas e a orientação heurística de fallback do planejador genérico é descartada. O encerramento espera o worker fora da admissão."
    )
    r.table(
        [
            ["Evidência atual", "Resultado"],
            ["Chamadas reais / falhas / fallback", "509 / 0 / 0"],
            ["Sessões com orientação disponível", "9.246/9.900 (93,39%)"],
            ["Decisão média / p99", "347,51 / 1.199,04 ms"],
            ["Resposta de admissão média", "347,87 ms, incluindo cópia e submissão do snapshot"],
        ],
        [235, r.available - 235],
        size=9,
        padding=7,
    )
    r.p(
        "O Qwen continua disputando CPU com o MAPPO. A ausência de espera pela resposta não significa custo zero. Esse custo e a utilidade das orientações ainda são pontos de otimização.",
        size=9.5,
        leading=14,
    )

    r.start(
        "3.1. Qwen e abordagens explicativas",
        "Separar o algoritmo que decide, a orientação que pode alterar a decisão e a explicação do que foi decidido.",
    )
    r.table(
        [
            ["Abordagem", "Decisão e explicação", "Benefício e limite"],
            [
                "Regras/heurística sem LLM",
                "Regras explícitas e custos programados escolhem a alocação; os critérios podem ser registrados.",
                "Pode ser transparente sem um modelo explicativo adicional. A clareza das regras não garante boa alocação.",
            ],
            [
                "DRL sem camada explicativa",
                "PPO, MAPPO, SAC ou DQN aprendem ações/valores; máscaras e logs podem mostrar restrições.",
                "Aprende propostas, mas os pesos da rede não fornecem automaticamente uma justificativa compreensível ou causal.",
            ],
            [
                "DRL + explicador pós-decisão [9,10]",
                "LIME aproxima localmente uma previsão; SHAP atribui contribuições a entradas. Não são LLMs.",
                "Ajuda a estudar influência de features. Explicar a política não explica sozinho máscara, ranking e resultado final; a análise depende das perturbações e referências usadas.",
            ],
            [
                "LLM apenas como narrador",
                "Recebe registros depois da escolha e produz uma descrição em linguagem natural.",
                "Facilita leitura, mas não muda a alocação. A narrativa precisa ser confrontada com os registros; plausibilidade não comprova fidelidade.",
            ],
            [
                "LLM como controlador direto",
                "O modelo propõe a própria ação de alocação, sujeita a validação física.",
                "Permite instruções contextuais, mas introduz custo de geração e risco de ações inválidas. Não é o desenho usado aqui.",
            ],
            [
                "MAPPO + Qwen orientador: nosso desenho",
                "Qwen fornece pesos, preferências e uma justificativa breve; MAPPO propõe hosts e o ranking usa a orientação.",
                "A orientação pode alterar escolhas viáveis. O texto do Qwen descreve sua recomendação; ainda não foi validado como explicação da escolha final.",
            ],
        ],
        [115, 195, r.available - 310],
        size=8.5,
        padding=6,
    )
    r.h("Por que essa abordagem se adequa ao projeto")
    r.p(
        "Queremos orientação contextual sem entregar ao LLM o controle irrestrito dos recursos. A saída estruturada é verificada e entra no ranking entre propostas dos atores. O worker assíncrono mantém o MAPPO disponível enquanto o Qwen processa o snapshot. Essa separação permite analisar o custo e a utilidade da orientação independentemente da segurança física."
    )
    r.h("O que é explicável hoje e o que falta validar")
    r.p(
        "Temos ações, propostas, hosts e orientações registrados. Ainda não armazenamos em cada escolha todos os termos do custo e as causas detalhadas de cada exclusão. Para validar explicações, precisamos registrar esses componentes, comparar a decisão com e sem orientação e testar alterações controladas das entradas. <b>Qwen é um orientador com justificativa textual; não um explicador causal validado.</b>",
        size=9.2,
        leading=13.2,
    )
    r.p(
        "Qwen2.5 é o LLM pré-treinado escolhido [11]; explicabilidade é um papel de arquitetura, não uma propriedade garantida por essa família. As alternativas da tabela não foram avaliadas no último benchmark.",
        size=9,
        leading=13,
    )

    r.start(
        "3.1.1. Justificativa técnica do Qwen e de seu papel",
        "Escolher uma camada de orientação é diferente de escolher um algoritmo de alocação ou um explicador causal.",
    )
    r.h("Por que acrescentar orientação contextual?")
    r.p(
        "Uma heurística usa pesos definidos pelo programador; o MAPPO produz propostas a partir da política treinada. A camada Qwen recebe um resumo do estado e recomenda prioridades de latência/carga e hosts conhecidos em uma saída estruturada. Isso permite experimentar ajustes contextuais no ranking sem retreinar o ator a cada alteração de prioridade. É uma capacidade adicional do desenho; não demonstra, por si só, aumento de aceitação.",
        size=9.3, leading=13,
    )
    r.h("Em relação a regras e DRL sem LLM")
    r.p(
        "Regras são baratas, reproduzíveis e podem gerar explicações exatas de seus próprios critérios. Um ranking fixo é suficiente quando os objetivos permanecem bem definidos; um LLM não é necessário para torná-lo compreensível. DRL sem Qwen elimina custo de inferência e dependência de orientação atualizada. Optamos por uma interface de orientação para testar flexibilidade contextual mantendo essas alternativas como baselines, em vez de atribuir ao texto gerado uma qualidade decisória automática.",
        size=9.3, leading=13,
    )
    r.h("Em relação a LIME, SHAP e um narrador pós-decisão")
    r.p(
        "LIME e SHAP investigam influência das entradas sobre uma previsão, conforme o método e a referência adotados [9,10]. Esse objetivo é diferente de alterar pesos de um ranking. Um narrador posterior facilita leitura, mas não participa da alocação. Qwen foi integrado como orientador porque sua saída pode modificar a preferência entre propostas válidas; para explicar o resultado final, ainda precisamos reconstruir máscara, custos, rotas e escolha. Esses papéis podem ser combinados.",
        size=9.3, leading=13,
    )
    r.h("Por que orientação limitada e assíncrona?")
    r.p(
        "Um LLM no controle direto teria de selecionar ações sob recursos variáveis e responder a cada admissão. Nosso desenho mantém o MAPPO e a validação física no caminho de decisão, restringe a resposta a um esquema e consulta Qwen em segundo plano. A orientação em cache tem validade limitada; sem orientação utilizável, a política continua operando. Isso reduz a espera por geração e limita o efeito de uma recomendação incorreta. A CPU ainda é compartilhada, portanto a inferência pode disputar processamento com o controlador.",
        size=9.3, leading=13,
    )
    r.h("Por que Qwen2.5:1.5b nesta máquina?")
    r.p(
        "A escolha do modelo pequeno, pré-treinado e executado localmente via Ollama atende à necessidade operacional de experimentar em CPU, sem exigir uma GPU nem treinar um LLM. A interface aceita instruções e uma resposta JSON que nosso código valida. Essas propriedades justificam sua adoção inicial, mas não provam superioridade sobre outras famílias ou tamanhos de LLM. Modelos menores podem perder qualidade de orientação; modelos maiores exigem mais memória e tempo de inferência, a medir no mesmo computador.",
        size=9.3, leading=13,
    )
    r.h("Como demonstrar benefício e fidelidade")
    r.p(
        "Comparar MAPPO sozinho, ranking fixo e MAPPO + Qwen nas mesmas sementes; medir aceitação, atraso, decisão, consumo de CPU/RAM e idade da orientação. Para o texto, confrontar cada justificativa com custos e exclusões registrados. O resultado atual demonstra uso real do Qwen, mas sem a ablação pareada de 33 sementes não atribui a ele o ganho observado nem valida sua narrativa como explicação causal.",
        size=9.2, leading=13.2,
    )

    r.start(
        "3.2. Qwen: orientação, decisão e explicação",
        "Separar a recomendação do LLM, o efeito no ranking e a justificativa que pode ser auditada.",
    )
    r.h("O que a telemetria oferece ao modelo")
    r.p(
        "O snapshot informa recursos livres, velocidade de processamento e distâncias da sessão. Ele é uma cópia: o worker não modifica a rede ativa. Esse resumo permite recomendar prioridades, mas não é uma descrição completa de todas as possibilidades futuras do DAG. O estado pode mudar enquanto o modelo responde; as propostas são sempre verificadas na rede atual."
    )
    r.h("Exemplo didático de orientação")
    r.table(
        [
            ["Parte da resposta", "Exemplo e efeito"],
            ["Peso de latência", "0,8: dá maior importância à parcela de atraso no ranking"],
            ["Peso de carga", "0,2: mantém influência da distribuição de recursos"],
            [
                "Preferência de host",
                "Um nó conhecido é favorecido se estiver entre as propostas elegíveis e cumprir as restrições",
            ],
            [
                "Justificativa textual",
                "“Priorizar caminhos curtos com folga de processamento”: descreve a intenção da recomendação",
            ],
        ],
        [160, r.available - 160],
        size=9,
        padding=7,
    )
    r.p(
        "Os valores são ilustrativos e não representam uma nova consulta ou uma decisão específica do experimento. Um peso favorável não autoriza exceder o SLA; um nó recomendado fora das propostas não se torna elegível por recomendação do LLM.",
        size=9,
        leading=13,
    )
    r.h("O que pode ser rastreado")
    r.p(
        "Os registros mostram a orientação recebida, as ações dos atores, os hosts propostos e a escolha final. Para explicar a decisão completa precisamos também dos termos do custo, das rotas e das causas de cada exclusão. O texto gerado pelo Qwen pode ser coerente com sua recomendação e ainda assim não explicar por que o ranking final escolheu outro nó."
    )
    r.h("Como uma explicação poderia ser validada")
    r.p(
        "Registrar o estado e os custos da escolha; reconstruir o ranking; comparar com e sem orientação; alterar uma feature mantendo as demais controladas; conferir se a explicação acompanha o efeito observado. Essas verificações são diferentes de pedir uma narrativa convincente. LIME/SHAP podem apoiar a análise da política, mas não substituem a análise do pipeline inteiro."
    )
    r.h("Disponibilidade e custo")
    r.p(
        "Uma consulta pendente não bloqueia o MAPPO; a orientação em cache permite continuar e a validade limita seu uso. Com zero falhas no ensaio atual, demonstramos execução real, não ausência de falhas em qualquer ambiente. Precisamos medir utilidade de cada atualização, orientação expirada, CPU/RAM do controlador e custo por sessão. O processamento em segundo plano remove a espera, mas não o consumo de recursos.",
        size=9.5,
        leading=14,
    )

    feature_groups = [
        (
            "4. Features: estrutura e recursos",
            "Funcionalidades F01-F07. Cada análise descreve a aplicação, o benefício esperado e o limite do modelo.",
            [
                (
                    "F01. Dependências no DAG",
                    "Preserva bifurcações e junções e organiza as funções por precedência. Evita representar como sequência única etapas independentes. Todas as dependências precisam ser reservadas; a representação não comprova execução paralela em um dispositivo real.",
                ),
                (
                    "F02. Perfil de demanda multiusuário",
                    "Gera partes comuns e exclusivas a partir de P(h) e P(f), separando BE e FI. Normaliza a sessão a 20 CPU/15 memória. Torna o experimento reprodutível, mas as probabilidades são sorteadas: não estamos calculando similaridade em imagens reais.",
                ),
                (
                    "F03. Reuso de instâncias",
                    "Funções compatíveis podem compartilhar CPU/memória com contadores de referência. Reduz duplicação lógica e mantém instâncias usadas por outras sessões. Compatibilidade é obrigatória; seu ganho isolado de aceitação ainda não foi quantificado no último benchmark.",
                ),
                (
                    "F04. Capacidade computacional e memória",
                    "Cada proposta testa reserva sem ultrapassar limites físicos e sem multiplicadores. A velocidade IPS entra no atraso de processamento. Evita aceitar por intenção apenas; memória é capacidade normalizada no campo cache, não RAM medida em megabytes.",
                ),
                (
                    "F05. Rotas, banda residual e fluxos compartilhados",
                    "Busca caminhos que comportem a demanda, ponderando atraso e pressão de banda. Reuso só ocorre para fluxos explicitamente compatíveis; FI fica exclusivo. Todas as arestas do DAG são consideradas. Não há simulação de filas de pacotes ou interferência radioelétrica.",
                ),
                (
                    "F06. Caminho crítico e SLA",
                    "Computa o maior atraso entre os caminhos da aplicação e exige até 6 ms na sessão aceita. Diferencia processamento paralelo de soma de todos os ramos. É latência modelada de serviço, não motion-to-photon; a margem próxima do limite merece avaliação de robustez.",
                ),
                (
                    "F07. Sincronização em junções",
                    "Compara chegadas dos predecessores e usa a tolerância modelada de 5 ms para evitar reunir ramos excessivamente defasados. Complementa o SLA de 6 ms, sem substituí-lo. Não mede sincronização de sensores, relógios reais ou apresentação no display.",
                ),
            ],
        ),
        (
            "4.1. Features: política e decisão",
            "Funcionalidades F08-F14. Aprendizado propõe ações; critérios e restrições determinam quais propostas podem ser usadas.",
            [
                (
                    "F08. Regionalização e cooperação MAPPO",
                    "Quatro atores regionais aprendem com um crítico centralizado. Organiza escolhas locais em um objetivo conjunto. As regiões mantêm os mesmos 25 servidores, sem aumentar recursos; a execução no simulador é centralizada e a escalabilidade distribuída ainda não foi medida.",
                ),
                (
                    "F09. Observações normalizadas",
                    "Carga, banda, distâncias, demandas, fork/join e velocidade contextualizam o ator. A análise detalhada está na seção 2.2. Resumos compactos podem omitir informações; a importância de cada entrada e suas escalas ainda precisa de ablação.",
                ),
                (
                    "F10. Máscaras de viabilidade",
                    "Eliminam hosts sem recursos, com entradas sem rotas/banda ou com atraso mínimo restante acima do SLA. Evitam gastar escolhas em impossibilidades conhecidas. Não garantem uma solução global futura: uma escolha viável agora pode limitar as funções seguintes.",
                ),
                (
                    "F11. Ranking multicritério de propostas",
                    "Entre hosts propostos, combina atraso, distância à entrega, carga, banda reservada, diferença de chegada e reuso. Preferências Qwen participam desse custo. Atores e regras atuam em conjunto; não é uma alocação decidida exclusivamente pelo LLM ou pela política.",
                ),
                (
                    "F12. Busca interna limitada",
                    "A primeira tentativa é determinística; alternativas amostram a política e excluem escolhas anteriores, até oito tentativas por jogador. Pode explorar outros planos sem fila de espera por recursos. A busca é limitada, não exaustiva; pode rejeitar mesmo quando existe outra solução.",
                ),
                (
                    "F13. Treinamento com demonstrações e MAPPO",
                    "Demonstrações aquecem os atores; depois PPO/GAE ajusta políticas e crítico em cenários de carga variados. Ajuda a aprender propostas sem treino durante cada chegada. A avaliação isolada permaneceu em 86,67%; melhoria causal dessa etapa não foi demonstrada.",
                ),
                (
                    "F14. Checkpoint validado e inferência fixa",
                    "Confere dimensões, registro de passos/atualizações e hash antes de usar os pesos. Impede apresentar modelo aleatório como treinado e permite identificar o experimento. Ter treinamento registrado não prova optimalidade ou adequação a qualquer novo cenário.",
                ),
            ],
        ),
        (
            "4.2. Features: controle e operação",
            "Funcionalidades F15-F21. Diferenciamos recursos ativos no teste de capacidades disponíveis que ainda não foram exercitadas em hardware.",
            [
                (
                    "F15. Telemetria compacta e JSON Qwen",
                    "Envia um snapshot com colunas numéricas e exige resposta estruturada, com preferências de até três nós conhecidos. Reduz ambiguidades e volume do prompt. Estrutura válida não comprova boa recomendação; o validador físico continua obrigatório.",
                ),
                (
                    "F16. Worker assíncrono e snapshot separado",
                    "Uma consulta por vez usa cópias da rede, sem alterar a rede ativa ou formar uma fila de consultas. O MAPPO não espera a resposta na admissão. Reduziu o tempo no comparativo interno, mas cópias e disputa de CPU continuam custando tempo.",
                ),
                (
                    "F17. Orientação em cache, validade e falhas",
                    "Reutiliza a orientação por até 120 segundos reais, tentando renová-la a cada 60 segundos virtuais. Ausência, expiração ou falha deixam o MAPPO seguir sozinho. Respostas de fallback são descartadas; snapshots antigos podem perder utilidade, embora as restrições atuais sejam revalidadas.",
                ),
                (
                    "F18. Admissão atômica e lifetime",
                    "Implanta a sessão em uma cópia e só publica a nova rede se todos os jogadores forem alocados. Rejeições preservam a rede atual; expiração libera recursos. Não espera vagas em fila, mas o cálculo tem duração própria. As chegadas do benchmark usam relógio virtual.",
                ),
                (
                    "F19. Realocação preparada antes da troca",
                    "A API reserva novas cadeias enquanto as antigas permanecem ativas e verifica o estado antes do commit. Falhas preservam a sessão atual. Foi implementada e testada, mas não acionada no cenário estático; ainda não transfere estado de serviços ou quadros reais.",
                ),
                (
                    "F20. Métricas de cauda e entrada de dados AR",
                    "Registra p95/p99, máximo e variação de decisão e serviço. O analisador aceita timestamps de movimento/display e quadros perdidos explicitamente registrados. No último experimento esses dados visuais não existem: AR está como not_measured e conforto não foi validado.",
                ),
                (
                    "F21. Auditoria e reprodução",
                    "Registra sementes, parâmetros TOML, hashes, ações, propostas e orientação usada; o simulador audita capacidade a cada implantação/liberação. Permite rastrear escolhas e repetir entradas. A resposta do LLM e a execução em segundo plano podem variar; o JSON não guarda todos os estados físicos intermediários.",
                ),
            ],
        ),
    ]
    for title, subtitle, features in feature_groups:
        r.start(title, subtitle)
        for name, analysis in features:
            r.h(name)
            r.p(analysis, size=9.3, leading=13.2, after=10)
    # All descriptions above refer to the active MUVR/MAPPO/Qwen path or explicitly
    # identify an available-but-inactive API. Legacy algorithms are not claimed active.

    r.start(
        "5. Como a implementação melhorou",
        "Resultados internos observados e mudanças de correção, separados de hipóteses ainda não isoladas.",
    )
    r.h("Comparação interna: Qwen síncrono e assíncrono")
    r.table(
        [
            ["Métrica", "Antes: síncrono", "Depois: assíncrono"],
            ["Carga", "50 sessões × 3 sementes", "50 sessões × 3 sementes"],
            [
                "Aceitação",
                num(before["acceptance_percent"]) + "%",
                num(after["acceptance_percent"]) + "%",
            ],
            [
                "Latência média SFC (ms)",
                num(before["mean_latency_ms"], 3),
                num(after["mean_latency_ms"], 3),
            ],
            [
                "Decisão média (ms)",
                num(before["mean_decision_ms"]),
                num(after["mean_decision_ms"]),
            ],
            ["Espera pela resposta Qwen na admissão", "Sim, em consultas periódicas", "Não"],
        ],
        [205, 150, r.available - 355],
        size=9,
        padding=7,
    )
    r.p(
        f"A decisão média caiu em <b>{num(reduction)}%</b>, com a mesma aceitação agregada de 84%. A carga e as topologias foram pareadas, mas o momento de uso e as respostas do LLM podem variar. A latência de serviço não melhorou nesse ensaio: ficou ligeiramente maior. A melhoria medida principal é a remoção da espera síncrona do tempo de admissão."
    )
    r.h("Melhorias de implementação")
    r.p(
        "<b>DAG:</b> dependências completas e caminhos críticos tornam os custos mais fiéis.<br/><b>Recursos:</b> reservas por dependência e liberação por referências preservam capacidade.<br/><b>MAPPO:</b> atores treinados, máscaras e propostas rastreáveis.<br/><b>Qwen:</b> snapshot independente, orientação em cache e tolerância a falhas.<br/><b>Realocação:</b> preparação da nova cadeia mantendo a antiga, com commit validado; disponível e testada, mas não exercitada por mobilidade no último cenário.<br/><b>Reprodução:</b> arquivo TOML editável, checkpoints identificados por hash e parâmetros efetivos salvos."
    )
    r.h("Ampliação da avaliação")
    r.p(
        "O último experimento passou a 33 sementes de 300 requisições cada: 9.029/9.900 aceitas, ou 91,20%, com IC95% entre sementes de 89,41%-93,00%. Isso fornece uma avaliação mais ampla. Não devemos atribuir a diferença em relação ao teste de 50 requisições a uma nova melhoria algorítmica: o conjunto de redes e solicitações mudou."
    )
    r.h("Interpretação para realidade aumentada")
    r.p(
        "A decisão afeta a entrada ou realocação de uma sessão; não deve ser executada no loop de quadros. A SFC modelada tem SLA de 6 ms, mas rastreamento, renderização e display ainda não foram medidos. O resultado preliminar não é uma garantia de conforto visual em um dispositivo real.",
        size=9.5,
        leading=14,
    )

    r.start(
        "6. Tabela comparativa de dados a priori",
        "Referências publicadas e resultado preliminar próprio. A tabela contextualiza os dados; não é uma comparação controlada entre algoritmos.",
    )
    r.table(
        [
            [
                "Indicador",
                "MuSFiCO / tese [1,2]",
                "DRL de energia [3]",
                "Nosso resultado preliminar",
            ],
            [
                "Abordagem",
                "Heurística multicritério",
                "PPO ciente de energia; máscaras",
                "DAG + MAPPO + Qwen assíncrono",
            ],
            [
                "Topologia / recursos",
                "ER: 25 nós; 30% borda pela Tabela 9*",
                "Luxemburgo: 35 nós; 1/3 MEC; CPU/GPU 17×",
                "ER: 25 servidores, sem multiplicadores",
            ],
            ["SLA de serviço", "6 ms", "13 ms", "6 ms"],
            [
                "Repetições / carga",
                "33; 300 solicitações",
                "33 rodadas",
                "33; 300 sessões por semente",
            ],
            ["Aceitação", "~65% em 300 solicitações**", ">95%", "91,20%"],
            [
                "Latência de serviço",
                "~5,1 ms**",
                "~10,6 ms (mediana)",
                "4,580 ms (média); 4,691 ms (mediana)",
            ],
            [
                "Tempo de decisão",
                "~2 s em 300 solicitações**",
                "<7 ms (conclusão)",
                "347,51 ms (média)",
            ],
            [
                "Energia / equidade",
                "Não usadas nesta comparação",
                "-30,8% energia móvel vs OSCIM; Jain ~0,93",
                "Ainda não medidas",
            ],
        ],
        [95, 130, 135, r.available - 360],
        size=8.3,
        padding=7,
    )
    r.p(
        "[2]: Tabela 9 e Figuras 28/29/33 da tese do autor; curvas rotuladas MSF e atribuídas ao artigo [1]. O original IEEE não foi obtido integralmente nesta consulta. ** Valores aproximados de leitura visual no ponto de 300 solicitações.",
        size=8.5,
        leading=12,
    )
    r.p(
        "* A fonte apresenta ambiguidade entre nós totais e servidores de borda e entre demanda por usuário/sessão. [3]: PDF oficial, seções 3-5. Os cenários diferem em capacidade, destinos, mobilidade e custos; os valores não demonstram superioridade global.",
        size=8.5,
        leading=12,
    )
    r.h("O que a tabela permite afirmar")
    r.p(
        "Nosso resultado combina boa aceitação com aderência ao orçamento de serviço escolhido. Alguns indicadores reportados são numericamente favoráveis à implementação; outros favorecem o estudo de energia. A leitura adequada é de posicionamento preliminar e de escolhas de projeto, não de uma classificação definitiva de desempenho."
    )
    r.h("Nossa contribuição neste estágio")
    r.p(
        "A contribuição é a integração auditável de dependências, aprendizado cooperativo e orientação contextual desacoplada da admissão. O SLA mais restritivo, sozinho, não torna o cenário mais difícil em todos os aspectos. O ganho de tempo que demonstramos vem da comparação interna síncrono/assíncrono. Ganhos sobre os artigos exigem aplicar todos os métodos ao mesmo cenário.",
        size=9.5,
        leading=14,
    )

    r.start(
        "7. Gráficos do resultado preliminar",
        "33 sementes × 300 sessões. Gráficos do nosso experimento, sem sobrepor séries publicadas como se fossem medições pareadas.",
        wide=True,
    )
    picture(r, DATA / "graficos_33_sementes_300.png")
    r.p(
        "Aceitação e CPU: médias entre sementes. Latência: média acumulada ponderada pelos fluxos aceitos. Decisão: média, p95 e p99 de todas as sessões, incluindo rejeições. Latência p99: 5,981 ms; máxima: 5,999833 ms. A aceitação entre redes variou de 77,67% a 99,67%.",
        size=10,
        leading=14,
    )
    r.p(
        "A implementação ainda está sujeita a melhorias. A curva média estável não prova desempenho uniforme em qualquer rede nem mede a latência visual de AR.",
        size=10,
        leading=14,
    )

    r.start(
        "7.1. Leitura aprofundada dos resultados",
        "O significado de cada indicador e o que ele não pode demonstrar sozinho.",
    )
    r.h("Aceitação e diversidade das redes")
    r.p(
        "9.029 das 9.900 sessões foram aceitas: 91,20%. Todas as sementes tiveram a mesma quantidade de solicitações, por isso a média entre sementes coincide com o percentual consolidado. A variação de 77,67% a 99,67% mostra sensibilidade à rede e à carga sorteada. O IC95% de 89,41%-93,00% descreve a média entre 33 execuções, não a chance de aceitar cada pedido."
    )
    r.h("Média, cauda e margem para o SLA")
    r.table(
        [
            ["Métrica", "Média", "p95", "p99", "Máximo"],
            ["Latência SFC (ms)", "4,580", "5,879", "5,981", "5,999833"],
            ["Decisão (ms)", "347,505", "728,808", "1199,045", "2330,277"],
        ],
        [155, 85, 85, 85, r.available - 410],
        size=8.8,
        padding=7,
    )
    r.p(
        "A latência usa somente os 36.116 fluxos aceitos. Rejeições não entram como zero para reduzir a média. A proximidade de algumas sessões ao limite pede cuidado: propagação, filas e processamento reais podem variar de formas não modeladas. O tempo de decisão inclui rejeições e não deve ser confundido com a latência da sessão já alocada."
    )
    r.h("Por que uso global baixo pode coexistir com rejeições")
    r.p(
        "CPU média nas chegadas foi de 4,83%. Sessões expiram em até 120 segundos; enviar 300 ao longo de cerca de 50 minutos virtuais não significa manter 300 ativas. Capacidade livre longe do destino pode não ser útil sob 6 ms. Também podem existir gargalos locais de banda ou escolhas que deixam funções posteriores sem opção."
    )
    r.h("O que a participação do Qwen demonstra")
    r.p(
        "509 chamadas concluídas e orientação usada em 9.246 sessões comprovam participação real. Chamadas concluídas apenas no encerramento não contam como orientação usada antes. Esse registro não mede quanto o LLM melhorou a aceitação: a hipótese precisa de MAPPO sozinho nas mesmas redes e de análise de diferenças pareadas."
    )
    r.h("Conforto em AR e energia")
    r.p(
        "O benchmark não tem timestamps reais de movimento/display, perdas de quadros, consumo elétrico ou bateria. O SLA da SFC não representa toda a latência visual. Baixo uso de recursos simulados não é evidência de economia energética do PC ou do headset. A implementação permite incorporar medições, mas o relatório mantém esses resultados como não medidos.",
        size=9.5,
        leading=14,
    )

    r.start(
        "8. Melhorias futuras e referências",
        "Os resultados atuais são uma base de desenvolvimento, não o limite final da implementação.",
    )
    r.h("O que vamos precisar aperfeiçoar")
    r.p(
        "<b>Política MAPPO:</b> estudar as rejeições não excluídas pelo limite mínimo de latência e melhorar generalização e busca de alternativas.<br/><b>Tempo de decisão:</b> reduzir cópias e cálculos repetidos, medir caudas e disputa de CPU com o Ollama.<br/><b>Utilidade do Qwen:</b> comparar com MAPPO sozinho nas mesmas 33 sementes, distinguindo ganho de aceitação e custo do controlador.<br/><b>Ablações:</b> medir os efeitos separados do DAG, das máscaras, do reuso e da orientação.<br/><b>Energia, equidade e mobilidade:</b> acrescentar métricas próprias e exercitar as realocações.<br/><b>AR real:</b> medir movimento até a imagem, jitter e perda de quadros, mantendo rastreamento e apresentação independentes da orquestração."
    )
    r.h("Limite físico e objetivo de aceitação")
    r.p(
        "Das 871 rejeições, 546 já excedem 6 ms no limite inferior otimista de comunicação e processamento. O teto calculado para essa amostra é 94,48%. As outras 325 não são automaticamente recuperáveis: também há ocupação e viabilidade conjunta. Chegar a 98%-100% exigiria rever premissas físicas ou carga além de aperfeiçoar a política; inserir um LLM não elimina esse limite.",
        size=9.5,
        leading=14,
    )
    r.h("Referências dos dados a priori")
    r.p(
        "[1] Santos, H.; Rosário, D.; Cerqueira, E.; Braun, T. <i>Multi-criteria Service Function Chaining Orchestration for Multi-user Virtual Reality Services</i>. GLOBECOM 2022. "
        + f'<link href="{SOURCES["musfico"]}" color="#3478b8">DOI: 10.1109/GLOBECOM48099.2022.10000827</link>.',
        size=8.2,
        leading=11.7,
    )
    r.p(
        "[2] Santos, H. L. M. <i>Multimedia Service Orchestration in Multi-tier Edge Computing Environments</i>. Tese, 2023. Cap. 5, Tabela 9 e Figuras 26-33. "
        + f'<link href="{SOURCES["thesis"]}" color="#3478b8">PDF do autor, Universidade de Berna</link>.',
        size=8.2,
        leading=11.7,
    )
    r.p(
        "[3] Galhego, D. et al. <i>Orquestração de Multimídia Imersiva Baseada em Aprendizado por Reforço Profundo em Computação Local e de Borda Ciente de Energia</i>. SBRC 2026. DOI: 10.5753/sbrc_estendido.2026.22955. "
        + f'<link href="{SOURCES["energy"]}" color="#3478b8">PDF oficial SBC, seções 3-5</link>.',
        size=8.2,
        leading=11.7,
    )
    r.h("Origem dos nossos resultados")
    r.p(
        "Última avaliação: results/meu_treino_33_sementes_300/analysis/summary.json e medições correspondentes. Comparação interna: results/mappo_qwen_muvr/ e results/mappo_qwen_muvr_async/. Código: core/sfc.py, core/net_v2.py, algorithms/mappo/muvr.py e mappo_core.py. Nenhum novo treino ou simulação foi executado para esta revisão. Fontes verificadas em 14/10/2026.",
        size=8.2,
        leading=11.7,
    )
    r.start(
        "9. Referências das comparações de modelos",
        "Fontes primárias para as diferenças conceituais. Sua aplicação ao orquestrador é uma análise de projeto, não uma reprodução dos benchmarks desses trabalhos.",
    )
    method_sources = [
        (
            "[4] PPO",
            "Schulman et al. Proximal Policy Optimization Algorithms (2017).",
            "https://arxiv.org/abs/1707.06347",
        ),
        (
            "[5] MAPPO e IPPO",
            "Yu et al. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games (NeurIPS 2022).",
            "https://arxiv.org/abs/2103.01955",
        ),
        (
            "[6] Máscaras de ação",
            "Huang e Ontañón. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms (2020).",
            "https://arxiv.org/abs/2006.14171",
        ),
        (
            "[7] SAC discreto",
            "Christodoulou. Soft Actor-Critic for Discrete Action Settings (2019).",
            "https://arxiv.org/abs/1910.07207",
        ),
        (
            "[8] DQN",
            "Mnih et al. Playing Atari with Deep Reinforcement Learning (2013).",
            "https://arxiv.org/abs/1312.5602",
        ),
        (
            "[9] LIME",
            "Ribeiro, Singh e Guestrin. Why Should I Trust You?: Explaining the Predictions of Any Classifier (2016).",
            "https://arxiv.org/abs/1602.04938",
        ),
        (
            "[10] SHAP",
            "Lundberg e Lee. A Unified Approach to Interpreting Model Predictions (2017).",
            "https://arxiv.org/abs/1705.07874",
        ),
        (
            "[11] Qwen",
            "Qwen Team. Qwen2.5 Technical Report (2024).",
            "https://arxiv.org/abs/2412.15115",
        ),
        (
            "[12] Uso da sigla MPPO",
            "Deep Reinforcement Learning for Rapid Spacecraft Science Operations (2025), seção Approach: Masked/Maskable PPO.",
            "https://prl-theworkshop.github.io/prl2025-icaps/papers/5.pdf",
        ),
    ]
    for label, description, url in method_sources:
        r.h(label)
        r.p(
            description + f' <link href="{url}" color="#3478b8">Fonte original</link>.',
            size=9,
            leading=13,
            after=10,
        )
    r.h("Escopo das comparações")
    r.p(
        "As seções 2.1, 2.1.1 e 3.1 comparam arquitetura e uso potencial no contexto MEC + IMS + MSC. Aceitação, latência e decisão apresentados no relatório continuam sendo do nosso experimento MAPPO + Qwen. Não executamos PPO, SAC, IPPO, DQN, LIME ou SHAP nesta revisão. MPPO foi definido explicitamente para evitar misturar siglas de trabalhos diferentes.",
        size=9,
        leading=13,
    )

    r.finish()
    print(OUT)


if __name__ == "__main__":
    main()
