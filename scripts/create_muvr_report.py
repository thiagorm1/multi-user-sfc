"""Create a reviewed technical report from the saved paired MUVR measurements."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/relatorio_modificacoes_muvr_50.pdf"
DATA = ROOT / "results/muvr_comparison"
SUMMARY = json.loads((DATA / "plots/summary.json").read_text())
ROBUST = json.loads((ROOT / "results/muvr_50_reference/summary.json").read_text())
DOCS = {
    name: json.loads((DATA / f"{name}.json").read_text())
    for name in ["joint", "replic", "musfico", "greedy", "darsppo"]
}
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
for name, filename in [
    ("Body", "DejaVuSans.ttf"),
    ("Bold", "DejaVuSans-Bold.ttf"),
    ("Italic", "DejaVuSans-Oblique.ttf"),
    ("Mono", "DejaVuSansMono.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
pdfmetrics.registerFontFamily(
    "Body", normal="Body", bold="Bold", italic="Italic", boldItalic="Bold"
)
NAVY = colors.HexColor("#183249")
BLUE = colors.HexColor("#3478b8")
TEAL = colors.HexColor("#2a9d8f")
GRAY = colors.HexColor("#526170")
LIGHT = colors.HexColor("#edf3f7")
BORDER = colors.HexColor("#d6e0e8")
STYLE = ParagraphStyle(
    "body",
    fontName="Body",
    fontSize=10,
    leading=14.7,
    textColor=NAVY,
    alignment=TA_LEFT,
    spaceAfter=0,
)
TOTAL = 7


def num(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


class Report:
    def __init__(self, output=OUT, total=TOTAL, date="13 de outubro de 2026"):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.total = total
        self.date = date
        self.pdf = canvas.Canvas(str(output), pagesize=A4)
        self.pdf.setTitle("Modificações e resultados - Orquestração MUVR com SLA de 6 ms")
        self.pdf.setAuthor("Projeto multi-user-sfc")
        self.pdf.setSubject(
            "Cenário, alocação física, baselines adaptados e gráficos de 50 requisições"
        )
        self.page = 0

    def start(self, title, subtitle="", wide=False):
        if self.page:
            self.pdf.showPage()
        self.page += 1
        self.width, self.height = landscape(A3) if wide else A4
        self.pdf.setPageSize((self.width, self.height))
        self.margin = 43
        self.available = self.width - self.margin * 2
        self.pdf.setStrokeColor(BORDER)
        self.pdf.line(self.margin, self.height - 36, self.width - self.margin, self.height - 36)
        self.pdf.setFillColor(GRAY)
        self.pdf.setFont("Body", 8)
        self.pdf.drawString(self.margin, self.height - 26, "MULTI-USER SFC / RELATÓRIO TÉCNICO")
        self.pdf.drawRightString(
            self.width - self.margin, self.height - 26, self.date
        )
        self.pdf.line(self.margin, 36, self.width - self.margin, 36)
        self.pdf.setFont("Body", 8)
        self.pdf.drawString(self.margin, 23, "MUVR - SLA 6 ms - admissão imediata")
        self.pdf.drawRightString(self.width - self.margin, 23, f"{self.page} / {self.total}")
        self.y = self.height - 55
        self.p(title, 19, 25, bold=True, after=9)
        if subtitle:
            self.p(subtitle, 9.5, 14, color=GRAY, after=17)

    def p(self, text, size=10, leading=14.7, bold=False, color=NAVY, after=9, mono=False):
        style = ParagraphStyle(
            "p",
            parent=STYLE,
            fontSize=size,
            leading=leading,
            fontName="Mono" if mono else "Bold" if bold else "Body",
            textColor=color,
        )
        paragraph = Paragraph(text, style)
        _, h = paragraph.wrap(self.available, self.y - 48)
        if self.y - h < 48:
            raise ValueError(f"Page {self.page} overflow: {text[:60]}")
        paragraph.drawOn(self.pdf, self.margin, self.y - h)
        self.y -= h + after

    def h(self, text):
        self.p(text, 11.7, 16, bold=True, after=5)

    def table(self, rows, widths, size=9, padding=7):
        cells = []
        for index, row in enumerate(rows):
            style = ParagraphStyle(
                "cell",
                parent=STYLE,
                fontName="Bold" if index == 0 else "Body",
                fontSize=size,
                leading=size + 3.5,
                textColor=colors.white if index == 0 else NAVY,
            )
            cells.append([Paragraph(str(cell), style) for cell in row])
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), padding),
                    ("RIGHTPADDING", (0, 0), (-1, -1), padding),
                    ("TOPPADDING", (0, 0), (-1, -1), padding),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, BORDER),
                ]
            )
        )
        _, h = table.wrap(self.available, self.y - 48)
        if self.y - h < 48:
            raise ValueError(f"Table overflow on page {self.page}")
        table.drawOn(self.pdf, self.margin, self.y - h)
        self.y -= h + 13

    def cards(self, items):
        gap = 12
        width = (self.available - gap * (len(items) - 1)) / len(items)
        for index, (value, label) in enumerate(items):
            x = self.margin + index * (width + gap)
            self.pdf.setFillColor(LIGHT)
            self.pdf.roundRect(x, self.y - 70, width, 70, 7, stroke=0, fill=1)
            self.pdf.setFillColor(BLUE)
            self.pdf.setFont("Bold", 23)
            self.pdf.drawString(x + 12, self.y - 29, value)
            self.pdf.setFont("Body", 8.8)
            self.pdf.setFillColor(GRAY)
            self.pdf.drawString(x + 12, self.y - 51, label)
        self.y -= 88

    def finish(self):
        assert self.page == self.total
        self.pdf.save()


def main():
    report = Report()
    own = SUMMARY[0]
    report.start(
        "Modificações e resultados<br/>da orquestração MUVR",
        "Registro técnico das alterações implementadas e da comparação com baselines. "
        "Foco: 50 requisições, recursos sem aumento e SLA de 6 ms.",
    )
    report.cards(
        [
            (num(own["acceptance_percent"]) + "%", "Aceitação imediata"),
            (num(own["mean_latency_ms"], 3) + " ms", "Latência média aceita"),
            (num(own["mean_decision_ms"]) + " ms", "Decisão média por sessão"),
        ]
    )
    report.h("Resultado principal")
    report.p(
        "No comparativo pareado de três sementes (42, 43 e 44), o modelo conjunto aceitou "
        "<b>133 de 150 sessões</b>. Cada repetição submeteu 50 sessões de quatro jogadores. "
        "Todas as sessões aceitas atenderam ao limite de 6 ms, com validação física de recursos."
    )
    rows = [
        ["Modelo", "Aceites", "Aceitação", "Latência<br/>(ms)", "CPU<br/>(%)", "Decisão<br/>(ms)"]
    ]
    labels = [
        "Nosso modelo",
        "REPLIC (adaptado)",
        "MuSFiCO (DP adaptado)",
        "Greedy (DAG)",
        "DARSPPO (adaptado)",
    ]
    for label, row in zip(labels, SUMMARY, strict=True):
        rows.append(
            [
                label,
                f"{row['accepted']}/150",
                num(row["acceptance_percent"]) + "%",
                num(row["mean_latency_ms"], 3),
                num(row["cpu_mean_at_arrivals_percent"]),
                num(row["mean_decision_ms"]),
            ]
        )
    report.table(rows, [145, 57, 66, 69, 61, report.available - 398], size=8.8)
    report.p(
        "Latência: média dos caminhos aceitos até a 50ª requisição. CPU: média do uso global "
        "instantâneo observado após as chegadas. Decisão: média de todas as tentativas, "
        "incluindo rejeições; carregamento inicial dos checkpoints excluído.",
        8.5,
        12.5,
        color=GRAY,
        after=13,
    )
    report.h("O que este resultado permite afirmar")
    report.p(
        "Neste conjunto de testes, houve mais aceites e menor latência média que nos baselines "
        "executados. A decisão do modelo conjunto foi mais lenta que a dos modelos DRL e Greedy, "
        "mas mais rápida que a busca DP adaptada. Menor uso de CPU também pode indicar mais rejeições."
    )
    report.p(
        "<b>A meta de aproximadamente 98% não foi atingida na média.</b> Os baselines foram "
        "adaptados ao DAG; a curva rotulada MuSFiCO não é a execução inalterada do algoritmo "
        "publicado. Estes dados não comprovam superioridade sobre o artigo.",
        size=9.5,
        leading=14,
    )

    report.start(
        "1. Cenário e recursos utilizados",
        "O perfil ativo substitui o cenário Luxembourg por uma infraestrutura aleatória de "
        "25 servidores homogêneos, conforme a especificação solicitada.",
    )
    params = [
        ["Parâmetro", "Configuração efetiva"],
        ["Servidores de borda", "25, todos com capacidade de processamento"],
        ["Topologia", "Erdős-Rényi G(25; 0,20); sem adicionar enlaces ou regenerar grafos"],
        [
            "CPU e memória por servidor",
            "100 e 100 unidades nominais; memória representada pelo campo cache",
        ],
        [
            "Velocidade de processamento",
            "IPS uniforme entre 1×10¹⁰ e 2×10¹⁰; sem multiplicador de capacidade",
        ],
        ["Capacidade de cada enlace", "1.000 Mbps (1 Gbps)"],
        ["Latência de cada enlace", "Uniforme contínua entre 1 e 2 ms"],
        ["Demanda lógica da sessão", "20 unidades de CPU e 15 de memória, nas SFs distintas"],
        ["SLA de latência", "6 ms; comunicação e processamento incluídos"],
        ["Requisições por repetição", "50 sessões completas; 4 jogadores por sessão"],
        ["Chegadas", "Determinísticas: uma a cada 10 s (0,1 Hz)"],
        ["Lifetime", "Uniforme contínuo em [0; 120] s"],
        [
            "Foreground Interactions (FI)",
            "0,266 Mbps por jogador; função e perspectiva individuais",
        ],
        ["Background Environment (BE)", "25 Mbps por jogador; frações comuns e únicas"],
        ["Similaridade P(h)", "Uniforme em [0,65; 0,90]"],
        ["Predição P(f)", "Uniforme em [0,17; 0,25]"],
        ["Origem e destino", "Origem aleatória; jogadores da sessão no mesmo acesso regional"],
        ["Admissão", "Imediata e atômica: os quatro jogadores devem ser aceitos"],
        ["Mobilidade, falhas e backups", "Desativados neste experimento"],
        ["LLM e política MAPPO treinada", "Desativados no perfil que produziu os gráficos"],
    ]
    report.table(params, [182, report.available - 182], size=8.7, padding=5.1)
    report.p(
        "<b>Precisão sobre a capacidade anterior:</b> foram mantidas as capacidades nominais "
        "de CPU/memória, sem os aumentos do perfil exploratório. A topologia antiga e seus "
        "subnós heterogêneos não foram preservados: o novo perfil possui 25 servidores homogêneos.",
        size=9,
        leading=13,
        after=5,
    )
    report.p(
        "O perfil principal não acrescenta atraso de acesso móvel, nuvem externa ou fila de "
        "processamento sem parâmetros fornecidos. O custo de processamento segue a fórmula "
        "existente no simulador, com 60 quadros/s e 10 ciclos por bit.",
        9,
        13,
    )

    report.start(
        "2. Alterações na cadeia e na alocação",
        "Correções de consistência e de contabilização sustentam a aceitação sob limites reais.",
    )
    report.h("Cadeia MUVR completa, com compartilhamento explícito")
    report.p(
        "Foi criado um gerador específico para MUVR. A cadeia deixou de depender apenas do "
        "modelo MUAR anterior: agora representa um DAG com funções comuns e individuais. "
        "Para quatro jogadores, são 18 SFs distintas, apesar de as funções comuns aparecerem "
        "nas estruturas de cada jogador."
    )
    report.table(
        [
            ["Componente", "SFs distintas", "Regra"],
            ["Sincronização / IA", "1", "Comum à sessão"],
            ["BE comum: predito e não predito", "2", "Compartilháveis dentro da sessão"],
            ["BE único por jogador", "8", "Duas funções por jogador"],
            ["FI por jogador", "4", "Uma função individual por jogador"],
            ["Renderização predita e de visão", "2", "Comuns à sessão"],
            ["Encoder / Transcoder (codec)", "1", "Comum à sessão"],
            ["Total", "18", "14 em Object Viewing e 4 nas demais etapas"],
        ],
        [225, 87, report.available - 312],
        size=9,
        padding=5.5,
    )
    report.p(
        "As frações de BE usam P(h)×P(f), P(h)×(1-P(f)), (1-P(h))×P(f) e "
        "(1-P(h))×(1-P(f)). CPU e memória por SF são distribuídas proporcionalmente ao "
        "tráfego e normalizadas para a demanda total pedida. Essa distribuição é uma hipótese "
        "do perfil; não foi recuperado um custo exato de cada SF no artigo.",
        9.5,
        14,
    )
    report.h("Reservas corretas e liberação simétrica")
    report.p(
        "Instâncias e fluxos comuns usam uma identidade que inclui a sessão. Reuso só ocorre "
        "quando a identidade e a demanda são compatíveis. Referências são contadas para liberar "
        "CPU, memória e banda apenas quando deixam de ser utilizadas. O tráfego codificado "
        "final de 25,266 Mbps por jogador é reservado individualmente, de forma conservadora."
    )
    report.h("Todas as dependências e o caminho crítico")
    report.p(
        "Cada aresta do DAG recebe rota e contabilização de banda, incluindo origem e destino. "
        "Nos pontos de junção, o tempo de chegada corresponde ao ramo mais lento. A latência "
        "E2E é o caminho crítico, sem somar indevidamente ramos paralelos nem omitir enlaces."
    )
    report.h("Estado isolado e admissão da sessão inteira")
    report.p(
        "A busca opera sobre uma cópia da rede. Uma solução só é aceita se todos os jogadores "
        "puderem ser implantados com CPU, memória, banda e SLA válidos. Rejeições não alteram "
        "a rede ativa. As sessões expiradas devolvem os recursos antes das próximas chegadas."
    )

    report.start(
        "3. Estratégias implementadas e estado atual",
        "As seis frentes anteriores foram tratadas, mas nem todas ficam ativas no experimento final.",
    )
    report.table(
        [
            ["Estratégia", "Estado neste teste", "Modificação e efeito esperado"],
            [
                "1. Compartilhar instâncias e fluxos",
                "Ativa",
                "Identidades por sessão e contadores de referências evitam cobrar a mesma instância "
                "comum várias vezes. FI permanece individual.",
            ],
            [
                "2. Usar uma cadeia DAG",
                "Ativa",
                "Modelagem de bifurcações e junções, rotas por dependência e latência por caminho crítico.",
            ],
            [
                "3. Aumentar gargalos da infraestrutura",
                "Retirada do perfil ativo",
                "CPU/cache/IPS ampliados, enlaces multiplicados e fonte de 40 Gbps foram usados em testes "
                "anteriores. O experimento atual não usa esses aumentos.",
            ],
            [
                "4. Posicionamento e roteamento conscientes de recursos",
                "Ativa",
                "Busca conjunta de hosts, alternativas em duas/três etapas e fallback no motor de "
                "viabilidade MAPPO; banda residual e SLA são restrições de aceitação.",
            ],
            [
                "5. Fila com prazo de admissão",
                "Implementada; desativada",
                "O módulo de fila com deadline e tentativas limitadas existe. Neste teste não há espera "
                "nem novas tentativas de admissão.",
            ],
            [
                "6. Treinar MAPPO com cargas variadas",
                "Exploratória; fora da medição atual",
                "Foi criado treinamento com cargas e SLAs variados e checkpoint separado. O perfil "
                "atual é uma heurística conjunta; não usa o ator treinado em outra topologia.",
            ],
        ],
        [124, 103, report.available - 227],
        size=9,
        padding=7,
    )
    report.h("Como o modelo conjunto procura a solução")
    report.p(
        "Primeiro calcula um limite otimista de latência. Depois enumera hosts para a sessão "
        "inteira. Se necessário, testa etapas separadas para processamento de objetos e "
        "renderização/codificação, incluindo uma busca em três etapas. O fallback distribuído "
        "usa o motor de viabilidade do MAPPO, com peso da política igual a zero. "
        "Busca interna por alternativas não representa uma segunda tentativa de admissão."
    )
    report.h("Qwen/Ollama: integração disponível, sem atribuição de ganho")
    report.p(
        "A integração com a API compatível com OpenAI do Ollama foi implementada, com timeout, "
        "saída JSON, validação dos nós sugeridos e fallback heurístico. Qwen pôde ser executado "
        "em CPU. No perfil final, o planejador cognitivo está desligado. Os gráficos não incluem "
        "inferência Qwen e não atribuem a ele melhora de aceitação."
    )
    report.p(
        "<b>Os 100% observados em testes anteriores não são este resultado:</b> foram obtidos "
        "em outro cenário, com capacidade ampliada e SLA diferente. Não são uma referência "
        "pareada de ganho exclusivamente algorítmico.",
        9.5,
        14,
    )

    report.start(
        "4. Comparação e definição das métricas",
        "Cinco implementações, mesmas topologias e demandas, e validação física comum.",
    )
    report.h("Adaptações dos baselines")
    report.table(
        [
            ["Modelo", "Execução efetiva no comparativo"],
            [
                "Nosso modelo",
                "Heurística conjunta do novo perfil MUVR; sem ator MAPPO treinado e sem LLM.",
            ],
            [
                "REPLIC",
                "Checkpoint MaskablePPO existente, 25 ações, observações originais e inferência "
                "determinística. Travessia reversa de todas as SFs DAG, coordenação de instâncias comuns "
                "e validação física ao final. Sem retreinamento.",
            ],
            [
                "MuSFiCO - DP adaptado",
                "Adaptador de programação dinâmica generalizado para DAG, limitado "
                "a quatro estados por camada. Usa o motor de viabilidade física. Não é a classe original "
                "especializada em MUAR nem uma reprodução do algoritmo publicado.",
            ],
            [
                "Greedy - DAG",
                "DAGGreedyAlgorithm existente, com coordenação de instâncias comuns e "
                "validação de todas as dependências, recursos e SLA.",
            ],
            [
                "DARSPPO",
                "Checkpoint MaskablePPO existente e ambiente de observação DARSPPO compatível "
                "com suas 25 ações. Travessia reversa do DAG, inferência determinística e validação física. "
                "Sem retreinamento.",
            ],
        ],
        [120, report.available - 120],
        size=9,
        padding=6,
    )
    report.h("Repetições pareadas e unidade de contagem")
    report.p(
        "Foram usadas as sementes 42, 43 e 44, com 50 sessões por semente: 150 requisições por "
        "modelo. Uma sessão conta como aceita somente se os quatro jogadores forem aceitos. "
        "Os gráficos param na 50ª requisição e agregam as três repetições. Hashes SHA-256 "
        "conferem identidade das topologias e das solicitações antes da plotagem."
    )
    report.h("Fórmulas usadas no painel")
    report.p(
        "<b>Aceitação acumulada:</b> aceites até N / N × 100, com média das sementes.<br/>"
        "<b>Latência média acumulada:</b> soma das latências dos caminhos aceitos até N / "
        "quantidade desses caminhos. Rejeições não entram como latência zero.<br/>"
        "<b>CPU global:</b> CPU física usada após cada chegada / capacidade total dos "
        "servidores × 100. A curva é instantânea e inclui liberações por lifetime.<br/>"
        "<b>Decisão média:</b> média do tempo de todas as tentativas, convertido de s para ms. "
        "Inclui inferência, busca e validação; exclui carregamento inicial dos checkpoints."
    )
    report.p(
        "O SLA de 6 ms se aplica à comunicação e ao processamento modelados da SFC. "
        "O tempo computacional de orquestração é medido separadamente. O ensaio usa tempo "
        "virtual para as chegadas e expirações; não é um teste de entrega de quadros VR em tempo real.",
        9.5,
        14,
    )

    report.start(
        "5. Gráficos comparativos",
        "Figura 1. Painel 2×2 - 50 requisições por repetição; três sementes pareadas. "
        "Página em A3 paisagem para preservar a legibilidade.",
        wide=True,
    )
    image = DATA / "plots/comparativo_muvr_50.png"
    from PIL import Image

    with Image.open(image) as source:
        iw, ih = source.size
    height = report.y - 50
    width = min(report.available, height * iw / ih)
    height = width * ih / iw
    x = (report.width - width) / 2
    report.pdf.drawImage(
        str(image),
        x,
        report.y - height,
        width=width,
        height=height,
        preserveAspectRatio=True,
        mask="auto",
    )

    report.start(
        "6. Robustez, limitações e reprodução",
        "A diferença entre três e 33 sementes deve permanecer explícita.",
    )
    report.table(
        [
            ["Indicador do nosso modelo", "Comparativo: 3 sementes", "Robustez: 33 sementes"],
            ["Requisições / aceites", "150 / 133", "1.650 / 1.532"],
            [
                "Aceitação média",
                num(own["acceptance_percent"]) + "%",
                num(ROBUST["acceptance_percent"]) + "%",
            ],
            [
                "Latência média aceita",
                num(own["mean_latency_ms"], 3) + " ms",
                num(ROBUST["mean_latency_ms"], 3) + " ms",
            ],
            ["Sementes", "42 a 44", "42 a 74"],
        ],
        [190, 156, report.available - 346],
        size=9,
        padding=6,
    )
    report.p(
        "No ensaio de 33 sementes, o intervalo de confiança de 95% entre sementes foi de "
        f"{num(ROBUST['acceptance_seed_ci95'][0])}% a {num(ROBUST['acceptance_seed_ci95'][1])}%. "
        f"A maior latência aceita foi {num(ROBUST['max_latency_ms'], 3)} ms. "
        f"O limite otimista médio de aceitação foi {num(ROBUST['physical_sla_upper_bound_percent'])}%, "
        "calculado pela menor comunicação possível somada ao caminho crítico de processamento "
        "nos servidores mais rápidos. Esse limite vale para o modelo e as solicitações usados, "
        "não para qualquer implementação ou para o artigo.",
        9.5,
        14,
    )
    report.p(
        "Hipóteses relevantes: quatro jogadores no mesmo acesso; CPU/memória agregadas por "
        "sessão; custos por SF proporcionais ao tráfego; processamento com a fórmula existente; "
        "sem mobilidade, falhas e backups. A tabela 9 da tese citada informa 25 nós totais e "
        "30% de borda, enquanto este teste segue os 25 servidores pedidos. Os checkpoints DRL "
        "não foram retreinados para esse novo workload. CPU baixa não significa, por si só, "
        "que um modelo é melhor.",
        9.3,
        13.5,
    )
    report.h("Verificação e arquivos principais")
    report.p(
        "<b>52 testes passaram</b> ao final da implementação. Os testes cobrem recursos "
        "compartilhados, rollback, todas as rotas DAG, admissão imediata, SLA, expiração e "
        "liberação. O benchmark audita CPU, memória e banda, verifica o SLA e exige recursos "
        "liberados ao término. A figura foi revisada visualmente.",
        9.3,
        13.5,
    )
    report.p(
        "Código: core/scenarios/muvr_reference.py; algorithms/muvr_baselines.py; "
        "core/orchestration/placement.py; core/net_v2.py; core/infrastructure/resource_allocator.py; "
        "algorithms/mappo/env_mappo.py. Caminhos relativos a src/muar_sfc/.",
        8.5,
        12,
    )
    report.h("Comandos para reproduzir")
    report.p(
        ".venv/bin/python scripts/benchmark_muvr_comparison.py<br/>"
        ".venv/bin/python scripts/plot_muvr_comparison.py<br/>"
        ".venv/bin/python scripts/benchmark_muvr_reference.py<br/>"
        ".venv/bin/python -m muar_sfc.main",
        8.4,
        12.7,
        mono=True,
    )
    report.h("Fontes e dados utilizados")
    report.p(
        "Dados do painel: results/muvr_comparison/{modelo}.json e plots/summary.json. "
        "Ensaio de robustez: results/muvr_50_reference/summary.json. "
        "Configuração: .env.muvr-reference.example. Os resultados anteriores com ampliação "
        "de recursos não foram combinados com estes dados.",
        8.5,
        12.2,
    )
    report.p(
        "Referência bibliográfica: <i>Multi-criteria Service Function Chaining Orchestration "
        "for Multi-user Virtual Reality Services</i>, GLOBECOM 2022, DOI "
        '<link href="https://doi.org/10.1109/GLOBECOM48099.2022.10000827" color="#3478b8">'
        "10.1109/GLOBECOM48099.2022.10000827</link>. "
        'Descrição MUVR na <link href="https://webarchive.inf.unibe.ch/cds/publications/files/Hugo_Santos.pdf" '
        'color="#3478b8">tese do autor, seção 5.3.1 e tabela 9</link>. '
        "O presente relatório documenta uma simulação com hipóteses explícitas, sem afirmar "
        "reprodução exata do artigo.",
        8.4,
        12,
    )
    report.finish()
    print(OUT)


if __name__ == "__main__":
    main()
