"""Create a new PDF from audited tradeoff data and explicitly separate historical baselines."""

import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from create_muvr_report import Report, num
from PIL import Image
from plot_muvr_comparison import validate_pairing

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/tradeoff_33_sementes_300_20261015_005012_755874"
AUDIT = RUN / "analysis_20261015_015123_981154/comparison.json"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUT = ROOT / f"output/pdf/relatorio_melhorias_comparativos_{STAMP}.pdf"
FIGURES = ROOT / f"output/figures/comparativos_{STAMP}"
FIGURES.mkdir(parents=True)
LABELS = {
    "mappo_qwen": "MAPPO + Qwen (peso 0)",
    "mappo": "MAPPO anterior",
    "joint": "Heurística anterior",
    "musfico": "MuSFiCO (DP adaptado)",
    "greedy": "Greedy (DAG)",
    "replic": "REPLIC (adaptado)",
    "darsppo": "DARSPPO (adaptado)",
}
COLORS = ["#e76f51", "#8054b3", "#526170", "#2a9d8f", "#e9a34a", "#3478b8", "#264653"]


def picture(r, path, maximum_height):
    with Image.open(path) as im:
        ratio = im.height / im.width
    width = min(r.available, maximum_height / ratio)
    height = width * ratio
    r.pdf.drawImage(
        str(path),
        r.margin + (r.available - width) / 2,
        r.y - height,
        width=width,
        height=height,
        mask="auto",
    )
    r.y -= height + 14


def historical():
    documents = {
        name: json.loads((ROOT / f"results/muvr_comparison/{name}.json").read_text())
        for name in ("joint", "musfico", "greedy", "replic", "darsppo")
    }
    paths = {name: ROOT / f"results/muvr_comparison/{name}.json" for name in documents}
    paths["mappo"] = ROOT / "results/mappo_qwen_muvr_async/mappo.json"
    paths["mappo_qwen"] = (
        ROOT
        / "results/tradeoff_cpu_aceitacao/maior_concentracao_live_3_sementes_50/mappo_qwen.json"
    )
    for name in ("mappo", "mappo_qwen"):
        documents[name] = json.loads(paths[name].read_text())
    validate_pairing(documents)
    summary = {}
    for name in LABELS:
        rows = documents[name]["rows"]
        assert [r["seed"] for r in rows] == [42, 43, 44]
        assert all(r["requests"] == 50 for r in rows)
        latencies = [v for r in rows for rec in r["records"] for v in rec["latencies_ms"]]
        summary[name] = {
            "acceptance": sum(r["accepted"] for r in rows) / 150 * 100,
            "accepted": sum(r["accepted"] for r in rows),
            "cpu": statistics.mean(
                s["cpu_percent"] for r in rows for s in r["utilization_samples"]
            ),
            "bandwidth": statistics.mean(
                r["time_weighted_utilization_percent"]["bandwidth"] for r in rows
            ),
            "latency": statistics.mean(latencies),
            "decision": statistics.mean(
                rec["decision_s"] * 1000 for r in rows for rec in r["records"]
            ),
        }
    return summary, paths


def charts(current, old):
    definitions = [
        ("acceptance", "Taxa de aceitação", "Requisições aceitas (%)", "acceptance_percent", 2),
        (
            "cpu",
            "Uso de CPU na rede",
            "CPU global média nas chegadas (%)",
            "cpu_global_mean_percent",
            3,
        ),
        (
            "bandwidth",
            "Uso de banda na rede",
            "Banda média ponderada pelo tempo (%)",
            "bandwidth_time_weighted_percent",
            3,
        ),
        (
            "latency",
            "Latência média das requisições aceitas",
            "Latência de comunicação + processamento (ms)",
            "latency_ms",
            3,
        ),
        (
            "decision",
            "Tempo médio de decisão por SFC",
            "Tempo real de decisão por sessão (ms)",
            "decision_ms",
            2,
        ),
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    for key, title, unit, field, digits in definitions:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.65]})
        values = [
            current[str(s)][field]["mean"]
            if field in ("latency_ms", "decision_ms")
            else current[str(s)][field]
            for s in (1, 0)
        ]
        axes[0].barh(["Peso 1,0", "Peso 0,0"], values, color=["#3478b8", "#e76f51"], height=0.38)
        axes[0].invert_yaxis()
        axes[0].set_title("A. Resultado atual | 33 sementes × 300", fontsize=12)
        others = [old[name][key] for name in LABELS]
        axes[1].barh(list(LABELS.values()), others, color=COLORS, height=0.65)
        axes[1].invert_yaxis()
        axes[1].set_title("B. Comparação histórica | 3 sementes × 50", fontsize=12)
        for ax, vals in zip(axes, (values, others), strict=True):
            maximum = max(vals)
            ax.set_xlim(0, 105 if key == "acceptance" else maximum * 1.28)
            ax.set_xlabel(unit, fontsize=11)
            ax.tick_params(labelsize=11)
            ax.grid(axis="y", visible=False)
            for index, value in enumerate(vals):
                ax.text(
                    value + maximum * 0.015, index, num(value, digits), va="center", fontsize=11
                )
        fig.suptitle(title, fontsize=17, fontweight="bold")
        fig.text(
            0.5,
            0.018,
            "Painéis independentes: tamanhos de carga e amostras diferentes. Baselines locais adaptados; não são curvas publicadas dos artigos.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.065, 1, 0.93), w_pad=3)
        for ext in ("png", "svg"):
            fig.savefig(FIGURES / f"{key}.{ext}", dpi=180)
        plt.close(fig)
    return definitions


def main():
    audit = json.loads(AUDIT.read_text())
    assert audit["audit"] == "passed"
    current = audit["metrics"]
    a = current["1"]
    b = current["0"]
    old, paths = historical()
    definitions = charts(current, old)
    provenance = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [AUDIT, RUN / "manifest.json", *paths.values()]
    }
    (FIGURES / "dados_e_fontes.json").write_text(
        json.dumps({"current": current, "historical_50": old, "sha256": provenance}, indent=2)
    )
    r = Report(output=OUT, total=13, date="15 de outubro de 2026")
    r.pdf.setTitle("Evolução do orquestrador MAPPO + Qwen e comparativos de desempenho")
    r.pdf.setSubject(
        "Mudanças técnicas, tradeoff de CPU e cinco métricas; cenários de 300 e 50 requisições separados"
    )
    r.start(
        "Evolução do orquestrador e comparação de desempenho",
        "MAPPO + Qwen, DAG, otimização de decisão e tradeoff de concentração. Relatório dos resultados salvos, sem novas simulações.",
    )
    r.cards(
        [
            ("92,70%", "Aceitação atual: 9.177 / 9.900"),
            ("+117", "Aceitas adicionais ao peso 1"),
            ("6 ms", "SLA SFC preservado"),
        ]
    )
    r.h("O que o último experimento demonstrou")
    r.p(
        "A remoção da penalidade explícita por concentração no ranking elevou a aceitação de 91,52% para 92,70% nas mesmas 33 sementes com 300 requisições. Houve melhora em 28 sementes, empate em 4 e piora em 1. O ganho médio pareado foi de 1,18 ponto percentual, com IC95% de 0,85 a 1,51 ponto percentual."
    )
    r.h("Qual foi o custo")
    r.p(
        "A CPU global média nas chegadas caiu de 4,84% para 4,77%, mas os recursos ficaram mais concentrados: o p99 do nó mais ocupado por chegada subiu de 65,52% para 80%. A banda média ponderada pelo tempo passou de 1,12% para 1,37%. Menos balanceamento favoreceu a aceitação média, com menor folga local em parte dos servidores."
    )
    r.h("Como ler os gráficos deste relatório")
    r.p(
        "Cada uma das cinco métricas tem dois painéis. O painel A mostra o experimento atual de 33 sementes × 300 requisições, comparando os pesos 1 e 0. O painel B compara modelos em uma base histórica pareada de 3 sementes × 50 requisições. Os números dos dois painéis não formam um ranking único."
    )
    r.h("Modelos comparados")
    r.p(
        "MAPPO + Qwen com peso zero, MAPPO anterior, heurística anterior, MuSFiCO com DP adaptado, Greedy em DAG, REPLIC adaptado e DARSPPO adaptado. Os baselines são implementações locais; os gráficos não reproduzem resultados oficiais dos respectivos artigos."
    )

    r.start(
        "1. Cenário, métricas e comparabilidade",
        "Recursos físicos mantidos; diferentes bases de dados apresentadas separadamente.",
    )
    r.table(
        [
            ["Parâmetro", "Valor utilizado"],
            [
                "Infraestrutura",
                "25 servidores; CPU 100 e memória 100 por nó; grafo Erdős-Rényi p = 0,20",
            ],
            ["Rede", "Enlaces de 1 Gbps; atrasos uniformes entre 1 e 2 ms"],
            [
                "Sessão MUVR",
                "Até 4 jogadores; 18 SFs distintas; demanda normalizada de 20 CPU e 15 memória por sessão completa",
            ],
            [
                "Tráfego",
                "FI 0,266 Mbps por usuário; BE 25 Mbps parcialmente compartilhado; similaridade 65%-90% e predição 17%-25%",
            ],
            [
                "Carga e SLA",
                "Uma chegada a cada 10 s virtuais; duração uniforme de 0 a 120 s; SLA de 6 ms; destinos no mesmo acesso",
            ],
            [
                "Admissão",
                "Imediata, sem fila de espera; até 16 tentativas internas no MAPPO atual",
            ],
        ],
        [135, r.available - 135],
        size=8.7,
        padding=5,
    )
    r.h("Duas bases com escopos distintos")
    r.p(
        "<b>Atual:</b> pesos 1 e 0; sementes 42-74; 9.900 requisições por peso; mesmo checkpoint e mesmas orientações Qwen reproduzidas. <b>Histórica:</b> sementes 42, 43 e 44; 150 requisições por modelo; mesmas topologias, cargas e validação física entre os modelos locais."
    )
    r.p(
        "CPU é a reserva simulada dos servidores, não o uso do processador do PC. Banda é a utilização média ponderada pelo tempo. Latência é calculada apenas para fluxos aceitos; rejeições não entram como zero. Decisão é tempo real para aceitar ou rejeitar uma sessão, medido separadamente do SLA.",
        size=9.3,
        leading=13.3,
    )
    r.p(
        "Os tempos históricos foram coletados em execuções anteriores, sem novo ensaio uniforme após as otimizações. Checkpoints e adaptação dos modelos DRL diferem. Comparar esses tempos descreve os registros disponíveis; não isola a vantagem computacional de cada algoritmo.",
        size=9.3,
        leading=13.3,
    )

    r.start(
        "2. Mudanças de arquitetura e sua finalidade",
        "Estrutura das dependências, aprendizado cooperativo e orientação contextual.",
    )
    for title, text in [
        (
            "DAG: dependências e recursos de todos os ramos",
            "O grafo direcionado acíclico representa bifurcações e junções sem serializar funções independentes. Todas as dependências reservam banda; o atraso corresponde ao caminho crítico. Isso evita tanto somar ramos paralelos como omitir seus recursos. O DAG da aplicação é distinto do grafo físico, que pode conter ciclos.",
        ),
        (
            "Compartilhamento e transações",
            "Instâncias e fluxos só são reutilizados quando compatíveis. Contadores de referência mantêm recursos ainda usados por outros jogadores. Planejamento em cópia e confirmação da sessão preservam a rede ativa quando uma tentativa falha. CPU, memória, banda e sincronização continuam sendo verificadas.",
        ),
        (
            "MAPPO: atores regionais com treino centralizado",
            "Quatro atores fazem propostas regionais; o crítico usa informação conjunta no treinamento. Cada ator tem seis hosts locais e uma opção de nó global, que continua sendo um servidor finito. O ranking escolhe apenas entre propostas dos atores. A execução do simulador permanece centralizada em Python; CTDE não significa que quatro controladores físicos já foram implantados.",
        ),
        (
            "Busca limitada e checkpoint",
            "A busca passou de até 8 para até 16 tentativas, incluindo alternativas com prefixo fixo e exclusões de hosts. Não há resgate por uma heurística que injete hosts fora das propostas. O checkpoint do usuário foi preservado; esse experimento não treinou novos atores.",
        ),
        (
            "Qwen: orientação limitada, não controlador irrestrito",
            "Qwen2.5:1.5b pré-treinado fornece pesos e preferências em JSON validado. Em operação ao vivo, um worker assíncrono produz orientação em cache com validade limitada. As recomendações não autorizam exceder recursos ou SLA. No experimento de 33 sementes, as orientações reais foram reproduzidas por requisição, sem novas chamadas ao Ollama.",
        ),
    ]:
        r.h(title)
        r.p(text, size=9.4, leading=13.4)

    r.start(
        "3. Mudanças para reduzir o custo da decisão",
        "Otimizações de implementação preservam as dimensões dos atores e os controles de viabilidade.",
    )
    r.table(
        [
            ["Mudança", "Aplicação e efeito técnico"],
            [
                "Cópia sob escrita",
                "Na avaliação de um host, destacam-se apenas seus atributos e os enlaces alterados. Evita copiar todos os recursos para cada candidato; as tentativas permanecem isoladas. Grafos direcionados/multigrafos mantêm cópia profunda.",
            ],
            [
                "Cache de candidatos",
                "Máscara e ranking reutilizam o mesmo resultado na etapa. Invalidação por grafo, função, exclusões e reset evita reaproveitar reservas de uma etapa anterior.",
            ],
            [
                "Cache de distâncias",
                "Uma busca por origem fornece as distâncias de propagação durante a tentativa. Rotas dependentes de banda residual continuam verificadas no estado corrente.",
            ],
            [
                "Observação única",
                "O estado global aproveita as observações dos atores; removem-se extrações repetidas e reset redundante.",
            ],
            [
                "Telemetria de CPU e caudas",
                "Benchmark registra média, p95, p99, máximo e CPU da thread. Amostragem Linux mede controlador e Ollama; uma condição descarta orientação para observar só a disputa de CPU.",
            ],
            [
                "Saída sem sobrescrita",
                "Diretórios únicos com data/hora, criação exclusiva de arquivos e gravação por semente. Uma interrupção preserva as sementes concluídas; não há retomada automática.",
            ],
        ],
        [135, r.available - 135],
        size=8.5,
        padding=5,
    )
    r.h("Evidência específica da otimização")
    r.p(
        "No ensaio anterior de 3 sementes × 50 requisições, MAPPO + Qwen caiu de 429,43 para 194,02 ms em média (54,8%). Sem orientação, as escolhas foram preservadas; com Qwen assíncrono, a disponibilidade das recomendações mudou e a aceitação variou em uma sessão. Esses tempos não são os do replay atual de 33 sementes."
    )
    r.p(
        "A instrumentação identificou interferência do Ollama: com consultas reais e orientação descartada, a média da política otimizada passou de 123,37 para 190,82 ms. Reduzir a espera por geração não elimina competição por CPU, memória e cache.",
        size=9.2,
        leading=13.2,
    )

    r.start(
        "4. Resultado atual: pesos 1,0 e 0,0",
        "33 sementes × 300 requisições por peso. Orientações Qwen reproduzidas; mesmo checkpoint e recursos.",
    )
    rows = [
        ["Métrica", "Peso 1,0", "Peso 0,0"],
        ["Aceitas / total", "9.060 / 9.900", "9.177 / 9.900"],
        ["Aceitação", "91,515%", "92,697%"],
        ["Rejeitadas", "840", "723"],
    ]
    for label, key, digits in [
        ("CPU global média (%)", "cpu_global_mean_percent", 3),
        ("Banda média no tempo (%)", "bandwidth_time_weighted_percent", 3),
        ("p99 do pico por nó (%)", "cpu_peak_node_percent", 2),
        ("Latência média (ms)", "latency_ms", 3),
        ("Latência p99 (ms)", "latency_ms", 3),
        ("Latência máxima (ms)", "latency_ms", 6),
        ("Decisão média (ms)", "decision_ms", 2),
        ("Decisão p95 (ms)", "decision_ms", 2),
        ("Decisão p99 (ms)", "decision_ms", 2),
    ]:
        sub = "mean"
        if "p99" in label:
            sub = "p99"
        if "máxima" in label:
            sub = "max"
        vals = [v[key][sub] if isinstance(v[key], dict) else v[key] for v in (a, b)]
        rows.append([label, *[num(v, digits) for v in vals]])
    r.table(rows, [265, 122, r.available - 387], size=8.5, padding=4)
    r.h("O parâmetro alterado")
    r.p(
        "A parcela de carga no custo é: peso Qwen de balanceamento × (utilização de CPU + utilização de memória do host) × load_balance_scale. O valor atual é 0,0, removendo essa parcela explícita do ranking. Atores, demais termos do custo, máscaras e capacidades não foram removidos."
    )
    r.p(
        "O ganho de 117 aceitações é saldo: 125 requisições antes rejeitadas passaram a ser aceitas e 8 antes aceitas passaram a ser rejeitadas. Houve redução de 13,93% no total de rejeições. Todas as latências aceitas permaneceram dentro de 6 ms.",
        size=9.3,
        leading=13.3,
    )

    notes = {
        "acceptance": (
            "Aceitação não é medida com a mesma carga entre painéis.",
            "No painel A, peso zero ganhou 1,18 ponto percentual. No B, a heurística anterior aceitou 133/150 (88,67%), MAPPO + Qwen atual 130/150 (86,67%) e MuSFiCO adaptado 128/150 (85,33%). Nosso modelo não lidera todos os comparadores nessa base de 50 requisições.",
        ),
        "cpu": (
            "CPU global baixa não representa folga em todos os nós.",
            "O painel A mostra leve redução da média global, acompanhada de maior concentração local. No painel B, baixo uso em REPLIC/DARSPPO acompanha baixa aceitação; não demonstra melhor eficiência para atender a mesma demanda.",
        ),
        "bandwidth": (
            "A banda é média ponderada pelo tempo, não o valor no último instante.",
            "No painel A, o uso passou de 1,115% para 1,370% (+22,85% relativo). A maior aceitação e a mudança dos mapeamentos alteram o tráfego. Consumo menor de um modelo com muitas rejeições não é vantagem isolada.",
        ),
        "latency": (
            "A média inclui somente fluxos das sessões aceitas.",
            "Todos os registros aceitos respeitam 6 ms. Como cada modelo aceita um conjunto diferente de solicitações, uma média menor não prova que atenderia melhor toda a carga. A latência visual de AR, filas reais e renderização física não foram medidas.",
        ),
        "decision": (
            "Tempos de decisão são reais; o SLA SFC é outra métrica.",
            "No painel A, média caiu de 115,71 para 112,08 ms, sem novas gerações Qwen. No B, há execuções históricas com checkpoints e datas diferentes; MAPPO + Qwen atual usa Ollama ao vivo. Não atribuir diferenças de tempo apenas ao algoritmo nem tratar o replay como custo completo de produção.",
        ),
    }
    for index, (key, title, _, _, _) in enumerate(definitions, 5):
        r.start(
            f"{index}. {title}",
            "A: comparação atual de 300 requisições. B: modelos na base histórica de 50 requisições.",
            wide=True,
        )
        picture(r, FIGURES / f"{key}.png", maximum_height=470)
        h, text = notes[key]
        r.h(h)
        r.p(text, size=12, leading=17)
        r.p(
            "Os rótulos MuSFiCO, REPLIC e DARSPPO referem-se aos adaptadores locais. Resultados oficiais de artigos não foram misturados nestas barras.",
            size=10.5,
            leading=15,
        )

    r.start(
        "10. Tabela dos modelos na base de 50 requisições",
        "Sementes 42, 43 e 44: 150 sessões por modelo. Fingerprints de topologia/carga e SLA conferidos.",
    )
    table = [["Modelo local", "Aceitação", "CPU média", "Banda média", "Latência", "Decisão"]]
    for name, label in LABELS.items():
        v = old[name]
        table.append(
            [
                label,
                num(v["acceptance"]) + "%",
                num(v["cpu"]) + "%",
                num(v["bandwidth"]) + "%",
                num(v["latency"], 3) + " ms",
                num(v["decision"], 1) + " ms",
            ]
        )
    r.table(table, [140, 70, 66, 67, 77, r.available - 420], size=7.6, padding=4)
    r.h("O que representam os comparadores")
    r.p(
        "<b>MuSFiCO adaptado:</b> recorrência de DP inspirada no método, estendida ao DAG completo e limitada a quatro estados por camada (beam). Não é uma reprodução certificada do artigo. <b>REPLIC e DARSPPO:</b> adaptadores com checkpoints legados MaskablePPO, observações compatíveis e validação física. Não houve novo treinamento uniforme desses modelos para esta comparação."
    )
    r.p(
        "<b>Greedy:</b> decisão gulosa em DAG. <b>Heurística anterior:</b> orquestração conjunta sem o checkpoint MAPPO treinado atual. <b>MAPPO anterior:</b> checkpoint e busca históricos, sem Qwen. <b>MAPPO + Qwen atual:</b> peso zero e execução assíncrona ao vivo; é o ensaio de 50 requisições validado separadamente do replay de 300."
    )
    r.h("Interpretação justa")
    r.p(
        "Os valores baixos de aceitação dos adaptadores DRL não demonstram inferioridade geral de REPLIC ou DARSPPO. Arquitetura, treinamento, adaptação e políticas de reserva diferem. A heurística anterior supera nosso modelo em aceitação nesse teste pequeno. Para um ranking de 300 requisições, faltam execuções equivalentes desses baselines nas 33 sementes."
    )
    r.p(
        "Fonte conceitual do MuSFiCO: Santos, Rosário, Cerqueira e Braun. Multi-criteria Service Function Chaining Orchestration for Multi-user Virtual Reality Services. GLOBECOM 2022, pp. 6360-6365. DOI: 10.1109/GLOBECOM48099.2022.10000827. O nome do método é mantido como referência, acompanhado da identificação da adaptação local.",
        size=8.7,
        leading=12.5,
    )

    r.start(
        "11. Concentração de CPU e consistência do ganho",
        "A unidade estatística é a semente; requisições da mesma execução não são réplicas independentes.",
    )
    r.table(
        [
            ["Indicador de concentração", "Peso 1,0", "Peso 0,0"],
            ["Média do nó mais ocupado por chegada", "31,17%", "37,64%"],
            ["p99 do nó mais ocupado por chegada", "65,52%", "80,00%"],
            ["Máximo observado por nó", "aprox. 100%", "aprox. 100%"],
            ["Desequilíbrio médio entre nós", "8,57 pp", "10,05 pp"],
        ],
        [265, 122, r.available - 387],
        size=9,
        padding=6,
    )
    seed_fig = RUN / "analysis_20261015_015123_981154/diferenca_por_semente.png"
    picture(r, seed_fig, maximum_height=195)
    r.h("O ganho persistiu fora das três sementes do ajuste inicial")
    r.p(
        "Nas 33 sementes: 28 melhoras, 4 empates e 1 piora; IC95% pareado do ganho: 0,854 a 1,510 pp. Excluindo 42-44, usadas na escolha inicial do peso, as outras 30 tiveram saldo de +101 aceitações, ganho médio de 1,122 pp e IC95% de 0,787 a 1,458 pp. Isso reforça a consistência no cenário avaliado, sem garantir generalização a outra topologia ou carga."
    )
    r.p(
        "A única piora foi na semente 69: 299 para 298 aceitações. Os empates ocorreram em 46, 57, 72 e 74. Ambos os pesos atingiram o limite de CPU em algum nó; desvios da ordem de 1e-13 acima de 100% são arredondamento, não capacidade adicional.",
        size=9.2,
        leading=13.2,
    )

    r.start(
        "12. Conclusão, limites e rastreabilidade",
        "A configuração atual é sustentada pelos dados deste cenário e continua sujeita a melhorias.",
    )
    r.h("Avaliação da implementação atual")
    r.p(
        "Manter peso zero é uma escolha sustentada pelo ensaio pareado de 33 × 300: mais aceitação média, latência aceita dentro do SLA e menor média de decisão no replay. O custo é maior concentração por nó e maior utilização de banda. Outras taxas de chegada, tempos de vida ou topologias podem mudar esse resultado."
    )
    r.h("O que não foi demonstrado")
    r.p(
        "Não há ranking completo dos artigos em 33 × 300, treino uniforme dos baselines, medição de energia, renderização real nem latência motion-to-photon. As 546 requisições excluídas por limite inferior físico de atraso implicam teto otimista de 94,485% neste modelo; isso não garante que todas as demais sejam alocáveis. Após o ajuste, ainda houve 723 rejeições."
    )
    r.h("Tempo simulado e Qwen")
    r.p(
        "Chegadas e expirações usam relógio virtual; decisão e inferência ao vivo usam relógio real. Essa combinação pode mudar quando uma orientação chega ao controlador. O replay padroniza a orientação por requisição para isolar o ranking. As 9.246 orientações usadas por peso foram reproduzidas; nenhuma chamada nova foi realizada no experimento de 33 sementes. Não existe comprovação de conforto visual em AR apenas por cumprir o SLA da SFC."
    )
    r.h("Fontes dos dados e reprodução")
    r.p(
        "Experimento atual: results/tradeoff_33_sementes_300_20261015_005012_755874. Auditoria: analysis_20261015_015123_981154/comparison.json. Modelos de 50 requisições: results/muvr_comparison, results/mappo_qwen_muvr_async e results/tradeoff_cpu_aceitacao/maior_concentracao_live_3_sementes_50.",
        size=8.5,
        leading=12,
    )
    r.p(
        "Código: scripts/benchmark_load_tradeoff.py, scripts/benchmark_decision_latency.py e src/muar_sfc/algorithms/mappo/{muvr,env_mappo}.py. Parâmetro atual: evaluation.load_balance_scale = 0.0 em config/mappo_qwen.toml. A suíte mais recente passou com 94 testes; este relatório não iniciou treino nem simulação.",
        size=8.8,
        leading=12.5,
    )
    r.p(
        "Referências técnicas: Yu et al., The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games (NeurIPS 2022), arXiv:2103.01955; Qwen Team, Qwen2.5 Technical Report, arXiv:2412.15115; MuSFiCO: DOI 10.1109/GLOBECOM48099.2022.10000827. As fontes bibliográficas fundamentam conceitos, não as barras locais.",
        size=8.5,
        leading=12,
    )
    r.p(
        "Os hashes SHA-256 das fontes e as tabelas que alimentam os gráficos estão no arquivo dados_e_fontes.json, junto às figuras exportadas. Arquivos anteriores foram preservados; este PDF e suas figuras usam nomes exclusivos.",
        size=8.5,
        leading=12,
    )
    r.finish()
    print(OUT)
    print(FIGURES)
    (ROOT / "tmp/pdfs/latest_tradeoff_report_path.txt").write_text(str(OUT))


if __name__ == "__main__":
    main()
