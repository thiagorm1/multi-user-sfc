"""Report for 33x50 tradeoff results and three-seed paired local baselines."""

import hashlib
import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from create_muvr_report import Report, num
from PIL import Image
from plot_muvr_comparison import curves, validate_pairing

from muar_sfc.core.scenarios.muvr_reference import MUVRProfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/tradeoff_33_sementes_50_20261015_022202_401462"
ANALYSIS = RUN / "analysis_20261015_024037_529297"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUT = ROOT / f"output/pdf/relatorio_33_sementes_50_musfico_greedy_{STAMP}.pdf"
FIGURES = ROOT / f"output/figures/comparativo_50_{STAMP}"


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


def stats(doc):
    rows = doc["rows"]
    total = sum(r["requests"] for r in rows)
    return {
        "accepted": sum(r["accepted"] for r in rows),
        "acceptance": 100 * sum(r["accepted"] for r in rows) / total,
        "cpu": statistics.mean(s["cpu_percent"] for r in rows for s in r["utilization_samples"]),
        "bandwidth": statistics.mean(
            r["time_weighted_utilization_percent"]["bandwidth"] for r in rows
        ),
        "latency": statistics.mean(
            v for r in rows for x in r["records"] for v in x["latencies_ms"]
        ),
        "decision": statistics.mean(x["decision_s"] * 1000 for r in rows for x in r["records"]),
    }


def main():
    FIGURES.mkdir(parents=True)
    audit = json.loads((ANALYSIS / "comparison.json").read_text())
    assert audit["audit"] == "passed"
    a, b = audit["metrics"]["1"], audit["metrics"]["0"]
    profile = asdict(MUVRProfile(requests=50))
    full = {
        scale: {
            "profile": profile,
            "rows": [
                json.loads((RUN / f"seeds/scale_{scale}/seed_{seed}.json").read_text())
                for seed in range(42, 75)
            ],
        }
        for scale in (1, 0)
    }
    validate_pairing({"weight_1": full[1], "weight_0": full[0]})
    models = {"current": {"profile": profile, "rows": full[0]["rows"][:3]}}
    for name in ("musfico", "greedy"):
        models[name] = json.loads((ROOT / f"results/muvr_comparison/{name}.json").read_text())
    validate_pairing(models)
    labels = {
        "current": "MAPPO + Qwen (replay, peso 0)",
        "musfico": "MuSFiCO (DP adaptado)",
        "greedy": "Greedy (DAG)",
    }
    colors = {"current": "#e76f51", "musfico": "#2a9d8f", "greedy": "#e9a34a"}
    summaries = {name: stats(doc) for name, doc in models.items()}
    full_curves = {k: curves(v) for k, v in full.items()}
    model_curves = {k: curves(v) for k, v in models.items()}
    definitions = [
        (
            "cpu",
            "Uso de CPU na rede",
            "CPU global média nas chegadas (%)",
            2,
            "cpu_global_mean_percent",
            3,
        ),
        (
            "bandwidth",
            "Utilização de banda",
            "Banda média ponderada pelo tempo (%)",
            None,
            "bandwidth_time_weighted_percent",
            3,
        ),
        (
            "decision",
            "Tempo médio de decisão por SFC",
            "Decisão por sessão (ms)",
            None,
            "decision_ms",
            2,
        ),
        (
            "latency",
            "Latência média das requisições aceitas",
            "Comunicação + processamento (ms)",
            1,
            "latency_ms",
            3,
        ),
        ("acceptance", "Taxa de aceitação", "Requisições aceitas (%)", 0, "acceptance_percent", 2),
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    for key, title, unit, curve_index, field, digits in definitions:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
        axes[0].set_title("A. Experimento completo | sementes 42-74", fontsize=12)
        axes[1].set_title("B. Comparação pareada | sementes 42, 43 e 44", fontsize=12)
        if curve_index is not None:
            for scale, label, color in [(1, "Peso 1,0", "#3478b8"), (0, "Peso 0,0", "#e76f51")]:
                axes[0].plot(
                    range(1, 51),
                    full_curves[scale][curve_index],
                    label=label,
                    color=color,
                    linewidth=2,
                )
            for name in labels:
                axes[1].plot(
                    range(1, 51),
                    model_curves[name][curve_index],
                    label=labels[name],
                    color=colors[name],
                    linewidth=2,
                )
            for ax in axes:
                ax.set_xlabel("Requisições submetidas por semente")
                ax.set_ylabel(unit)
                ax.legend(fontsize=10)
                if key == "acceptance":
                    ax.set_ylim(0, 105)
                if key == "latency":
                    ax.axhline(6, color="gray", linestyle="--", label="SLA de 6 ms")
                    ax.legend(fontsize=10)
        else:
            values = [v[field]["mean"] if field == "decision_ms" else v[field] for v in (a, b)]
            axes[0].barh(
                ["Peso 1,0", "Peso 0,0"], values, color=["#3478b8", "#e76f51"], height=0.4
            )
            values_b = [summaries[name][key] for name in labels]
            axes[1].barh(list(labels.values()), values_b, color=list(colors.values()), height=0.5)
            for ax, vals in zip(axes, (values, values_b), strict=True):
                ax.invert_yaxis()
                ax.set_xlim(0, max(vals) * 1.25)
                ax.set_xlabel(unit)
                for i, v in enumerate(vals):
                    ax.text(v + max(vals) * 0.015, i, num(v, digits), va="center", fontsize=11)
        fig.suptitle(title, fontsize=17, fontweight="bold")
        fig.text(
            0.5,
            0.018,
            "50 requisições por semente nos dois painéis. A: 33 sementes; B: 3 sementes comuns. Baselines locais adaptados, medidos anteriormente.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.065, 1, 0.93), w_pad=3)
        for ext in ("png", "svg"):
            fig.savefig(FIGURES / f"{key}.{ext}", dpi=180)
        plt.close(fig)
    sources = [
        ANALYSIS / "comparison.json",
        RUN / "manifest.json",
        ROOT / "results/muvr_comparison/musfico.json",
        ROOT / "results/muvr_comparison/greedy.json",
    ]
    (FIGURES / "dados_e_fontes.json").write_text(
        json.dumps(
            {
                "audit": audit,
                "paired_three_seed_metrics": summaries,
                "sha256": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sources
                },
            },
            indent=2,
        )
    )
    r = Report(output=OUT, total=10, date="15 de outubro de 2026")
    r.pdf.setTitle(
        "Simulação 33 sementes x 50 requisições: MAPPO + Qwen, MuSFiCO adaptado e Greedy"
    )
    r.pdf.setSubject(
        "Resultados completos e comparação pareada nas mesmas três sementes; CPU, banda, decisão, latência e aceitação"
    )
    r.start(
        "Simulação de 50 requisições: resultados e comparação de modelos",
        "Relatório da execução concluída em 33 sementes. Comparação com MuSFiCO adaptado e Greedy nas três sementes comuns, sem novas simulações.",
    )
    r.cards(
        [
            ("92,18%", "Aceitação: 1.521 / 1.650"),
            ("+26", "Aceitas extras frente ao peso 1"),
            ("6 ms", "SLA SFC respeitado"),
        ]
    )
    r.h("Resultado do experimento completo")
    r.p(
        "Com peso zero para a penalidade explícita de concentração, o modelo aceitou 1.521 sessões, contra 1.495 no peso 1: ganho de 1,58 ponto percentual. Melhorou em 19 sementes e empatou em 14; nenhuma semente teve redução do total aceito. O IC95% do ganho médio pareado é de 0,99 a 2,16 pontos percentuais."
    )
    r.h("Resultado do recorte comparativo")
    r.p(
        "Nas sementes 42, 43 e 44, todas com 50 requisições, MAPPO + Qwen com orientações reproduzidas aceitou 130/150 (86,67%); MuSFiCO adaptado aceitou 128/150 (85,33%); Greedy aceitou 125/150 (83,33%). As topologias e cargas desse recorte foram conferidas por fingerprint."
    )
    r.p(
        "Nos registros disponíveis, MuSFiCO e Greedy apresentaram menor latência média aceita; Greedy teve decisão mais rápida. Nosso modelo aceitou mais nesse recorte, mas isso não demonstra superioridade universal, nem equivale a uma comparação de 33 sementes para todos os algoritmos."
    )
    r.h("O que o documento contém")
    r.p(
        "Cenário e funcionamento do orquestrador; tabela completa dos pesos; tabela dos três modelos nas mesmas sementes; gráficos de CPU, banda, tempo médio de decisão e latência. A taxa de aceitação também foi incluída para dar contexto ao consumo de recursos."
    )

    r.start(
        "1. Cenário e desenho da comparação",
        "As 33 sementes não são misturadas com as três sementes dos baselines.",
    )
    r.table(
        [
            ["Item", "Descrição"],
            [
                "Infraestrutura",
                "25 servidores, CPU/memória 100 por nó; Erdős-Rényi p = 0,20; enlaces de 1 Gbps e 1-2 ms",
            ],
            [
                "Carga",
                "50 requisições por semente; chegadas a cada 10 s virtuais; duração uniforme em [0,120] s",
            ],
            [
                "Sessões",
                "Até 4 jogadores; 18 SFs distintas por sessão completa; CPU 20 e memória 15 normalizadas; FI exclusivo e BE parcialmente compartilhado",
            ],
            [
                "SLA e admissão",
                "6 ms para comunicação/processamento da SFC; admissão imediata sem espera em fila; até 16 tentativas no MAPPO",
            ],
            [
                "Experimento A",
                "Sementes 42-74; 1.650 requisições por peso; pesos 1 e 0; mesmo checkpoint e orientações reproduzidas",
            ],
            [
                "Comparação B",
                "Sementes 42, 43 e 44; 150 requisições por modelo; mesmas topologias/cargas, validadas por hash",
            ],
        ],
        [130, r.available - 130],
        size=8.7,
        padding=5,
    )
    r.h("Como o orquestrador foi aplicado")
    r.p(
        "O DAG preserva ramificações e junções, reserva recursos em todos os ramos e calcula o caminho crítico. Quatro atores MAPPO treinados propõem hosts regionais; o ranking escolhe apenas entre as propostas viáveis. O crítico global é usado no treino. A execução dos atores permanece centralizada em Python."
    )
    r.p(
        "Qwen pré-treinado pode orientar os pesos e preferências do ranking. Neste ensaio, orientações reais do experimento anterior foram reproduzidas nos mesmos índices de requisição. Não houve novas chamadas ao Ollama nem treino. O valor load_balance_scale = 0 remove só a parcela explícita de carga; máscaras e limites físicos continuam obrigatórios."
    )
    r.p(
        "A implementação usa cópia sob escrita, cache de candidatos/distâncias e extração única das observações. Saídas usam pasta exclusiva com data/hora, arquivos por semente e criação sem sobrescrita.",
        size=9.3,
        leading=13.3,
    )

    r.start(
        "2. Resultado completo das 33 sementes",
        "50 requisições por semente, 3.300 sessões avaliadas entre os dois pesos.",
    )
    table = [
        ["Métrica", "Peso 1,0", "Peso 0,0"],
        ["Aceitas / total", "1.495 / 1.650", "1.521 / 1.650"],
        ["Aceitação", "90,606%", "92,182%"],
        ["Rejeições", "155", "129"],
    ]
    for label, field, sub, digits in [
        ("CPU global média nas chegadas (%)", "cpu_global_mean_percent", None, 3),
        ("Banda média no tempo (%)", "bandwidth_time_weighted_percent", None, 3),
        ("p99 do nó mais ocupado por chegada (%)", "cpu_peak_node_percent", "p99", 2),
        ("CPU máxima por nó (%)", "cpu_node_percent", "max", 2),
        ("Latência média aceita (ms)", "latency_ms", "mean", 3),
        ("Latência máxima aceita (ms)", "latency_ms", "max", 6),
        ("Decisão média (ms)", "decision_ms", "mean", 2),
        ("Decisão p95 (ms)", "decision_ms", "p95", 2),
        ("Decisão p99 (ms)", "decision_ms", "p99", 2),
        ("Decisão máxima (ms)", "decision_ms", "max", 2),
    ]:
        table.append([label, *[num(v[field][sub] if sub else v[field], digits) for v in (a, b)]])
    r.table(table, [270, 118, r.available - 388], size=8.4, padding=4)
    r.h("Interpretação do tradeoff")
    r.p(
        "A CPU média global caiu, mas o p99 do nó mais ocupado subiu de 61,01% para 80% e o máximo por nó atingiu 100%. A banda média aumentou de 0,982% para 1,241%. O ganho de aceitação exige maior concentração e maior consumo de banda, sem aumentar a capacidade dos servidores."
    )
    r.p(
        "Média, p95 e p99 de decisão melhoraram, porém o máximo piorou de 638,32 para 809,02 ms. A vantagem não aparece em toda decisão individual. Foram 27 novas aceitações e 1 nova rejeição, saldo de +26. As 6.084 latências de fluxos aceitos com peso zero ficaram dentro de 6 ms.",
        size=9.3,
        leading=13.3,
    )

    r.start(
        "3. Comparação com MuSFiCO e Greedy",
        "Recorte das mesmas três sementes 42-44; não usar os 92,18% das 33 sementes para ranquear estes baselines.",
    )
    table = [["Modelo", "Aceitação", "CPU média", "Banda média", "Latência", "Decisão"]]
    for name, label in labels.items():
        v = summaries[name]
        table.append(
            [
                label,
                num(v["acceptance"]) + "%",
                num(v["cpu"]) + "%",
                num(v["bandwidth"]) + "%",
                num(v["latency"], 3) + " ms",
                num(v["decision"], 2) + " ms",
            ]
        )
    r.table(table, [145, 68, 64, 66, 78, r.available - 421], size=7.7, padding=5)
    r.h("MAPPO + Qwen, peso zero")
    r.p(
        "Maior aceitação nesse recorte: 130/150. A orientação foi reproduzida dos registros; o tempo de 110,77 ms não inclui novas gerações de texto nem a disputa de CPU com Ollama. O checkpoint e os mapeamentos são os da nova execução de 33 sementes, restritos aqui às três sementes comuns."
    )
    r.h("MuSFiCO adaptado")
    r.p(
        "Adaptador local com recorrência de programação dinâmica inspirada no MuSFiCO, estendida ao DAG completo e limitada a quatro estados por camada. Aceitou 128/150, usou menos banda e apresentou menor latência média aceita. O tempo registrado foi maior. Não é o código oficial nem uma reprodução certificada dos resultados do artigo."
    )
    r.h("Greedy em DAG")
    r.p(
        "Heurística gulosa local com validação física: aceitou 125/150 e teve o menor tempo de decisão. Sua latência média aceita foi menor que a nossa, mas a utilização média de banda foi maior. A velocidade pode ser útil; o custo observado foi menor aceitação nesse recorte."
    )
    r.h("Limites da leitura")
    r.p(
        "Os baselines foram medidos em execuções anteriores e não foram novamente executados após as otimizações. Os tempos descrevem versões e condições diferentes do controlador. As latências usam apenas o conjunto aceito por cada modelo. A diferença de 2 aceitações frente ao MuSFiCO, em três sementes, não sustenta uma conclusão estatística geral de superioridade.",
        size=9.3,
        leading=13.3,
    )

    notes = {
        "cpu": (
            "CPU simulada, não consumo do processador do PC",
            "O painel A resume a utilização global ao longo das 50 chegadas em 33 sementes. O painel B usa apenas as três sementes comuns. Média global baixa pode coexistir com saturação local; o custo do peso zero aparece principalmente no pico por nó e no desequilíbrio.",
        ),
        "bandwidth": (
            "Banda média ponderada pelo tempo",
            "O painel B compara 1,158% no nosso recorte, 0,762% no MuSFiCO adaptado e 1,406% no Greedy. Esses valores incluem o efeito das diferentes sessões aceitas. Menor consumo isolado não prova maior eficiência para atender a mesma carga.",
        ),
        "decision": (
            "Tempo real da decisão, separado do SLA",
            "O nosso replay não gera texto novo; MuSFiCO e Greedy não usam LLM. Os baselines históricos não receberam uma nova medição uniforme. As barras não comprovam o custo relativo dos algoritmos sob uma implementação e máquina controladas em uma mesma rodada.",
        ),
        "latency": (
            "Latência média acumulada apenas das sessões aceitas",
            "Todos os modelos têm os registros aceitos dentro de 6 ms. MuSFiCO e Greedy têm médias menores nesse recorte, enquanto nosso modelo aceita mais. Conjuntos aceitos diferentes e a ausência de renderização real limitam conclusões sobre a experiência visual em AR.",
        ),
        "acceptance": (
            "Mesma carga por semente; amostras distintas entre painéis",
            "A aceitação das 33 sementes é 92,18% no peso zero. Para confrontar os baselines, utiliza-se somente o recorte de três sementes: 86,67% contra 85,33% e 83,33%. Os percentuais do painel A não devem substituir o recorte pareado do painel B.",
        ),
    }
    for index, (key, title, *_) in enumerate(definitions, 4):
        r.start(
            f"{index}. {title}",
            "A: resultado completo de 33 sementes. B: comparação dos modelos nas mesmas três sementes.",
            wide=True,
        )
        picture(r, FIGURES / f"{key}.png", 470)
        h, text = notes[key]
        r.h(h)
        r.p(text, size=12, leading=17)
        r.p(
            "MuSFiCO e Greedy são implementações locais. Cada semente submete 50 requisições. CPU é média nas chegadas; banda é ponderada pelo tempo; decisão é medida por sessão.",
            size=10.5,
            leading=15,
        )

    r.start(
        "9. Avaliação, limites e fontes",
        "Vantagem de aceitação no cenário avaliado, acompanhada de custos locais de concentração.",
    )
    r.h("Consistência do resultado nas 33 sementes")
    r.p(
        "O peso zero melhorou o total aceito em 19 sementes e empatou em 14, sem piora no agregado por semente. IC95% pareado do ganho: 0,994 a 2,157 pontos percentuais. Excluindo as três sementes do ajuste inicial, o ganho médio nas outras 30 foi de 1,533 ponto percentual, com IC95% de 0,923 a 2,144 pontos. Isso não é uma prova para toda rede ou carga."
    )
    r.h("Limite físico e relação com o ensaio de 300")
    r.p(
        "Há 96 requisições excluídas pelo limite inferior físico de atraso ou por ausência de caminho; o teto otimista calculado é 94,182%, sem garantir a alocação de todas as demais. Restaram 129 rejeições no peso zero. Os resultados destas primeiras 50 requisições reproduzem os prefixos do experimento de 300: são dados relacionados, não uma validação estatística independente."
    )
    r.h("Qwen, tempo simulado e experiência visual")
    r.p(
        "Orientação Qwen reproduzida em 996/1.650 requisições por peso. Chegadas e expirações são virtuais; tempos de decisão são reais. O replay padroniza quando a orientação é usada, mas não mede geração ao vivo ou interferência do Ollama. O SLA SFC não inclui a decisão e não comprova motion-to-photon, ausência de enjoo, filas reais ou capacidade de renderização."
    )
    r.h("Fontes e rastreabilidade")
    r.p(
        "Execução: results/tradeoff_33_sementes_50_20261015_022202_401462. Auditoria: analysis_20261015_024037_529297/comparison.json. Baselines: results/muvr_comparison/{musfico,greedy}.json. Perfis, fingerprints e admissão imediata foram conferidos na comparação. Hashes SHA-256 e tabelas exportadas acompanham as figuras em dados_e_fontes.json.",
        size=8.7,
        leading=12.5,
    )
    r.p(
        "MuSFiCO: Santos, Rosário, Cerqueira e Braun. Multi-criteria Service Function Chaining Orchestration for Multi-user Virtual Reality Services. GLOBECOM 2022, pp. 6360-6365. DOI 10.1109/GLOBECOM48099.2022.10000827. A publicação fundamenta a referência do método; os gráficos usam medições do adaptador local, não números oficiais do artigo.",
        size=8.7,
        leading=12.5,
    )
    r.p(
        "Foram preservados os arquivos e relatórios anteriores. Este documento e suas figuras têm nomes exclusivos. Nenhum treino ou simulação foi iniciado para gerar este relatório.",
        size=8.7,
        leading=12.5,
    )
    r.finish()
    print(OUT)
    print(FIGURES)


if __name__ == "__main__":
    main()
