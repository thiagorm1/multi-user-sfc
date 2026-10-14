"""Create the audited 33-seed report from saved measurements, without simulation."""

import csv
import json
from pathlib import Path

from create_muvr_report import Report, num
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/meu_treino_33_sementes_300"
OUT = ROOT / "output/pdf/relatorio_mappo_qwen_33_sementes_300.pdf"


def main():
    summary = json.loads((DATA / "analysis/summary.json").read_text())
    config = json.loads((DATA / "experiment_config.json").read_text())["configuration"]
    with (DATA / "analysis/per_seed.csv").open() as stream:
        seeds = list(csv.DictReader(stream))
    assert summary["status"] == "passed" and len(seeds) == 33
    r = Report(output=OUT, total=7, date="14 de outubro de 2026")
    r.pdf.setTitle("MAPPO + Qwen - análise de 33 sementes e 300 requisições")
    r.pdf.setSubject("Resultados do experimento executado pelo usuário; SLA SFC 6 ms")
    r.start(
        "MAPPO + Qwen assíncrono<br/>Análise de 33 sementes",
        "300 requisições por execução; 9.900 sessões avaliadas. Relatório elaborado a partir dos arquivos salvos pelo usuário, sem executar novos treinos ou simulações.",
    )
    r.cards(
        [
            (num(summary["acceptance_percent"]) + "%", "Aceitação consolidada"),
            (num(summary["latency_ms"]["mean"], 3) + " ms", "Latência média da SFC"),
            (num(summary["decision_ms"]["mean"]) + " ms", "Decisão média"),
        ]
    )
    r.h("Resultado principal")
    r.p(
        "<b>9.029 sessões aceitas e 871 rejeitadas.</b> As 33 sementes, de 42 a 74, foram concluídas. Todas as sessões aceitas respeitaram o SLA de 6 ms no modelo do simulador. O maior atraso observado foi de 5,999833 ms, muito próximo do limite."
    )
    rows = [["Métrica (ms)", "Média", "p95", "p99", "Máximo"]]
    for label, key in [
        ("Latência SFC", "latency_ms"),
        ("Decisão", "decision_ms"),
        ("Resposta de admissão", "admission_response_ms"),
    ]:
        d = summary[key]
        rows.append(
            [label]
            + [
                num(d[k], 6 if key == "latency_ms" and k == "max" else 3)
                for k in ("mean", "p95", "p99", "max")
            ]
        )
    r.table(rows, [155, 85, 85, 85, r.available - 410], size=8.7, padding=7)
    r.p(
        "Latência: 36.116 fluxos aceitos, quatro por sessão. Decisão e resposta: 9.900 sessões, incluindo rejeições. A resposta de admissão inclui o custo de preparar/submeter o snapshot ao worker; a decisão não espera o Qwen.",
        size=9,
        leading=13,
    )
    r.h("Variação entre sementes")
    r.p(
        "IC95% da média de aceitação entre sementes: <b>89,41% a 93,00%</b>, calculado com distribuição t de Student sobre 33 execuções. Desvio padrão: 5,06 pontos percentuais. Melhor semente: 69, com 299/300 (99,67%). Pior: 47, com 233/300 (77,67%). O intervalo caracteriza a variação desta amostra; não garante uma taxa em qualquer nova rede."
    )
    r.h("Participação do Qwen")
    r.p(
        "<b>509 chamadas reais, zero falhas e zero fallback.</b> Orientação efetivamente usada em 9.246/9.900 sessões (93,39%). As demais prosseguiram com MAPPO sem orientação disponível, conforme o fluxo assíncrono."
    )

    r.start(
        "1. Modelo, treinamento e cenário",
        "Apenas MAPPO + Qwen foi avaliado neste experimento. MAPPO sozinho e os baselines não foram executados nas 33 sementes.",
    )
    r.h("O que é MAPPO + Qwen")
    r.p(
        "MAPPO (Multi-Agent Proximal Policy Optimization) é aprendizado por reforço cooperativo com múltiplos agentes. Neste projeto, quatro atores regionais treinados propõem servidores para as funções da sessão. O Qwen2.5:1.5b, um modelo de linguagem pré-treinado no Ollama, recebe telemetria e orienta a classificação dessas propostas. Ele não altera capacidades, SLA ou os pesos do MAPPO durante a avaliação."
    )
    r.h("Treinamento realizado")
    r.p(
        "Foram usados 100 episódios de demonstração e 300 episódios MAPPO, com 2.534 passos e 60 atualizações. O checkpoint tem quatro agentes, observações de 48 valores, sete ações por agente e estado global de 196 valores. O Qwen não foi treinado neste fluxo. Na avaliação isolada de 30 casos, o sucesso permaneceu em 86,67% antes e depois; esse conjunto não demonstra ganho isolado do treino."
    )
    t = config["training"]
    r.table(
        [
            ["Hiperparâmetro", "Valor"],
            ["Taxas de aprendizado ator / crítico", f"{t['lr_actor']} / {t['lr_critic']}"],
            ["PPO: clipping / épocas", f"{t['clip_param']} / {t['ppo_epoch']}"],
            ["Gamma / lambda GAE", f"{t['gamma']} / {t['gae_lambda']}"],
            [
                "Entropia / perda do crítico / gradiente",
                f"{t['entropy_coef']} / {t['value_loss_coef']} / {t['max_grad_norm']}",
            ],
            [
                "Atualização / clonagem",
                f"A cada {t['update_every']} episódios; {t['cloning_epochs']} épocas, lote {t['cloning_batch_size']}",
            ],
            ["Semente de treino / threads CPU", f"{t['seed']} / {config['execution']['threads']}"],
        ],
        [260, r.available - 260],
        size=8.8,
        padding=5,
    )
    r.h("Recursos e carga preservados")
    r.p(
        "25 servidores com CPU/memória nominal 100/100; grafo Erdős-Rényi p=0,20; enlaces de 1 Gbps e atraso entre 1 e 2 ms. Demanda da cadeia completa: 20% de CPU e 15% de memória antes dos efeitos de compartilhamento. Quatro jogadores por sessão, até 18 SFs; FI 0,266 Mbps exclusivo e BE 25 Mbps compartilhável. P(h): 65%-90%; P(f): 17%-25%."
    )
    r.p(
        "Uma sessão chega a cada 10 segundos; lifetime uniforme [0,120] s. Cada semente começa com uma rede nova. Portanto, 300 requisições não são 300 sessões simultâneas. O cenário usa destino same_access, admissão sem fila e SLA de 6 ms para comunicação e processamento. Memória é representada pelo campo cache; custos de processamento são hipóteses do simulador.",
        size=9.5,
        leading=14,
    )

    r.start(
        "2. Gráficos do experimento",
        "Curvas agregadas das 33 sementes. O painel de decisão mostra média e percentis; não há comparação com outros algoritmos.",
        wide=True,
    )
    image = DATA / "analysis/graficos_33_sementes_300.png"
    with Image.open(image) as im:
        ratio = im.height / im.width
    width = min(r.available, (r.y - 125) / ratio)
    height = width * ratio
    r.pdf.drawImage(
        str(image),
        r.margin + (r.available - width) / 2,
        r.y - height,
        width=width,
        height=height,
        mask="auto",
    )
    r.y -= height + 12
    r.p(
        "Aceitação: média entre sementes do percentual acumulado até cada chegada. Latência: média acumulada ponderada pelos fluxos aceitos. CPU: média do uso global observado após cada chegada, incluindo liberação das sessões expiradas. Decisão: tempo por sessão, incluindo rejeições, sem espera pela resposta do LLM.",
        size=10,
        leading=14,
    )
    r.p(
        "A estabilidade da curva média não elimina a diferença entre topologias: a aceitação final variou de 77,67% a 99,67%. A latência SFC não é uma medição completa de latência visual em realidade aumentada.",
        size=10,
        leading=14,
    )

    r.start(
        "3. Rejeições, recursos e limites",
        "O limite inferior físico de latência ajuda a distinguir impedimentos do cenário de casos ainda não excluídos por esse limite.",
    )
    r.table(
        [
            ["Categoria", "Quantidade", "Interpretação"],
            ["Aceitas", "9.029", "Todas com latência SFC de até 6 ms"],
            [
                "Rejeitadas por limite mínimo físico",
                "546",
                "Limite inferior supera 6 ms, ou não existe caminho",
            ],
            [
                "Outras rejeições",
                "325",
                "Não excluídas pelo limite otimista; podem envolver ocupação e propostas inviáveis",
            ],
            ["Total de rejeições", "871", "Motivo registrado: nenhuma proposta treinada viável"],
        ],
        [175, 65, r.available - 240],
        size=9,
        padding=7,
    )
    r.h("Teto otimista de aceitação")
    r.p(
        "O limite inferior soma a menor comunicação possível entre acesso e destino ao caminho crítico de processamento usando os servidores mais rápidos. Ele ignora ocupação e pode supor escolhas que não são realizáveis em conjunto. Sob esses custos, as 546 sessões excluídas impedem superar <b>94,48%</b> neste conjunto. Assim, 98%-100% não é uma meta alcançável para estas requisições sem rever premissas ou requisitos físicos."
    )
    r.p(
        "As outras 325 rejeições representam uma diferença de até 3,28 pontos percentuais em relação ao teto otimista, mas não uma garantia de ganho. Para avaliar recuperação, é necessário examinar a viabilidade real de recursos e rotas e as alternativas propostas pelos atores."
    )
    r.h("Uso de recursos da rede")
    r.table(
        [
            ["Métrica", "Uso médio (%)", "Agregação"],
            [
                "CPU nas chegadas",
                num(summary["cpu_mean_at_arrivals_percent"], 3),
                "Média dos registros após cada chegada",
            ],
            [
                "CPU temporal",
                num(summary["time_weighted_cpu_mean_percent"], 3),
                "Média entre sementes das médias ponderadas pelo tempo",
            ],
            [
                "Memória temporal",
                num(summary["time_weighted_memory_mean_percent"], 3),
                "Mesma agregação temporal",
            ],
            [
                "Banda temporal",
                num(summary["time_weighted_bandwidth_mean_percent"], 3),
                "Mesma agregação temporal",
            ],
        ],
        [175, 95, r.available - 270],
        size=9,
        padding=6,
    )
    r.p(
        "O uso global baixo não comprova ausência de gargalos locais. Aqui, o limite de latência pode rejeitar sessões mesmo com CPU livre em outros servidores. Os percentuais temporais incluem a liberação final das sessões e não devem ser confundidos com ocupação de pico.",
        size=9.5,
        leading=14,
    )
    r.h("Implicações para realidade aumentada")
    r.p(
        "O tempo de decisão afeta o início ou a realocação de uma sessão. Mesmo com média de 347,51 ms, o p99 é de 1.199,04 ms e o máximo de 2.330,28 ms: esse cálculo deve ficar fora do loop de quadros. Rastreamento, renderização e display não foram medidos; o SLA de 6 ms da SFC não comprova conforto visual ou ausência de enjoo."
    )

    for page, subset in enumerate((seeds[:17], seeds[17:]), start=1):
        r.start(
            f"4. Resultados por semente ({page}/2)",
            "Cada linha corresponde a uma rede independente com 300 requisições; os percentuais não incluem sessões de outras sementes.",
        )
        rows = [
            [
                "Semente",
                "Aceitas",
                "Aceitação",
                "Latência média (ms)",
                "Decisão média (ms)",
                "Chamadas / sessões orientadas",
            ]
        ]
        for x in subset:
            rows.append(
                [
                    x["seed"],
                    x["accepted"] + "/300",
                    num(float(x["acceptance_percent"])) + "%",
                    num(float(x["mean_latency_ms"]), 3),
                    num(float(x["mean_decision_ms"])),
                    x["qwen_calls"] + " / " + x["guidance_used_requests"],
                ]
            )
        r.table(rows, [48, 62, 76, 95, 90, r.available - 371], size=8.8, padding=6)
        r.p(
            "Latência média: somente fluxos aceitos. Decisão média: todas as 300 sessões. Chamadas: consultas Qwen reais concluídas, incluindo as drenadas após a última admissão. Sessões orientadas: decisões que efetivamente usaram orientação já disponível; não correspondem ao número de chamadas.",
            size=9,
            leading=13,
        )
        if page == 2:
            r.h("Leitura da variação")
            r.p(
                "A média consolidada de 91,20% representa igualmente as 33 sementes, porque todas receberam 300 requisições. Selecionar apenas a melhor semente produziria uma impressão de quase 100% que não descreve o desempenho geral. As diferenças entre redes devem ser mantidas na análise."
            )

    r.start(
        "5. Validação e conclusões", "Resultados auditados sem alterar o experimento original."
    )
    r.h("O que foi conferido")
    r.p(
        "Foram verificadas as 33 sementes previstas, 300 registros por semente, correspondência entre aceitação e evidência dos atores, latências dentro do SLA, recursos nominais da topologia e hashes do checkpoint e dos metadados de treino. A auditoria examinou <b>358.589 decisões dos atores</b>; os hosts selecionados pertenciam às propostas elegíveis. As chamadas Qwen ocorreram em segundo plano, sem tempo de espera dentro da admissão."
    )
    r.p(
        "O simulador audita recursos a cada implantação e liberação. O arquivo de resultados não contém todos os snapshots de reservas para reconstruir de forma independente cada estado físico; a revisão dos registros não substitui essa auditoria durante a execução.",
        size=9.5,
        leading=14,
    )
    r.h("O que os resultados permitem concluir")
    r.p(
        "A execução concluiu 9.900 solicitações com 91,20% de aceitação e respeitou o SLA modelado nas sessões aceitas. O Qwen foi consultado de fato e suas diretrizes foram usadas em 93,39% das sessões. O modelo não atingiu a meta de 98%-100%, e o teto otimista físico calculado é inferior a essa meta."
    )
    r.p(
        "Como não há MAPPO sozinho nem MuSFiCO nas mesmas 33 sementes, não é possível atribuir ao Qwen o ganho em relação a experimentos anteriores ou comprovar superioridade sobre esses modelos. A comparação adequada exige uma ablação pareada de topologias, cargas e checkpoints. A telemetria visual de AR permanece <b>not_measured</b>."
    )
    r.h("Arquivos de origem")
    r.p(
        "Pasta do experimento: results/meu_treino_33_sementes_300/<br/>Medições: mappo_qwen.json.<br/>Configuração efetiva: experiment_config.json.<br/>Análise: analysis/summary.json e analysis/per_seed.csv.<br/>Checkpoint: rl_saved_models/MAPPO_muvr_meu_treino.pt.<br/>Treino: rl_saved_models/MAPPO_muvr_meu_treino.training.json.",
        size=8.5,
        leading=12,
    )
    for label, key in [
        ("SHA-256 do checkpoint", "model_sha256"),
        ("SHA-256 do arquivo de medições", "input_sha256"),
    ]:
        digest = summary[key]
        r.p(label + ":<br/>" + digest[:32] + "<br/>" + digest[32:], size=8, leading=11, mono=True)
    r.p(
        "Os registros do experimento original foram preservados. Este relatório e seus gráficos derivam somente da análise dos dados salvos pelo usuário.",
        size=9,
        leading=13,
    )
    r.finish()
    print(OUT)


if __name__ == "__main__":
    main()
