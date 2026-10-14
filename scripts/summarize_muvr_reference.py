"""Summarize all independent seeds without selecting favorable topologies."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    document = json.loads(args.input.read_text())
    rows = document["rows"]
    accepted = sum(row["accepted"] for row in rows)
    requested = sum(row["requests"] for row in rows)
    latencies = [lat for row in rows for r in row["records"] for lat in r["latencies_ms"]]
    decisions = [r["decision_s"] for row in rows for r in row["records"]]
    acceptance = np.array([row["acceptance_percent"] for row in rows])
    half = (
        float(t.ppf(0.975, len(rows) - 1) * acceptance.std(ddof=1) / np.sqrt(len(rows)))
        if len(rows) > 1
        else 0
    )
    upper = np.mean([row["physical_sla_upper_bound_percent"] for row in rows])
    assert all(lat <= document["profile"]["sla"] + 1e-9 for lat in latencies)
    assert all(r["attempts"] == 1 and r["wait_s"] == 0 for row in rows for r in row["records"])
    summary = {
        "repetitions": len(rows),
        "requests": requested,
        "accepted": accepted,
        "acceptance_percent": 100 * accepted / requested,
        "acceptance_seed_ci95": [
            max(0, acceptance.mean() - half),
            min(100, acceptance.mean() + half),
        ],
        "physical_sla_upper_bound_percent": float(upper),
        "mean_latency_ms": float(np.mean(latencies)),
        "max_latency_ms": max(latencies),
        "accumulated_latency_ms": sum(latencies),
        "latency_samples": len(latencies),
        "mean_session_max_latency_ms": float(
            np.mean(
                [
                    max(r["latencies_ms"])
                    for row in rows
                    for r in row["records"]
                    if r["accepted_immediate"]
                ]
            )
        ),
        "mean_decision_s": float(np.mean(decisions)),
        "decision_p95_s": float(np.percentile(decisions, 95)),
        "mean_time_weighted_utilization_percent": {
            name: float(np.mean([r["time_weighted_utilization_percent"][name] for r in rows]))
            for name in ("cpu", "memory", "bandwidth")
        },
    }
    output = args.input.parent
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output / "seeds.csv").open("w", newline="") as handle:
        fields = [
            "seed",
            "requests",
            "accepted",
            "acceptance_percent",
            "physical_sla_upper_bound_percent",
            "mean_latency_ms",
            "max_latency_ms",
            "mean_decision_s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = f"""# MUVR: 50 requisições, admissão imediata, SLA 6 ms

{len(rows)} repetições independentes, sementes {rows[0]["seed"]} a {rows[-1]["seed"]}.
Aceitação: **{accepted}/{requested} ({summary["acceptance_percent"]:.2f}%)**.
IC de 95% entre sementes: {summary["acceptance_seed_ci95"][0]:.2f}% a
{summary["acceptance_seed_ci95"][1]:.2f}%.
Latência média por jogador: **{summary["mean_latency_ms"]:.3f} ms**;
pior latência aceita: **{summary["max_latency_ms"]:.3f} ms**.
Soma das latências dos {len(latencies)} caminhos de jogadores aceitos:
{summary["accumulated_latency_ms"]:.3f} ms. Não é o atraso de uma única requisição.
Tempo médio de decisão medido neste PC: {summary["mean_decision_s"]:.3f} s;
p95: {summary["decision_p95_s"]:.3f} s. Execuções concorrentes podem afetar esse tempo.

Uso médio ponderado pelo tempo, incluindo liberações até a última sessão expirar:
CPU {summary["mean_time_weighted_utilization_percent"]["cpu"]:.2f}%,
memória {summary["mean_time_weighted_utilization_percent"]["memory"]:.2f}%,
banda {summary["mean_time_weighted_utilization_percent"]["bandwidth"]:.2f}%.

## Limite e comparação

A meta de aproximadamente 98% {"foi" if summary["acceptance_percent"] >= 98 else "não foi"}
atingida na média. O teto otimista imposto pelo modelo de latência é {upper:.2f}%:
menor caminho de comunicação mais caminho crítico de processamento usando, para cada SF,
o servidor mais rápido. Esse teto ignora disputa por recursos e restrições de sincronização.
É um limite deste modelo e dessas solicitações, não um limite universal do artigo.

Capacidades nominais: CPU/memória 100/100 por servidor; IPS original sem multiplicador
uniforme entre 1e10 e 2e10; banda 1 Gbps por enlace. Não preserva a antiga topologia
Luxembourg nem seus subnós heterogêneos: são 25 servidores homogêneos conforme o pedido.
Erdős–Rényi p=0,20 sem regenerar grafos desconectados; enlaces uniformes [1,2] ms.
Chegadas determinísticas de 10 s; lifetime contínuo [0,120] s; 4 jogadores por sessão.
Uma requisição inclui todos os jogadores e só conta se todos forem implantados imediatamente.

Hipótese principal: jogadores da mesma sessão usam o mesmo acesso regional, mantendo origem
aleatória independente. Há teste separado com destinos espalhados. O SLA soma todos os
saltos, inclusive origem/destino, e processamento. O custo de processamento usa a fórmula
existente (60 fps, 10 ciclos por bit); custos por SF não foram recuperados do artigo.
CPU e memória das 18 SFs distintas são proporcionais ao tráfego, normalizados para 20 e 15
unidades **por sessão inteira**, seguindo o pedido. O texto da tese também menciona consumo
por usuário; essa ambiguidade impede uma reprodução exata.

FI mantém uma SF distinta de 0,266 Mbps por jogador. BE divide 25 Mbps conforme P(h) e P(f),
com duas SFs comuns e duas únicas por jogador. IA, dois renderizadores e codec são comuns;
total 18 SFs. Compartilhamento entre jogadores é restrito à própria sessão. Tráfego final
codificado de 25,266 Mbps por jogador é reservado individualmente de forma conservadora.
Predição não elimina custo de processamento nem usa capacidade gratuita.

O resultado usa uma heurística conjunta com o motor de viabilidade do MAPPO como fallback;
não atribui ganhos ao checkpoint treinado em outra topologia nem ao Qwen.
Os números do MuSFiCO fornecidos pelo usuário são referência, não execução reproduzida.
Fonte primária consultada: tese do autor, seção 5.3.1 e tabela 9:
https://webarchive.inf.unibe.ch/cds/publications/files/Hugo_Santos.pdf
A tabela informa 25 nós totais e 30% de borda; este perfil segue os **25 servidores** pedidos.

## Reproduzir

```bash
.venv/bin/python scripts/benchmark_muvr_reference.py
# saída padrão pode ser definida com --output
.venv/bin/python scripts/summarize_muvr_reference.py results/muvr_50_reference/joint_final.json
.venv/bin/python -m muar_sfc.main
```

O último comando usa `.env` e executa uma semente em tempo virtual.
Para destinos espalhados, use `--destinations scattered` e uma saída diferente com `--output`.
Os JSONs registram topologia, demandas, aceitação, latências, decisão e utilização por requisição.
"""
    (output / "report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
