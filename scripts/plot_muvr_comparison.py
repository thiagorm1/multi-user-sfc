"""Four-panel comparison from paired, physically validated MUVR benchmark JSONs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "joint": ("Nosso modelo", "#3478b8", "D", "-"),
    "replic": ("REPLIC (DRL; adaptado)", "#2a9d8f", "o", "-"),
    "musfico": ("MuSFiCO (DP; adaptado)", "#e76f51", "s", "-"),
    "greedy": ("Greedy (DAG)", "#f4a261", "^", "-"),
    "darsppo": ("DARSPPO (DRL; adaptado)", "#264653", "v", "-"),
}


def validate_pairing(documents):
    reference = next(iter(documents.values()))
    seeds = [row["seed"] for row in reference["rows"]]
    for name, document in documents.items():
        if document["profile"] != reference["profile"]:
            raise ValueError(f"Profile differs for {name}")
        if [row["seed"] for row in document["rows"]] != seeds:
            raise ValueError(f"Seeds differ for {name}")
        for row, expected in zip(document["rows"], reference["rows"], strict=True):
            for key in ("topology_sha256", "workload_sha256"):
                if row[key] != expected[key]:
                    raise ValueError(f"{key} differs for {name}")
            if len(row["records"]) != reference["profile"]["requests"]:
                raise ValueError(f"Incomplete trace for {name}")
            for request in row["records"]:
                if request["attempts"] != 1 or request["wait_s"] != 0:
                    raise ValueError(f"Not immediate admission for {name}")
                if request["accepted_immediate"] and not request["latencies_ms"]:
                    raise ValueError(f"Accepted request has no measured latency: {name}")
                if any(
                    lat > reference["profile"]["sla"] + 1e-9 for lat in request["latencies_ms"]
                ):
                    raise ValueError(f"SLA violation for {name}")


def curves(document):
    rows = document["rows"]
    n = document["profile"]["requests"]
    accepted = np.array(
        [[r["accepted_immediate"] for r in row["records"]] for row in rows], dtype=float
    )
    acceptance = (accepted.cumsum(axis=1) / np.arange(1, n + 1) * 100).mean(axis=0)
    cpu = np.array([[s["cpu_percent"] for s in row["utilization_samples"]] for row in rows])
    latency_sum = np.zeros(n)
    latency_count = np.zeros(n)
    for row in rows:
        for index, request in enumerate(row["records"]):
            if request["accepted_immediate"]:
                latency_sum[index] += sum(request["latencies_ms"])
                latency_count[index] += len(request["latencies_ms"])
    cumulative_count = latency_count.cumsum()
    latency = np.divide(
        latency_sum.cumsum(), cumulative_count, out=np.full(n, np.nan), where=cumulative_count > 0
    )
    decision = np.array([[r["decision_s"] * 1000 for r in row["records"]] for row in rows])
    return acceptance, latency, cpu.mean(axis=0), float(decision.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/muvr_comparison"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mappo-input", type=Path)
    args = parser.parse_args()
    output = args.output or args.input / "plots"
    output.mkdir(parents=True, exist_ok=True)
    documents = {model: json.loads((args.input / f"{model}.json").read_text()) for model in MODELS}
    models = dict(MODELS)
    if args.mappo_input:
        models = {
            "mappo_qwen": ("MAPPO + Qwen", "#8054b3", "D", "-"),
            "mappo": ("MAPPO treinado", "#3478b8", "o", "-"),
            **models,
        }
        models["joint"] = ("Heurística anterior", "#888888", "x", ":")
        for model in ("mappo", "mappo_qwen"):
            documents[model] = json.loads((args.mappo_input / f"{model}.json").read_text())
            for row in documents[model]["rows"]:
                evidence = row["inference_evidence"]
                if not evidence["checkpoint_loaded"] or not evidence["policy_candidates_only"]:
                    raise ValueError("Trained actor evidence missing")
                if model == "mappo_qwen" and (
                    evidence["qwen_successful_calls"] == 0
                    or evidence["qwen_fallback_calls"] != 0
                    or (
                        evidence.get("planning_mode") == "background_nonblocking"
                        and evidence["guidance_applied_requests"] == 0
                    )
                ):
                    raise ValueError("Real Qwen evidence missing")
        if all(
            r["inference_evidence"].get("planning_mode") == "background_nonblocking"
            for r in documents["mappo_qwen"]["rows"]
        ):
            models["mappo_qwen"] = ("MAPPO + Qwen (assíncrono)", "#8054b3", "D", "-")
    validate_pairing(documents)
    profile = documents["joint"]["profile"]
    repetitions = len(documents["joint"]["rows"])
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "grid.alpha": 0.55,
            "lines.linewidth": 2.2,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5))
    fig.suptitle(
        f"Comparação MUVR — {profile['requests']} requisições e SLA de {profile['sla']:g} ms",
        fontsize=18,
        y=0.985,
    )
    x = np.arange(1, profile["requests"] + 1)
    summaries, curve_records = [], []
    for model, (label, color, marker, linestyle) in models.items():
        acceptance, latency, cpu, decision = curves(documents[model])
        for axis, values in zip(
            (axes[0, 0], axes[0, 1], axes[1, 0]), (acceptance, latency, cpu), strict=True
        ):
            axis.plot(
                x,
                values,
                color=color,
                marker=marker,
                markersize=4,
                markevery=max(1, len(x) // 12),
                linestyle=linestyle,
                label=label,
                zorder=10 if model == "joint" else 3,
            )
        rows = documents[model]["rows"]
        valid_latency = latency[np.isfinite(latency)]
        summaries.append(
            {
                "model": model,
                "label": label,
                "repetitions": repetitions,
                "requests": sum(row["requests"] for row in rows),
                "accepted": sum(row["accepted"] for row in rows),
                "acceptance_percent": acceptance[-1],
                "mean_latency_ms": valid_latency[-1] if len(valid_latency) else None,
                "cpu_mean_at_arrivals_percent": float(cpu.mean()),
                "mean_decision_ms": decision,
            }
        )
        for index in range(len(x)):
            curve_records.append(
                {
                    "model": model,
                    "requests_submitted": int(x[index]),
                    "acceptance_percent": float(acceptance[index]),
                    "cumulative_mean_latency_ms": float(latency[index])
                    if np.isfinite(latency[index])
                    else None,
                    "cpu_percent": float(cpu[index]),
                }
            )
    axes[0, 0].set_title("Taxa de Aceitação de Requisições (%)")
    axes[0, 0].set_ylabel("Taxa de aceitação (%)")
    axes[0, 0].set_ylim(-3, 105)
    axes[0, 1].set_title("Latência Média Acumulada (ms)")
    axes[0, 1].set_ylabel("Latência E2E (ms)")
    axes[0, 1].axhline(
        profile["sla"], color="#777777", ls="--", lw=1.4, label=f"SLA: {profile['sla']:g} ms"
    )
    axes[0, 1].set_ylim(bottom=0)
    missing = [
        item["label"].split(" (")[0] for item in summaries if item["mean_latency_ms"] is None
    ]
    if missing:
        axes[0, 1].text(
            0.02,
            0.03,
            f"Sem aceites: {', '.join(missing)} (latência N/D)",
            transform=axes[0, 1].transAxes,
            fontsize=10,
        )
    axes[1, 0].set_title("Utilização de CPU na Rede (%)")
    axes[1, 0].set_ylabel("Uso de CPU (%)")
    axes[1, 0].set_ylim(bottom=0)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("Número de Requisições SFC Submetidas")
        axis.set_xlim(1, len(x))
        axis.set_xticks([1, *range(10, len(x) + 1, 10)])
        axis.legend(loc="best", fontsize=9, frameon=True)
    times = [item["mean_decision_ms"] for item in summaries]
    bars = axes[1, 1].bar(
        range(len(models)),
        times,
        width=0.62,
        color=[item[1] for item in models.values()],
        edgecolor="#333333",
        alpha=0.9,
    )
    axes[1, 1].set_title("Tempo Médio de Decisão por SFC (ms)")
    axes[1, 1].set_ylabel("Tempo de decisão (ms)")
    axes[1, 1].set_xticks(
        range(len(models)),
        [
            {
                "joint": "Heurística\nanterior" if args.mappo_input else "Nosso\nmodelo",
                "mappo": "MAPPO",
                "mappo_qwen": "MAPPO +\nQwen",
                "replic": "REPLIC",
                "musfico": "MuSFiCO\n(DP adaptado)",
                "greedy": "Greedy\n(DAG)",
                "darsppo": "DARSPPO",
            }[model]
            for model in models
        ],
    )
    axes[1, 1].set_ylim(0, max(times) * 1.2)
    axes[1, 1].bar_label(
        bars,
        labels=[f"{value:.2f} ms" for value in times],
        padding=4,
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.026,
        f"{repetitions} repetições pareadas • 25 servidores • admissão imediata • "
        f"{profile['requests']} sessões de 4 jogadores • CPU sem aumento",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.009,
        (
            "MAPPO treinado para MUVR; Qwen real. Baselines adaptados ao DAG. "
            if args.mappo_input
            else "Baselines adaptados ao DAG; checkpoints DRL sem retreinamento. "
        )
        + "Resultados deste simulador, não resultados publicados do MuSFiCO.",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96), h_pad=2.3, w_pad=2.7)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"comparativo_muvr_50.{extension}", dpi=200, facecolor="white")
    plt.close(fig)
    for filename, records in [("summary.csv", summaries), ("curves.csv", curve_records)]:
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    if args.mappo_input:
        background = all(
            r["inference_evidence"].get("planning_mode") == "background_nonblocking"
            for r in documents["mappo_qwen"]["rows"]
        )
        timing = (
            "Qwen em segundo plano: decisão não espera a chamada; custo de submissão "
            "é reportado separadamente e incluído no tempo de resposta de admissão. "
            if background
            else "Tempo de decisão inclui a chamada síncrona ao LLM. "
        )
        (output / "mappo_qwen_method.md").write_text(
            "MAPPO: checkpoint específico MUVR, quatro atores treinados; só suas propostas "
            "são elegíveis. Busca interna limitada a oito alternativas da política, sem fila. "
            "Qwen: chamadas reais periódicas em CPU, com diretrizes aplicadas ao ranking. "
            + timing
            + "Perfil e hashes pareados com os baselines. "
            "Os relatórios anteriores permanecem registros dos experimentos históricos.\n"
        )
    (output / "method.md").write_text("""# Método de cálculo

Mesmas topologias e solicitações para todos os modelos, conferidas por hashes.
Uma requisição é uma sessão completa de quatro jogadores. Todos os caminhos aceitos
respeitam 6 ms e o alocador físico audita CPU, memória e banda e libera as sessões expiradas.

- Aceitação: aceites acumulados / requisições submetidas, média das sementes.
- Latência média acumulada: soma das latências dos caminhos aceitos até N / número
  desses caminhos. Rejeições não entram como zero. Não é uma soma acumulada de atrasos.
- CPU: uso global instantâneo após cada chegada, incluindo as liberações por lifetime,
  dividido pela capacidade de todos os servidores; média entre sementes. Já está em %.
- Decisão: média de todas as tentativas, inclusive rejeições, convertida de s para ms.
  Exclui carregamento inicial dos checkpoints, inclui inferência, busca e validação.

REPLIC/DARSPPO usam os checkpoints existentes e suas funções originais de observação;
a ordem reversa percorre todas as SFs do DAG, com coordenação dos serviços comuns e
validação física ao final. Não houve retreinamento. Greedy usa o algoritmo DAG existente,
com validação física completa. A curva MuSFiCO usa um adaptador de programação dinâmica
para DAG, com feixe limitado a 4 estados por camada: não é a execução inalterada do artigo
ou da classe original especializada em MUAR. Essas adaptações são parte da comparação.
Logo, a figura não comprova superioridade sobre as implementações publicadas.
""")
    print(json.dumps(summaries, indent=2))
    print(output / "comparativo_muvr_50.png")


if __name__ == "__main__":
    main()
