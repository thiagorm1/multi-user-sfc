"""Plot and audit paired MAPPO/Qwen measurements for any request count."""

import argparse
import json
from pathlib import Path

from plot_muvr_comparison import curves, plt, validate_pairing

from muar_sfc.core.ar_metrics import distribution

MODELS = {
    "mappo": ("MAPPO treinado", "#3478b8"),
    "mappo_qwen": ("MAPPO + Qwen assíncrono", "#8054b3"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    docs = {v: json.loads((args.input / f"{v}.json").read_text()) for v in MODELS}
    validate_pairing(docs)
    profile = docs["mappo"]["profile"]
    count = profile["requests"]
    out = args.input / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    summary = []
    for index, (variant, (label, color)) in enumerate(MODELS.items()):
        doc = docs[variant]
        rows = doc["rows"]
        acceptance, latency, cpu, decision = curves(doc)
        x = range(1, count + 1)
        for ax, data in zip(axes.flat, (acceptance, latency, cpu), strict=False):
            ax.plot(x, data, label=label, color=color, linewidth=1.8)
        axes[1, 1].bar(index, decision, color=color, width=0.55)
        axes[1, 1].text(index, decision, f"{decision:.2f} ms", ha="center", va="bottom")
        decisions, response, latencies = [], [], []
        for row in rows:
            e = row["inference_evidence"]
            assert e["checkpoint_loaded"] and e["policy_candidates_only"]
            assert e["planning_mode"] == "background_nonblocking" and not e["background_pending"]
            if variant == "mappo_qwen":
                assert e["qwen_successful_calls"] > 0 and e["guidance_applied_requests"] > 0
            for request in e["request_evidence"]:
                assert request["qwen_seconds"] == 0
                for d in request["actor_decisions"]:
                    assert (
                        d["selected_host"] is None or d["selected_host"] in d["proposals"].values()
                    )
            decisions += [r["decision_s"] * 1000 for r in row["records"]]
            response += [r["admission_response_s"] * 1000 for r in row["records"]]
            latencies += [v for r in row["records"] for v in r["latencies_ms"]]
        accepted = sum(r["accepted"] for r in rows)
        total = count * len(rows)
        summary.append(
            {
                "model": variant,
                "label": label,
                "requests": total,
                "requests_per_seed": count,
                "seeds": [r["seed"] for r in rows],
                "accepted": accepted,
                "acceptance_percent": 100 * accepted / total,
                "latency_ms": distribution(latencies),
                "decision_ms": distribution(decisions),
                "admission_response_ms": distribution(response),
                "cpu_mean_at_arrivals_percent": float(cpu.mean()),
                "qwen_successful_calls": sum(
                    r["inference_evidence"]["qwen_successful_calls"] for r in rows
                ),
                "qwen_failed_calls": sum(
                    r["inference_evidence"].get("qwen_failed_calls", 0) for r in rows
                ),
                "guidance_applied_requests": sum(
                    r["inference_evidence"]["guidance_applied_requests"] for r in rows
                ),
                "ar_telemetry": "not_measured",
            }
        )
    titles = [
        "Taxa de aceitação acumulada",
        "Latência média acumulada",
        "Utilização de CPU na rede",
        "Tempo médio de decisão por sessão SFC",
    ]
    units = ["Aceitação (%)", "Latência (ms)", "CPU (%)", "Decisão (ms)"]
    for index, (ax, title, unit) in enumerate(zip(axes.flat, titles, units, strict=True)):
        ax.set_title(title)
        ax.set_ylabel(unit)
        if index < 3:
            ax.set_xlabel("Número de sessões submetidas")
            ax.legend()
    axes[0, 0].set_ylim(0, 105)
    axes[0, 1].axhline(profile["sla"], color="gray", linestyle="--", label="SLA 6 ms")
    axes[0, 1].legend()
    axes[1, 1].set_xticks(range(len(MODELS)), ["MAPPO", "MAPPO + Qwen\nassíncrono"])
    axes[1, 1].set_ylim(0, max(s["decision_ms"]["mean"] for s in summary) * 1.25)
    fig.suptitle(f"MUVR - {count} sessões por semente; SLA de {profile['sla']:g} ms", fontsize=18)
    fig.text(
        0.5,
        0.015,
        f"{len(rows)} sementes pareadas • 25 servidores • recursos nominais • "
        "Qwen em CPU e segundo plano\n"
        "Latência SFC não equivale a motion-to-photon; telemetria AR não medida.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"comparativo_mappo_qwen_{count}.{extension}", dpi=180)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (args.input / "verification.json").write_text(
        json.dumps({"status": "passed", "summary": summary}, indent=2)
    )
    lines = [
        f"# MAPPO + Qwen assíncrono: {count} sessões por semente",
        "",
        f"Sementes pareadas: {summary[0]['seeds']}. "
        "SLA de 6 ms, recursos nominais e admissão sem fila.",
        "",
        "| Modelo | Aceitas/total | Aceitação | Latência média (ms) | Decisão média (ms) | "
        "Decisão p95 (ms) | Decisão p99 (ms) | CPU (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summary:
        d = s["decision_ms"]
        lines.append(
            f"| {s['label']} | {s['accepted']}/{s['requests']} | "
            f"{s['acceptance_percent']:.2f}% | {s['latency_ms']['mean']:.3f} | "
            f"{d['mean']:.2f} | {d['p95']:.2f} | {d['p99']:.2f} | "
            f"{s['cpu_mean_at_arrivals_percent']:.3f} |"
        )
    q = summary[1]
    lines += [
        "",
        f"Qwen: {q['qwen_successful_calls']} chamadas reais, {q['qwen_failed_calls']} falhas; "
        f"orientação aplicada em {q['guidance_applied_requests']}/{q['requests']} sessões.",
        "",
        "O teste usa tempo virtual. Chamadas Qwen em andamento não bloqueiam a admissão, "
        "e o MAPPO segue sozinho sem orientação válida. O número de consultas depende "
        "também do tempo real de execução. Não há medida de rastreamento/display "
        "ou validação de conforto em AR.",
        "",
        "A carga total não é simultânea: sessões chegam a cada 10 segundos e expiram "
        f"em até 120 segundos. Enviar {count} ao longo do tempo "
        f"não equivale a manter {count} ativas.",
        "",
        f"[Gráficos](plots/comparativo_mappo_qwen_{count}.png).",
    ]
    (args.input / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
