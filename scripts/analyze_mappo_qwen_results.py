"""Audit and summarize one completed MAPPO + Qwen experiment, without rerunning it."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from plot_muvr_comparison import curves, plt, validate_pairing
from scipy.stats import t

from muar_sfc.core.ar_metrics import distribution


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    source = args.input / "mappo_qwen.json"
    with source.open() as stream:
        doc = json.load(stream)
    config = json.loads((args.input / "experiment_config.json").read_text())
    expected = config["configuration"]["evaluation"]
    rows = doc["rows"]
    seeds = list(range(expected["seed_start"], expected["seed_start"] + expected["seed_count"]))
    assert [r["seed"] for r in rows] == seeds, "Sementes incompletas ou diferentes da configuração"
    assert doc["profile"]["requests"] == expected["requests"]
    validate_pairing({"mappo_qwen": doc})
    model = Path(config["configuration"]["execution"]["model_path"])
    if not model.is_absolute():
        model = Path(config["project_root"]) / model
    model_hash = sha256(model)
    training = json.loads(model.with_suffix(".training.json").read_text())
    assert model_hash == training["checkpoint_sha256"]
    decisions, response, latency, cpu, per_seed = [], [], [], [], []
    reason_counts = Counter()
    actor_decisions = 0
    impossible, potential = 0, 0
    for row in rows:
        assert row["requests"] == expected["requests"]
        e = row["inference_evidence"]
        assert e["model_sha256"] == model_hash and e["checkpoint_loaded"]
        assert e["policy_candidates_only"] and e["planning_mode"] == "background_nonblocking"
        assert not e["background_pending"]
        assert len(e["request_evidence"]) == row["requests"]
        assert sum(r["accepted_immediate"] for r in row["records"]) == row["accepted"]
        assert (
            sum(r["guidance_used"] for r in e["request_evidence"])
            == e["guidance_applied_requests"]
        )
        assert sum(r["qwen_success"] for r in e["planning_records"]) == e["qwen_successful_calls"]
        assert all(
            n["cpu_capacity"] == n["cache_capacity"] == 100 for _, n in row["topology"]["nodes"]
        )
        assert len(row["topology"]["nodes"]) == 25
        assert all(
            edge["bandwidth_capacity"] == 1000 and 1 <= edge["latency"] <= 2
            for _, _, edge in row["topology"]["edges"]
        )
        for record, evidence in zip(row["records"], e["request_evidence"], strict=True):
            assert record["accepted_immediate"] == evidence["accepted"]
            assert evidence["qwen_seconds"] == 0
            if record["accepted_immediate"]:
                assert record["latency_lower_bound_ms"] is not None
                assert record["latency_lower_bound_ms"] <= doc["profile"]["sla"]
            if not record["accepted_immediate"]:
                reason_counts[record["placement"]] += 1
                bound = record["latency_lower_bound_ms"]
                if bound is None or bound > doc["profile"]["sla"]:
                    impossible += 1
                else:
                    potential += 1
            for decision in evidence["actor_decisions"]:
                assert (
                    decision["selected_host"] is None
                    or decision["selected_host"] in decision["proposals"].values()
                )
                assert "action_log_probabilities" in decision
                actor_decisions += 1
        decisions += [r["decision_s"] * 1000 for r in row["records"]]
        response += [r["admission_response_s"] * 1000 for r in row["records"]]
        latency += [v for r in row["records"] for v in r["latencies_ms"]]
        cpu += [s["cpu_percent"] for s in row["utilization_samples"]]
        per_seed.append(
            {
                "seed": row["seed"],
                "accepted": row["accepted"],
                "requests": row["requests"],
                "acceptance_percent": row["acceptance_percent"],
                "mean_latency_ms": row["mean_latency_ms"],
                "max_latency_ms": row["max_latency_ms"],
                "mean_decision_ms": row["mean_decision_s"] * 1000,
                "qwen_calls": e["qwen_successful_calls"],
                "guidance_used_requests": e["guidance_applied_requests"],
            }
        )
    rates = [r["acceptance_percent"] for r in rows]
    mean = statistics.mean(rates)
    margin = t.ppf(0.975, len(rows) - 1) * statistics.stdev(rates) / np.sqrt(len(rows))
    total = sum(r["requests"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    result = {
        "status": "passed",
        "seeds": seeds,
        "requests_per_seed": expected["requests"],
        "total_requests": total,
        "accepted": accepted,
        "rejected": total - accepted,
        "acceptance_percent": 100 * accepted / total,
        "acceptance_seed_ci95_percent": [max(0, mean - margin), min(100, mean + margin)],
        "acceptance_seed_stddev_percentage_points": statistics.stdev(rates),
        "best_seed": max(per_seed, key=lambda r: r["accepted"]),
        "worst_seed": min(per_seed, key=lambda r: r["accepted"]),
        "latency_ms": distribution(latency),
        "decision_ms": distribution(decisions),
        "admission_response_ms": distribution(response),
        "cpu_mean_at_arrivals_percent": statistics.mean(cpu),
        "time_weighted_cpu_mean_percent": statistics.mean(
            r["time_weighted_utilization_percent"]["cpu"] for r in rows
        ),
        "time_weighted_memory_mean_percent": statistics.mean(
            r["time_weighted_utilization_percent"]["memory"] for r in rows
        ),
        "time_weighted_bandwidth_mean_percent": statistics.mean(
            r["time_weighted_utilization_percent"]["bandwidth"] for r in rows
        ),
        "qwen_successful_calls": sum(
            r["inference_evidence"]["qwen_successful_calls"] for r in rows
        ),
        "qwen_failed_calls": sum(r["inference_evidence"]["qwen_failed_calls"] for r in rows),
        "qwen_fallback_calls": sum(r["inference_evidence"]["qwen_fallback_calls"] for r in rows),
        "guidance_applied_requests": sum(
            r["inference_evidence"]["guidance_applied_requests"] for r in rows
        ),
        "rejection_reasons": dict(reason_counts),
        "rejected_with_physical_latency_lower_bound_above_sla": impossible,
        "rejected_not_excluded_by_latency_lower_bound": potential,
        "optimistic_latency_ceiling_percent": 100 * (total - impossible) / total,
        "actor_decisions_audited": actor_decisions,
        "model_sha256": model_hash,
        "input_sha256": sha256(source),
        "training": {
            k: training[k]
            for k in (
                "episodes",
                "demonstration_episodes",
                "training_steps",
                "MAPPO_updates",
                "held_out_before_percent",
                "held_out_after_MAPPO_percent",
            )
        },
        "ar_telemetry": "not_measured",
    }
    out = args.input / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (out / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    acceptance_curve, latency_curve, cpu_curve, _ = curves(doc)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(1, expected["requests"] + 1)
    for ax, values, title, unit in zip(
        axes.flat,
        (acceptance_curve, latency_curve, cpu_curve),
        ("Aceitação acumulada", "Latência média acumulada", "Utilização de CPU"),
        ("Aceitação (%)", "Latência SFC (ms)", "CPU (%)"),
        strict=False,
    ):
        ax.plot(x, values, color="#8054b3", label="MAPPO + Qwen assíncrono", linewidth=2)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlabel("Requisições submetidas")
        ax.legend()
    axes[0, 0].set_ylim(0, 105)
    axes[0, 1].axhline(doc["profile"]["sla"], color="gray", linestyle="--", label="SLA 6 ms")
    axes[0, 1].legend()
    d = result["decision_ms"]
    axes[1, 1].bar(
        ["Média", "p95", "p99"],
        [d["mean"], d["p95"], d["p99"]],
        color=["#8054b3", "#a68bc3", "#c7b5da"],
    )
    for i, v in enumerate([d["mean"], d["p95"], d["p99"]]):
        axes[1, 1].text(i, v, f"{v:.2f} ms", ha="center", va="bottom")
    axes[1, 1].set_ylim(0, d["p99"] * 1.2)
    axes[1, 1].set_title("Tempo de decisão por sessão SFC")
    axes[1, 1].set_ylabel("Decisão (ms)")
    fig.suptitle(
        f"MAPPO + Qwen — {len(rows)} sementes × {expected['requests']} requisições", fontsize=18
    )
    fig.text(
        0.5,
        0.018,
        "Mesmos recursos nominais; SLA SFC de 6 ms. Latência visual de AR não medida.\nCurvas agregadas das sementes; sem execução de novos experimentos.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"graficos_33_sementes_300.{extension}", dpi=180)
    plt.close(fig)
    lines = [
        "# Análise do experimento MAPPO + Qwen",
        "",
        f"{len(rows)} sementes completas, {expected['requests']} requisições cada; {accepted}/{total} aceitas ({result['acceptance_percent']:.2f}%).",
        "",
        f"IC95% da média entre sementes (t de Student): {mean - margin:.2f}% a {mean + margin:.2f}%. Desvio padrão entre sementes: {statistics.stdev(rates):.2f} pontos percentuais.",
        "",
        "| Métrica | Média | p95 | p99 | Máximo |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key in [
        ("Latência SFC (ms)", "latency_ms"),
        ("Decisão (ms)", "decision_ms"),
        ("Resposta de admissão (ms)", "admission_response_ms"),
    ]:
        stats = result[key]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{stats[k]:.3f}" for k in ("mean", "p95", "p99", "max"))
            + " |"
        )
    lines += [
        "",
        f"Qwen: {result['qwen_successful_calls']} chamadas reais, {result['qwen_failed_calls']} falhas, {result['qwen_fallback_calls']} fallbacks; orientação aplicada em {result['guidance_applied_requests']}/{total} sessões.",
        "",
        f"CPU média nas chegadas: {result['cpu_mean_at_arrivals_percent']:.3f}%. CPU média temporal entre sementes: {result['time_weighted_cpu_mean_percent']:.3f}%. Memória temporal: {result['time_weighted_memory_mean_percent']:.3f}%; banda temporal: {result['time_weighted_bandwidth_mean_percent']:.3f}%.",
        "",
        f"Rejeições: {impossible} com limite inferior físico de latência acima do SLA ou sem caminho; {potential} não excluídas por esse limite. Estas últimas não são necessariamente recuperáveis: o limite ignora ocupação e restrições adicionais. Teto otimista de latência: {result['optimistic_latency_ceiling_percent']:.2f}%, não uma promessa de aceitação.",
        "",
        "O conjunto isolado de 30 casos de treinamento permaneceu em 86,67% antes e depois. A avaliação completa confirmou execução do checkpoint treinado, mas não demonstra um ganho isolado de treinamento.",
        "",
        "Somente MAPPO + Qwen foi avaliado nas 33 sementes. Sem ablação pareada de MAPPO sozinho, não é possível atribuir ao Qwen o aumento de aceitação comparado a experimentos menores ou comprovar superioridade sobre MuSFiCO. O objetivo de 98%-100% não foi atingido na média.",
        "",
        "O tempo de decisão afeta admissão/realocação, não deve bloquear quadros. A SFC respeitou 6 ms, mas não foram medidos rastreamento, renderização e display: conforto em AR permanece não validado.",
        "",
        "A auditoria verificou sementes, registros, SLA, propostas elegíveis, recursos nominais da topologia e hashes do checkpoint/metadados. Recursos após cada alocação são auditados pelo simulador; os JSONs não contêm snapshots completos de todas as reservas para reconstruir independentemente essa auditoria.",
        "",
        "Resultados preservados: a análise não executou treino ou simulação e não modificou o arquivo de medições original.",
        "",
        "[Gráficos](graficos_33_sementes_300.png) · [Tabela por semente](per_seed.csv).",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
