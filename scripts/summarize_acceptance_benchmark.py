"""Validate paired runs and write a report and standalone chart."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "before": "Anterior",
    "improved": "Melhorias",
    "heuristic": "Melhorias + heurística",
    "qwen": "Melhorias + Qwen",
}
WORKLOADS = {"": "Rajada (média 3 s)", "moderate_": "Chegadas espaçadas (média 10 s)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    datasets = {}
    aggregates = []
    seed_rows = []
    for prefix, workload in WORKLOADS.items():
        paired = {}
        model_hashes = set()
        for variant in LABELS:
            payload = json.loads((directory / f"{prefix}{variant}.json").read_text())
            assert len(payload["rows"]) == 3, f"Incomplete run: {prefix}{variant}"
            datasets[prefix, variant] = payload
            model_hashes.add(payload["model_sha256"])
            for row in payload["rows"]:
                assert row["checkpoint_dimensions_match"]
                paired.setdefault(row["seed"], set()).add(
                    (row["trace_sha256"], row["topology_sha256"])
                )
                seed_rows.append(
                    {
                        "workload": workload,
                        "variant": variant,
                        "seed": row["seed"],
                        "accepted": row.get("accepted", 0),
                        "requests": row["requests"],
                        "acceptance_pct": row["acceptance_pct"],
                        "planning_calls": row["planning_calls"],
                        "llm_success": row.get("llm_success", 0),
                        "llm_fallback": row.get("llm_fallback", 0),
                    }
                )
            accepted = sum(r.get("accepted", 0) for r in payload["rows"])
            count = sum(r["requests"] for r in payload["rows"])
            calls = sum(r["planning_calls"] for r in payload["rows"])
            planning_seconds = sum(r["planning_seconds"] for r in payload["rows"])
            aggregates.append(
                {
                    "workload": workload,
                    "variant": variant,
                    "accepted": accepted,
                    "requests": count,
                    "acceptance_pct": accepted / count * 100,
                    "planning_calls": calls,
                    "mean_planning_seconds": planning_seconds / calls if calls else 0,
                    "llm_success": sum(r.get("llm_success", 0) for r in payload["rows"]),
                    "llm_fallback": sum(r.get("llm_fallback", 0) for r in payload["rows"]),
                }
            )
        assert len(model_hashes) == 1, "Checkpoints differ between variants"
        assert all(len(hashes) == 1 for hashes in paired.values()), "Input traces differ"
    for filename, rows in [("summary.csv", aggregates), ("seeds.csv", seed_rows)]:
        with (directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["#8d99ae", "#2a6f97", "#2a9d8f", "#e9a03b"]
    for axis, (prefix, workload) in zip(axes, WORKLOADS.items(), strict=True):
        values = [
            next(
                r["acceptance_pct"]
                for r in aggregates
                if r["workload"] == workload and r["variant"] == variant
            )
            for variant in LABELS
        ]
        axis.bar(range(4), values, color=colors, width=0.66)
        for i, variant in enumerate(LABELS):
            per_seed = [r["acceptance_pct"] for r in datasets[prefix, variant]["rows"]]
            axis.scatter(
                np.linspace(i - 0.13, i + 0.13, 3), per_seed, color="#18232f", s=19, zorder=3
            )
        for i, (variant, value) in enumerate(zip(LABELS, values, strict=True)):
            highest = max(r["acceptance_pct"] for r in datasets[prefix, variant]["rows"])
            axis.text(i, max(value, highest) + 2, f"{value:.1f}%", ha="center", fontweight="bold")
        axis.set_xticks(range(4), ["Anterior", "Melhorias", "+ Heurística", "+ Qwen"])
        axis.set_title(workload, loc="left", fontweight="bold")
        axis.set_ylim(0, 62)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Requisições de jogador aceitas (%)")
    figure.suptitle(
        "Aceitação: comparação com tráfego idêntico",
        x=0.07,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.07,
        0.015,
        "LuxembourgV2 · 3 sementes · 360 requisições por variante/carga · "
        "Pontos: resultados por semente",
        fontsize=9,
        color="#45505c",
    )
    figure.tight_layout(rect=[0, 0.05, 1, 0.94])
    figure.savefig(directory / "acceptance.png", dpi=180)
    plt.close(figure)

    table = ["| Carga | Variante | Aceitas / total | Aceitação |", "|---|---|---:|---:|"]
    for row in aggregates:
        table.append(
            f"| {row['workload']} | {LABELS[row['variant']]} | "
            f"{row['accepted']} / {row['requests']} | {row['acceptance_pct']:.2f}% |"
        )
    lines = [
        "# Resultado da comparação de aceitação",
        "",
        "Execução em 13/10/2026. Cada variante recebeu exatamente as mesmas sessões, "
        "topologia e características de usuários dentro de cada semente.",
        "",
        *table,
        "",
        "## Método",
        "",
        "- Topologia LuxembourgV2; proporção eco/eficiência 0,7; sementes 42, 43 e 44.",
        "- 30 sessões de quatro jogadores por semente. Cada jogador exige duas SFCs "
        "(cache e unique), aceitas em conjunto.",
        "- 120 requisições de jogador (240 SFCs) por execução. "
        "24 execuções: 2.880 requisições de jogador e 5.760 SFCs avaliadas.",
        "- Chegadas Poisson com média de 3 ou 10 segundos; duração Poisson média 120 segundos.",
        "- Tempo virtual para chegadas, expiração e planejamento; o tempo da LLM "
        "não muda a sequência de tráfego.",
        "- Sem mobilidade SUMO, falhas, backups ou compartilhamento. "
        "Estas opções foram desativadas igualmente para isolar a alocação.",
        "- Checkpoint MAPPO existente reutilizado sem novo treinamento. "
        "Dimensões compatíveis em todas as execuções; hash idêntico.",
        "- Linha de base extraída do commit 6ba4abf49324730cd402f5e403cf60f746bdc14f. "
        "As demais variantes usam o código atual.",
        "- Mesmos módulos de geração, admissão, implantação física e expiração do projeto. "
        "Estado especulativo isolado em cópias para todas as variantes; "
        "somente implantação concluída e aprovada na auditoria de CPU, cache e banda "
        "conta como aceitação.",
        "- Hashes de tráfego e topologia comparados automaticamente por semente. "
        "Resultados brutos incluem cada requisição e resposta cognitiva.",
        "",
        "## Consultas Qwen e custo",
        "",
    ]
    for row in aggregates:
        if row["variant"] == "qwen":
            lines.append(
                f"- {row['workload']}: {row['llm_success']} respostas válidas, "
                f"{row['llm_fallback']} fallbacks; média de "
                f"{row['mean_planning_seconds']:.2f} s por planejamento."
            )
    rows_by_key = {(r["workload"], r["variant"]): r for r in aggregates}
    lines.extend(["", "## Interpretação", ""])
    for workload in WORKLOADS.values():
        old = rows_by_key[workload, "before"]["acceptance_pct"]
        improved = rows_by_key[workload, "improved"]["acceptance_pct"]
        qwen = rows_by_key[workload, "qwen"]["acceptance_pct"]
        lines.append(
            f"- {workload}: as correções mudaram a aceitação de {old:.2f}% para "
            f"{improved:.2f}% ({improved - old:+.2f} pontos percentuais). "
            f"Com Qwen: {qwen:.2f}% ({qwen - improved:+.2f} pontos em relação "
            "às correções sem planejador)."
        )
    lines.extend(
        [
            "",
            "## Limites da conclusão",
            "",
            "Os valores são resultados descritivos de três sementes, não uma garantia de "
            "ganho em outras topologias ou cargas. O tempo virtual não mede o impacto de "
            "consultas síncronas do Qwen na fila do simulador em tempo real. Tempos de "
            "execução são observacionais, sem isolamento de processos concorrentes. "
            "Respostas da LLM usam temperatura 0,1 e podem variar entre repetições.",
            "",
            "Na rajada, as variantes novas ocupam 4.800 dos 5.000 Mb/s do enlace 0–34. "
            "Cada jogador precisa de 300 Mb/s de entrada para as duas cadeias; "
            "os 200 Mb/s restantes não comportam outra requisição. Este gargalo comum "
            "impede aumento adicional de aceitação apenas mudando a política.",
            "",
            "Reprodução: use scripts/benchmark_acceptance.py com --variant before, "
            "improved, heuristic ou qwen; --seeds 42 43 44; --sessions 30; --players 4; "
            "--arrival-mean 3 ou 10; --planner-interval 30 e o mesmo --model-path. "
            "A variante before requer --source-root apontando para a cópia do commit anterior.",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregates, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
