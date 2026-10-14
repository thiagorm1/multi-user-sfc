"""Validate the high-acceptance experiment and export a report and chart."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(payload):
    rows = payload["rows"]
    assert len(rows) == 3
    assert all(r["pending"] == 0 and r["checkpoint_dimensions_match"] for r in rows)
    requests = sum(r["requests"] for r in rows)
    accepted = sum(r["accepted"] for r in rows)
    accepted_sfcs = sum(r["accepted_sfcs"] for r in rows)
    latency = sum(r["accumulated_latency_ms"] for r in rows)
    return {
        "requests": requests,
        "accepted": accepted,
        "acceptance_pct": 100 * accepted / requests,
        "accepted_immediate": sum(r["accepted_immediate"] for r in rows),
        "accepted_after_wait": sum(r["accepted_after_wait"] for r in rows),
        "rejected": sum(r["rejected"] for r in rows),
        "accumulated_latency_ms": round(latency, 2),
        "mean_latency_ms": round(latency / accepted_sfcs, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    datasets = {
        name: json.loads((root / f"{name}.json").read_text())
        for name in ["control_burst", "control_spaced", "full_burst", "full_spaced"]
    }
    summary = {name: summarize(payload) for name, payload in datasets.items()}
    for workload in ["burst", "spaced"]:
        baseline, full = datasets[f"control_{workload}"], datasets[f"full_{workload}"]
        assert baseline["settings"]["latency"] == full["settings"]["latency"] == 50
        for old, new in zip(baseline["rows"], full["rows"], strict=True):
            assert old["seed"] == new["seed"]
            assert old["arrival_sha256"] == new["arrival_sha256"]
        assert summary[f"full_{workload}"]["accepted_immediate"] == 360
    model = Path("rl_saved_models/MAPPO_acceptance_model.pt")
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    assert all(
        datasets[name]["model_sha256"] == model_hash for name in ["full_burst", "full_spaced"]
    )
    training = json.loads(model.with_suffix(".training.json").read_text())
    assert training["checkpoint_sha256"] == model_hash
    with (root / "summary.csv").open("w", newline="") as stream:
        rows = [{"case": name, **values} for name, values in summary.items()]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = ["Rajada: chegadas médias de 3 s", "Chegadas médias de 10 s"]
    controls = [summary[f"control_{w}"]["acceptance_pct"] for w in ["burst", "spaced"]]
    fulls = [summary[f"full_{w}"]["acceptance_pct"] for w in ["burst", "spaced"]]
    bars = axis.bar(
        [-0.18, 0.82], controls, width=0.34, color="#718096", label="Controle: capacidade original"
    )
    axis.bar_label(bars, labels=[f"{v:.1f}%" for v in controls], padding=6)
    bars = axis.bar(
        [0.18, 1.18],
        fulls,
        width=0.34,
        color="#138c70",
        label="Perfil completo: recursos ampliados",
    )
    axis.bar_label(bars, labels=["100% · 360/360", "100% · 360/360"], padding=6, fontweight="bold")
    axis.set_xticks([0, 1], labels)
    axis.set_ylim(0, 115)
    axis.set_yticks(range(0, 101, 20))
    axis.set_ylabel("Requisições de jogador aceitas (%)")
    axis.set_title(
        "Aceitação com SLA de rede/processamento de 50 ms",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.17)
    axis.set_axisbelow(True)
    handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    figure.text(
        0.12,
        0.02,
        "LuxembourgV2 · 3 sementes por carga · "
        "Perfil completo: todas aceitas de imediato, sem pendências",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.15, 1, 1])
    figure.savefig(root / "acceptance.png", dpi=180)
    plt.close(figure)
    lines = [
        "# Seis estratégias de aceitação: implementação e validação",
        "",
        "**Resultado final: 720/720 requisições aceitas de imediato (100%).** "
        "Sementes 42, 43 e 44; 30 sessões × 4 jogadores em cada execução. "
        "Cada jogador tem um DAG completo no perfil final.",
        "",
        "| Carga | Controle atual | Perfil completo | Latência média do DAG | "
        "Latência acumulada dos DAGs |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload, label in [("burst", "Rajada, média 3 s"), ("spaced", "Média 10 s")]:
        old, new = summary[f"control_{workload}"], summary[f"full_{workload}"]
        lines.append(
            f"| {label} | {old['accepted']}/360 ({old['acceptance_pct']:.2f}%) | "
            f"360/360 (100%) | {new['mean_latency_ms']:.2f} ms | "
            f"{new['accumulated_latency_ms']:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "## O que foi aplicado",
            "",
            "1. Compartilhamento de instâncias com mesmo serviço, sessão e demandas "
            "compatíveis de CPU/cache. Fluxos lógicos idênticos compartilham banda; "
            "fluxos distintos e backups usam identidades separadas. Liberação mantém "
            "recursos até a última referência compartilhada.",
            "2. DAG completo: reservas de todas as dependências, inclusive bifurcações, "
            "junções e destino; latência pelo caminho crítico e sincronização máxima de 5 ms. "
            "A entrada do transcodificador foi corrigida para 240 Mb/s, com saída de 97 Mb/s.",
            "3. Dimensionamento finito e explícito: CPU/cache ×3,6, enlaces internos ×8, "
            "canal sem fio ×4 e entrada total de 40.000 Mb/s. "
            "Não foram usados recursos ilimitados nem retirada de verificações.",
            "4. Custos de CPU/cache residual, comunicação, reutilização, proximidade do "
            "destino e preferência dos atores MAPPO; máscaras físicas e busca limitada "
            "com alternativas preservam os limites de recursos e SLA.",
            "5. Fila com prazo máximo de 30 s e novas tentativas a cada 1 s. "
            "Aceitação imediata, após espera, rejeição, expiração e pendências são separadas. "
            "Implantação que termina depois do prazo é desfeita. "
            "A fila foi validada também com saturação real e liberação de recursos.",
            "6. Treinamento MAPPO em 200 episódios na CPU, com cargas de fundo variadas, "
            "DAG/linear e SLAs de 10, 25, 50 ou 100 ms. Checkpoint anterior preservado; "
            "novo arquivo MAPPO_acceptance_model.pt, gravação atômica e metadados de treino.",
            "",
            "## Treinamento isolado",
            "",
            f"Validação em {training['held_out_count']} ambientes reservados: "
            f"{training['held_out_before_pct']:.1f}% antes e "
            f"{training['held_out_after_pct']:.1f}% depois. O treino foi realizado, "
            "mas não demonstrou aumento isolado nessa validação. "
            "Não atribuímos os 100% somente ao aprendizado.",
            "",
            "## Critérios e limites",
            "",
            "- Controle e perfil completo usam as mesmas chegadas, durações e localização "
            "dos jogadores por semente, verificadas por hash. A infraestrutura foi "
            "intencionalmente dimensionada no perfil completo.",
            "- SLA configurado em MUAR_LATENCY=50 ms, respeitado pelo gerador. "
            "O gerador antigo usava 1.000 ms fixos; estes resultados usam um limite mais estrito.",
            "- Latência da SFC é rede/processamento do caminho crítico. Espera de admissão "
            "e tempo computacional do decisor são métricas separadas. "
            "Os testes em tempo virtual não representam garantia de tempo de resposta "
            "real no computador do usuário.",
            "- Todos os recursos foram auditados após implantação física. "
            "Não há requisições pendentes ou contabilizadas apenas por entrar na fila.",
            "- Sem falhas, SUMO ou backups nas seis rodadas principais. "
            "Os 100% se aplicam a essas cargas e sementes, com os recursos ampliados. "
            "Não garantem 100% para qualquer carga ou falha.",
            "- O perfil ativo usa planejamento heurístico local; Qwen continua disponível "
            "como opção. O ganho medido aqui não é atribuído ao Qwen.",
            "- Os números do controle diferem do relatório anterior porque a validação "
            "agora reserva também todos os enlaces de destino, corrige entradas do DAG "
            "e aplica o SLA de 50 ms.",
            "",
            "## Execução",
            "",
            "O .env local já foi configurado. Execute `uv run muar-sim`. "
            "O modelo está em `rl_saved_models/MAPPO_acceptance_model.pt`. "
            "O perfil reproduzível está em `.env.high-acceptance.example`.",
            "",
            "Treino: `.venv/bin/python scripts/train_mappo.py --episodes 200`.",
            "",
            "Benchmark: `.venv/bin/python scripts/benchmark_acceptance.py "
            "--variant heuristic --profile full --model-path "
            "rl_saved_models/MAPPO_acceptance_model.pt --output results/nova_comparacao.json "
            "--arrival-mean 3` (ou `10`).",
            "",
            "Metadados, por requisição e por semente, estão nos JSONs full_burst.json "
            "e full_spaced.json. O comando principal foi validado com uma execução curta "
            "de duas sessões que terminou com código zero e oito requisições aceitas.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
