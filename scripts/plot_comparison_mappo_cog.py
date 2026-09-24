import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuração de estilo elegante para publicação científica
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.titlesize": 15,
        "grid.alpha": 0.5,
        "lines.linewidth": 2.2,
    }
)

BASE_DIR = Path("results/results_flows")
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuração dos Modelos DAG (Benchmark: 15 sessões, 4 players)
MODELS_CONFIG = {
    "daggreedy_s_15_p_4_a_1.0_c_0": {
        "label": "DAG Greedy",
        "color": "#E63946",
        "marker": "x",
        "linestyle": ":",
        "zorder": 4,
    },
    "greedybdag_s_15_p_4_a_1.0_c_0": {
        "label": "Greedy Boosted",
        "color": "#F4A261",
        "marker": "s",
        "linestyle": "--",
        "zorder": 5,
    },
    "musficodag_s_15_p_4_a_1.0_c_0": {
        "label": "MuSFiCO (DP)",
        "color": "#9C27B0",
        "marker": "^",
        "linestyle": "-.",
        "zorder": 6,
    },
    "mappodag_s_15_p_4_a_1.0_c_0": {
        "label": "MAPPO + DAG",
        "color": "#1E88E5",
        "marker": "o",
        "linestyle": "-",
        "zorder": 8,
    },
    "mappodagcog_s_15_p_4_a_1.0_c_0": {
        "label": "MAPPO + Cog (LLM)",
        "color": "#2A9D8F",
        "marker": "D",
        "linestyle": "-",
        "zorder": 10,
    },
}


def carregar_dados(pasta_relativa: str) -> pd.DataFrame | None:
    caminho = BASE_DIR / pasta_relativa
    if not caminho.exists():
        print(f"[AVISO] Diretório não encontrado: {caminho}")
        return None
    arquivos = sorted(glob.glob(str(caminho / "*.csv")))
    if not arquivos:
        print(f"[AVISO] Nenhum CSV em: {caminho}")
        return None
    df = pd.read_csv(arquivos[-1])
    return df


def plot_painel_comparativo():
    dados = {}
    for pasta, info in MODELS_CONFIG.items():
        df = carregar_dados(pasta)
        if df is not None:
            dados[pasta] = {"df": df, **info}

    if not dados:
        print("Nenhum dado encontrado.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Avaliação 6G: Orquestração Cognitiva (Cloud Planner + MAPPO) vs Baselines DAG",
        fontweight="bold",
        y=0.98,
    )

    # 1. Taxa de Aceitação (%)
    ax1 = axes[0, 0]
    for _, info in dados.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        acc = pd.to_numeric(df["acceptance_rate"], errors="coerce")
        ax1.plot(
            x,
            acc,
            label=info["label"],
            color=info["color"],
            marker=info["marker"],
            linestyle=info["linestyle"],
            markevery=max(1, len(df) // 10),
            zorder=info["zorder"],
        )
    ax1.set_title("Taxa de Aceitação de SFCs (%)", fontweight="bold")
    ax1.set_xlabel("Número de SFCs Submetidas")
    ax1.set_ylabel("Taxa de Aceitação (%)")
    ax1.set_ylim(-5, 105)
    ax1.legend(loc="lower left", frameon=True)

    # 2. Latência Média E2E Acumulada (ms)
    ax2 = axes[0, 1]
    for _, info in dados.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        lat = pd.to_numeric(df["latency"], errors="coerce")
        lat_cum = lat.expanding().mean()
        ax2.plot(
            x,
            lat_cum,
            label=info["label"],
            color=info["color"],
            marker=info["marker"],
            linestyle=info["linestyle"],
            markevery=max(1, len(df) // 10),
            zorder=info["zorder"],
        )
    ax2.axhline(20.0, color="gray", linestyle="--", alpha=0.7, label="SLA Limite (20 ms)")
    ax2.set_title("Latência Média Acumulada E2E (ms)", fontweight="bold")
    ax2.set_xlabel("Número de SFCs Submetidas")
    ax2.set_ylabel("Latência E2E (ms)")
    ax2.legend(loc="upper left", frameon=True)

    # 3. Latência de Comunicação Acumulada (ms)
    ax3 = axes[1, 0]
    for _, info in dados.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        comm = pd.to_numeric(df["comm_latency"], errors="coerce")
        comm_cum = comm.expanding().mean()
        ax3.plot(
            x,
            comm_cum,
            label=info["label"],
            color=info["color"],
            marker=info["marker"],
            linestyle=info["linestyle"],
            markevery=max(1, len(df) // 10),
            zorder=info["zorder"],
        )
    ax3.set_title("Latência de Comunicação de Rede (ms)", fontweight="bold")
    ax3.set_xlabel("Número de SFCs Submetidas")
    ax3.set_ylabel("Latência de Comunicação (ms)")
    ax3.legend(loc="upper left", frameon=True)

    # 4. Tempo de Decisão Algorítmica (ms)
    ax4 = axes[1, 1]
    for _, info in dados.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        t_dec = pd.to_numeric(df["decision_time_ms"], errors="coerce")
        t_cum = t_dec.expanding().mean()
        ax4.plot(
            x,
            t_cum,
            label=info["label"],
            color=info["color"],
            marker=info["marker"],
            linestyle=info["linestyle"],
            markevery=max(1, len(df) // 10),
            zorder=info["zorder"],
        )
    ax4.set_title("Tempo de Decisão Médio por SFC (ms)", fontweight="bold")
    ax4.set_xlabel("Número de SFCs Submetidas")
    ax4.set_ylabel("Tempo de Decisão (ms)")
    ax4.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    out_path = PLOTS_DIR / "comparativo_mappo_cog_painel.png"
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Painel comparativo salvo em: {out_path}")
    plt.close()


def plot_barras_resumo():
    labels = []
    lat_tot = []
    lat_comp = []
    lat_comm = []
    acc_rates = []
    dec_times = []
    colors = []

    for pasta, info in MODELS_CONFIG.items():
        df = carregar_dados(pasta)
        if df is not None:
            succ = df[df["success"] == 1]
            labels.append(info["label"])
            lat_tot.append(succ["latency"].mean())
            lat_comp.append(succ["comp_latency"].mean())
            lat_comm.append(succ["comm_latency"].mean())
            acc_rates.append(df["acceptance_rate"].iloc[-1])
            dec_times.append(df["decision_time_ms"].mean())
            colors.append(info["color"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Decomposição da Latência (Comp vs Comm)
    x = np.arange(len(labels))
    width = 0.35

    ax1 = axes[0]
    bars1 = ax1.bar(x - width / 2, lat_comp, width, label="Computação", color="#457B9D")
    bars2 = ax1.bar(x + width / 2, lat_comm, width, label="Comunicação", color="#E76F51")

    ax1.set_title("Decomposição da Latência: Computação vs Comunicação", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Latência Média (ms)")
    ax1.legend()
    ax1.bar_label(bars1, fmt="%.2f", padding=3, fontsize=9)
    ax1.bar_label(bars2, fmt="%.2f", padding=3, fontsize=9)

    # Gráfico 2: Taxa de Aceitação (%)
    ax2 = axes[1]
    bars3 = ax2.bar(x, acc_rates, width=0.5, color=colors)
    ax2.set_title("Taxa de Aceitação Final (%)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Taxa de Aceitação (%)")
    ax2.set_ylim(0, max(acc_rates) * 1.25)
    ax2.bar_label(bars3, fmt="%.1f%%", padding=3, fontweight="bold")

    plt.tight_layout()
    out_path = PLOTS_DIR / "comparativo_mappo_cog_barras.png"
    plt.savefig(out_path, dpi=300)
    print(f"[OK] Gráfico de barras salvo em: {out_path}")
    plt.close()


if __name__ == "__main__":
    plot_painel_comparativo()
    plot_barras_resumo()
