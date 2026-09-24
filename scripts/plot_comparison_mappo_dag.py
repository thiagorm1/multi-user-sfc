import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuração de estilo elegante para publicação
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

# Configuração dos Modelos DAG (Benchmark Isomórfico: 15 sessões, 4 players)
DAG_MODELS = {
    "mappodag_s_15_p_4_a_1.0_c_0": {
        "label": "MAPPO + DAG (O Nosso)",
        "color": "#1E88E5",
        "marker": "o",
        "linestyle": "-",
        "zorder": 10,
    },
    "greedybdag_s_15_p_4_a_1.0_c_0": {
        "label": "Greedy Boosted (DAG)",
        "color": "#F4A261",
        "marker": "s",
        "linestyle": "--",
        "zorder": 5,
    },
    "musficodag_s_15_p_4_a_1.0_c_0": {
        "label": "MuSFiCO (DP DAG)",
        "color": "#9C27B0",
        "marker": "^",
        "linestyle": "-.",
        "zorder": 6,
    },
    "daggreedy_s_15_p_4_a_1.0_c_0": {
        "label": "DAG Greedy (Heurística)",
        "color": "#E63946",
        "marker": "x",
        "linestyle": ":",
        "zorder": 4,
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


def plot_comparativo_dag():
    """Gera painel comparativo 2x2 com os modelos sob requisições em DAG."""
    dados_dag = {}
    for pasta, info in DAG_MODELS.items():
        df = carregar_dados(pasta)
        if df is not None:
            dados_dag[pasta] = {"df": df, **info}

    if not dados_dag:
        print("Nenhum dado DAG encontrado.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Comparativo de Desempenho: MAPPO + DAG vs Algoritmos de Referência em DAG",
        fontweight="bold",
        y=0.98,
    )

    # 1. Taxa de Aceitação (%)
    ax1 = axes[0, 0]
    for _, info in dados_dag.items():
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
    ax1.set_title("Taxa de Aceitação de Requisições SFC (%)", fontweight="bold")
    ax1.set_xlabel("Número de SFCs Submetidas")
    ax1.set_ylabel("Taxa de Aceitação (%)")
    ax1.set_ylim(-5, 105)
    ax1.legend(loc="lower left", frameon=True)

    # 2. Latência Média E2E Acumulada (ms)
    ax2 = axes[0, 1]
    for _, info in dados_dag.items():
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
    ax2.axhline(20.0, color="gray", linestyle="--", alpha=0.7, label="SLA Máx. (20 ms)")
    ax2.set_title("Latência Média Acumulada E2E (ms)", fontweight="bold")
    ax2.set_xlabel("Número de SFCs Submetidas")
    ax2.set_ylabel("Latência (ms)")
    ax2.legend(loc="upper left", frameon=True)

    # 3. Utilização de Recursos de CPU da Rede (%)
    ax3 = axes[1, 0]
    for _, info in dados_dag.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        col_cpu = (
            "network_cpu_utilization"
            if "network_cpu_utilization" in df.columns
            else "cpu_utilization"
        )
        cpu = pd.to_numeric(df[col_cpu], errors="coerce")
        ax3.plot(
            x,
            cpu,
            label=info["label"],
            color=info["color"],
            marker=info["marker"],
            linestyle=info["linestyle"],
            markevery=max(1, len(df) // 10),
            zorder=info["zorder"],
        )
    ax3.set_title("Utilização de CPU na Rede (%)", fontweight="bold")
    ax3.set_xlabel("Número de SFCs Submetidas")
    ax3.set_ylabel("Utilização de CPU (%)")
    ax3.legend(loc="upper left", frameon=True)

    # 4. Tempo Médio de Decisão (ms)
    ax4 = axes[1, 1]
    labels = []
    tempos = []
    cores = []
    for _, info in dados_dag.items():
        df = info["df"]
        if "decision_time_ms" in df.columns:
            dec = pd.to_numeric(df["decision_time_ms"], errors="coerce").dropna()
            dec_mean = (
                dec.iloc[1:].mean() if len(dec) > 1 and dec.iloc[0] > 100 else dec.mean()
            )
            labels.append(info["label"].split(" (")[0])
            tempos.append(dec_mean)
            cores.append(info["color"])

    bars = ax4.bar(labels, tempos, color=cores, width=0.5, edgecolor="black", alpha=0.88)
    ax4.axhline(15.0, color="red", linestyle="--", alpha=0.7, label="Target SLA (<15 ms)")
    ax4.set_title("Tempo Médio de Inferência / Decisão (ms)", fontweight="bold")
    ax4.set_ylabel("Tempo de Decisão (ms)")
    ax4.set_ylim(0, max(tempos) * 1.25 if tempos else 35)
    ax4.legend(loc="upper left", frameon=True)

    for bar in bars:
        h = bar.get_height()
        ax4.annotate(
            f"{h:.2f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = Path("results/comparativo_mappo_dag_vs_outros.png")
    plt.savefig(out_path, dpi=300)
    print(f"[SUCESSO] Gráfico salvo em: {out_path.resolve()}")


def plot_comparativo_multidimensional():
    """Gera painel estendido 2x3 explorando Balanceamento de Carga (Jain's) e Energia."""
    dados_dag = {}
    for pasta, info in DAG_MODELS.items():
        df = carregar_dados(pasta)
        if df is not None:
            dados_dag[pasta] = {"df": df, **info}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Avaliação Multicritério: MAPPO + DAG vs Modelos de Orquestração em Grafos DAG",
        fontweight="bold",
        y=0.98,
    )

    # 1. Taxa de Aceitação (%)
    ax1 = axes[0, 0]
    for _, info in dados_dag.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        acc = pd.to_numeric(df["acceptance_rate"], errors="coerce")
        ax1.plot(x, acc, label=info["label"], color=info["color"], lw=2.2)
    ax1.set_title("Taxa de Aceitação (%)", fontweight="bold")
    ax1.set_xlabel("SFCs Submetidas")
    ax1.set_ylabel("Acceptance Rate (%)")
    ax1.legend(loc="lower left", fontsize=9)

    # 2. Latência Ponta a Ponta
    ax2 = axes[0, 1]
    for _, info in dados_dag.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        lat = pd.to_numeric(df["latency"], errors="coerce").expanding().mean()
        ax2.plot(x, lat, label=info["label"], color=info["color"], lw=2.2)
    ax2.axhline(20.0, color="gray", linestyle="--", alpha=0.7, label="SLA (20ms)")
    ax2.set_title("Latência Média Acumulada (ms)", fontweight="bold")
    ax2.set_xlabel("SFCs Submetidas")
    ax2.set_ylabel("Latência (ms)")
    ax2.legend(loc="upper left", fontsize=9)

    # 3. Tempo de Decisão (ms)
    ax3 = axes[0, 2]
    labels, tempos, cores = [], [], []
    for _, info in dados_dag.items():
        df = info["df"]
        dec = pd.to_numeric(df["decision_time_ms"], errors="coerce").dropna()
        dec_mean = dec.iloc[1:].mean() if len(dec) > 1 and dec.iloc[0] > 100 else dec.mean()
        labels.append(info["label"].split(" (")[0])
        tempos.append(dec_mean)
        cores.append(info["color"])
    bars = ax3.bar(labels, tempos, color=cores, width=0.5, edgecolor="black", alpha=0.88)
    ax3.axhline(15.0, color="red", linestyle="--", alpha=0.7, label="SLA (<15 ms)")
    ax3.set_title("Tempo Médio de Decisão (ms)", fontweight="bold")
    ax3.set_ylabel("Tempo (ms)")
    ax3.legend(loc="upper left", fontsize=9)
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(
            f"{h:.1f}ms",
            (bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    # 4. CPU Utilizada na Rede (%)
    ax4 = axes[1, 0]
    for _, info in dados_dag.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        col_cpu = (
            "network_cpu_utilization"
            if "network_cpu_utilization" in df.columns
            else "cpu_utilization"
        )
        cpu = pd.to_numeric(df[col_cpu], errors="coerce")
        ax4.plot(x, cpu, label=info["label"], color=info["color"], lw=2.2)
    ax4.set_title("Utilização de CPU na Rede (%)", fontweight="bold")
    ax4.set_xlabel("SFCs Submetidas")
    ax4.set_ylabel("CPU (%)")
    ax4.legend(loc="upper left", fontsize=9)

    # 5. Jain's Fairness Index (Balanceamento de Carga)
    ax5 = axes[1, 1]
    for _, info in dados_dag.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        if "jain_cpu" in df.columns:
            jain = pd.to_numeric(df["jain_cpu"], errors="coerce")
            ax5.plot(x, jain, label=info["label"], color=info["color"], lw=2.2)
    ax5.set_title("Equidade no Uso de CPU (Jain's Fairness Index)", fontweight="bold")
    ax5.set_xlabel("SFCs Submetidas")
    ax5.set_ylabel("Jain's Index (0 a 1)")
    ax5.legend(loc="lower left", fontsize=9)

    # 6. Consumo Energético Total (Joules / Watts)
    ax6 = axes[1, 2]
    for _, info in dados_dag.items():
        df = info["df"]
        x = np.arange(1, len(df) + 1)
        if "total_energy_consumption" in df.columns:
            energy = pd.to_numeric(df["total_energy_consumption"], errors="coerce")
            ax6.plot(x, energy, label=info["label"], color=info["color"], lw=2.2)
    ax6.set_title("Consumo Energético Total da Rede", fontweight="bold")
    ax6.set_xlabel("SFCs Submetidas")
    ax6.set_ylabel("Energia Acumulada")
    ax6.legend(loc="lower right", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_ext_path = Path("results/comparativo_mappo_dag_multidim.png")
    plt.savefig(out_ext_path, dpi=300)
    print(f"[SUCESSO] Gráfico multidimensional salvo em: {out_ext_path.resolve()}")


if __name__ == "__main__":
    plot_comparativo_dag()
    plot_comparativo_multidimensional()
