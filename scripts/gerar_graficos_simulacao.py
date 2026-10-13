import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Estilo para publicação
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 14,
    "grid.alpha": 0.4,
    "lines.linewidth": 2.0,
})

BASE_DIR = Path("results/results_flows")
OUTPUT_DIR = Path("results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapeamento de algoritmos com 50 sessões e 6 players
CONFIGS = {
    "replic_s_50_p_6_a_0.99_c_3": {
        "label": "REPLIC (DRL - PPO)",
        "color": "#2A9D8F",
        "marker": "o",
        "linestyle": "-",
        "zorder": 10
    },
    "musfico_s_50_p_6_a_0.99_c_3": {
        "label": "MuSFiCO (Dynamic Prog.)",
        "color": "#E76F51",
        "marker": "^",
        "linestyle": "-.",
        "zorder": 8
    },
    "greedyb_s_50_p_6_a_0.99_c_3": {
        "label": "Greedy Boosted",
        "color": "#F4A261",
        "marker": "s",
        "linestyle": "--",
        "zorder": 6
    },
    "vegeta_s_50_p_6_a_0.99_c_3": {
        "label": "Vegeta",
        "color": "#9C27B0",
        "marker": "d",
        "linestyle": ":",
        "zorder": 5
    },
    "msf_s_50_p_6_a_0.99_c_3": {
        "label": "MSF",
        "color": "#457B9D",
        "marker": "v",
        "linestyle": "--",
        "zorder": 4
    },
}

def carregar_dados():
    dados = {}
    for pasta, info in CONFIGS.items():
        caminho = BASE_DIR / pasta
        if not caminho.exists():
            continue
        csvs = sorted(glob.glob(str(caminho / "*.csv")))
        if not csvs:
            continue
        # Carrega o CSV mais recente dessa configuração
        ultimo_csv = csvs[-1]
        df = pd.read_csv(ultimo_csv)
        dados[pasta] = {
            "df": df,
            "arquivo": Path(ultimo_csv).name,
            **info
        }
        print(f"Carregado: {info['label']} ({len(df)} registros) de {Path(ultimo_csv).name}")
    return dados

def plotar_aceitacao_e_latencia(dados):
    """Gera figura focada em Aceitação e Latência"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Desempenho da Simulação: Taxa de Aceitação e Latência", fontweight="bold", fontsize=14, y=1.02)

    # 1. Taxa de Aceitação (%)
    ax1 = axes[0]
    for pasta, info in dados.items():
        df = info["df"]
        if "acceptance_rate" in df.columns:
            x = np.arange(1, len(df) + 1)
            y = pd.to_numeric(df["acceptance_rate"], errors="coerce")
            ax1.plot(x, y, label=info["label"], color=info["color"], marker=info["marker"],
                     linestyle=info["linestyle"], markersize=3.5, markevery=max(1, len(x)//20),
                     zorder=info["zorder"])
    ax1.set_title("Taxa de Aceitação de Requisições (%)", fontweight="bold")
    ax1.set_xlabel("Número de SFCs Submetidas")
    ax1.set_ylabel("Taxa de Aceitação (%)")
    ax1.set_ylim(-2, 103)
    ax1.legend(loc="lower left", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Latência Média Acumulada (ms)
    ax2 = axes[1]
    for pasta, info in dados.items():
        df = info["df"]
        if "latency" in df.columns:
            lat = pd.to_numeric(df["latency"], errors="coerce")
            lat_valid = lat[lat > 0]
            lat_cum = lat_valid.expanding().mean()
            x = np.arange(1, len(lat_cum) + 1)
            ax2.plot(x, lat_cum, label=info["label"], color=info["color"], marker=info["marker"],
                     linestyle=info["linestyle"], markersize=3.5, markevery=max(1, len(x)//20),
                     zorder=info["zorder"])
    ax2.set_title("Latência Média Acumulada (ms)", fontweight="bold")
    ax2.set_xlabel("Número de SFCs Submetidas")
    ax2.set_ylabel("Latência Ponta-a-Ponta (ms)")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out1 = OUTPUT_DIR / "grafico_aceitacao_latencia.png"
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out1.resolve()}")

def plotar_painel_completo(dados):
    """Gera painel completo 3x2 com todas as métricas-chave"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Resultados Completos da Simulação de Redes SFC (MUAR-SFC)", fontweight="bold", fontsize=15, y=0.99)

    # [0, 0] Taxa de Aceitação
    ax = axes[0, 0]
    for pasta, info in dados.items():
        df = info["df"]
        if "acceptance_rate" in df.columns:
            x = np.arange(1, len(df) + 1)
            ax.plot(x, pd.to_numeric(df["acceptance_rate"], errors="coerce"),
                    label=info["label"], color=info["color"], marker=info["marker"],
                    markevery=max(1, len(x)//20), linestyle=info["linestyle"], zorder=info["zorder"])
    ax.set_title("1. Taxa de Aceitação de Requisições (%)", fontweight="bold")
    ax.set_xlabel("SFCs Submetidas")
    ax.set_ylabel("Aceitação (%)")
    ax.set_ylim(-2, 103)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # [0, 1] Latência Acumulada
    ax = axes[0, 1]
    for pasta, info in dados.items():
        df = info["df"]
        if "latency" in df.columns:
            lat = pd.to_numeric(df["latency"], errors="coerce")
            lat_valid = lat[lat > 0]
            lat_cum = lat_valid.expanding().mean()
            x = np.arange(1, len(lat_cum) + 1)
            ax.plot(x, lat_cum, label=info["label"], color=info["color"], marker=info["marker"],
                    markevery=max(1, len(x)//20), linestyle=info["linestyle"], zorder=info["zorder"])
    ax.set_title("2. Latência Média Ponta a Ponta (ms)", fontweight="bold")
    ax.set_xlabel("SFCs Submetidas")
    ax.set_ylabel("Latência (ms)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # [1, 0] Utilização de CPU da Rede
    ax = axes[1, 0]
    for pasta, info in dados.items():
        df = info["df"]
        col = "network_cpu_utilization" if "network_cpu_utilization" in df.columns else "cpu_utilization"
        if col in df.columns:
            x = np.arange(1, len(df) + 1)
            ax.plot(x, pd.to_numeric(df[col], errors="coerce"),
                    label=info["label"], color=info["color"], marker=info["marker"],
                    markevery=max(1, len(x)//20), linestyle=info["linestyle"], zorder=info["zorder"])
    ax.set_title("3. Utilização de CPU dos Servidores de Borda (%)", fontweight="bold")
    ax.set_xlabel("SFCs Submetidas")
    ax.set_ylabel("Uso de CPU (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # [1, 1] Utilização de Largura de Banda
    ax = axes[1, 1]
    for pasta, info in dados.items():
        df = info["df"]
        if "bandwidth_utilization" in df.columns:
            x = np.arange(1, len(df) + 1)
            ax.plot(x, pd.to_numeric(df["bandwidth_utilization"], errors="coerce"),
                    label=info["label"], color=info["color"], marker=info["marker"],
                    markevery=max(1, len(x)//20), linestyle=info["linestyle"], zorder=info["zorder"])
    ax.set_title("4. Utilização de Largura de Banda dos Links (%)", fontweight="bold")
    ax.set_xlabel("SFCs Submetidas")
    ax.set_ylabel("Uso de Banda (%)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    # [2, 0] Tempo Médio de Decisão (Barplot)
    ax = axes[2, 0]
    nomes, tempos, cores = [], [], []
    for pasta, info in dados.items():
        df = info["df"]
        if "decision_time_ms" in df.columns:
            dec = pd.to_numeric(df["decision_time_ms"], errors="coerce").dropna()
            if len(dec) > 0:
                nomes.append(info["label"].split(" (")[0])
                tempos.append(dec.mean())
                cores.append(info["color"])
    if tempos:
        bars = ax.bar(nomes, tempos, color=cores, width=0.5, edgecolor="black", alpha=0.85)
        ax.set_title("5. Tempo Médio de Decisão por SFC (ms)", fontweight="bold")
        ax.set_ylabel("Tempo de Execução (ms)")
        ax.set_yscale("log")
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}ms", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # [2, 1] Consumo Energético Total
    ax = axes[2, 1]
    for pasta, info in dados.items():
        df = info["df"]
        if "total_energy_consumption" in df.columns:
            x = np.arange(1, len(df) + 1)
            ax.plot(x, pd.to_numeric(df["total_energy_consumption"], errors="coerce"),
                    label=info["label"], color=info["color"], marker=info["marker"],
                    markevery=max(1, len(x)//20), linestyle=info["linestyle"], zorder=info["zorder"])
    ax.set_title("6. Consumo Energético Total da Infraestrutura (Watts)", fontweight="bold")
    ax.set_xlabel("SFCs Submetidas")
    ax.set_ylabel("Consumo Energético (W)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out2 = OUTPUT_DIR / "grafico_desempenho_completo.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvo: {out2.resolve()}")

def main():
    dados = carregar_dados()
    if not dados:
        print("Erro: Nenhum dado encontrado em", BASE_DIR)
        return
    plotar_aceitacao_e_latencia(dados)
    plotar_painel_completo(dados)
    print("\nTodos os gráficos foram gerados com sucesso!")

if __name__ == "__main__":
    main()
