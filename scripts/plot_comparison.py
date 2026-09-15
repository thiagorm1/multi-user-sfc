import glob
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Configurações de estilo
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
})

BASE_DIR = Path("results/results_flows")

ALGORITMOS = {
    "daggreedy_s_5_p_2_a_0.99_c_0": ("DAG Greedy", "#264653"),
    "musfico_s_5_p_2_a_0.99_c_0": ("MuSFiCO (DP)", "#E76F51"),
    "greedyb_s_5_p_2_a_0.99_c_0": ("Greedy Boosted", "#F4A261"),
    "replic_s_5_p_2_a_0.99_c_0": ("REPLIC (DRL)", "#2A9D8F"),
}

dados = {}

for pasta, (nome, cor) in ALGORITMOS.items():
    caminho_pasta = BASE_DIR / pasta
    if not caminho_pasta.exists():
        print(f"Aviso: Pasta não encontrada: {caminho_pasta}")
        continue
    
    arquivos_csv = sorted(glob.glob(str(caminho_pasta / "*.csv")))
    if not arquivos_csv:
        print(f"Aviso: Nenhum CSV encontrado em {caminho_pasta}")
        continue
    
    # Pega o arquivo mais recente
    ultimo_csv = arquivos_csv[-1]
    df = pd.read_csv(ultimo_csv)
    dados[nome] = {"df": df, "cor": cor}
    print(f"Carregado {nome}: {len(df)} registros de {Path(ultimo_csv).name}")

if not dados:
    print("Nenhum dado encontrado para plotar.")
    exit(1)

# Cria figura com 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Taxa de Aceitação (Acceptance Rate)
ax1 = axes[0, 0]
for nome, info in dados.items():
    df = info["df"]
    if "acceptance_rate" in df.columns:
        ax1.plot(df.index + 1, df["acceptance_rate"], label=nome, color=info["cor"], lw=2.5, marker="o", markersize=4)
ax1.set_title("Taxa de Aceitação de Requisições (%)")
ax1.set_xlabel("Número de SFCs Submetidas")
ax1.set_ylabel("Acceptance Rate (%)")
ax1.set_ylim(-5, 105)
ax1.legend()

# 2. Latência Média Ponta a Ponta
ax2 = axes[0, 1]
for nome, info in dados.items():
    df = info["df"]
    if "latency" in df.columns:
        lat = pd.to_numeric(df["latency"], errors="coerce")
        lat_cum = lat.expanding().mean()
        ax2.plot(df.index + 1, lat_cum, label=nome, color=info["cor"], lw=2.5, marker="s", markersize=4)
ax2.set_title("Latência Média Acumulada (ms)")
ax2.set_xlabel("Número de SFCs Submetidas")
ax2.set_ylabel("Latência (ms)")
ax2.legend()

# 3. Utilização de Recursos de Rede (CPU)
ax3 = axes[1, 0]
for nome, info in dados.items():
    df = info["df"]
    col_cpu = "network_cpu_utilization" if "network_cpu_utilization" in df.columns else "cpu_utilization"
    if col_cpu in df.columns:
        cpu = pd.to_numeric(df[col_cpu], errors="coerce")
        ax3.plot(df.index + 1, cpu, label=nome, color=info["cor"], lw=2.5, marker="^", markersize=4)
ax3.set_title("Utilização de CPU na Rede (%)")
ax3.set_xlabel("Número de SFCs Submetidas")
ax3.set_ylabel("Uso de CPU (%)")
ax3.legend()

# 4. Tempo de Decisão do Algoritmo (Decision Time)
ax4 = axes[1, 1]
tempos_medios = []
nomes_alg = []
cores_bar = []
for nome, info in dados.items():
    df = info["df"]
    if "decision_time_ms" in df.columns:
        dec_time = pd.to_numeric(df["decision_time_ms"], errors="coerce").dropna()
        tempos_medios.append(dec_time.mean())
        nomes_alg.append(nome)
        cores_bar.append(info["cor"])

if tempos_medios:
    bars = ax4.bar(nomes_alg, tempos_medios, color=cores_bar, width=0.55, edgecolor="black", alpha=0.85)
    ax4.set_title("Tempo Médio de Decisão por SFC (ms)")
    ax4.set_ylabel("Tempo de Decisão (ms)")
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f"{height:.2f} ms",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
output_fig = Path("results/comparativo_algoritmos.png")
plt.savefig(output_fig, dpi=300)
output_fig_dag = Path("results/comparativo_algoritmos_dag.png")
plt.savefig(output_fig_dag, dpi=300)
print(f"\n Gráfico salvo com sucesso em: {output_fig.resolve()} e {output_fig_dag.resolve()}")
