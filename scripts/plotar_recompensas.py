import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ==============================================================================
# CONFIGURAÇÕES DE ESTILO GLOBAIS (Padrão Artigo Científico)
# ==============================================================================
# Define a fonte para Serif (estilo Times New Roman), idêntico à imagem da sua legenda
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "serif"],
    }
)

# Dicionários baseados nas suas constantes
NOME_ALGORITMOS_MAP = {"REPLIC": "RELIC", "DRL": "HDRLB"}

CORES_ALGORITMOS = {
    "RELIC": "#90BE6D",  # Verde Claro/Mudo
    "HDRLB": "#4D774E",  # Verde Escuro/Mudo
}


# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================
def suavizacao_ema(valores, peso=0.95):
    suavizado = []
    ultimo = valores[0]
    for valor in valores:
        valor_suavizado = ultimo * peso + (1 - peso) * valor
        suavizado.append(valor_suavizado)
        ultimo = valor_suavizado
    return np.array(suavizado)


def formatar_eixo_x_em_k(x, pos):
    if x >= 1000:
        return f"{int(x / 1000)}k"
    return str(int(x))


# ==============================================================================
# FUNÇÃO PRINCIPAL DE PLOTAGEM
# ==============================================================================
def plotar_grafico_final(projetos, base_log_dir="logs/"):
    dados_carregados = {}
    min_global = float("inf")
    max_global = float("-inf")

    # 1. Carregamento e Extração
    for projeto in projetos:
        evaluations_path = os.path.join(base_log_dir, projeto, "evaluations.npz")

        if not os.path.exists(evaluations_path):
            print(f"⚠️ Aviso: Arquivo não encontrado para '{projeto}' em: {evaluations_path}")
            continue

        dados = np.load(evaluations_path)
        mean_rewards = np.mean(dados["results"], axis=1)
        dados_carregados[projeto] = {"timesteps": dados["timesteps"], "mean_rewards": mean_rewards}

        min_global = min(min_global, np.min(mean_rewards))
        max_global = max(max_global, np.max(mean_rewards))

    if not dados_carregados:
        print("Nenhum dado foi encontrado para plotar.")
        return

    # 2. Configuração da Figura
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Processamento e Plotagem
    legend_handles = []
    for projeto, dados in dados_carregados.items():
        timesteps = dados["timesteps"]
        recompensas_cruas = dados["mean_rewards"]

        # Normalização global e Suavização
        recompensas_norm = (recompensas_cruas - min_global) / (max_global - min_global + 1e-8)
        recompensas_suaves = suavizacao_ema(recompensas_norm, peso=0.95)

        # Mapeamento de Nome e Cor (usa o nome da pasta se não achar no dict)
        nome_final = NOME_ALGORITMOS_MAP.get(projeto, projeto)
        cor_final = CORES_ALGORITMOS.get(nome_final, "#000000")

        # Plota a linha (Ambas sólidas para bater com sua referência da legenda)
        (linha,) = ax.plot(
            timesteps,
            recompensas_suaves,
            label=nome_final,
            color=cor_final,
            linestyle="-",
            linewidth=4.5,
        )
        legend_handles.append(linha)

    # 4. Estilo Visual dos Eixos
    ax.set_ylim(0.0, 1.0)

    ax.set_ylabel("MRW", fontsize=26, fontweight="bold")
    ax.set_xlabel("Training steps", fontsize=26, fontweight="bold")

    # Formatação dos Ticks
    ax.xaxis.set_major_formatter(FuncFormatter(formatar_eixo_x_em_k))
    ax.tick_params(axis="both", which="major", labelsize=20)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # Grid tracejado
    ax.grid(axis="both", linestyle="--", alpha=0.5, linewidth=1.0)

    # ==========================================================================
    # 5. LEGENDA IDÊNTICA À REFERÊNCIA (Borda Forte e Serif)
    # ==========================================================================
    leg = ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        handlelength=2.0,
        fontsize=24,
        facecolor="white",
        framealpha=1.0,  # Opaco 1.0 para o grid não vazar por trás da legenda
    )

    # Textos da legenda em negrito
    for text in leg.get_texts():
        text.set_fontweight("bold")

    # Borda preta grossa
    leg.get_frame().set_linewidth(2.5)
    leg.get_frame().set_edgecolor("black")

    # 6. Salvar Imagem em PDF
    nome_arquivo = "grafico_recompensas_artigo_final.pdf"
    plt.savefig(nome_arquivo, format="pdf", bbox_inches="tight", pad_inches=0.05)
    print(f"\n✅ Gráfico salvo com sucesso como '{nome_arquivo}'")

    plt.show()


if __name__ == "__main__":
    # Mantemos os nomes reais das pastas dos logs aqui
    modelos_para_comparar = ["REPLIC", "DRL"]
    plotar_grafico_final(modelos_para_comparar)
