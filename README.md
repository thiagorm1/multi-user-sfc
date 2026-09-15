# MUAR-SFC (Multi-User Service Function Chaining Simulator)

Um simulador de redes avançado e de alto desempenho, focado em resiliência, mobilidade e alocação adaptativa de Service Function Chains (SFC) para serviços imersivos.

O MUAR-SFC integra infraestrutura de topologia de redes (**NetworkX**), simulação de tráfego urbano realístico (**Eclipse SUMO**) e inteligência artificial de ponta (**Deep Reinforcement Learning** via Stable-Baselines3) para orquestrar serviços de rede sob condições severas de falhas e mobilidade.

---

## ✨ Diferenciais de Engenharia

Diferente de simuladores acadêmicos convencionais, o MUAR-SFC foi refatorado seguindo padrões industriais de software:

-   **Performance Atômica:** Lógica de falhas e backups otimizada com complexidade $O(1)$, substituindo varreduras lineares por acessos diretos a conjuntos e dicionários.
-   **Arquitetura Robusta:** Organizado em `src-layout` para garantir isolamento total e evitar a "ilusão de corretude" no desenvolvimento.
-   **Resiliência Baseada em EAFP:** Tratamento de erros idiomático (*Easier to Ask for Forgiveness than Permission*), eliminando verificações lentas e silenciamento de bugs críticos.
-   **Multiplataforma Nativa:** Gerenciamento de caminhos via `Pathlib`, garantindo que o simulador rode sem alterações em Linux, macOS ou Windows.
-   **Configuração Centralizada:** Baseado na metodologia *Twelve-Factor App* através do `Pydantic-Settings`.

---

## 🛠️ Tecnologias e Dependências

-   **Runtime:** Python >= 3.12 (Aproveitando melhorias de performance e tipagem).
-   **Gerenciador:** `uv` (Hiper-velocidade escrita em Rust).
-   **IA/ML:** stable-baselines3, sb3-contrib (PPO/MaskablePPO), tensorboard.
-   **Rede:** networkx, simpy.
-   **Tráfego:** Eclipse SUMO (TraCI).
-   **Qualidade:** Ruff (Linting), Pyright (Tipagem Estática).

---

## 🚀 Instalação e Setup

O projeto utiliza o `uv` para garantir que o seu ambiente seja **exatamente igual** ao dos desenvolvedores.

1. **Instale o uv**
    ```bash
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
    ```

2. **Prepare o ambiente**
    ```bash
    git clone [https://github.com/seu-usuario/multi-user-sfc.git](https://github.com/seu-usuario/multi-user-sfc.git)
    cd multi-user-sfc
    uv sync
    ```

3. **Configuração (.env)**
    Crie o arquivo de variáveis de ambiente:
    ```bash
    cp .env.example .env
    ```
    *Ajuste algoritmos, níveis de confiabilidade e topologia diretamente no `.env`.*

---

## 🖥️ Como Executar

Como o projeto agora é um pacote instalado, você não precisa mais caçar scripts em pastas. Use os comandos registrados:

-   **Simulação Única:**
    ```bash
    uv run muar-sim
    ```
-   **Simulação em Lote (Paralelo):**
    ```bash
    uv run muar-sim-paralelo
    ```
-   **Execução Sequencial:**
    ```bash
    uv run muar-sim-seq
    ```

---

## 🏗️ Estrutura do Repositório

```text
multi-user-sfc/
├── src/
│   └── muar_sfc/           # Código-fonte principal (Pacote)
│       ├── main.py         # Entrypoint da simulação
│       ├── algorithms/     # REPLIC, DARSPPO, Hephaestus, etc.
│       ├── controllers/    # Orchestrator, SFCManager, Crasher
│       ├── core/           # Configurações (Pydantic) e Net_V2
│       └── utils/          # Helpers de rede e falhas
├── tests/                  # Testes automatizados (Pytest)
├── rl_saved_models/        # Modelos de rede neural (.zip)
├── pyproject.toml          # Definições de CLI e dependências
├── uv.lock                 # Trava determinística de versões
└── .env                    # Configurações locais de simulação

🧑‍💻 Desenvolvimento e Qualidade
Mantenha a integridade do código com as ferramentas integradas:

Linting e Estilo: uv run ruff check . --fix

Verificação de Tipagem: uv run pyright

Bateria de Testes: uv run pytest

Autores

Hugo Leonardo — hugosantos@ufpa.br

David Galhego — david.galhego@icen.ufpa.br

Matheus Morais de Brito

Erick