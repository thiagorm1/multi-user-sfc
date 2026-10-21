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

## Qwen para planejamento da aceitação

O Qwen analisa CPU, cache e enlaces congestionados e orienta os custos de alocação
MAPPO. IDs sugeridos são validados contra a topologia e os pesos são normalizados.
Preferências para evitar nós não retiram candidatos viáveis. Falhas de conexão,
timeout ou JSON inválido acionam o planejador heurístico local.

```bash
ollama pull qwen2.5:1.5b
cp .env.qwen.example .env
uv run muar-sim
```

O servidor Ollama deve estar ativo (`ollama serve` se necessário).
A integração usa `/v1/chat/completions` com respostas JSON, conforme a
[documentação oficial do Ollama](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx).
Para outro tamanho de Qwen ou endpoint compatível, ajuste `MUAR_COGNITIVE_LLM_MODEL`
e `MUAR_COGNITIVE_API_BASE_URL`; endpoints autenticados usam `MUAR_COGNITIVE_API_KEY`.
O intervalo de planejamento e o timeout são configuráveis no exemplo.

Na inferência, MAPPO avalia alternativas às propostas dos atores, respeitando CPU,
cache, banda e latência também na nuvem. A rota final precisa de banda disponível,
e a latência da origem não é contabilizada duas vezes. Estas correções evitam
rejeições por propostas inviáveis e aceitações sem recursos; o ganho na taxa de
aceitação deve ser medido comparando cenários com a mesma carga e topologia.


## Comparação reproduzível da aceitação

O script `scripts/benchmark_acceptance.py` repete as sessões MUAR em tempo virtual,
com geração, admissão, implantação física e expiração do projeto. Assim, a demora
do Qwen não altera as chegadas. Cada semente usa a mesma topologia, tráfego e
checkpoint entre variantes; compartilhamento, mobilidade, falhas e backups ficam
desativados neste experimento controlado.

```bash
.venv/bin/python scripts/benchmark_acceptance.py \
  --variant qwen \
  --model-path rl_saved_models/MAPPO_allocation_model.pt \
  --output results/comparacao/qwen.json \
  --seeds 42 43 44 --sessions 30 --players 4 --arrival-mean 3
```

Repita com `--variant improved` e `heuristic`. Para a linha de base `before`, use
`--source-root` apontando para uma cópia do código anterior, com pasta `logs/`
existente e o mesmo checkpoint. A segunda carga usa `--arrival-mean 10`.

Após gerar `before.json`, `improved.json`, `heuristic.json`, `qwen.json` e suas
variantes `moderate_*.json`, o comando abaixo valida os hashes das entradas e gera
relatório, CSVs e gráfico:

```bash
.venv/bin/python scripts/summarize_acceptance_benchmark.py results/comparacao
```


## Perfil completo de alta aceitação

As seis estratégias estão integradas: compartilhamento de instâncias e fluxos,
DAG com reservas em todos os ramos, capacidade configurável, colocação considerando
recursos e destino, fila com prazo máximo e treinamento MAPPO com cargas variadas.

O arquivo `.env.high-acceptance.example` configura CPU/cache em 3,6 vezes a capacidade
original, enlaces internos em 8 vezes, canal sem fio em 4 vezes e entrada de 40 Gb/s.
O SLA de rede/processamento é explicitamente 50 ms; a espera de admissão tem prazo
separado de 30 s. Estes são recursos da rede simulada, não requisitos do seu PC.
O gerador agora respeita `MUAR_LATENCY`, inclusive no DAG. A latência considera o
caminho crítico e todas as dependências, com tolerância de sincronização de 5 ms.

```bash
.venv/bin/python scripts/train_mappo.py --episodes 200
cp .env.high-acceptance.example .env
uv run muar-sim
```

O treino salva `rl_saved_models/MAPPO_acceptance_model.pt` e um arquivo
`.training.json` com a curva, sementes e validação separada, preservando o modelo
anterior. O comando principal respeita `MUAR_MAPPO_MODEL_PATH` e `MUAR_N_MAPPO_REGIONS`.
O perfil usa a heurística local; para Qwen, altere `MUAR_COGNITIVE_LLM_MODEL` e
`MUAR_COGNITIVE_API_BASE_URL` conforme a seção de integração.

Cada simulação grava também um arquivo `.admission.json`, com aceitação imediata,
após espera, rejeições, expirações, pendências e latências. Requisições enfileiradas
não contam como aceitas e implantações tardias são desfeitas.

O benchmark aceita `--profile basic`, `sharing`, `dag`, `scaled` ou `full`.
O perfil completo foi validado em duas cargas e três sementes: 720/720 requisições
aceitas de imediato, com SLA de 50 ms e auditoria de recursos. O ganho depende do
dimensionamento; estes resultados não garantem 100% em outras cargas ou falhas.
A validação isolada do treino não mostrou aumento de sucesso, e o ganho do conjunto
não deve ser atribuído apenas ao MAPPO ou ao Qwen.

## MUVR: 50 sessões com SLA de 6 ms e capacidade original nominal

O perfil `.env.muvr-reference.example` ativa `MUAR_APPLICATION=muvr_reference`.
Esse exemplo reproduz a heurística anterior. O `.env` local agora ativa MAPPO + Qwen,
conforme o perfil descrito abaixo. O comando principal executa tempo virtual:

```bash
.venv/bin/python -m muar_sfc.main
.venv/bin/python scripts/benchmark_muvr_reference.py
```

São 25 servidores homogêneos com CPU/memória nominal 100/100, sem multiplicadores,
IPS uniforme original 1e10–2e10, grafo Erdős–Rényi p=0,20, enlaces de 1 Gbps e
latências uniformes entre 1 e 2 ms. Não mantém as conexões nem a divisão heterogênea
de Luxembourg: segue a topologia nova solicitada. Uma requisição é uma sessão
inteira de quatro jogadores, com 18 SFs distintas e demanda agregada 20 CPU/15 memória.
FI é único (0,266 Mbps/jogador); BE (25 Mbps) divide funções comuns e únicas por
P(h) uniforme [0,65;0,90] e P(f) [0,17;0,25]. Lifetime é uniforme contínuo [0;120] s;
chegadas a cada 10 s. A admissão é atômica, imediata, sem fila, com auditoria e
liberação física de CPU, memória e banda. O SLA inclui comunicação e processamento.

Jogadores da mesma sessão usam um acesso regional comum, como no cenário anterior.
Use `--destinations scattered` para avaliar jogadores espalhados. Nenhuma semente
é selecionada em função do resultado, nem se adicionam enlaces para conectar o grafo.
A heurística procura uma alocação conjunta, em duas/três etapas, e usa o motor de
viabilidade do MAPPO como fallback. Não usa o checkpoint aprendido em Luxembourg
nem atribui resultados ao Qwen. Tempos de decisão são medidos separadamente do SLA.

As demandas por SF são proporcionais ao tráfego e normalizadas para a demanda
agregada pedida. Processamento segue a fórmula existente: 60 fps e 10 ciclos/bit.
Predição também reserva recursos; os fluxos finais codificados são individuais.
Essas são hipóteses explícitas: não representam uma reprodução exata do MuSFiCO.
A tese do autor, seção 5.3.1/tabela 9, informa 1 Gbps, mas traz 25 nós totais e
30% de borda, além de uma ambiguidade sobre demanda por usuário/sessão.
Este perfil segue os 25 **servidores** e 20% CPU/15% memória **por sessão** pedidos.
Com essa carga, o pico possível de CPU global é 9,6% (12 sessões ×20 /2500);
portanto os 23–25% citados não podem ser impostos como resultado desse perfil.

Para consolidar o teste com as 33 sementes padrão:

```bash
.venv/bin/python scripts/benchmark_muvr_reference.py --output results/muvr_50_reference/joint_final.json
.venv/bin/python scripts/summarize_muvr_reference.py results/muvr_50_reference/joint_final.json
```

O relatório mostra aceitação média e IC95%, latência, decisão, utilização e teto
otimista de aceitação imposto pelas distâncias e pelo processamento. A meta de 98%
não é uma condição artificial do gerador nem uma autorização para relaxar o SLA.
Fonte: https://webarchive.inf.unibe.ch/cds/publications/files/Hugo_Santos.pdf

### Painel comparativo MUVR no formato 2×2

```bash
.venv/bin/python scripts/benchmark_muvr_comparison.py
.venv/bin/python scripts/plot_muvr_comparison.py
```

O benchmark executa 50 sessões por modelo nas mesmas três sementes (42, 43, 44),
com admissão imediata e o perfil MUVR ativo sem aumento de capacidade. A figura
compara nosso modelo conjunto, REPLIC, MuSFiCO, Greedy e DARSPPO em aceitação,
latência média acumulada, CPU na rede e tempo médio de decisão por sessão.
Os hashes de topologia e workload precisam coincidir antes de plotar.
A figura é exportada em PNG, PDF e SVG, com os valores em CSV, em
`results/muvr_comparison/plots/`.

**As adaptações são explícitas:** REPLIC e DARSPPO reutilizam os checkpoints e
as funções de observação existentes, com travessia reversa de todas as funções
DAG, coordenação das instâncias comuns e validação física completa. Greedy usa
DAGGreedyAlgorithm com a mesma validação. MuSFiCO usa um adaptador de programação
dinâmica generalizado para DAG, limitado a quatro estados por camada; não é a
implementação inalterada do artigo ou da classe especializada em MUAR. O benchmark
não retreina os modelos e não reproduz os resultados publicados. Tempos de decisão
excluem o carregamento inicial dos checkpoints e incluem todas as tentativas.
Latência é calculada somente nos caminhos aceitos, nunca preenchida com zero
quando não há aceites. CPU é o uso instantâneo após cada chegada, respeitando
expiração das sessões; não é uma curva de consumo acumulado sem liberação.

## MAPPO treinado + Qwen no cenário MUVR de 6 ms

O `.env` ativo usa `.env.mappo-qwen-muvr.example`: `MUAR_ALG=mappo`,
`MUAR_COGNITIVE_PLANNER=true` e `qwen2.5:1.5b` no Ollama local. Não há
aumento de CPU, memória, IPS ou banda. São os mesmos 25 servidores, 50 sessões
com quatro jogadores, chegada a cada 10 s e admissão imediata sem fila.

O checkpoint `MAPPO_muvr_model.pt` é treinado especificamente para essa topologia,
com quatro atores, 48 valores de observação, sete ações por ator e estado global 196.
O treinamento usa uma inicialização por demonstrações seguida por 300 episódios
on-policy MAPPO/GAE. As sementes de treinamento e validação são diferentes das
sementes 42, 43 e 44 do benchmark; não se treina o Qwen. Checkpoints incompatíveis
ou sem registro de atualizações MAPPO geram erro, em vez de inferência aleatória.

Na inferência, **somente hosts propostos pelos atores** podem ser escolhidos. A primeira
busca é determinística; buscas internas revisitam escolhas com ações
amostradas das distribuições treinadas e máscaras de ação. A semente é fixa por
sessão e jogador. Essas buscas não representam espera ou novas tentativas de admissão.
O padrão permite até 16 tentativas por jogador: mantém as oito tentativas da busca
original e, se necessário, explora alternativas que preservam o prefixo de funções
já alocadas, com exclusões cumulativas de hosts. Essa etapa adicional pode aumentar
o tempo de decisão. Estados de exclusão duplicados não consomem o orçamento de
tentativas. Cada jogador registra
`search_diagnostics`: tentativa, função onde falhou, quantidade de funções alocadas,
motivo e ausência de ações viáveis na máscara daquele ramo. Uma máscara vazia em um
ramo explorado não prova que todas as alocações possíveis sejam inviáveis.
Não existe resgate pela heurística conjunta. A observação inclui velocidade de
processamento e distância ao destino; a máscara considera um limite otimista
de latência restante, sem relaxar o SLA. O nó global compartilhado continua sendo
um servidor de borda finito, sem adicionar capacidade de nuvem.

O treinamento inclui cargas de até 85% e gargalos correlacionados de CPU, memória
ou banda, sem alterar capacidades. Após atualizações MAPPO, avalia candidatos a
cada 50 episódios e ao final; salva o primeiro com maior aceitação em 30 cenários
de validação. Outros 30 cenários separados medem o resultado final, sem escolher
o checkpoint. Os metadados distinguem o total de atualizações executadas das
atualizações do checkpoint selecionado. Essa seleção e diversidade de treinamento
precisam ser avaliadas por experimentos pareados; não garantem maior aceitação.

O checkpoint experimental local `rl_saved_models/MAPPO_muvr_generalization_v2.pt`
usa esse treino e não substitui automaticamente um modelo anterior.
Para avaliá-lo com Qwen e a nova busca, execute:

```bash
python scripts/run_mappo_qwen.py --config config/mappo_qwen_generalization_v2.toml
```

A configuração v2 usa `stage = "evaluate"` e um diretório próprio de resultados.
O analisador agrupa os diagnósticos das rejeições cujo limite inferior de latência
não excede o SLA; arquivos antigos permanecem legíveis, com diagnósticos indisponíveis
identificados explicitamente.

O Qwen recebe uma cópia da telemetria e produz orientações em um worker de
segundo plano. A admissão **não espera o LLM**: o MAPPO usa a última orientação
válida ou decide sozinho quando ela ainda não existe ou expirou. A validade é de
120 segundos de relógio monotônico (`MUAR_COGNITIVE_GUIDANCE_MAX_AGE`). A tentativa
de atualização ocorre a cada 60 segundos virtuais, mas há no máximo uma chamada
em andamento: consultas não se acumulam em fila. No replay acelerado isso significa
menos chamadas que uma execução com chegadas reais a cada 10 segundos.

A API usa JSON Schema com IDs conhecidos e até três preferências de nós.
As diretrizes afetam apenas propostas viáveis; não alteram capacidades ou SLA.
Uma falha mantém o MAPPO disponível e é registrada como falha Qwen; o fallback
heurístico gerado pelo planejador genérico é descartado. Os resultados distinguem
chamadas bem-sucedidas, falhas e sessões que efetivamente usaram orientação.

```bash
.venv/bin/python scripts/train_mappo_muvr.py
.venv/bin/python scripts/benchmark_mappo_qwen_muvr.py
.venv/bin/python -m muar_sfc.main
```

Novos resultados: `results/mappo_qwen_muvr_async/`. O tempo de decisão não inclui
espera pela resposta Qwen. O custo de copiar/enviar o snapshot é medido separadamente
em `background_submission_s` e incluído em `admission_response_s`. Ao fim do teste,
o worker é encerrado e drenado fora da admissão para registrar chamadas pendentes.
O Qwen continua disputando CPU com a inferência MAPPO; assíncrono não elimina esse custo.
Resultados e PDFs em `results/mappo_qwen_muvr/` descrevem o experimento síncrono histórico.

### Realocação sem liberar a cadeia atual antes da preparação

`core/orchestration/handover.py` fornece `prepare_handover(net, old_ids, new_batch,
admitter)`. A nova cadeia precisa de IDs novos e é reservada com a antiga ainda ativa,
sem aumentar recursos. Se não couber ou violar SLA, retorna `None` e mantém a antiga.
`prepared.commit(net)` retorna a rede substituta depois de liberar a antiga e validar
recursos; rejeita alterações de estado desde a preparação e commits repetidos.
O chamador deve instalar a rede retornada no seu loop de orquestração. Não há gatilho
automático de mobilidade no cenário estático; o helper permite preparar a troca.
Isso modela a continuidade das reservas, não a transferência de estado de serviços,
o encaminhamento de pacotes ou a apresentação de quadros em um dispositivo real.

### Latência visual e métricas de cauda em AR

Cada simulação agora fornece p95, p99, máximo e desvio padrão da latência da SFC,
da decisão e da resposta de admissão. A SFC continua com SLA de 6 ms. O simulador
não tem rastreador/tela/renderizador: `ar_telemetry.status=not_measured` é o padrão.
Não se transforma a latência da SFC em uma medida de conforto visual.

Para dados reais, `simulate(..., frame_samples=..., ar_budget_ms=...)` aceita
quadros com `motion_s` e `display_s` no mesmo relógio monotônico; quadros perdidos
precisam ser explicitamente registrados como `{"dropped": true}`. Também há CLI:

```bash
.venv/bin/python scripts/analyze_ar_frames.py frames.json --budget-ms 10
```

O valor 10 é apenas um exemplo de orçamento explicitamente escolhido, sem promessa
de ausência de enjoo. A análise mede motion-to-photon, p95/p99, jitter (diferença
absoluta da latência entre quadros exibidos consecutivos), intervalos de exibição,
quadros perdidos e violações do orçamento. `MUAR_AR_MOTION_TO_PHOTON_BUDGET_MS`
configura o orçamento do perfil; permanece indefinido até escolher o dispositivo.
A renderização/rastreamento real deve continuar em seu loop próprio, sem chamar o
orquestrador a cada quadro. Esse loop não existe neste repositório e não é simulado.

### Avaliação com 300 sessões

O perfil local ativo foi ajustado para 300 sessões. Para comparar MAPPO com e sem
Qwen nas mesmas três sementes e manter os resultados de 50 sessões separados:

```bash
.venv/bin/python scripts/benchmark_mappo_qwen_muvr.py --requests 300 --seeds 42 43 44 --output results/mappo_qwen_muvr_async_300
.venv/bin/python scripts/plot_mappo_benchmark.py results/mappo_qwen_muvr_async_300
```

Cada repetição começa com uma rede nova. São 300 sessões por semente, não 900
simultâneas. O SLA de 6 ms, as capacidades, as chegadas a cada 10 s e os lifetimes
até 120 s permanecem iguais. O gerador dos quatro gráficos audita o pareamento e
salva um resumo com aceitação, CPU, latência e percentis de decisão.

### Configuração editável para o seu experimento MAPPO + Qwen

Edite `config/mappo_qwen.toml`. Ele contém comentários em português e controla
episódios, demonstrações, taxas de aprendizado, PPO/GAE, exploração, clipping de
gradientes, atualização e aprendizado por demonstrações. Também controla o modelo
Qwen, timeout, tokens, validade da orientação e parâmetros da avaliação.

O padrão treina uma vez e depois executa **somente MAPPO + Qwen**, com 33 sementes
(42 a 74) e 300 requisições por semente. CPU, memória, topologia e SLA do cenário
não são alterados pelo arquivo. Os caminhos novos preservam o checkpoint anterior.

```bash
# Apenas verificar o arquivo, sem treino, simulação ou contato com Qwen:
./executar_mappo_qwen.sh --check

# Iniciar o experimento quando você decidir:
./executar_mappo_qwen.sh
```

Em `[execution]`, escolha `stage = "train"` para somente treinar,
`stage = "evaluate"` para avaliar um checkpoint já existente, ou
`stage = "train_then_evaluate"` para as duas etapas. Para usar o checkpoint anterior,
selecione `evaluate` e `model_path = "rl_saved_models/MAPPO_muvr_model.pt"`.
O Ollama deve estar ativo com o modelo escolhido para a avaliação. O lançador
confere a lista de modelos antes de iniciar as etapas que incluem avaliação.

As seções são `[execution]`, `[training]`, `[evaluation]` e `[qwen]`.
`seed_count = 33` representa 33 sementes de avaliação, não 33 treinamentos.
`episodes = 300` representa episódios MAPPO; `requests = 300` são sessões por
semente na avaliação. O Qwen não é treinado neste fluxo.

O lançador registra os parâmetros efetivos na pasta de resultados; os scripts
salvam também os hiperparâmetros de treinamento e avaliação nos metadados.
`overwrite = false` impede substituir checkpoints ou resultados existentes.
Para repetir um experimento, prefira novos `model_path` e `output_dir`.
Não há retomada automática: `overwrite = true` autoriza reiniciar e substituir.
A configuração é validada antes de executar, incluindo valores inválidos,
chaves desconhecidas e sobreposição entre sementes de treino e avaliação.


### Medição e otimização do tempo de decisão

O ambiente MAPPO reutiliza candidatos entre a máscara e o ranking na mesma etapa.
O cache é invalidado ao mudar o grafo planejado, a função ou as exclusões de hosts,
e é reconstruído no reset. Distâncias de propagação são calculadas uma vez por
origem durante a tentativa; rotas com banda residual continuam sendo verificadas
no estado da etapa. Observações são extraídas uma vez por transição e reutilizadas
para formar o estado global.

As tentativas em grafos físicos simples usam cópia sob escrita: apenas os atributos
do host e dos enlaces reservados são destacados antes da alteração. O ambiente
não modifica a topologia durante uma tentativa. Grafos direcionados ou multigrafos
mantêm a cópia profunda. CPU, memória, banda, sincronização e SLA continuam sendo
verificados. Checkpoints existentes mantêm as mesmas dimensões de entrada e saída;
não é necessário retreinar para usar essa otimização.

Para medir média, p95, p99 e máximo em três sementes pareadas:

```bash
.venv/bin/python scripts/benchmark_decision_latency.py \
  --baseline-ref fda4e87 \
  --model rl_saved_models/MAPPO_muvr_meu_treino.pt \
  --requests 50 --seeds 42 43 44 --threads 4 \
  --output results/comparacao_tempo_decisao
```

Use uma pasta de saída nova. O benchmark não treina o modelo. Ele executa a
implementação anterior e a atual, com e sem Qwen. A condição
`optimized_ollama_load` gera consultas reais mas descarta suas orientações: assim
podemos observar a disputa de CPU sem alterar o ranking por recomendação do LLM.
`--variants` permite selecionar condições específicas.

No Linux, amostras de `/proc` a cada 0,5 s registram CPU do controlador, processos
Ollama e ocupação total da máquina. CPU de processo pode superar 100%: cada 100%
equivale a um núcleo lógico ocupado. A medição de CPU inclui o encerramento do
worker; tempos por decisão excluem essa espera final. O benchmark registra também
tempo de CPU da thread de decisão, permitindo distinguir trabalho da thread de
parte da espera/escalonamento. As amostras não medem diretamente largura de banda
de memória ou pressão de cache. Consulte `summary.json`, `manifest.json` e os traces
por condição. A disponibilidade assíncrona da orientação pode mudar as escolhas do
Qwen quando a política fica mais rápida; compare essa condição separadamente da
equivalência da política sem Qwen.

### Tradeoff de balanceamento para aceitação

Em `[evaluation]` de `config/mappo_qwen.toml`, `load_balance_scale = 0.0`
remove a parcela explícita de penalidade por concentração de CPU/memória no
ranking do MUVR MAPPO. A seleção pode concentrar funções em hosts próximos,
sempre sujeita às mesmas máscaras e limites físicos. O ajuste anterior era `1.0`;
a referência original `2.0` continua disponível para comparação.
O parâmetro é usado apenas para ordenar os hosts propostos pelos atores; não
altera suas máscaras, pesos treinados, orçamento de busca, recursos dos nós,
reuso, sincronização ou SLA de 6 ms. O peso efetivo da parcela continua sendo
multiplicado por `load_balance_weight` da orientação Qwen, quando disponível.

A configuração principal e o exemplo `.env.mappo-qwen-muvr.example` ativam o
tradeoff. Configurações que omitem a opção conservam a referência 2.0 para
compatibilidade. Na CLI de avaliação, use `--load-balance-scale 0.0`; por variável
de ambiente, `MUAR_MAPPO_LOAD_BALANCE_SCALE=0.0`. Não é necessário retreinar.

Para comparar pesos sem confundir o resultado com a disponibilidade assíncrona
ou com diferentes respostas do LLM:

```bash
.venv/bin/python scripts/benchmark_load_tradeoff.py \
  --guidance-source results/otimizacao_decisao/comparacao_3_sementes_50/optimized_qwen.json \
  --scales 2.0 1.0 0.75 0.5 0.25 0.0 \
  --output results/comparacao_tradeoff_cpu
```

Esse ensaio **reproduz orientações reais já registradas**, por requisição. Não
faz novas chamadas Qwen. Usa o mesmo checkpoint, topologias, carga e orçamento
de busca; o peso 2.0 precisa reproduzir a aceitação da fonte. O resumo inclui
aceitação por semente, CPU média global, CPU p95/p99/máxima por nó, desequilíbrio,
latência e caudas do tempo de decisão. CPU nessa análise é a dos servidores
simulados, não a CPU do computador que executa Ollama. A validação com o LLM ao
vivo deve ser registrada separadamente. Ganhos em três sementes são exploratórios
e ainda precisam de confirmação em uma avaliação maior e sementes independentes.


No ensaio controlado de maior concentração (três sementes, 50 requisições cada),
reduzir de 1.0 para 0.0 elevou a aceitação de 126/150 para 130/150. A CPU global
média passou de 4,184% para 4,254%; o pico de CPU por nó passou de 65,93% para 80%.
Isso representa maior concentração, não ampliação de capacidade. Pesos 0.75, 0.5
e 0.25 não melhoraram a aceitação em relação a 1.0. Os resultados são preliminares
e não garantem o mesmo ganho em outras sementes ou em 300 requisições. O Qwen
continua orientando latência e preferências de hosts; com peso zero, seu peso de
balanceamento não contribui para essa parcela específica do ranking. O treino
MAPPO e os demais termos do custo permanecem os mesmos.


### Saída protegida da comparação de tradeoff

`--output` é um prefixo. Cada execução de `benchmark_load_tradeoff.py` cria uma
pasta exclusiva com data/hora, por exemplo
`results/tradeoff_33_sementes_300_20261015_143000_123456`. O caminho efetivo é
mostrado no terminal e registrado no manifesto. Se houver colisão de nomes,
é acrescentado um sufixo numérico por criação atômica do diretório.

Todos os arquivos são criados em modo exclusivo; nenhum resultado existente é
substituído. Cada semente concluída é gravada uma única vez em
`seeds/scale_1/seed_42.json`, por exemplo. Ao concluir um peso, são criados
`scale_1.json` e `summary_scale_1.json`. `summary.json` só é criado quando todos
os pesos terminam. Se houver interrupção, os arquivos das sementes já concluídas
permanecem disponíveis. Reexecutar cria outra pasta e começa um experimento novo;
ainda não há retomada automática das sementes anteriores.

Comando para comparar os dois pesos nas 33 sementes e 300 requisições da fonte:

```bash
.venv/bin/python -u scripts/benchmark_load_tradeoff.py \
  --guidance-source results/meu_treino_33_sementes_300/mappo_qwen.json \
  --model rl_saved_models/MAPPO_muvr_meu_treino.pt \
  --scales 1.0 0.0 --threads 4 \
  --output results/tradeoff_33_sementes_300
```

O arquivo de origem e o checkpoint são apenas lidos. Para analisar os resultados,
use a pasta com data/hora que foi mostrada no início da execução.


Para repetir a comparação com **33 sementes e 50 requisições por peso**, use a
mesma fonte de 300 requisições e acrescente `--requests 50`. São utilizadas as
primeiras 50 requisições e as orientações correspondentes de cada semente. O
fingerprint da carga é recalculado para o prefixo e conferido com a execução;
os arquivos de origem não são alterados. As 33 sementes vêm da fonte indicada.

```bash
.venv/bin/python -u scripts/benchmark_load_tradeoff.py \
  --guidance-source results/meu_treino_33_sementes_300/mappo_qwen.json \
  --model rl_saved_models/MAPPO_muvr_meu_treino.pt \
  --requests 50 --scales 1.0 0.0 --threads 4 \
  --output results/tradeoff_33_sementes_50
```

Total: 3.300 requisições nos dois pesos. Não há retreinamento nem novas chamadas
Qwen. A saída continua em uma pasta exclusiva com data/hora. Omitir `--requests`
preserva o comportamento anterior de avaliar toda a carga da fonte.
