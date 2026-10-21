# AGENTS.md — Guia de operação para agentes

> **Projeto:** MUAR-SFC / multi-user-sfc — orquestração de cadeias de funções de
> serviço para serviços imersivos em computação de borda, com DAG, MAPPO e Qwen.
> **Workspace atual:** `/home/thiagorm/Projects/multi-user-sfc`.
> Este guia registra as regras operacionais deste repositório. Consulte também
> [README.md](README.md), as configurações e o código antes de alterar premissas.
> Não transportar configurações de outros projetos para este ambiente.

## 1. Objetivo e escopo

1. Desenvolver e avaliar o posicionamento de SFs e suas rotas, respeitando CPU,
   memória, banda, sincronização e SLA de latência.
2. Combinar propostas cooperativas de atores MAPPO treinados com orientação
   contextual de um Qwen pré-treinado, sem entregar ao LLM o controle dos recursos.
3. Usar o cenário MUVR de referência para comparar mudanças na aceitação, latência,
   uso de recursos e custo de decisão, com sementes e cargas pareadas.
4. Manter rastreáveis as diferenças entre artigos, implementações locais e
   resultados observados. O projeto é uma adaptação/extensão; não presumir uma
   reprodução integral dos artigos-base.

Este é um simulador Python/NetworkX. SUMO/TraCI participa de cenários legados de
mobilidade. Não há neste projeto um ambiente MODA/ns-3/5G-LENA nem integração
NORI, E2/RIC ou controle de PRBs implementada como parte do perfil MUVR.

## 2. Artigos-base e limites da reprodução

### 2.1. MuSFiCO — referência de SFC para MUVR

**Hugo Santos, Denis Rosário, Eduardo Cerqueira e Torsten Braun.**
*Multi-criteria Service Function Chaining Orchestration for Multi-user Virtual
Reality Services*. IEEE GLOBECOM, 2022, pp. 6360–6365.

- DOI: <https://doi.org/10.1109/GLOBECOM48099.2022.10000827>.
- Referência para orquestração multicritério, decomposição de serviços MUVR,
  restrições de recursos/atraso e compartilhamento entre usuários.
- Fonte complementar primária: tese de Hugo Santos, *Multimedia Service
  Orchestration in Multi-tier Edge Computing Environments*, 2023, capítulo 5:
  <https://home.inf.unibe.ch/~rvs/research/pub_files/Hugo_Santos.pdf>.
- O adaptador em `algorithms/muvr_baselines.py` usa DP com busca limitada
  (beam padrão de quatro estados por camada), estendida ao DAG completo. Rotular
  seus resultados como **MuSFiCO adaptado**, sem atribuí-los ao código dos autores.
- Há também código legado em `algorithms/musfico.py`; não confundi-lo com o
  adaptador utilizado em `benchmark_muvr_comparison.py`.

### 2.2. AI-native cloud-edge — referência de LLM + DRL multiagente

**Daniel Ayepah-Mensah et al.**
*AI-native cloud-edge orchestration for 6G metaverse networks: an LLM-guided
multi-agent DRL approach*. **Complex & Intelligent Systems**, volume 12,
artigo 141, publicado em 16 de março de 2026.

- DOI: <https://doi.org/10.1007/s40747-026-02249-9>.
- Texto primário: <https://link.springer.com/article/10.1007/s40747-026-02249-9>.
- Referência para planejamento cloud-edge com LLM, DAGs de microsserviços e
  agentes MAPPO cooperativos. O artigo descreve TopoRAG, priors ponderados por
  confiança e recompensa baseada no desvio entre custos previstos e realizados.
- **Nossa integração atual:** Qwen2.5:1.5b local fornece pesos/preferências em JSON;
  o MAPPO propõe hosts e um ranking físico seleciona entre propostas viáveis.
- Não há reprodução de TopoRAG, ajuste parcial do LLM ou de todo o mecanismo de
  confiança desse artigo. O shaping existente no ambiente não comprova equivalência
  à recompensa proposta pelos autores. Tratar essas diferenças explicitamente.

### 2.3. Regra de rastreabilidade científica

Distinguir sempre **[ARTIGO]**, **[ASSUMIDO NO SIMULADOR]** e **[MEDIDO]**.
Premissas novas devem ser registradas na documentação da mudança, com fonte ou
justificativa e impacto na comparação. Antes de adotar um parâmetro como publicado,
consultar seção/tabela do artigo ou identificar a fonte complementar utilizada.

Não apresentar narrativa do Qwen como explicação causal validada. Não afirmar que
MAPPO supera PPO, IPPO, SAC discreto ou DQN sem comparação controlada. A escolha
atual tem justificativa de arquitetura cooperativa; superioridade quantitativa é
uma pergunta experimental.

## 3. Fontes e mapa do repositório

Todos os caminhos abaixo são relativos à raiz, salvo indicação contrária.

| Caminho | Papel |
|---|---|
| `README.md` | Instruções, cenários, medições e execução dos experimentos |
| `pyproject.toml`, `uv.lock` | Dependências, Python suportado, ferramentas e CLIs |
| `config/mappo_qwen.toml` | Arquivo editável de treino/avaliação com Qwen |
| `config/mappo_qwen_generalization_v2.toml` | Configuração alternativa de generalização |
| `executar_mappo_qwen.sh`, `scripts/run_mappo_qwen.py` | Validação e execução por configuração |
| `src/muar_sfc/config.py` | Configuração geral por Pydantic/variáveis `MUAR_*` |
| `src/muar_sfc/experiment_config.py` | Schema TOML e construção dos comandos |
| `src/muar_sfc/core/scenarios/muvr_reference.py` | Perfil MUVR, geração de carga, relógio virtual e auditoria |
| `src/muar_sfc/core/sfc.py` | DAG de serviço, dependências e caminho crítico |
| `src/muar_sfc/core/net_v2.py` | Estado e ciclo de vida dos recursos físicos |
| `src/muar_sfc/core/orchestration/placement.py` | Identidades, reuso e reservas de CPU/memória/banda |
| `src/muar_sfc/core/orchestration/handover.py` | Preparação e confirmação de realocação; fora do benchmark estático |
| `src/muar_sfc/algorithms/mappo/mappo_core.py` | Atores, crítico, buffer e atualizações MAPPO |
| `src/muar_sfc/algorithms/mappo/env_mappo.py` | Ambiente, máscaras, ranking e recompensa |
| `src/muar_sfc/algorithms/mappo/muvr.py` | Ambiente MUVR, checkpoint treinado, Qwen e busca limitada |
| `src/muar_sfc/controllers/cloud_llm_planner.py` | Planejador cognitivo compartilhado |
| `src/muar_sfc/algorithms/muvr_baselines.py` | Adaptadores locais de MuSFiCO, Greedy, REPLIC e DARSPPO |
| `src/muar_sfc/core/ar_metrics.py` | Métricas de frames/AR quando há dados apropriados |
| `src/muar_sfc/core/controller_metrics.py` | Amostragem Linux da CPU do controlador/Ollama |
| `scripts/train_mappo_muvr.py` | Treinamento específico do MAPPO MUVR em CPU |
| `scripts/benchmark_mappo_qwen_muvr.py` | Avaliação MAPPO com ou sem Qwen ao vivo |
| `scripts/benchmark_load_tradeoff.py` | Comparação de pesos com replay da orientação Qwen |
| `scripts/benchmark_decision_latency.py` | Comparação de decisão, caudas e disputa de CPU |
| `scripts/benchmark_muvr_comparison.py` | Comparação dos baselines MUVR |
| `scripts/analyze_mappo_qwen_results.py` | Auditoria de saída de avaliação padrão MAPPO + Qwen |
| `scripts/create_*report*.py` | Geradores de relatórios; conferir as fontes selecionadas |
| `tests/` | Testes de viabilidade, recursos, política, configuração e persistência |
| `results/`, `rl_saved_models/` | Resultados e checkpoints locais; ignorados pelo Git |
| `output/pdf/`, `output/figures/`, `tmp/` | Artefatos finais e intermediários; ignorados por padrão |

O analisador de avaliação padrão espera `mappo_qwen.json` e
`experiment_config.json`. Não aplicá-lo diretamente a `scale_0.json` ou
`scale_1.json`: os experimentos de tradeoff têm outro contrato de saída.
Os geradores de relatório podem apontar para campanhas locais específicas;
verificar caminhos e dados antes de reutilizá-los.

## 4. Perfil MUVR de referência

Os valores abaixo descrevem **a implementação local**, não uma declaração de que
todos foram publicados exatamente assim nos artigos.

| Parâmetro | Perfil local |
|---|---|
| Servidores | 25; capacidade nominal CPU 100 e memória/cache 100 por nó |
| Topologia | Grafo físico Erdős–Rényi, probabilidade 0,20; não forçar conectividade adicional |
| Enlaces | 1.000 Mbps; atraso uniforme de 1 a 2 ms por enlace |
| Processamento | IPS amostradas entre `1e10` e `2e10`; preservar a geração do perfil |
| Requisição | Sessão completa de até quatro jogadores; até 18 SFs distintas |
| Demanda completa | 20 unidades de CPU e 15 de memória, normalizadas por sessão |
| Tráfego | FI 0,266 Mbps exclusivo por usuário; BE 25 Mbps parcialmente compartilhado |
| Similaridade/predição | `P(h)` em [0,65; 0,90] e `P(f)` em [0,17; 0,25] |
| SLA | 6 ms de comunicação + processamento da SFC |
| Chegadas | Intervalo determinístico de 10 segundos virtuais |
| Duração | Uniforme contínua entre 0 e 120 segundos virtuais |
| Destinos | `same_access` por padrão |
| Admissão | Imediata, sem fila de espera no perfil de referência |
| Mobilidade/falhas | Não exercitadas no benchmark estático de referência |

A divisão de recursos entre SFs e o cálculo de processamento com 60 fps e
10 ciclos/bit são hipóteses do modelo. Não constituem renderização real nem
medição de capacidade de hardware para uma aplicação XR completa.
Não ativar escalonamento de capacidade, nuvem infinita ou o perfil legado
`.env.high-acceptance.example` para melhorar uma comparação de recursos nominais.

## 5. Loop de decisão e invariantes

```text
Tempo virtual: chegada -> expirar sessões antigas -> gerar DAG/carga
  -> preparar orientação (worker ao vivo ou replay por requisição)
  -> atores MAPPO propõem hosts usando máscaras
  -> ranking entre propostas -> reservar todos os recursos/dependências
  -> validar sessão completa -> confirmar ou rejeitar atomicamente
  -> registrar decisão, utilização e evidências -> liberar no vencimento
```

### 5.1. DAG e reservas

- A rede física pode conter ciclos; o DAG representa precedências da aplicação.
- Preservar forks/joins, somar entradas nas junções e reservar todos os ramos,
  mesmo os que não determinam o atraso máximo.
- Medir caminho crítico, não somar artificialmente ramos paralelos.
- Reuso exige identidade e compatibilidade de instância/fluxo. FI não é um fluxo
  compartilhável apenas por existir similaridade entre usuários.
- Falha de uma tentativa não altera a rede ativa. Liberação mantém recursos
  ainda referenciados; reservas nunca ultrapassam limites salvo tolerância numérica.

### 5.2. MAPPO, Qwen e peso de concentração

- Quatro atores regionais, seis hosts locais por região e uma opção de nó global
  finito; observação 48, ação 7 e estado global 196 no perfil padrão.
- CTDE: crítico global no treino. A execução atual dos atores é centralizada no
  simulador, não uma implantação de controladores distribuídos.
- Validar dimensões, hash e evidência de treino do checkpoint. Não substituir um
  checkpoint ausente/incompatível por política aleatória silenciosa.
- Busca atual: até 16 tentativas por jogador, preservando propostas dos atores.
  Não inserir resgate heurístico nem relaxar SLA para aumentar aceitação.
- Qwen ao vivo: saída estruturada validada, worker não bloqueante, cache com
  validade real e continuidade com MAPPO se não há orientação válida. Registrar
  falhas; não contabilizar fallback como consulta Qwen bem-sucedida.
- O Qwen usado é pré-treinado; treinar MAPPO não é treinar o LLM.
- `load_balance_scale` altera a penalidade de concentração no ranking, não as
  capacidades ou máscaras. O arquivo principal usa **0,0**; a referência original
  é **2,0** e o ajuste intermediário é **1,0**.
- `SimulationSettings` e a CLI de avaliação mantêm default **2,0** quando o peso
  não é informado. Sempre registrar o valor efetivo e passar o argumento quando
  necessário; não presumir que toda execução usa o valor do TOML.
- Não confundir o treinador atualmente no código com o histórico de um checkpoint
  salvo anteriormente. Consultar seus metadados `.training.json`.

### 5.3. Desempenho e isolamento

A cópia sob escrita destaca os atributos de hosts/enlaces antes de reservar.
O cache de candidatos é válido apenas na etapa correspondente e é invalidado
por grafo, função, exclusões e reset. Distâncias de propagação podem ser
reutilizadas durante uma tentativa de topologia fixa; banda residual continua
sendo verificada. Mudanças nesses caminhos exigem testes de isolamento e de
invalidação. Nunca trocar verificação física por estimativa em cache obsoleta.

## 6. Regras obrigatórias de trabalho e avaliação

1. **Preservar autorização do usuário.** Análise, gráfico ou PDF não autoriza
   retreinamento, campanha, download de modelos ou consultas reais ao LLM.
   Quando o usuário disser que rodará, preparar/validar o comando sem executá-lo.
2. **Não sobrescrever saídas.** Usar diretório exclusivo por execução, com
   data/hora, e arquivos por semente. Não apagar resultados ou checkpoints para
   contornar proteção. Nova execução não é retomada automática de uma interrompida.
3. **Não alterar premissas silenciosamente.** Registrar configuração, sementes,
   número de requisições, checkpoint/hash, busca, peso e modo Qwen. Alterar
   capacidade, SLA, destinos ou chegadas exige explicitar uma nova condição.
4. **Comparações pareadas.** Conferir `topology_sha256`, `workload_sha256`, perfil,
   sementes e horizonte. Comparar os mesmos subconjuntos quando os baselines têm
   menos sementes; não ranquear 33 sementes contra 3 como se fossem equivalentes.
5. **Separar artigos e adaptadores.** MuSFiCO local, REPLIC/DARSPPO legados e
   Greedy são implementações desta base. Não rotular medições locais como números
   oficiais, nem alegar equivalência de treinamento dos modelos sem evidência.
6. **Separar tempos e recursos.** CPU de borda simulada não é CPU do PC/Ollama.
   SLA de 6 ms não inclui decisão, inferência, display ou motion-to-photon.
   Não inferir energia, conforto ou ausência de enjoo apenas desse SLA.
7. **Replay não é LLM ao vivo.** Usar as mesmas orientações por requisição para
   isolar o ranking. O replay não mede geração nem disputa de CPU com Ollama.
   Relógio virtual de chegadas e relógio real do worker podem alterar a orientação
   disponível em uma execução ao vivo.
8. **Prefixos não são novas amostras independentes.** As primeiras 50 requisições
   de uma execução de 300 são relacionadas àquela carga. Recalcular/verificar o
   fingerprint do prefixo; não reutilizar o hash da carga inteira.
9. **Estatística e interpretação.** Usar a semente como unidade para diferenças
   pareadas/IC; separar treino, ajuste e avaliação. Informar média, p95, p99 e
   máximo. Medir concentração por nó e banda, não só CPU global. Latência das
   aceitas não inclui rejeições como zero.
10. **Validação antes da campanha.** Testes relevantes -> smoke curto quando
    autorizado -> campanha multisseed explicitamente solicitada. Corrigir os
    testes que falharem antes de aumentar a carga.
11. **Mudanças focadas.** Preservar cenários legados fora do escopo, evitar
    reformatar arquivos não relacionados e não alterar arquivos gerados para
    esconder problemas de implementação.
12. **Git e privacidade.** Fazer commits/push somente quando solicitados. Uma
    solicitação explícita já autoriza a operação, sem confirmação repetida.
    Separar commits por finalidade, em português, com `feat`, `fix` ou `chore`.
    Não versionar segredos, resultados brutos ou checkpoints; PDFs finais podem
    ser incluídos explicitamente quando fazem parte da entrega.

## 7. Comandos úteis

Executar da raiz. Usar `.venv/bin/python` para não depender do interpretador do
sistema. Se o ambiente não existir, conferir `pyproject.toml` e `uv.lock` antes de
instalar dependências; não recriar um ambiente funcional sem necessidade.

### 7.1. Verificações sem campanha

```bash
./executar_mappo_qwen.sh --check
.venv/bin/python scripts/benchmark_load_tradeoff.py --help
.venv/bin/python -m pytest -q
.venv/bin/ruff check <arquivos_python_alterados>
git diff --check
```

Os dois primeiros comandos não treinam nem chamam Qwen. Sem `--check`, o launcher
segue `execution.stage`: o TOML atual define `train_then_evaluate`, 300 requisições
e 33 sementes. Não executá-lo supondo que é apenas avaliação de 50 requisições.
`overwrite = false` é a proteção padrão; manter essa proteção.

### 7.2. Comparação de tradeoff: 33 sementes × 50 requisições

**Executar somente quando solicitado.** Requer a fonte e o checkpoint locais.
Usa as primeiras 50 requisições de cada semente da fonte de 300; são 3.300
requisições nos dois pesos. Não chama Ollama nem retreina.

```bash
.venv/bin/python -u scripts/benchmark_load_tradeoff.py \
  --guidance-source results/meu_treino_33_sementes_300/mappo_qwen.json \
  --model rl_saved_models/MAPPO_muvr_meu_treino.pt \
  --requests 50 \
  --scales 1.0 0.0 \
  --threads 4 \
  --output results/tradeoff_33_sementes_50
```

Para comparar 300 requisições, trocar por `--requests 300` e usar o prefixo de
saída `results/tradeoff_33_sementes_300` (19.800 requisições nos dois pesos).
A quantidade de sementes vem da fonte, não do nome da pasta. Conferi-la.

A saída efetiva recebe data/hora e é mostrada no terminal:

- `manifest.json`: parâmetros, hashes e origem da orientação;
- `seeds/scale_0/seed_42.json`: exemplo de semente concluída;
- `scale_0.json`, `scale_1.json`: agregados por peso, quando concluídos;
- `summary_scale_0.json`, `summary_scale_1.json`: resumo de cada peso;
- `summary.json`: criado ao concluir todos os pesos.

### 7.3. Avaliação com Qwen ao vivo

Só executar por solicitação. Requer Ollama já instalado, servidor disponível e
`qwen2.5:1.5b` presente. Não iniciar download ou alterar o modelo silenciosamente.
A pasta abaixo é apenas um exemplo e deve ser nova, pois essa CLI direta não tem
as mesmas garantias de criação exclusiva do comparador de tradeoff.

```bash
.venv/bin/python scripts/benchmark_mappo_qwen_muvr.py \
  --requests 50 --seeds 42 43 44 --variants mappo_qwen \
  --model rl_saved_models/MAPPO_muvr_meu_treino.pt \
  --search-attempts 16 --load-balance-scale 0.0 --threads 4 \
  --qwen-model qwen2.5:1.5b --api-base-url http://localhost:11434/v1 \
  --output results/nova_avaliacao_qwen_50
```

Para treinar, editar o TOML com etapa e caminhos apropriados ou consultar
`scripts/train_mappo_muvr.py --help`. Não transformar uma solicitação de avaliação
em treino, nem sobrescrever `MAPPO_muvr_meu_treino.pt`.

## 8. Resultados registrados e escopo

Valores medidos em campanhas existentes; não são metas obrigatórias nem números
dos artigos. Sementes 42-74; SLA de 6 ms; mesmo checkpoint no comparador;
orientações Qwen reproduzidas, sem inferência ao vivo.

| Carga por semente | Peso 1: aceitas | Peso 0: aceitas | Ganho |
|---|---:|---:|---:|
| 33 × 300 | 9.060/9.900 — 91,52% | 9.177/9.900 — 92,70% | +117; +1,18 pp |
| 33 × 50 | 1.495/1.650 — 90,61% | 1.521/1.650 — 92,18% | +26; +1,58 pp |

- Campanha de 300: `results/tradeoff_33_sementes_300_20261015_005012_755874`.
  Auditoria: `analysis_20261015_015123_981154/comparison.json` dentro da campanha.
- Campanha de 50: `results/tradeoff_33_sementes_50_20261015_022202_401462`.
  Auditoria: `analysis_20261015_024037_529297/comparison.json` dentro da campanha.
- Peso zero melhorou em 28 sementes, empatou em 4 e piorou em 1 no ensaio de 300.
  No de 50, melhorou em 19 e empatou em 14. Não garante preservar cada aceitação.
- A média global de CPU caiu ligeiramente, mas o p99 do nó mais ocupado chegou a
  80% e o uso de banda aumentou. Não descrever como aumento da capacidade física.
- Para MuSFiCO/Greedy, os resultados salvos em `results/muvr_comparison` têm
  apenas três sementes de 50 requisições. Usar o recorte correspondente do nosso
  modelo, não confrontar diretamente suas taxas com a média das 33 sementes.
- O melhor resultado não é aceitação de 100%. Limites físicos de atraso e de
  recursos continuam produzindo rejeições; não contornar isso adulterando o perfil.
- Última suíte completa registrada antes deste guia: **98 testes aprovados**.
  Esse número é histórico, não substitui a validação das próximas mudanças.

Resultados, checkpoints e documentos locais podem não existir em outro clone,
pois várias dessas pastas são ignoradas pelo Git. Não fabricar dados faltantes.

## 9. Fluxo de trabalho dos agentes

1. Ler este guia, identificar a solicitação e consultar os arquivos relevantes.
2. Separar fatos do código, premissas do modelo e ideias de artigos.
3. Implementar a menor mudança que atende ao pedido, preservando experimentos.
4. Testar os invariantes afetados; relatar evidências e limitações reais.
5. Executar campanha, treino, chamadas reais, commit ou push apenas no escopo
   autorizado pelo usuário.
6. Em relatórios, renderizar/verificar todas as páginas e explicitar população,
   modo Qwen, unidades, período de medição e condições dos comparadores.

### Orquestrador dos agentes de desenvolvimento

O modelo definido pelo usuário para o papel de **orquestrador (Maestro)** é
**GLM 5.3 Flash Free**. Ele coordena o planejamento, a distribuição de tarefas e
a integração dos resultados dos agentes de desenvolvimento. Essa definição não
altera o Qwen nem os atores MAPPO usados pelo simulador.

Não presumir que contas de nuvem, credenciais ou APIs desse modelo já estejam
configuradas. Os quatro atores MAPPO são agentes do algoritmo, não agentes de
programação. Delegação entre agentes de desenvolvimento só deve ocorrer quando
explicitamente solicitada ou quando houver uma instrução aplicável que a exija;
não confundir essa organização com a arquitetura executada pelo simulador.

## 10. Referências complementares

- MAPPO: Yu et al., *The Surprising Effectiveness of PPO in Cooperative,
  Multi-Agent Games*, NeurIPS 2022: <https://arxiv.org/abs/2103.01955>.
- Qwen: Qwen Team, *Qwen2.5 Technical Report*:
  <https://arxiv.org/abs/2412.15115>.
- Ponto de partida operacional: [README.md](README.md).
- Contrato de execução editável: [config/mappo_qwen.toml](config/mappo_qwen.toml).

---

*Atualizado em 21/10/2026. Os dois artigos-base foram identificados por título e
DOI; as diferenças para a implementação local foram mantidas explícitas. Este
arquivo não autoriza campanhas, alterações de capacidade ou operações Git por si só.*
