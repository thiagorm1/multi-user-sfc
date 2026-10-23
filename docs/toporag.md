# TopoRAG local: contrato e premissas

Esta é uma **adaptação experimental**, não reprodução integral de TopoRAG.
Fonte do parecer fornecido pelo pesquisador: Ayepah-Mensah et al., *AI-native
cloud-edge orchestration for 6G metaverse networks*, Complex & Intelligent Systems
12:141 (2026), DOI https://doi.org/10.1007/s40747-026-02249-9.
Os números de equação abaixo seguem esse parecer; não se atribuem parâmetros
locais ao artigo. Não foram medidos resultados nesta implementação.

## P1–P12: rastreabilidade

1. **P1 [ARTIGO]** TopoRAG recupera top-k traços topológicos por similaridade
   para contextualizar decisões do LLM (descrição de TopoRAG). O parecer não
   fornece número de seção; essa localização bibliográfica permanece pendente.
2. **P2 [ASSUMIDO NO SIMULADOR]** Memória só em RAM, por adaptador/execução/semente.
   `reset_toporag()` limpa traços, confiança e evidência TopoRAG. O benchmark cria
   outro adaptador por semente. Não carregar dados entre sementes/treino/teste.
3. **P3 [ASSUMIDO NO SIMULADOR]** Assinatura canônica exata do DAG tipado via
   refinamento, individualização e poda de automorfismos; SHA-256 de JSON canônico.
   Não usa IDs de sessão/jogador. Nos nomes MUVR, o tipo semântico distingue
   IA/BE/renderers/codec compartilhados e FI/BE privados; fora deles usa `VNF.type`.
   Isso NÃO autoriza reuso de instâncias ou de fluxos.
4. **P4 [ASSUMIDO NO SIMULADOR]** Chave inclui conjunto de tipos e quartil da
   fração livre de banda física agregada, descontando used e reserved. Quartis
   são [0,.25), [.25,.5), [.5,.75), [.75,1]. Sem capacidade, quartil zero.
   Substitui embeddings/métrica composta; pode perder informações de gargalos.
5. **P5 [ASSUMIDO NO SIMULADOR]** Match exato primeiro; depois similaridade =
   (Jaccard dos tipos + igualdade do DAG + igualdade do quartil)/3.
   Empates usam ordem de inserção, mais antigo primeiro. Top-k default 4, sem
   truncamento da memória nesta iteração. Retrieval retorna cópias independentes.
6. **P6 [ASSUMIDO NO SIMULADOR]** Prior NÃO é produzido pelo LLM. Para cada
   tipo/host soma similaridades dos traços aceitos com SLA atendido, contando um
   host no máximo uma vez por traço/tipo. Divide pelo máximo entre propostas
   fisicamente viáveis no passo; denominador zero implica nenhum bônus.
7. **P7 [ARTIGO / ASSUMIDO NO SIMULADOR]** Eq. 26 descreve mistura convexa de
   priors LLM/agentes. Aqui **não há mistura convexa**, nem alteração dos logits:
   `cost -= kappa * prior_scale * score`. Só ranking entre propostas dos atores.
   Máscaras, reservas atômicas, SLA, sincronização e capacidade ficam intactos.
8. **P8 [ASSUMIDO NO SIMULADOR]** Confiança fixa em [0,1), default 0.0; ou
   `lagged_expost`: fração de requisições com orientação usada que foram aceitas
   e atenderam SLA no ciclo anterior, limitada por `kappa_max` (<1). Ciclo local
   é um batch de admissão, isto é, uma requisição de sessão com vários jogadores;
   a fração é portanto 0 ou 1 antes do teto. Sem orientação usada/denominador
   vazio, zero. Primeiro ciclo zero. Sem dados do próprio ciclo/futuro no prior.
   Não é confiança aprendida/calibrada nem prova de qualidade causal do Qwen.
9. **P9 [ARTIGO / ASSUMIDO NO SIMULADOR]** Eq. 34 cita KL, Eq. 22 gatilho de
   ajuste do LLM. **Não implementados**: sem KL, treino/ajuste de LLM/checkpoint.
10. **P10 [ASSUMIDO NO SIMULADOR]** Nenhuma mudança em `_compute_reward`.
    O abstract menciona desvio de custos, mas o parecer não identifica termo
    aditivo correspondente no corpo nem valores de R-hat, alpha ou kappa_max.
    Não inventamos recompensa nem valores publicados para essas lacunas.
11. **P11 [ASSUMIDO NO SIMULADOR]** Recuperação/ranking local por requisição e
    jogador; worker Qwen assíncrono mantém a cadência configurada de submissão
    (e pode pular por intervalo ou worker ocupado), não 600 s do artigo. Prompt
    recebe contexto só com `enabled=true`, `prompt_context=true` e memória não
    vazia. Schema/validação da resposta Qwen não mudam. O worker só lê cópias.
12. **P12 [ARTIGO / ASSUMIDO NO SIMULADOR]** Valores reportados no artigo de
    44,3 ms / 6,5% / p99 71,5 ms **não são metas** nem resultados deste código.
    CPU simulada não é CPU do PC/Ollama. Decisão não é latência SFC (SLA 6 ms)
    nem motion-to-photon. Replay não mede Qwen ao vivo.

## Decisões D1–D8 aprovadas para esta iteração

- **D1 [ASSUMIDO NO SIMULADOR]** Memória isolada/reset e hash de estado por semente.
- **D2 [ASSUMIDO NO SIMULADOR]** Recuperação estrutural determinística P3–P5,
  substituta explícita da métrica do artigo.
- **D3 [ASSUMIDO NO SIMULADOR]** Prior dos traços, somente bônus de ranking P6–P7.
- **D4 [ASSUMIDO NO SIMULADOR]** Confiança fixa/defasada P8; default inerte.
- **D5 [ASSUMIDO NO SIMULADOR]** Recompensa intocada, sem termo por desvio (P10).
- **D6 [ASSUMIDO NO SIMULADOR]** Contexto no prompt opt-in separado (P11).
- **D7 [ASSUMIDO NO SIMULADOR]** `load_balance_scale` intocado.
- **D8 [ASSUMIDO NO SIMULADOR]** Sem KL, treino LLM ou ajuste de checkpoint (P9).

## Configuração (somente exemplo, não executado)

Defaults dos dois TOMLs: `enabled=false`, `kappa=0.0`, contexto off. Nenhuma
consulta/memória/prior é usada no caminho padrão. Para estudar apenas ranking:

```toml
[toporag]
enabled = true
top_k = 4
kappa = 0.2
confidence_mode = "fixed"
kappa_max = 0.9
prior_scale = 1.0
prompt_context = false
```

Com `enabled=true`, `kappa=0`, contexto off, registram-se traços mas decisões e
métricas físicas devem coincidir com o controle; **tempo real de decisão pode
aumentar** pelo custo de instrumentação. Ao vivo, qualquer overhead pode mudar
qual resposta assíncrona chega a tempo; igualdade temporal absoluta não é
garantida. Para equivalência de cadência/inferência ao vivo, mantenha disabled.

## Proveniência e limites

`request_evidence.toporag` contém metadados por jogador: kappa efetivo, quantidade
recuperada, tamanho, assinatura e hashes antes/depois. `evidence().toporag`
contém parâmetros, hash final e registros independentes. `planning_records`
mantém sua ordem/contagem e os campos de guidance usados pelo replay: adições
`toporag` (snapshot da submissão) e `toporag_admissions` (decisões que usaram
aquele ciclo). Sem planning record, a evidência fica apenas na requisição e
no bloco independente. O JSON do benchmark inclui `toporag_manifest` por semente.
Nenhum formato de guidance ou hash de carga/topologia é substituído.

O comparador `scripts/benchmark_load_tradeoff.py` recusa fontes com TopoRAG `enabled=true` na evidência, em `toporag_manifest` ou em `execution_parameters`, antes de ler checkpoint ou criar saída, evitando mudança silenciosa de condição.
Fontes legadas/desabilitadas seguem aceitas com defaults inertes no replay e parâmetros da fonte e do replay registrados no `manifest.json`.
A guarda é conservadora: recusa também `enabled=true` com `kappa=0` (controle), pois confiança defasada/contexto podem alterar condições.

Traços por jogador são gravados após saber o resultado atômico da sessão:
rejeição posterior marca todos os jogadores já tentados como rejeitados; hosts
parciais não produzem prior positivo. Jogadores não tentados não geram traço.
Hash usa JSON canônico em vez de pickle para portabilidade dos dados primitivos;
segue o princípio SHA-256 do fingerprint existente. Não serializar rede viva.

Memória cresce com decisões, retrieval é linear e canonização pode ser cara em
DAGs maiores/assimétricos. Não há TTL do traço: viabilidade atual continua sendo
verificada, mas orientação histórica pode ficar pouco útil. Comparar sementes,
perfil, checkpoint, prefixo verificável e modo Qwen pareados. Avaliar benefícios
exige campanha separadamente autorizada, ainda não realizada.

## Avaliação exploratória autorizada em 21/10/2026

**[ASSUMIDO NO SIMULADOR]** A condição D2 usa `kappa=0.2` fixo,
`top_k=4`, `prior_scale=1.0` e `prompt_context=false`. O valor 0.2 é
uma escolha de projeto, não um parâmetro publicado no artigo. O efeito
esperado é um termo aditivo de prior no ranking **somente entre propostas**
dos atores, sem mudar logits, máscaras, peso de concentração ou recursos.
A memória permanece local à execução/semente, inicialmente vazia, sem
transferência do treino ou das demais condições. Não se presume ganho de
aceitação nem equivalência a uma implementação integral do artigo.
Resultados e caminhos efetivos serão registrados no relatório da campanha;
esta condição deve ser rotulada exploratória, separada da comparação D1.
