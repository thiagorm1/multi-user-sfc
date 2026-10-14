"""Create the PDF report for the audited MAPPO + Qwen MUVR experiment."""

import json
from pathlib import Path

from create_muvr_report import Report, num

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/mappo_qwen_muvr_async"
OUT = ROOT / "output/pdf/relatorio_mappo_qwen_muvr_50.pdf"


def main():
    summary = json.loads((DATA / "plots/summary.json").read_text())
    verification = json.loads((DATA / "verification.json").read_text())
    assert verification["status"] == "passed"
    docs = {v: json.loads((DATA / f"{v}.json").read_text()) for v in ("mappo", "mappo_qwen")}
    measured = verification["summary"]
    model_hash = docs["mappo_qwen"]["rows"][0]["inference_evidence"]["model_sha256"]
    r = Report(output=OUT, total=9, date="14 de outubro de 2026")
    r.pdf.setTitle("MAPPO + Qwen assíncrono - resultados e melhorias para AR")
    r.pdf.setSubject(
        "Avaliação pareada de 50 requisições por semente, SLA de 6 ms e recursos nominais"
    )
    r.start(
        "MAPPO + Qwen assíncrono<br/>Melhorias para realidade aumentada",
        "Relatório técnico do experimento executado em CPU. 50 requisições por repetição, três sementes pareadas e admissão imediata.",
    )
    own = summary[0]
    r.cards(
        [
            (num(own["acceptance_percent"]) + "%", "126 de 150 sessões"),
            (num(own["mean_latency_ms"], 3) + " ms", "Latência média aceita"),
            ("10", "Chamadas reais ao Qwen"),
        ]
    )
    r.h("Resultado principal")
    r.p(
        "O perfil ativo utiliza <b>atores MAPPO treinados</b> e o <b>Qwen2.5:1.5b via Ollama</b>. Foram aceitas 126 de 150 sessões, com planejamento em segundo plano e sem fallback heurístico do Qwen. Todos os fluxos das sessões aceitas respeitaram o SLA de 6 ms, com validação de recursos físicos."
    )
    r.h("Comparação consolidada")
    rows = [["Modelo", "Aceitas", "Aceitação", "Latência<br/>ms", "CPU<br/>%", "Decisão<br/>ms"]]
    for s in summary:
        rows.append(
            [
                s["label"],
                str(s["accepted"]) + "/150",
                num(s["acceptance_percent"]) + "%",
                num(s["mean_latency_ms"], 3),
                num(s["cpu_mean_at_arrivals_percent"], 3),
                num(s["mean_decision_ms"]),
            ]
        )
    r.table(rows, [145, 54, 65, 65, 58, r.available - 387], size=8, padding=6)
    r.p(
        "Latência: média ponderada pelos fluxos aceitos. CPU: média global observada nas chegadas. Decisão: média por sessão, incluindo rejeições, sem esperar a resposta Qwen; carregamento inicial do checkpoint excluído. O custo de submissão é medido separadamente e incluído na resposta de admissão.",
        size=8.5,
        leading=12,
    )
    r.h("Interpretação do ganho")
    r.p(
        "A decisão média com Qwen caiu de <b>1.378,89 ms no fluxo síncrono para 374,69 ms no assíncrono</b> (redução de 72,83%). A aceitação agregada permaneceu em 84%. O MAPPO sozinho obteve 83,33% e decisão média de 252,53 ms; a heurística anterior aceitou 88,67%. Três sementes não demonstram ganho estatístico robusto, e a meta de 98% ou 100% não foi atingida.",
        size=9.5,
        leading=14,
    )

    r.start(
        "1. O que são os três modelos",
        "Os nomes nas tabelas e nos gráficos identificam três formas de decidir onde hospedar as funções de uma sessão MUVR.",
    )
    r.h("MAPPO treinado: decisão com aprendizado por reforço")
    r.p(
        "MAPPO significa Multi-Agent Proximal Policy Optimization: uma versão do PPO para múltiplos agentes cooperativos. Neste projeto, quatro agentes regionais observam a rede e propõem servidores para hospedar as funções da sessão. O treinamento ajusta as redes neurais desses agentes a partir de experiências de alocação e recompensas."
    )
    r.p(
        "O rótulo <b>MAPPO treinado</b> representa a execução do checkpoint salvo após o treinamento, sem consultar o Qwen. Durante a simulação, os pesos ficam fixos: os agentes fazem inferência, não um novo treinamento. A seleção é limitada às suas propostas, com até oito tentativas internas por jogador e validação de recursos e latência. Resultado: <b>125/150 sessões, 83,33%</b>."
    )
    r.h("MAPPO + Qwen: política aprendida com orientação do LLM")
    r.p(
        "Este modelo combina o <b>mesmo MAPPO treinado</b> com o <b>Qwen2.5:1.5b</b>, um modelo de linguagem (LLM) executado localmente pelo Ollama em CPU. O Qwen recebe um resumo do estado da rede e devolve preferências de servidores e pesos para orientar a classificação das propostas dos agentes."
    )
    r.p(
        "O MAPPO gera os candidatos; o Qwen ajuda a escolher entre candidatos viáveis. Uma atualização é tentada a cada 60 segundos virtuais, com no máximo uma chamada em andamento. O MAPPO usa a orientação já disponível ou decide sozinho quando ela está ausente ou expirada. A orientação vale por até 120 segundos reais. O Qwen não é treinado aqui e não altera os pesos do MAPPO. Houve dez chamadas reais e orientação usada em 92/150 sessões. Resultado: <b>126/150 sessões, 84,00%</b>. A admissão não espera o LLM, mas ambos disputam CPU."
    )
    r.h("Heurística anterior: busca com regras programadas")
    r.p(
        "É o algoritmo conjunto usado antes da ativação dos atores treinados e do Qwen, identificado como <b>joint</b> no código. Ele aplica regras explícitas de busca e classificação. Primeiro testa hospedar toda a sessão em um servidor; depois busca distribuições em duas e três etapas, classificadas pela latência estimada do caminho crítico e validadas fisicamente. Se necessário, tenta uma alocação distribuída com classificação determinística por recursos e latência."
    )
    r.p(
        "Essa abordagem não usa checkpoint aprendido nem chamadas a um LLM. Embora reutilize componentes de viabilidade do ambiente MAPPO, as escolhas são feitas por regras, sem inferência dos atores. Resultado: <b>133/150 sessões, 88,67%</b>, a maior aceitação entre essas três variantes neste teste."
    )
    r.h("O que permanece comum")
    r.p(
        "As três variantes usam a mesma carga e topologia por semente, compartilhamento permitido pelo cenário, recursos nominais, admissão imediata e SLA de 6 ms. A diferença principal é como produzem e classificam as opções de alocação. A heurística é uma comparação independente; ela não resgata as rejeições do MAPPO nas variantes avaliadas.",
        size=9.5,
        leading=14,
    )

    r.start(
        "2. Cenário preservado",
        "A comparação usa a mesma infraestrutura e a mesma carga em cada par de execuções. A integração do MAPPO + Qwen não amplia recursos.",
    )
    rows = [
        ["Parâmetro", "Configuração do experimento"],
        ["Servidores / topologia", "25 servidores; Erdős-Rényi p = 0,20"],
        [
            "CPU / memória",
            "100 unidades nominais de cada recurso por servidor; sem multiplicadores",
        ],
        [
            "Demanda por sessão",
            "20% de CPU e 15% de memória para a cadeia completa, antes dos efeitos de compartilhamento",
        ],
        ["Processamento", "IPS entre 1×10¹⁰ e 2×10¹⁰; sem aumento na integração"],
        ["Enlaces", "1 Gbps; atraso uniforme entre 1 e 2 ms"],
        ["SLA", "6 ms, incluindo comunicação e processamento"],
        ["Carga", "50 sessões por semente; quatro jogadores por sessão; até 18 SFs"],
        ["Chegada / lifetime", "Uma chegada a cada 10 s; lifetime uniforme contínuo [0,120] s"],
        ["Banda por fluxo", "FI: 0,266 Mbps por jogador, exclusivo; BE: 25 Mbps, compartilhável"],
        ["Visão / predição", "P(h): 65%-90%; P(f): 17%-25%"],
        ["Admissão / destino", "Imediata, sem fila de espera; destino same_access"],
        ["Sementes", "42, 43 e 44; 150 sessões por modelo"],
    ]
    r.table(rows, [155, r.available - 155], size=9, padding=6)
    r.h("Premissas que afetam a comparação")
    r.p(
        "A memória é representada pelo recurso cache do simulador. A banda de 1 Gbps e os custos de processamento são parâmetros do cenário adotado, não uma garantia de reprodução exata do artigo MuSFiCO. As topologias variam entre sementes; dentro de cada semente, os modelos recebem o mesmo grafo e a mesma carga."
    )
    r.p(
        "A divisão em quatro regiões é uma organização dos agentes de decisão. Ela não adiciona nós, enlaces ou capacidade à infraestrutura física."
    )

    r.start(
        "3. O que mudou no MAPPO",
        "A seleção passou a usar um checkpoint treinado e propostas efetivas dos atores, com limites físicos aplicados em cada decisão.",
    )
    items = [
        (
            "Ambiente específico para MUVR",
            "Criado MUVRMAPPOEnv com quatro agentes regionais. As observações incluem recursos livres, distâncias ao destino e velocidade relativa de processamento. O checkpoint usa 48 valores de observação, sete ações por agente e estado global de 196 valores.",
        ),
        (
            "Máscaras e SLA",
            "As propostas são filtradas por disponibilidade de recursos e por um limite inferior de latência até o destino, incluindo o processamento restante. A validação física final mantém o limite de 6 ms.",
        ),
        (
            "Treinamento em CPU",
            "O treinamento inclui aquecimento por demonstrações e, em seguida, 300 episódios MAPPO com 60 atualizações. São usados cenários de carga variados e sementes distintas das sementes do benchmark. O checkpoint é validado antes da execução; pesos aleatórios não substituem um checkpoint inválido.",
        ),
        (
            "Busca com a política treinada",
            "Somente nós propostos pelos atores podem ser escolhidos. São permitidas até oito tentativas internas por jogador: a primeira é determinística e as demais amostram a política treinada. Isso não é uma fila de espera por capacidade nem altera a taxa de chegada.",
        ),
        (
            "Rastreabilidade",
            "Cada decisão registra ações, propostas, nó escolhido, tentativa e log-probabilidades. O carregamento e o hash do checkpoint são registrados nos resultados.",
        ),
    ]
    for title, body in items:
        r.h(title)
        r.p(body)
    r.h("Limite da evidência de treinamento")
    r.p(
        "No conjunto isolado de avaliação do treinamento, a taxa permaneceu em 86,67% antes e depois das etapas avaliadas. Portanto, esse conjunto não demonstra ganho isolado do treinamento. O resultado integrado de 84% vem do benchmark completo de sessões, com ocupação e compartilhamento ao longo do tempo.",
        size=9.5,
        leading=14,
    )

    r.start(
        "4. Como o Qwen participa",
        "O Qwen orienta a escolha entre propostas viáveis; os atores MAPPO continuam gerando os candidatos.",
    )
    r.h("Fluxo de decisão")
    r.p(
        "<b>1.</b> A chegada da sessão atualiza o estado físico e libera sessões expiradas.<br/><b>2.</b> Fora da decisão medida, o loop agenda uma consulta Qwen com uma cópia independente da rede e da sessão; se o worker estiver ocupado, não cria outra consulta.<br/><b>3.</b> O MAPPO prossegue com a última orientação concluída, ou sozinho quando ela não existe ou expirou.<br/><b>4.</b> O worker devolve JSON estruturado. O loop publica essa orientação para decisões seguintes, sem alterar a rede ativa.<br/><b>5.</b> O simulador aceita a sessão somente após validar recursos, caminhos e SLA."
    )
    r.h("Integração e limites")
    r.p(
        "Modelo: Qwen2.5:1.5b, executado em CPU. A API usada é compatível com OpenAI, em localhost:11434/v1. A resposta usa schema estruturado, IDs válidos de nós, até três preferências e limite de geração de 256 tokens. Timeout configurado em 120 segundos."
    )
    r.p(
        "A telemetria informa CPU e memória livres, velocidade de processamento e distâncias. Ela é enviada como colunas e linhas numéricas para reduzir o tamanho do prompt. O Qwen não pode aumentar capacidades nem flexibilizar o SLA. Falhas e exceções são registradas sem interromper a admissão pelo MAPPO. A orientação heurística de fallback do planejador genérico é descartada; ela não é apresentada como resposta real do Qwen."
    )
    r.h("Evidência de execução real")
    r.table(
        [["Semente", "Chamadas Qwen", "Sessões com orientação", "Fallback", "Tempo Qwen (s)"]]
        + [
            [
                str(row["seed"]),
                str(row["inference_evidence"]["qwen_successful_calls"]),
                str(row["inference_evidence"]["guidance_applied_requests"]),
                str(row["inference_evidence"]["qwen_fallback_calls"]),
                num(row["inference_evidence"]["qwen_planning_seconds"]),
            ]
            for row in docs["mappo_qwen"]["rows"]
        ],
        [65, 95, 135, 65, r.available - 360],
        size=9,
    )
    r.h("Latência de serviço e tempo de decisão")
    r.p(
        "O SLA de 6 ms se aplica à comunicação e ao processamento da SFC depois de alocada. O tempo de decisão não espera a resposta Qwen. Cópia e submissão do snapshot têm custo próprio, incluído na resposta de admissão. Ao final, o worker é drenado fora da admissão. No replay acelerado, há menos consultas e algumas terminam após a última sessão; chamadas concluídas não equivalem a sessões orientadas. O teste não comprova deadlines em tempo real."
    )

    r.start(
        "5. Latência e limites para AR",
        "O SLA da SFC e a latência do movimento até a imagem são métricas diferentes.",
    )
    r.h("Decisão e serviço: percentis do teste atual")
    rows = [["Métrica (ms)", "Média", "p95", "p99", "Máximo"]]
    for label, key in [
        ("Decisão MAPPO", "mappo"),
        ("Decisão MAPPO + Qwen", "mappo_qwen"),
    ]:
        d = measured[key]["decision_ms"]
        rows.append([label] + [num(d[v]) for v in ("mean", "p95", "p99", "max")])
    for label, key in [
        ("Resposta de admissão + Qwen", "admission_response_ms"),
        ("Latência SFC + Qwen", "sfc_latency_ms"),
    ]:
        d = measured["mappo_qwen"][key]
        rows.append([label] + [num(d[v], 3) for v in ("mean", "p95", "p99", "max")])
    r.table(rows, [180, 80, 80, 80, r.available - 420], size=8.5, padding=6)
    r.p(
        "Decisão e resposta: 150 amostras, incluindo rejeições. Latência SFC com Qwen: 504 fluxos aceitos. Percentis calculados com interpolação linear sobre o conjunto completo de amostras. A resposta de admissão inclui a preparação/submissão do snapshot.",
        size=9,
        leading=13,
    )
    r.h("Impacto do tempo de decisão")
    r.p(
        "Antes de iniciar a experiência, a decisão aumenta a espera de entrada. Se uma realocação interromper o serviço enquanto calcula uma nova cadeia, esse tempo pode se tornar uma pausa visível. Mesmo reduzida, a decisão média de 375 ms não é um orçamento para atualizar quadros. O loop de rastreamento e renderização precisa continuar independente da orquestração."
    )
    r.h("Motion-to-photon e desconforto visual")
    r.p(
        "Em AR, a latência completa vai do movimento capturado até a atualização exibida. Ela inclui rastreamento, comunicação, processamento, renderização e display. Os 6 ms da SFC cobrem apenas a parte modelada no simulador. Não existe aqui uma medida da latência visual nem um limite validado que garanta ausência de enjoo."
    )
    r.h("Métricas implementadas para medições reais")
    r.p(
        "O analisador aceita timestamps motion_s e display_s no mesmo relógio monotônico e quadros perdidos explicitamente registrados. Calcula média, p95/p99, máximo, desvio padrão, jitter (diferença absoluta de latência entre quadros consecutivos), intervalos de exibição, perda de quadros e violações de um orçamento configurado para o dispositivo."
    )
    r.p(
        "Neste benchmark, a telemetria AR está como <b>not_measured</b>. Não foram inventados valores de rastreamento, display ou perda de quadros. O orçamento motion-to-photon permanece indefinido até ser escolhido para a aplicação. Medição instrumental e avaliação com usuários continuam necessárias para concluir sobre conforto.",
        size=9.5,
        leading=14,
    )

    r.start(
        "6. Realocação preparada antes da troca",
        "Mudança de cadeia com preservação das reservas da sessão atual durante a preparação.",
    )
    r.h("O que foi implementado")
    r.p(
        "O helper prepare_handover recebe a rede ativa, as cadeias antigas, novas cadeias com IDs próprios e o algoritmo de admissão. A preparação usa uma cópia da rede mantendo as cadeias antigas implantadas. A nova versão precisa caber junto com a antiga e respeitar o SLA de cada fluxo, sem aumentar CPU, memória ou banda."
    )
    r.h("Sequência de preparação e commit")
    r.p(
        "<b>1.</b> Conferir as cadeias atuais e os IDs exclusivos das substitutas.<br/><b>2.</b> Reservar e validar a nova versão sem liberar a antiga.<br/><b>3.</b> Se não houver capacidade, preservar a atual e retornar falha de preparação.<br/><b>4.</b> Antes da troca, verificar se a rede permaneceu igual à usada na preparação.<br/><b>5.</b> No commit, retornar a rede com a nova versão e liberar as reservas antigas. Commits repetidos ou estado alterado são rejeitados."
    )
    r.h("Como isso reduz o risco de interrupção")
    r.p(
        "O cálculo e a preparação deixam a sessão atual disponível. O orquestrador pode instalar a rede retornada apenas quando a substituta estiver pronta. A continuidade das reservas depende de capacidade livre para a coexistência temporária; a melhoria não flexibiliza o SLA para forçar uma migração."
    )
    r.h("O que o simulador valida")
    r.p(
        "Os testes verificam que a rede ativa não muda durante a preparação, ambas as cadeias existem antes do commit, a antiga é liberada na rede substituta e falhas ou estado desatualizado preservam a sessão atual. O cenário permanece estático, sem gatilho automático de mobilidade; a API está disponível para o loop de orquestração."
    )
    r.h("Limites da continuidade em um dispositivo real")
    r.p(
        "A implementação modela reservas e commit da rede. Ela não transfere o estado de um serviço real, buffers de imagem ou conexões, nem executa a troca de pacotes de um dispositivo AR. O repositório não inclui um renderizador. Portanto, este teste não comprova migração sem perda de quadros em hardware; essa integração deve manter a apresentação e o rastreamento independentes do worker Qwen."
    )
    r.h("Configurações preservadas")
    r.p(
        "A topologia, os 25 servidores, as capacidades nominais, o SLA de 6 ms e a carga de chegada permanecem iguais. Não houve novo treinamento do MAPPO ou do Qwen para a melhoria assíncrona; o mesmo checkpoint foi reutilizado."
    )

    r.start(
        "7. Gráficos comparativos",
        "Curvas com 50 requisições por repetição e médias das três sementes. Baselines REPLIC, MuSFiCO e DARSPPO adaptados ao simulador.",
        wide=True,
    )
    from PIL import Image

    image = DATA / "plots/comparativo_muvr_50.png"
    with Image.open(image) as im:
        ratio = im.height / im.width
    width = min(r.available, (r.y - 125) / ratio)
    height = width * ratio
    r.pdf.drawImage(
        str(image),
        r.margin + (r.available - width) / 2,
        r.y - height,
        width=width,
        height=height,
        mask="auto",
    )
    r.y -= height + 12
    r.p(
        "Aceitação: percentual acumulado até cada chegada. Latência: média acumulada dos fluxos aceitos. CPU: ocupação global média observada em cada chegada. Decisão: tempo médio por sessão, incluindo rejeições. Todas as curvas compartilham a mesma carga e topologia por semente.",
        size=10,
        leading=14,
    )
    r.p(
        "Os resultados rotulados REPLIC, MuSFiCO e DARSPPO são de adaptações para este DAG e estes recursos. Não representam uma reprodução inalterada dos algoritmos dos artigos.",
        size=10,
        leading=14,
    )

    r.start(
        "8. Validação e reprodução",
        "Arquivos de implementação, evidências utilizadas e limites do resultado.",
    )
    r.h("Verificação concluída")
    r.p(
        "A suíte final passou com <b>66 testes</b>. Foram testados worker lento, falhas, orientação expirada, snapshots separados, métricas de quadros e realocação transacional. O smoke test aceitou duas sessões antes de a consulta Qwen terminar. A auditoria conferiu hashes pareados da carga/topologia, o checkpoint, propostas dos atores e SLA nas três sementes. Ruff e git diff --check passaram nos arquivos da melhoria."
    )
    r.h("Principais arquivos alterados ou adicionados")
    r.table(
        [
            ["Arquivo", "Papel"],
            [
                "algorithms/mappo/muvr.py",
                "Ambiente MUVR, checkpoint, busca dos atores e integração Qwen",
            ],
            [
                "algorithms/mappo/env_mappo.py",
                "Seleção restrita às propostas e pesos de classificação",
            ],
            [
                "core/scenarios/muvr_reference.py",
                "Planejamento assíncrono, relógio virtual e métricas de cauda",
            ],
            ["core/orchestration/handover.py", "Preparação e commit de realocação"],
            ["core/ar_metrics.py", "Percentis e análise de quadros reais"],
            ["scripts/analyze_ar_frames.py", "CLI para telemetria de AR"],
            ["tests/test_ar_async_orchestration.py", "Falhas, não bloqueio e realocação"],
            ["scripts/benchmark_mappo_qwen_muvr.py", "Experimentos pareados com e sem Qwen"],
            ["scripts/plot_muvr_comparison.py", "Comparação visual com os modelos anteriores"],
            [".env / .env.mappo-qwen-muvr.example", "Ativação e parâmetros do perfil"],
            [
                "tests/test_mappo_muvr_qwen.py",
                "Verificações de propostas, checkpoint, telemetria e limites",
            ],
        ],
        [230, r.available - 230],
        size=8.5,
        padding=5,
    )
    r.h("Fontes dos números deste relatório")
    r.p(
        "Resultados: results/mappo_qwen_muvr_async/mappo.json e mappo_qwen.json.<br/>Consolidação: results/mappo_qwen_muvr_async/plots/summary.json.<br/>Auditoria: results/mappo_qwen_muvr_async/verification.json.<br/>Checkpoint: rl_saved_models/MAPPO_muvr_model.pt.",
        size=8.5,
        leading=12,
    )
    r.p(
        "SHA-256 do checkpoint:<br/>" + model_hash[:32] + "<br/>" + model_hash[32:],
        size=8,
        leading=11,
        mono=True,
    )
    r.h("Execução do perfil ativo")
    r.p(".venv/bin/python -m muar_sfc.main", size=9, leading=13, mono=True)
    r.p(
        "Conclusão: a integração MAPPO + Qwen foi executada e auditada, mantendo os recursos nominais e o SLA. O modo assíncrono reduz a decisão sem reduzir a aceitação agregada do ensaio anterior. A heurística ainda aceita mais sessões. Conforto em AR não foi medido no simulador.",
        size=9.5,
        leading=14,
    )
    r.finish()
    print(OUT)


if __name__ == "__main__":
    main()
