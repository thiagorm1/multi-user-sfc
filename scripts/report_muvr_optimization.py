"""Audit completed optimization/training/live runs without changing source artifacts."""

import argparse
import hashlib
import json
import pstats
import re
from datetime import datetime
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_log(path):
    data = path.read_bytes()
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return data.decode(encoding)


def save_new(path, text):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def profile_rows(path):
    stats = pstats.Stats(str(path))
    return sorted([
        {"file": Path(file).name, "line": line, "function": name,
         "calls": values[1], "self_s": values[2], "cumulative_s": values[3]}
        for (file, line, name), values in stats.stats.items()
    ], key=lambda row: row["cumulative_s"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--d2-dir", type=Path)
    parser.add_argument("--d3-dir", type=Path)
    args = parser.parse_args()
    output = args.evaluations / f"audit_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(exist_ok=False)
    documents = {"baseline": load(args.baseline / "mappo_qwen.json")}
    directories = {label: args.evaluations / label for label in [
        "D1", "D2_exploratory_toporag", "D3_exploratory_attempts32"
    ]}
    if args.d2_dir:
        directories["D2_exploratory_toporag"] = args.d2_dir
    if args.d3_dir:
        directories["D3_exploratory_attempts32"] = args.d3_dir
    for label in ["D1", "D2_exploratory_toporag", "D3_exploratory_attempts32"]:
        documents[label] = load(directories[label] / "mappo_qwen.json")
    before = load(args.profile_dir / "before/mappo.json")["rows"][0]
    after = load(args.profile_dir / "after_fixed/mappo.json")["rows"][0]
    equivalence = {
        key: before[key] == after[key] for key in [
            "accepted", "mean_latency_ms", "max_latency_ms", "latency_samples",
            "topology_sha256", "workload_sha256", "time_weighted_utilization_percent",
        ]
    }
    equivalence["all_request_evidence"] = (
        before["inference_evidence"]["request_evidence"]
        == after["inference_evidence"]["request_evidence"]
    )
    if not all(equivalence.values()):
        raise ValueError(f"Optimization equivalence failed: {equivalence}")
    baseline = {row["seed"]: row for row in documents["baseline"]["rows"]}
    rows = []
    for label, document in documents.items():
        if document["profile"] != documents["baseline"]["profile"]:
            raise ValueError(f"Profile mismatch: {label}")
        if {r["seed"] for r in document["rows"]} != {42, 43, 44}:
            raise ValueError(f"Unexpected seed population: {label}")
        for row in document["rows"]:
            control = baseline[row["seed"]]
            for key in ["topology_sha256", "workload_sha256", "requests"]:
                if row[key] != control[key]:
                    raise ValueError(f"Pairing failed: {label}, {row['seed']}, {key}")
            evidence = row["inference_evidence"]
            rows.append({
                "condition": label, "seed": row["seed"], "accepted": row["accepted"],
                "acceptance_percent": row["acceptance_percent"],
                "delta_accepted": row["accepted"] - control["accepted"],
                "decision_ms": row["decision_distribution_ms"],
                "delta_decision_mean_ms": row["decision_distribution_ms"]["mean"]
                - control["decision_distribution_ms"]["mean"],
                "latency_ms": row["latency_distribution_ms"],
                "utilization_percent": row["time_weighted_utilization_percent"],
                "model_sha256": evidence["model_sha256"],
                "topology_sha256": row["topology_sha256"],
                "workload_sha256": row["workload_sha256"],
                "qwen_successful_calls": evidence["qwen_successful_calls"],
                "qwen_failed_calls": evidence["qwen_failed_calls"],
                "qwen_fallback_calls": evidence["qwen_fallback_calls"],
                "guidance_applied_requests": evidence["guidance_applied_requests"],
            })
    profiles = {
        "before": profile_rows(args.profile_dir / "before.pstats"),
        "after": profile_rows(args.profile_dir / "after_fixed.pstats"),
    }
    training = load(args.profile_dir / "training_execution.json")
    metadata = load(Path(training["model"]).with_suffix(".training.json"))
    workflow_started = datetime.strptime(
        args.profile_dir.name.removeprefix("decision_optimization_"), "%Y%m%d_%H%M%S_%f"
    ).astimezone()
    report_at = datetime.now().astimezone()
    workflow_wall_s = (report_at - workflow_started).total_seconds()
    executions = {label: load(directories[label] / "execution.json")
                  for label in documents if label != "baseline"}
    validation = {
        name: read_log(args.profile_dir / name)
        for name in ["pytest_final.log", "ruff_final.log"]
    }
    if not re.search(r"\d+ passed", validation["pytest_final.log"]):
        raise ValueError("Missing successful pytest evidence")
    if "failed" in validation["pytest_final.log"] or "All checks passed!" not in validation[
        "ruff_final.log"
    ]:
        raise ValueError("Final validation is not clean")
    hashes = {}
    for path in [
        Path("src/muar_sfc/core/orchestration/placement.py"),
        Path("src/muar_sfc/algorithms/mappo/env_mappo.py"),
        Path("src/muar_sfc/algorithms/mappo/muvr.py"),
        Path("tests/test_mappo_routing_equivalence.py"),
        args.baseline / "mappo_qwen.json",
    ]:
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {"rows": rows, "equivalence": equivalence, "profiles": profiles,
               "training": training, "training_metadata": metadata,
               "executions": executions, "source_hashes": hashes,
               "validation": validation,
               "workflow_started_at": workflow_started.isoformat(),
               "report_at": report_at.isoformat(), "workflow_wall_s": workflow_wall_s,
               "directories": {label: str(path) for label, path in directories.items()}}
    save_new(output / "comparison.json", json.dumps(summary, indent=2, allow_nan=False))
    text = [
        "# MUAR-SFC — treino completo e otimização (21/10/2026)",
        "", "Relatório técnico local; resultados [MEDIDO], não números dos artigos.",
        "## Condições e limites", "",
        "25 servidores nominais, SLA SFC 6 ms, admissão imediata e reservas atômicas; "
        "quatro atores (48/7/196), scale 0.0, threads 4; sementes pareadas 42/43/44 × 50.",
        "D1: comparação principal, attempts 16. D2: exploratória TopoRAG fixo "
        "kappa=0.2/top_k=4/prior_scale=1/contexto off, memória nova por semente. "
        "D3: exploratória attempts 32, TopoRAG desligado. Todas com Qwen ao vivo "
        "assíncrono qwen2.5:1.5b em localhost:11434/v1 (não replay).",
        "Três sementes são amostra pequena; sem alegação de ganho causal do Qwen. "
        "Treino completo versus reduzido é a comparação de política; disponibilidade "
        "assíncrona de guidance também pode variar com o tempo real de execução. "
        "D2/D3 alteram condições exploratórias. Rejeições físicas continuam possíveis; "
        "não se promete 100%. CPU simulada não é CPU do PC, decisão não é latência "
        "SFC nem motion-to-photon.",
        "", "## Aceitação e tempo de decisão por semente", "",
        "| Condição | Seed | Aceitas/50 | Δ aceitas | Média ms | p95 | p99 | Máx | Δ média ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        d = row["decision_ms"]
        text.append(f"| {row['condition']} | {row['seed']} | {row['accepted']} | "
                    f"{row['delta_accepted']:+d} | {d['mean']:.2f} | {d['p95']:.2f} | "
                    f"{d['p99']:.2f} | {d['max']:.2f} | {row['delta_decision_mean_ms']:+.2f} |")
    text += ["", "Agregação descritiva de três sementes (mesmo denominador):"]
    for label in documents:
        group = [r for r in rows if r["condition"] == label]
        total = sum(r["accepted"] for r in group)
        mean = sum(r["decision_ms"]["mean"] for r in group) / 3
        text.append(f"- {label}: {total}/150 ({total / 1.5:.2f}%), média decisão {mean:.2f} ms.")
    text += ["", "## Qwen: consultas válidas/falhas/fallback e requisições orientadas", "",
             "| Condição | Seed | Sucesso | Falhas | Fallback | Com guidance |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['condition']} | {row['seed']} | {row['qwen_successful_calls']} | "
                    f"{row['qwen_failed_calls']} | {row['qwen_fallback_calls']} | "
                    f"{row['guidance_applied_requests']} |")
    text += ["", "## Perfil antes/depois e equivalência", "",
             "Sem Qwen, checkpoint reduzido, semente 42 × 50; "
             "tempos instrumentados por cProfile, não diretamente comparáveis aos tempos ao vivo.",
             f"Decisão média: {before['mean_decision_s'] * 1000:.2f} → "
             f"{after['mean_decision_s'] * 1000:.2f} ms "
             f"({100 * (1 - after['mean_decision_s'] / before['mean_decision_s']):.2f}% menor).",
             "Todos os traces de propostas/selected_host/tentativas são exatamente iguais; "
             "aceitações, todas as amostras de latência SFC, utilização e hashes também iguais. "
             "Fixture determinística compara referência sem cache contra versão otimizada.",
             "Uma busca Dijkstra de origem única é compartilhada só para a primeira "
             "dependência de cada candidato sobre snapshot idêntico. Demais dependências "
             "usam estado residual próprio; toda reserva ainda verifica banda. "
             "Cache invalida por grafo/função/exclusões/reset. RNG não é consumido pelo cache. "
             "COW existente foi preservado; não foi ampliado para Net2 sem prova de isolamento."]
    for label, profile in profiles.items():
        text += ["", f"### Top-10 acumulado — {label}", "",
                 "| Função | Chamadas | Próprio s | Acumulado s | Acumulado ms/req |",
                 "|---|---:|---:|---:|---:|"]
        for item in [entry for entry in profile if entry["function"] != "<module>"][:10]:
            text.append(f"| {item['file']}:{item['line']} {item['function']} | "
                        f"{item['calls']} | {item['self_s']:.3f} | "
                        f"{item['cumulative_s']:.3f} | {item['cumulative_s'] * 20:.2f} |")
    text += ["", "Acumulados são aninhados e NÃO devem ser somados. "
             "Custos próprios e lista completa constam em comparison.json.",
             "", "### Hotspots por item (acumulado, segundos)", "",
             "| Função | Antes | Depois | Redução |", "|---|---:|---:|---:|"]
    for name in ["_evaluated_candidate", "_route_inputs", "available_path",
                 "_dijkstra_multisource", "deepcopy", "select_actions", "get_observations"]:
        totals = [sum(x["cumulative_s"] for x in profiles[label] if x["function"] == name)
                  for label in ["before", "after"]]
        a, b = totals
        text.append(f"| {name} | {a:.3f} | {b:.3f} | {100 * (1-b/a) if a else 0:.1f}% |")
    text += ["", "## Treino e duração", "",
             f"Treino CPU completo: {training['wall_seconds']:.2f} s; 300 episódios, "
             "100 demonstrações, seed 137, threads 4. Nenhum retreino de Qwen.",
             f"Seleção existente por validação escolheu episódio {metadata['selected_episode']}; "
             f"foram executadas {metadata['MAPPO_updates']} atualizações MAPPO, mas o checkpoint "
             f"selecionado contém {metadata['selected_checkpoint_MAPPO_updates']} atualizações. "
             "Treino completo não significa obrigatoriamente salvar a política "
             "do último episódio.",
             f"Checkpoint: `{training['model']}`; SHA-256 `{metadata['checkpoint_sha256']}`."]
    for label, execution in executions.items():
        text.append(f"- {label}: {execution['wall_seconds']:.2f} s; "
                    f"comando e timestamps: `{directories[label] / 'execution.json'}`.")
    total_seconds = training['wall_seconds'] + sum(e['wall_seconds'] for e in executions.values())
    text.append(f"Total treino + D1/D2/D3: {total_seconds:.2f} s "
                "(não inclui perfil, análise, testes ou espera da sessão).")
    text.append(f"Tempo de parede desde a criação da pasta da fase A até este relatório: "
                f"{workflow_wall_s:.2f} s; inclui implementação, testes, perfil, "
                "interrupção D2 e espera/coordenação da sessão.")
    text += ["", "## GPU e fontes oficiais", "",
             "[MEDIDO] torch 2.10.0+cpu; cuda.is_available=False, device_count=0, hip=None. "
             "RX 6600 é gfx1032 e consta sem suporte Runtime/HIP SDK Windows. "
             "A matriz PyTorch Windows 7.2.1 lista gfx1201/gfx1200/gfx1100/gfx1101; "
             "o SDK também lista gfx1102, mas isso não comprova suporte PyTorch. "
             "Logo a GPU do usuário não é aproveitada nesta configuração Windows 11; "
             "troca de SO seria necessária para investigar alternativas Linux (sem garantia "
             "automática de suporte da RX 6600). Nenhum driver/ROCm/modelo foi instalado.",
             "Fontes consultadas em 21/10/2026 (tabelas GPU):",
             "- https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html",
             "- https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html",
             "", "## Artefatos e verificações", "",
             f"- Perfil e logs: `{args.profile_dir}` (before.pstats e after_fixed.pstats).",
             f"- Avaliações, logs, manifests e JSON por semente: `{args.evaluations}`.",
             f"- D2 efetiva: `{directories['D2_exploratory_toporag']}`.",
             f"- D3 efetiva: `{directories['D3_exploratory_attempts32']}`.",
             "- A primeira D2 foi interrompida pelo limite de 120 s da ferramenta; "
             "artefatos parciais preservados e excluídos da comparação. "
             "D2 e D3 efetivas usam nova pasta, sem retomada nem sobrescrita.",
             f"- Baseline lido sem alteração: `{args.baseline}`.",
             "- `comparison.json`: hashes, pareamento verificado, tempos próprios/acumulados "
             "completos, métricas SFC e utilização, treino e execuções.",
             "- Validação: `.venv\\Scripts\\python.exe -m pytest -q` e "
             "`.venv\\Scripts\\python.exe -m ruff check <arquivos Python alterados>`. "
             f"Resultado: {validation['pytest_final.log'].strip()}; "
             f"Ruff: {validation['ruff_final.log'].strip()}.",
             "- Sem Git; sem alteração do checkpoint protegido, capacidades, SLA, máscaras, "
             "load_balance_scale ou resgate heurístico.",
             "", "Relatório gerado de dados locais; "
             "revisão humana científica permanece necessária."]
    save_new(output / "report.md", "\n".join(text) + "\n")
    print(output)


if __name__ == "__main__":
    main()
