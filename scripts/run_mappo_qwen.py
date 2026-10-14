"""Read the editable TOML and execute only when explicitly launched by the user."""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

from muar_sfc.experiment_config import command_plan, load_config

ROOT = Path(__file__).resolve().parents[1]


def check_qwen_available(config):
    """Read model inventory only; never generate tokens during preflight."""
    endpoint = config.qwen.api_base_url.rstrip("/") + "/models"
    with urllib.request.urlopen(endpoint, timeout=min(10, config.qwen.api_timeout)) as response:
        models = json.load(response)
    if config.qwen.model not in {model["id"] for model in models.get("data", [])}:
        raise ValueError(f"Modelo {config.qwen.model} não encontrado no Ollama")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/mappo_qwen.toml")
    parser.add_argument("--check", action="store_true", help="Validar sem executar ou gravar")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    plan = command_plan(config, ROOT, sys.executable)
    e = config.execution
    model, output = (ROOT / e.model_path).resolve(), (ROOT / e.output_dir).resolve()
    print(f"Etapa: {e.stage}; dispositivo: CPU; threads: {e.threads}", flush=True)
    print(f"Modelo: {model}", flush=True)
    if e.stage != "train":
        v = config.evaluation
        print(
            f"Somente MAPPO + Qwen: {v.seed_count} sementes, {v.requests} requisições "
            f"cada, {v.seed_count * v.requests} no total.",
            flush=True,
        )
        print(f"Resultados: {output}", flush=True)
    if args.check:
        print("Configuração válida. Nenhum treino, chamada Qwen ou simulação executados.")
        return
    if e.stage == "evaluate" and not model.is_file():
        parser.error("Checkpoint inexistente: execute a etapa train primeiro")
    if not e.overwrite:
        if e.stage != "evaluate" and (
            model.exists() or model.with_suffix(".training.json").exists()
        ):
            parser.error("Modelo já existe. Escolha outro model_path ou overwrite=true")
        if e.stage != "train" and (
            (output / "mappo_qwen.json").exists() or (output / "experiment_config.json").exists()
        ):
            parser.error("Resultados já existem. Escolha outro output_dir ou overwrite=true")
    if e.stage != "train":
        try:
            check_qwen_available(config)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"Confira o servidor/modelo Ollama antes de executar: {exc}")
    # Save the effective parameters even for interrupted runs, before launching.
    output.mkdir(parents=True, exist_ok=True)
    manifest = "training_config.json" if e.stage == "train" else "experiment_config.json"
    (output / manifest).write_text(
        json.dumps(
            {
                "configuration": config.model_dump(mode="json"),
                "commands": plan,
                "config_path": str(args.config.resolve()),
                "project_root": str(ROOT),
            },
            indent=2,
        )
    )
    for command in plan:
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    print("Execução concluída. Peça a análise dos resultados quando desejar.")


if __name__ == "__main__":
    main()
