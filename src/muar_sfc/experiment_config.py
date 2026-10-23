"""Validated editable configuration for CPU MAPPO training and Qwen evaluation."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Execution(Section):
    stage: Literal["train", "evaluate", "train_then_evaluate"] = "train_then_evaluate"
    overwrite: bool = False
    threads: int = Field(default=4, ge=1)
    model_path: Path = Path("rl_saved_models/MAPPO_muvr_meu_treino.pt")
    output_dir: Path = Path("results/meu_treino_33_sementes_300")


class Training(Section):
    episodes: int = Field(default=300, ge=5)
    demonstrations: int = Field(default=100, ge=1)
    seed: int = Field(default=137, ge=0, le=2**32 - 1)
    lr_actor: float = Field(default=1e-4, gt=0)
    lr_critic: float = Field(default=5e-4, gt=0)
    clip_param: float = Field(default=0.2, gt=0, lt=1)
    ppo_epoch: int = Field(default=4, ge=1)
    gamma: float = Field(default=0.99, ge=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    entropy_coef: float = Field(default=0.01, ge=0)
    value_loss_coef: float = Field(default=0.5, ge=0)
    max_grad_norm: float = Field(default=0.5, gt=0)
    update_every: int = Field(default=5, ge=1)
    cloning_epochs: int = Field(default=15, ge=1)
    cloning_batch_size: int = Field(default=128, ge=1)


class Evaluation(Section):
    requests: int = Field(default=300, ge=1)
    seed_start: int = Field(default=42, ge=0)
    seed_count: int = Field(default=33, ge=1)
    search_attempts: int = Field(default=16, ge=1)
    load_balance_scale: float = Field(default=2.0, ge=0)

    @property
    def seeds(self):
        return list(range(self.seed_start, self.seed_start + self.seed_count))


class Qwen(Section):
    model: str = "qwen2.5:1.5b"
    api_base_url: str = "http://localhost:11434/v1"
    api_timeout: float = Field(default=120, gt=0)
    max_tokens: int = Field(default=256, ge=1)
    planner_interval: float = Field(default=60, gt=0)
    guidance_max_age: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def valid_qwen(self):
        if not self.model.lower().startswith("qwen"):
            raise ValueError("Use um modelo Qwen neste experimento")
        if not self.api_base_url.startswith(("http://", "https://")):
            raise ValueError("api_base_url precisa começar com http:// ou https://")
        return self


class TopoRAG(Section):
    enabled: bool = False
    top_k: int = Field(default=4, ge=1)
    kappa: float = Field(default=0.0, ge=0, lt=1)
    confidence_mode: Literal["fixed", "lagged_expost"] = "fixed"
    kappa_max: float = Field(default=0.9, ge=0, lt=1)
    prior_scale: float = Field(default=1.0, ge=0)
    prompt_context: bool = False


class ExperimentConfig(Section):
    execution: Execution = Field(default_factory=Execution)
    training: Training = Field(default_factory=Training)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    qwen: Qwen = Field(default_factory=Qwen)
    toporag: TopoRAG = Field(default_factory=TopoRAG)

    @model_validator(mode="after")
    def disjoint_seeds(self):
        # Keep evaluation seeds separate from generated training/held-out topologies.
        if self.execution.stage != "evaluate":
            t = self.training
            intervals = [
                (200000 + t.seed, 200000 + t.seed),
                (200000 + t.seed * 1000, 200000 + t.seed * 1000 + t.demonstrations - 1),
                (500000 + t.seed * 1000, 500000 + t.seed * 1000 + t.episodes - 1),
                (800000, 800029),
                (900000, 900029),
            ]
            start = self.evaluation.seed_start
            end = start + self.evaluation.seed_count - 1
            if any(start <= upper and end >= lower for lower, upper in intervals):
                raise ValueError("Sementes de avaliação coincidem com treino/validação")
        return self


def load_config(path):
    with Path(path).open("rb") as stream:
        return ExperimentConfig.model_validate(tomllib.load(stream))


def command_plan(config, root, python):
    """Build arguments only; never run training, Ollama or simulation here."""
    root = Path(root)
    execution = config.execution
    model = (root / execution.model_path).resolve()
    output = (root / execution.output_dir).resolve()
    commands = []
    if execution.stage in ("train", "train_then_evaluate"):
        args = [str(python), str(root / "scripts/train_mappo_muvr.py")]
        for key, value in config.training.model_dump().items():
            args += ["--" + key.replace("_", "-"), str(value)]
        args += ["--threads", str(execution.threads), "--output", str(model)]
        commands.append(args)
    if execution.stage in ("evaluate", "train_then_evaluate"):
        e, q = config.evaluation, config.qwen
        commands.append(
            [
                str(python),
                str(root / "scripts/benchmark_mappo_qwen_muvr.py"),
                "--requests",
                str(e.requests),
                "--seeds",
                *map(str, e.seeds),
                "--variants",
                "mappo_qwen",
                "--model",
                str(model),
                "--output",
                str(output),
                "--threads",
                str(execution.threads),
                "--search-attempts",
                str(e.search_attempts),
                "--load-balance-scale",
                str(e.load_balance_scale),
                "--qwen-model",
                q.model,
                "--api-base-url",
                q.api_base_url,
                "--api-timeout",
                str(q.api_timeout),
                "--max-tokens",
                str(q.max_tokens),
                "--planner-interval",
                str(q.planner_interval),
                "--guidance-max-age",
                str(q.guidance_max_age),
            ]
        )
        for key, value in config.toporag.model_dump().items():
            flag = "--toporag-" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    commands[-1].append(flag)
            else:
                commands[-1] += [flag, str(value)]
    return commands
