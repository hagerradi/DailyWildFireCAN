"""Experiment configuration (YAML -> dataclasses).

See ``configs/default.yaml`` for the values used in the paper. Unknown YAML keys are
ignored with a warning so that older config files still load.
"""
from __future__ import annotations

import warnings

import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class DatasetConfig:
    height: int = 256
    width: int = 256
    channels: int = 50
    num_classes: int = 1

@dataclass
class ModelConfig:
    input_channels: int = 50
    num_classes: int = 1
    hidden_features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_skip_connections: bool = True
    use_activation_after_upsampling: bool = False
    architecture: str = "unet"
    utae_decoder_widths: list[int] | None = None
    utae_out_conv_channels: list[int] = field(default_factory=lambda: [32])
    utae_agg_mode: str = "att_group"
    utae_encoder_norm: str = "group"
    utae_n_head: int = 16
    utae_d_model: int = 256
    utae_d_k: int = 4
    utae_pad_value: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10


@dataclass
class CometConfig:
    enabled: bool = False
    project_name: str = ""
    workspace: str = ""
    experiment_name: str = "default-run"
    experiment_tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    comet: CometConfig = field(default_factory=CometConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a Config from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Populated Config instance.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            seed=raw.get("seed", 42),
            dataset=_build(DatasetConfig, raw.get("dataset"), "dataset"),
            model=_build(ModelConfig, raw.get("model"), "model"),
            training=_build(TrainingConfig, raw.get("training"), "training"),
            comet=_build(CometConfig, raw.get("comet"), "comet"),
        )


def _build(cls, values: dict | None, section: str):
    values = dict(values or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        warnings.warn(f"Ignoring unknown '{section}' config keys: {unknown}", stacklevel=3)
    return cls(**{k: v for k, v in values.items() if k in known})
