"""Explicit experiment settings, loaded from one JSON file."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Signal dimensions are inferred from the prepared data file."""

    num_classes: int = 5
    eeg_channels: int = 62
    eeg_samples: int = 250
    et_samples: int = 250
    et_features: int = 4
    student_dropout: float = 0.4
    private_dropout: float = 0.4

    def __post_init__(self):
        if self.num_classes < 2 or self.eeg_channels < 1 or self.eeg_samples < 32:
            raise ValueError(
                "Require >=2 classes, >=1 EEG channel and >=32 EEG samples."
            )
        if self.et_samples < 4 or self.et_features < 1:
            raise ValueError("Require >=4 ET samples and >=1 ET feature.")
        if not all(0 <= x < 1 for x in (self.student_dropout, self.private_dropout)):
            raise ValueError("Dropout must be in [0, 1).")


@dataclass
class ExperimentConfig:
    data: str = "data/trials.h5"
    images: str = "data/images"
    splits: str = "data/splits.json"
    output: str = "outputs"
    num_classes: int = 5
    seed: int = 2025
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0002
    lr_gamma: float = 0.99
    workers: int = 0
    device: str = "auto"
    pretrained: bool = True
    student_dropout: float = 0.4
    private_dropout: float = 0.4
    teacher_metric: str = "image_mean"
    teacher_losses: dict[str, float] = field(default_factory=lambda: {"loss_MI": 0.1})
    student_losses: dict[str, float] = field(
        default_factory=lambda: {
            "loss_distill_weights_eeg": 2.0,
            "loss_distill_feature_eeg": 2.0,
            "loss_distill_weights_et": 2.0,
            "loss_distill_feature_et": 2.0,
            "loss_dec": 0.1,
        }
    )

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        config = cls(**json.loads(path.read_text(encoding="utf-8")))
        # Paths are relative to the configuration file, independent of the shell cwd.
        for name in ("data", "images", "splits", "output"):
            value = Path(getattr(config, name))
            setattr(config, name, str((path.parent / value).resolve()))
        if config.epochs < 1 or config.batch_size < 1 or config.workers < 0:
            raise ValueError("epochs/batch_size must be positive; workers must be >=0.")
        if config.learning_rate <= 0 or config.weight_decay < 0 or config.lr_gamma <= 0:
            raise ValueError("Invalid optimizer settings.")
        return config

    def to_dict(self):
        return asdict(self)
