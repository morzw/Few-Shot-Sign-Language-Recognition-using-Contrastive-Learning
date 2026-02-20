from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class MeslTrainConfig:
    dataset_root: str
    output_root: str = "artifacts/mesl_train"
    model_name: str = "gru"  # lstm | bi_lstm | gru
    extraction_condition: Optional[str] = None  # None | hands | hands_and_pose | hands_and_face
    k_folds: int = 10
    epochs: int = 200
    batch_size: int = 256
    validation_split: float = 0.2
    learning_rate: float = 1e-4
    lr_factor: float = 0.2
    lr_patience: int = 5
    early_stopping_patience: int = 22
    seed: int = 42
    k_eval: int = 3


@dataclass
class FeslEvalConfig:
    fesl_root: str
    pretrained_model_path: str
    output_root: str = "artifacts/fesl_eval"
    extraction_condition: Optional[str] = None
    k: int = 5
    sizes: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 9), (3, 7), (5, 5), (7, 3), (9, 1)]
    )
    seed: int = 42
    run_name: str = "pretrained_model"


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_mesl_train_config(path: str) -> MeslTrainConfig:
    data = _load_yaml(path)
    return MeslTrainConfig(**data)


def load_fesl_eval_config(path: str) -> FeslEvalConfig:
    data = _load_yaml(path)
    if "sizes" in data:
        data["sizes"] = [tuple(x) for x in data["sizes"]]
    return FeslEvalConfig(**data)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

