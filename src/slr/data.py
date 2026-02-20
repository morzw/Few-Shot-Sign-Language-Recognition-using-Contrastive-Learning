from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _numeric_dirs(root: Path) -> List[Path]:
    dirs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            idx = int(p.name)
        except ValueError:
            continue
        dirs.append((idx, p))
    return [p for _, p in sorted(dirs, key=lambda x: x[0])]


def _list_classes(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def _collect_landmark_files_for_class(class_dir: Path) -> List[Path]:
    direct_samples = _numeric_dirs(class_dir)
    if direct_samples and (direct_samples[0] / "landmarks.npy").exists():
        return [s / "landmarks.npy" for s in direct_samples if (s / "landmarks.npy").exists()]

    files: List[Path] = []
    signer_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for signer in signer_dirs:
        for sample in _numeric_dirs(signer):
            f = sample / "landmarks.npy"
            if f.exists():
                files.append(f)
    return files


def _load_npy_files(files: List[Path]) -> np.ndarray:
    arrays = [np.load(str(f)) for f in files]
    return np.asarray(arrays)


def load_mesl_dataset(dataset_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(dataset_root)
    class_dirs = _list_classes(root)
    class_names = np.array([d.name for d in class_dirs])

    x_list: List[np.ndarray] = []
    y_list: List[int] = []
    for label, class_dir in enumerate(class_dirs):
        files = _collect_landmark_files_for_class(class_dir)
        if not files:
            continue
        x = _load_npy_files(files)
        x_list.append(x)
        y_list.extend([label] * len(x))

    X = np.concatenate(x_list, axis=0) if x_list else np.empty((0, 30, 1662))
    y = np.asarray(y_list, dtype=np.int32)
    return X, y, class_names


def load_fesl_support_query(
    fesl_root: str,
    support: int,
    query: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(fesl_root)
    class_dirs = _list_classes(root)
    class_names = np.array([d.name for d in class_dirs])

    x_support: List[np.ndarray] = []
    y_support: List[int] = []
    x_query: List[np.ndarray] = []
    y_query: List[int] = []

    needed = support + query
    for label, class_dir in enumerate(class_dirs):
        files = _collect_landmark_files_for_class(class_dir)
        if len(files) < needed:
            raise ValueError(
                f"Class '{class_dir.name}' has {len(files)} samples, but {needed} required "
                f"for support={support}, query={query}."
            )
        support_files = files[:support]
        query_files = files[support : support + query]

        xs = _load_npy_files(support_files)
        xq = _load_npy_files(query_files)

        x_support.append(xs)
        y_support.extend([label] * len(xs))
        x_query.append(xq)
        y_query.extend([label] * len(xq))

    Xs = np.concatenate(x_support, axis=0)
    ys = np.asarray(y_support, dtype=np.int32)
    Xq = np.concatenate(x_query, axis=0)
    yq = np.asarray(y_query, dtype=np.int32)
    return Xs, ys, Xq, yq, class_names


def apply_extraction_condition(X: np.ndarray, condition: Optional[str]) -> np.ndarray:
    if condition is None:
        return X
    if condition == "hands":
        return X[:, :, 1536:]
    if condition == "hands_and_pose":
        poses = X[:, :, :132]
        hands = X[:, :, 1536:]
        return np.concatenate([poses, hands], axis=2)
    if condition == "hands_and_face":
        return X[:, :, 132:]
    raise ValueError(
        "Unsupported extraction_condition. Use one of: None, hands, hands_and_pose, hands_and_face."
    )

