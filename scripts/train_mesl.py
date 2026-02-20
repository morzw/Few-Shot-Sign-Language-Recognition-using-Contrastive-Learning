import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slr.configs import ensure_dir, load_mesl_train_config
from slr.data import apply_extraction_condition, load_mesl_dataset
from slr.metrics import knn_leave_one_out_accuracy
from slr.models import create_embedding_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train MESL embedding model with triplet loss.")
    parser.add_argument(
        "--config",
        default="configs/mesl_train.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    args = parse_args()
    cfg = load_mesl_train_config(args.config)
    set_seed(cfg.seed)

    output_root = ensure_dir(cfg.output_root)
    X, y, class_names = load_mesl_dataset(cfg.dataset_root)
    X = apply_extraction_condition(X, cfg.extraction_condition)

    if len(X) == 0:
        raise RuntimeError("No MESL samples were loaded. Check dataset_root.")

    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.seed)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train,
                y_train,
                test_size=cfg.validation_split,
                random_state=cfg.seed,
                stratify=y_train,
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train,
                y_train,
                test_size=cfg.validation_split,
                random_state=cfg.seed,
                stratify=None,
            )

        model = create_embedding_model(cfg.model_name, (X.shape[1], X.shape[2]))
        model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss=tfa.losses.TripletSemiHardLoss(distance_metric="squared-L2"),
        )

        fold_dir = ensure_dir(str(output_root / cfg.model_name / str(fold)))
        ckpt_path = fold_dir / "model_weights.hdf5"
        callbacks = [
            ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.lr_factor,
                patience=cfg.lr_patience,
                min_delta=1e-4,
                mode="min",
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                min_delta=1e-3,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        test_embeddings = model.predict(X_test, verbose=0)
        _, knn_acc = knn_leave_one_out_accuracy(test_embeddings, y_test, k=cfg.k_eval)
        best_val = float(np.min(history.history["val_loss"]))
        rows.append(
            {
                "model": cfg.model_name,
                "fold": fold,
                "best_val_loss": best_val,
                "knn_test_acc": knn_acc,
                "checkpoint": str(ckpt_path),
            }
        )
        print(f"[Fold {fold}] best_val_loss={best_val:.6f}, knn_test_acc={knn_acc:.4f}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = output_root / f"{cfg.model_name}_kfold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved fold metrics to {metrics_path}")


if __name__ == "__main__":
    main()


