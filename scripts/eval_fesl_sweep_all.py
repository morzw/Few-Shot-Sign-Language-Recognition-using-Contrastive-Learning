import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slr.configs import ensure_dir, load_fesl_eval_config
from slr.data import apply_extraction_condition, load_fesl_support_query
from slr.metrics import knn_support_query_accuracy
from slr.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all local pretrained model_weights on FESL and plot combined sweep."
    )
    parser.add_argument("--config", default="configs/fesl_eval.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--models-root",
        default="model_weights",
        help="Directory that contains model folders (e.g., model_weights/lstm/model_weights.hdf5).",
    )
    parser.add_argument(
        "--output-name",
        default="results_fesl_sweep_all",
        help="Base name for output files (without extension).",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_model_paths(models_root: Path):
    model_names = ["lstm", "bi_lstm", "gru"]
    paths = {}
    for name in model_names:
        p = models_root / name / "model_weights.hdf5"
        if p.exists():
            paths[name] = p
        else:
            print(f"Missing checkpoint: {p}")
    if not paths:
        raise FileNotFoundError(f"No model_weights.hdf5 files found under {models_root}")
    return paths


def main():
    args = parse_args()
    cfg = load_fesl_eval_config(args.config)
    set_seed(cfg.seed)

    models_root = (ROOT / args.models_root).resolve()
    model_paths = resolve_model_paths(models_root)
    out_dir = ensure_dir(cfg.output_root)

    rows = []
    for model_name, model_path in model_paths.items():
        model = load_model(str(model_path))
        for support, query in cfg.sizes:
            Xs, ys, Xq, yq, _ = load_fesl_support_query(cfg.fesl_root, support=support, query=query)
            Xs = apply_extraction_condition(Xs, cfg.extraction_condition)
            Xq = apply_extraction_condition(Xq, cfg.extraction_condition)

            support_emb = model.predict(Xs, verbose=0)
            query_emb = model.predict(Xq, verbose=0)
            _, acc = knn_support_query_accuracy(
                support_embeddings=support_emb,
                support_labels=ys,
                query_embeddings=query_emb,
                query_labels=yq,
                k=cfg.k,
            )
            rows.append(
                {
                    "model": model_name,
                    "k": cfg.k,
                    "support": support,
                    "query": query,
                    "accuracy": acc,
                    "model_path": str(model_path),
                }
            )
            print(f"[{model_name}] support={support}, query={query}, accuracy={acc:.4f}")

    df = pd.DataFrame(rows).sort_values(["model", "support"])

    csv_path = out_dir / f"{args.output_name}.csv"
    png_path = out_dir / f"{args.output_name}.png"
    svg_path = out_dir / f"{args.output_name}.svg"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5), dpi=220)
    for model_name, sub in df.groupby("model"):
        sub = sub.sort_values("support")
        plt.plot(
            sub["support"],
            sub["accuracy"],
            marker="o",
            markersize=6,
            linewidth=2.2,
            label=model_name.upper(),
        )

    plt.xlabel("Support shots per class")
    plt.ylabel("Accuracy")
    plt.title(f"FESL Few-Shot Sweep (k={cfg.k})")
    plt.xticks(sorted(df["support"].unique()))
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=420, bbox_inches="tight")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"Saved results to {csv_path}")
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {svg_path}")


if __name__ == "__main__":
    main()
