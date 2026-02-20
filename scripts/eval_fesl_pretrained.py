import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slr.configs import ensure_dir, load_fesl_eval_config
from slr.data import apply_extraction_condition, load_fesl_support_query
from slr.metrics import knn_support_query_accuracy
from slr.models import load_model
from slr.plotting import plot_fesl_sweep


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained MESL model on FESL support/query sweep.")
    parser.add_argument(
        "--config",
        default="configs/fesl_eval.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    args = parse_args()
    cfg = load_fesl_eval_config(args.config)
    set_seed(cfg.seed)

    out_dir = ensure_dir(cfg.output_root)
    model = load_model(cfg.pretrained_model_path)

    rows = []
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
                "run_name": cfg.run_name,
                "k": cfg.k,
                "support": support,
                "query": query,
                "accuracy": acc,
                "model_path": cfg.pretrained_model_path,
            }
        )
        print(f"[support={support}, query={query}] accuracy={acc:.4f}")

    df = pd.DataFrame(rows).sort_values("support")
    csv_path = out_dir / f"{cfg.run_name}_fesl_sweep.csv"
    png_path = out_dir / f"{cfg.run_name}_fesl_sweep.png"
    svg_path = out_dir / f"{cfg.run_name}_fesl_sweep.svg"
    df.to_csv(csv_path, index=False)

    plot_fesl_sweep(
        df=df,
        output_png=png_path,
        output_svg=svg_path,
        title=f"FESL Few-Shot Sweep ({cfg.run_name}, k={cfg.k})",
    )
    print(f"Saved results to {csv_path}")
    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {svg_path}")


if __name__ == "__main__":
    main()

