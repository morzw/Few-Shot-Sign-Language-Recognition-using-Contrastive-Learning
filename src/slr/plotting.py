from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_fesl_sweep(
    df: pd.DataFrame,
    output_png: Path,
    output_svg: Path,
    title: str,
):
    plt.figure(figsize=(8, 5), dpi=220)
    plt.plot(
        df["support"],
        df["accuracy"],
        marker="o",
        markersize=6,
        linewidth=2.2,
    )
    plt.xlabel("Support shots per class")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(sorted(df["support"].unique()))
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=420, bbox_inches="tight")
    plt.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close()

