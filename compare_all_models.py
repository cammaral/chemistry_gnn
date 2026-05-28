#!/usr/bin/env python3
"""Collect all result summaries and create a model-comparison table/plot."""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    args = p.parse_args()
    rows = []
    for path in sorted(args.results_dir.glob("*/summary_mean_prediction.csv")):
        model = path.parent.name
        df = pd.read_csv(path)
        df.insert(0, "model", model)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No summary_mean_prediction.csv files found under {args.results_dir}")
    all_df = pd.concat(rows, ignore_index=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.results_dir / "comparison_all_models_by_molecule.csv", index=False)
    overall = all_df.groupby("model", as_index=False).agg(
        mean_mape_percent=("mape_percent", "mean"),
        mean_max_percent_difference=("max_percent_difference", "mean"),
        worst_molecule_max_percent_difference=("max_percent_difference", "max"),
    ).sort_values("mean_max_percent_difference")
    overall.to_csv(args.results_dir / "comparison_all_models_overall.csv", index=False)
    print("\nMODEL COMPARISON")
    print("=" * 100)
    print(overall.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(overall)), 5))
    ax.bar(overall["model"], overall["mean_max_percent_difference"])
    ax.set_ylabel("Mean of per-molecule max percent error (%)")
    ax.set_title("Model comparison")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.results_dir / "comparison_all_models.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison files in: {args.results_dir}")

if __name__ == "__main__":
    main()
