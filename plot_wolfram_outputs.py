#!/usr/bin/env python3
"""Aggregate and plot outputs created by run_official_wolfram.wl."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str, help="e.g. results/wolfram_partition2_Net20_seeds1-2-3")
    ap.add_argument("--tolerance-pp", type=float, default=10.0)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    metrics = []
    preds = []
    for seed_dir in sorted(run_dir.glob("seed_*")):
        m = pd.read_csv(seed_dir / "metrics.csv")
        p = pd.read_csv(seed_dir / "predictions_long.csv")
        metrics.append(m)
        preds.append(p)
    if not metrics:
        raise FileNotFoundError(f"No seed_*/metrics.csv found in {run_dir}")
    metrics = pd.concat(metrics, ignore_index=True)
    pred_long = pd.concat(preds, ignore_index=True)
    metrics.to_csv(run_dir / "metrics_all_seeds.csv", index=False)

    molecules = list(dict.fromkeys(pred_long["molecule"].tolist()))
    energies = np.sort(pred_long["energy_eV"].unique())

    rows = []
    fig, axes = plt.subplots(len(molecules), 1, figsize=(10, 2.4 * len(molecules)), sharex=True)
    if len(molecules) == 1:
        axes = [axes]
    for ax, mol in zip(axes, molecules):
        g = pred_long[pred_long["molecule"] == mol]
        actual = g.groupby("energy_eV")["actual_sigma_a0_2"].first().reindex(energies).to_numpy()
        pred_mat = []
        for seed, gs in g.groupby("seed"):
            pred_mat.append(gs.sort_values("energy_eV")["pred_sigma_a0_2"].to_numpy())
        pred_mat = np.vstack(pred_mat)
        mean = pred_mat.mean(axis=0)
        std = pred_mat.std(axis=0)
        pct = np.abs((actual - mean) / actual) * 100
        article = metrics.loc[metrics["molecule"] == mol, "article_table2_max_percent"].dropna()
        article_val = float(article.iloc[0]) if len(article) else np.nan
        rows.append({
            "molecule": mol,
            "mean_curve_mape_percent": float(np.mean(pct)),
            "mean_curve_max_percent_difference": float(np.max(pct)),
            "energy_at_max_error_eV": float(energies[int(np.argmax(pct))]),
            "article_table2_max_percent": article_val,
            "delta_vs_article_pp": float(np.max(pct) - article_val) if not np.isnan(article_val) else np.nan,
            "close_to_article": bool(abs(np.max(pct) - article_val) <= args.tolerance_pp) if not np.isnan(article_val) else False,
        })
        ax.scatter(energies, actual, s=18, label="Experiment/interpolated")
        ax.plot(energies, mean, label="Wolfram NetTrain mean")
        if pred_mat.shape[0] > 1:
            ax.fill_between(energies, mean - std, mean + std, alpha=0.22, label="±1 seed std")
        ax.text(0.02, 0.82, f"{mol}\nmean max error={np.max(pct):.1f}%", transform=ax.transAxes, fontsize=11)
        ax.set_ylabel(r"$\sigma$ ($a_0^2$)")
        ax.grid(True, alpha=0.25)
        if mol == molecules[0]:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Electron energy (eV)")
    fig.suptitle(run_dir.name + ": mean prediction over seeds")
    fig.tight_layout(rect=[0,0,1,0.98])
    fig.savefig(run_dir / "figure_mean_predictions_wolfram.png", dpi=200)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary_mean_prediction_vs_article.csv", index=False)
    print("\nFINAL CHECK: ficou próximo do artigo? [Wolfram outputs]")
    print("="*100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("="*100)
    nclose = int(summary["close_to_article"].sum())
    total = len(summary)
    if nclose == total:
        print(f"VEREDITO: SIM. Ficou próximo para {nclose}/{total} moléculas.")
    elif nclose >= max(1, total-1):
        print(f"VEREDITO: PARCIALMENTE MUITO BOM. Ficou próximo para {nclose}/{total}.")
    else:
        print(f"VEREDITO: PARCIAL/NÃO. Ficou próximo para {nclose}/{total}; verifique seeds, epochs e versão do Wolfram.")
    print(f"Saved: {run_dir / 'figure_mean_predictions_wolfram.png'}")


if __name__ == "__main__":
    main()
