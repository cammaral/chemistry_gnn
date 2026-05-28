#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_mean_predictions(
    outdir: Path,
    energies: np.ndarray,
    test_names: List[str],
    y_true_curves: np.ndarray,
    pred_curves_by_seed: np.ndarray,
    title: str,
    article_max: Optional[Dict[str, float]] = None,
):
    outdir.mkdir(parents=True, exist_ok=True)
    mean_pred = pred_curves_by_seed.mean(axis=0)
    std_pred = pred_curves_by_seed.std(axis=0)
    n = len(test_names)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)
    for i, mol in enumerate(test_names):
        yt = y_true_curves[i]
        yp = mean_pred[i]
        ys = std_pred[i]
        ax = axes[i, 0]
        ax.plot(energies, yt, "o", ms=3, label="experiment/interpolated")
        ax.plot(energies, yp, "-", lw=2, label="mean prediction")
        ax.fill_between(energies, yp - ys, yp + ys, alpha=0.25, label="±1 std seeds")
        ax.set_title(mol)
        ax.set_xlabel("Electron energy (eV)")
        ax.set_ylabel(r"Cross section ($a_0^2$)")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)
        pct = np.abs((yp - yt) / np.maximum(np.abs(yt), 1e-12)) * 100.0
        ax2 = axes[i, 1]
        ax2.plot(energies, pct, "-", lw=2)
        if article_max and mol in article_max:
            ax2.axhline(article_max[mol], ls="--", lw=1, label=f"article max {article_max[mol]:.0f}%")
            ax2.legend(fontsize=8)
        ax2.set_title(f"Percent error: max={pct.max():.1f}%, mean={pct.mean():.1f}%")
        ax2.set_xlabel("Electron energy (eV)")
        ax2.set_ylabel("Absolute percent error (%)")
        ax2.grid(alpha=0.25)
    fig.suptitle(title, y=0.995, fontsize=14)
    fig.tight_layout()
    path = outdir / "figure_mean_predictions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loss_curves(outdir: Path, loss_dfs: List[pd.DataFrame], title: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for df in loss_dfs:
        seed = df["seed"].iloc[0] if "seed" in df else "?"
        ax.plot(df["epoch"], df["loss"], lw=1.5, label=f"seed {seed}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "loss_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
