#!/usr/bin/env python3
"""Generate article-level reference tables and figures for Harris & Nepomuceno reproduction.

This script covers the parts of the paper that go beyond Table 2:
  - Figure 2: all interpolated experimental curves with partition test sets highlighted.
  - Figures 3/4/5-style panels from our result directories, when available.
  - Figure 6-style Net25 predictions using the official Figshare saved weights/biases.
  - CSV reference tables for Table 2 and Table 3 molecules.

It does not claim to digitize external literature curves used in Figure 6; the official Figshare ZIP
contains the all-25 trained weights, not all BEB/experimental comparison curves cited in Fig. 6.
"""
from __future__ import annotations
import argparse, zipfile, shutil, math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OFFICIAL_ZIP = ROOT / "official_mathematica" / "pmx_codes.zip"
OFFICIAL_DIR = ROOT / "official_mathematica" / "pmx_codes_extracted"
ARTICLE_DIR = RESULTS / "article_reference"

ARTICLE_TABLE2_ROWS = [
    (1, "Propanone", 14, 16),
    (1, "2-Methylpropanal", 4, 5),
    (1, "Hexan-3-one", 7, 7),
    (1, "3,3-Dimethylbutan-2-one", 5, 6),
    (1, "Methanol", 12, 9),
    (2, "Ethanal", 13, 14),
    (2, "Ethanol", 26, 23),
    (2, "Propanal", 6, 8),
    (2, "3-Methylbutan-2-one", 13, 12),
    (2, "Molecular Nitrogen", 30, 1940),
]

TABLE3_MOLECULES = [
    ("Alkanes", "Methane", "CH4", 1,4,0,0),
    ("Alkanes", "Ethane", "C2H6", 2,6,0,0),
    ("Alkanes", "Propane", "C3H8", 3,8,0,0),
    ("Alkanes", "Butane", "C4H10", 4,10,0,0),
    ("Alkanes", "Pentane", "C5H12", 5,12,0,0),
    ("Alkanes", "Hexane", "C6H14", 6,14,0,0),
    ("Alkanes", "Heptane", "C7H16", 7,16,0,0),
    ("Alkanes", "Octane", "C8H18", 8,18,0,0),
    ("Alkenes", "Ethene", "C2H4", 2,4,0,0),
    ("Alkenes", "Prop-1-ene", "C3H6", 3,6,0,0),
    ("Alkenes", "But-1-ene", "C4H8", 4,8,0,0),
    ("Alkenes", "Pent-1-ene", "C5H10", 5,10,0,0),
    ("Alkenes", "Hex-1-ene", "C6H12", 6,12,0,0),
    ("Alkenes", "Hept-1-ene", "C7H14", 7,14,0,0),
    ("Alkenes", "Oct-1-ene", "C8H16", 8,16,0,0),
    ("Ring structures", "Benzene", "C6H6", 6,6,0,0),
    ("Ring structures", "Cyclopropane", "C3H6", 3,6,0,0),
    ("Ring structures", "Cyclobutane", "C4H8", 4,8,0,0),
    ("Ring structures", "Cyclopentane", "C5H10", 5,10,0,0),
    ("Ring structures", "Vanillin", "C8H8O3", 8,8,0,3),
    ("Ring structures", "Naphthalene", "C10H8", 10,8,0,0),
    ("Ring structures", "Pyridine", "C5H5N", 5,5,1,0),
    ("Ring structures", "Pyrimidine", "C4H4N2", 4,4,2,0),
    ("Nucleotide bases", "Adenine", "C5H5N5", 5,5,5,0),
    ("Nucleotide bases", "Thymine", "C5H6N2O2", 5,6,2,2),
    ("Nucleotide bases", "Cytosine", "C4H5N3O", 4,5,3,1),
    ("Nucleotide bases", "Guanine", "C5H5N5O", 5,5,5,1),
]


def load_meta_curves():
    meta = pd.read_csv(DATA / "molecules_table1.csv")
    raw = pd.read_csv(DATA / "ionization_cross_sections_25.csv")
    energies = np.sort(raw.energy_eV.unique()).astype(float)
    curves = {m: g.sort_values("energy_eV").sigma_a0_2.to_numpy(float) for m, g in raw.groupby("molecule")}
    return meta, energies, curves


def save_article_reference_tables():
    ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
    table2 = pd.DataFrame(ARTICLE_TABLE2_ROWS, columns=["partition", "molecule", "Net20_max_percent", "Net20Ip_max_percent"])
    table2.to_csv(ARTICLE_DIR / "article_table2_reference.csv", index=False)
    table3 = pd.DataFrame(TABLE3_MOLECULES, columns=["category", "molecule", "formula", "C", "H", "N", "O"])
    table3.to_csv(ARTICLE_DIR / "article_table3_new_molecules.csv", index=False)
    return table2, table3


def figure2():
    meta, energies, curves = load_meta_curves()
    ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
    p1 = set(meta.loc[meta.partition1_test.astype(bool), "molecule"])
    p2 = set(meta.loc[meta.partition2_test.astype(bool), "molecule"])
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in meta["molecule"]:
        y = curves[m]
        if m in p1:
            ax.plot(energies, y, color="tab:red", lw=1.8, alpha=0.85, label="Partition 1 test" if "Partition 1 test" not in ax.get_legend_handles_labels()[1] else None)
        elif m in p2:
            ax.plot(energies, y, color="tab:blue", lw=1.8, alpha=0.85, label="Partition 2 test" if "Partition 2 test" not in ax.get_legend_handles_labels()[1] else None)
        else:
            ax.plot(energies, y, color="0.55", lw=0.9, alpha=0.50, label="Training pool" if "Training pool" not in ax.get_legend_handles_labels()[1] else None)
    ax.set_xlabel("Electron energy (eV)")
    ax.set_ylabel(r"Ionization cross section ($a_0^2$)")
    ax.set_title("Figure 2-style: official interpolated experimental datasets")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTICLE_DIR / "figure2_official_interpolated_data.png", dpi=220)
    plt.close(fig)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensure_official_extracted():
    if not OFFICIAL_DIR.exists():
        OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OFFICIAL_ZIP) as z:
            z.extractall(OFFICIAL_DIR)
    return OFFICIAL_DIR / "pmx_codes"


def official_net25_predict():
    pmx = ensure_official_extracted()
    wdir = pmx / "weights.biases.all25.mols.400.epochs"
    W1 = np.loadtxt(wdir / "weights_layer_1.csv", delimiter=",")
    W2 = np.loadtxt(wdir / "weights_layer_2.csv", delimiter=",")
    b1 = np.loadtxt(wdir / "biases_layer_1.csv", delimiter=",")
    b2 = np.loadtxt(wdir / "biases_layer_2.csv", delimiter=",")
    in_min, in_max = np.loadtxt(wdir / "input_max_min.csv", delimiter=",")
    out_min, out_max = np.loadtxt(wdir / "output_max_min.csv", delimiter=",")
    energies = np.linspace(25.0, 100.0, 101)
    table3 = pd.DataFrame(TABLE3_MOLECULES, columns=["category", "molecule", "formula", "C", "H", "N", "O"])
    X = table3[["C", "H", "N", "O"]].to_numpy(float)
    Xs = np.abs(0.05 + 0.9 * (X - in_min) / (in_max - in_min))
    yscaled = sigmoid(Xs) @ W1.T + b1
    yscaled = sigmoid(yscaled) @ W2.T + b2
    y = np.abs((yscaled - 0.05) * (out_max - out_min) / 0.9 + out_min)
    long_rows = []
    for i, row in table3.iterrows():
        for e, yy in zip(energies, y[i]):
            long_rows.append({"category": row.category, "molecule": row.molecule, "formula": row.formula, "energy_eV": e, "sigma_a0_2": yy})
    pred_df = pd.DataFrame(long_rows)
    pred_df.to_csv(ARTICLE_DIR / "official_net25_table3_predictions.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.ravel()
    for ax, cat in zip(axes, ["Alkanes", "Alkenes", "Ring structures", "Nucleotide bases"]):
        sub = pred_df[pred_df.category == cat]
        for mol, g in sub.groupby("molecule"):
            ax.plot(g.energy_eV, g.sigma_a0_2, lw=1.5, label=mol)
        ax.set_title(cat)
        ax.set_xlabel("Electron energy (eV)")
        ax.set_ylabel(r"Predicted $\sigma$ ($a_0^2$)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Figure 6-style: official Figshare Net25 saved-weight predictions")
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(ARTICLE_DIR / "figure6_official_net25_predictions.png", dpi=220)
    plt.close(fig)
    return pred_df


def find_prediction_dir(partition: int, net: int, ip: bool, optimizer: str = "adam"):
    suffix = f"partition{partition}_Net{net}{'Ip' if ip else ''}_{optimizer}_seeds"
    dirs = [p for p in RESULTS.glob(suffix + "*") if (p / "predictions_all_seeds.npz").exists()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.stat().st_mtime)[-1]


def load_pred(partition, net, ip, optimizer="adam"):
    d = find_prediction_dir(partition, net, ip, optimizer)
    if d is None:
        return None
    z = np.load(d / "predictions_all_seeds.npz", allow_pickle=True)
    # different scripts use actual or y_true naming
    y_true = z["actual"] if "actual" in z.files else z["y_true"]
    test_names = z["test_names"].astype(object).tolist()
    return d, z["energies"], test_names, y_true, z["predictions"]


def plot_article_grid(partition: int, include_ip_overlay: bool, name: str):
    nets = [10, 15, 20]
    base = load_pred(partition, 20, False)
    if base is None:
        print(f"[skip] Cannot build {name}: missing Net20 partition {partition} results")
        return
    _, energies, test_names, y_true, _ = base
    fig, axes = plt.subplots(len(test_names), 3, figsize=(13, 2.45 * len(test_names)), sharex=True)
    if len(test_names) == 1:
        axes = axes[None, :]
    for col, net in enumerate(nets):
        noip = load_pred(partition, net, False)
        ip = load_pred(partition, net, True) if include_ip_overlay else None
        for row, mol in enumerate(test_names):
            ax = axes[row, col]
            ax.scatter(energies, y_true[row], s=10, color="black", alpha=0.7, label="experiment" if row == 0 and col == 0 else None)
            if noip is not None:
                _, _, names_noip, _, preds = noip
                idx = names_noip.index(mol)
                mean = preds.mean(axis=0)[idx]
                std = preds.std(axis=0)[idx]
                ax.plot(energies, mean, color="tab:blue", lw=1.6, label="no Ip mean" if row == 0 and col == 0 else None)
                if preds.shape[0] > 1:
                    ax.fill_between(energies, mean-std, mean+std, color="tab:blue", alpha=0.17)
            if ip is not None:
                _, _, names_ip, _, preds_ip = ip
                idx = names_ip.index(mol)
                mean = preds_ip.mean(axis=0)[idx]
                std = preds_ip.std(axis=0)[idx]
                ax.plot(energies, mean, color="tab:red", lw=1.4, label="Ip mean" if row == 0 and col == 0 else None)
                if preds_ip.shape[0] > 1:
                    ax.fill_between(energies, mean-std, mean+std, color="tab:red", alpha=0.12)
            if row == 0:
                ax.set_title(f"Net{net}")
            if col == 0:
                ax.set_ylabel(mol + "\n" + r"$\sigma$ ($a_0^2$)")
            ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Energy (eV)")
    handles, labels = axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(name)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(ARTICLE_DIR / (name.lower().replace(" ", "_").replace("/", "_") + ".png"), dpi=220)
    plt.close(fig)


def compare_all_summaries_to_article():
    rows = []
    article = pd.DataFrame(ARTICLE_TABLE2_ROWS, columns=["partition", "molecule", "article_Net20", "article_Net20Ip"])
    for p in RESULTS.glob("*/summary_mean_prediction.csv"):
        model = p.parent.name
        df = pd.read_csv(p)
        if "max_percent_difference" not in df and "mean_curve_max_percent_difference" in df:
            df["max_percent_difference"] = df["mean_curve_max_percent_difference"]
        if "mape_percent" not in df and "mean_curve_mape_percent" in df:
            df["mape_percent"] = df["mean_curve_mape_percent"]
        # infer partition from model name
        part = 1 if "partition1" in model else (2 if "partition2" in model else None)
        if part is None:
            continue
        for _, r in df.iterrows():
            ref = article[(article.partition == part) & (article.molecule == r.molecule)]
            if len(ref):
                rows.append({
                    "model": model,
                    "partition": part,
                    "molecule": r.molecule,
                    "mape_percent": r.get("mape_percent", np.nan),
                    "max_percent_difference": r["max_percent_difference"],
                    "article_Net20": float(ref.article_Net20.iloc[0]),
                    "article_Net20Ip": float(ref.article_Net20Ip.iloc[0]),
                    "delta_vs_article_Net20": r["max_percent_difference"] - float(ref.article_Net20.iloc[0]),
                    "delta_vs_article_Net20Ip": r["max_percent_difference"] - float(ref.article_Net20Ip.iloc[0]),
                })
    if rows:
        out = pd.DataFrame(rows)
        out.to_csv(RESULTS / "comparison_to_article_table2_all_models.csv", index=False)
        overall = out.groupby("model", as_index=False).agg(
            mean_mape_percent=("mape_percent", "mean"),
            mean_max_percent_difference=("max_percent_difference", "mean"),
            mean_delta_vs_article_Net20=("delta_vs_article_Net20", "mean"),
            worst_max_percent_difference=("max_percent_difference", "max"),
        ).sort_values("mean_max_percent_difference")
        overall.to_csv(RESULTS / "comparison_to_article_table2_overall.csv", index=False)
        print("\nArticle Table 2 comparison written to results/.")
        print(overall.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fig2", action="store_true")
    ap.add_argument("--fig345", action="store_true")
    ap.add_argument("--fig6", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.all or not any([args.fig2, args.fig345, args.fig6, args.compare]):
        args.fig2 = args.fig345 = args.fig6 = args.compare = True
    save_article_reference_tables()
    if args.fig2:
        figure2()
    if args.fig6:
        official_net25_predict()
    if args.fig345:
        plot_article_grid(1, False, "Figure 3-style partition 1 no Ip")
        plot_article_grid(1, True, "Figure 4-style partition 1 no Ip vs Ip")
        plot_article_grid(2, True, "Figure 5-style partition 2 no Ip vs Ip")
    if args.compare:
        compare_all_summaries_to_article()
    print(f"Saved article reference outputs in {ARTICLE_DIR}")

if __name__ == "__main__":
    main()
