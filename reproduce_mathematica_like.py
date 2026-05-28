#!/usr/bin/env python3
"""
Faithful PyTorch port of Harris & Nepomuceno's Mathematica PMX notebook.

This script follows the official Figshare Mathematica notebook logic:
  - uses the official converted_interped data files (already converted here to CSV),
  - train/test partitions from Table 1,
  - input = [C,H,N,O] or [C,H,N,O,Ip],
  - output = 101 cross-section values on 25--100 eV,
  - global MinMax scaling to [0.05, 0.95] with Abs, matching the notebook,
  - network: Sigmoid -> Linear(Floor[nmols/3]) -> Sigmoid -> Linear(101),
  - full-batch training on rules input -> output.

It also adds reproducibility utilities that the original notebook did not automate:
  - multiple seeds/trials,
  - mean/std plots over seeds,
  - comparison against Table 2 max-error values,
  - final printed verdict: close or not close.

Note: exact numeric equivalence to Mathematica NetTrain is not guaranteed because
Mathematica's default optimizer/initializer are not reproduced bit-for-bit in PyTorch.
For the closest possible reproduction, run the included Wolfram script.
"""
from __future__ import annotations

import argparse
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

ARTICLE_TABLE2 = {
    (1, False): {
        "Propanone": 14.0,
        "2-Methylpropanal": 4.0,
        "Hexan-3-one": 7.0,
        "3,3-Dimethylbutan-2-one": 5.0,
        "Methanol": 12.0,
    },
    (1, True): {
        "Propanone": 16.0,
        "2-Methylpropanal": 5.0,
        "Hexan-3-one": 7.0,
        "3,3-Dimethylbutan-2-one": 6.0,
        "Methanol": 9.0,
    },
    (2, False): {
        "Ethanal": 13.0,
        "Ethanol": 26.0,
        "Propanal": 6.0,
        "3-Methylbutan-2-one": 13.0,
        "Molecular Nitrogen": 30.0,
    },
    (2, True): {
        "Ethanal": 14.0,
        "Ethanol": 23.0,
        "Propanal": 8.0,
        "3-Methylbutan-2-one": 12.0,
        "Molecular Nitrogen": 1940.0,
    },
}


@dataclass
class Scaling:
    in_min: float
    in_max: float
    out_min: float
    out_max: float


def scale_abs(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """Mathematica notebook scale[x_,min_,max_] := Abs[0.05 + 0.9*(x-min)/(max-min)]."""
    denom = xmax - xmin
    if abs(denom) < 1e-15:
        denom = 1.0
    return np.abs(0.05 + 0.9 * (x - xmin) / denom)


def unscale_abs(y: np.ndarray, ymin: float, ymax: float) -> np.ndarray:
    """Mathematica notebook unscale[y_,min_,max_] := Abs[(y-0.05)*(max-min)/0.9 + min]."""
    return np.abs((y - 0.05) * (ymax - ymin) / 0.9 + ymin)


class HarrisNet(nn.Module):
    """NetChain[{ElementwiseLayer[LogisticSigmoid], LinearLayer[Floor[nmols/3]], ElementwiseLayer[LogisticSigmoid], LinearLayer[101]}]."""
    def __init__(self, n_in: int, nmols: int, n_out: int):
        super().__init__()
        n_hidden = math.floor(nmols / 3)
        if n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")
        self.net = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(n_in, n_hidden),
            nn.Sigmoid(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data(include_ip: bool, partition: int, nmols: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], List[str], List[str], Scaling]:
    meta = pd.read_csv(DATA_DIR / "molecules_table1.csv")
    raw = pd.read_csv(DATA_DIR / "ionization_cross_sections_25.csv")

    energies = np.sort(raw["energy_eV"].unique())
    curves: Dict[str, np.ndarray] = {}
    for mol, g in raw.groupby("molecule"):
        g2 = g.sort_values("energy_eV")
        if len(g2) != len(energies):
            raise ValueError(f"{mol} has {len(g2)} points, expected {len(energies)}")
        curves[mol] = g2["sigma_a0_2"].to_numpy(dtype=float)

    test_col = f"partition{partition}_test"
    test_names = meta.loc[meta[test_col].astype(bool), "molecule"].tolist()
    training_pool = meta.loc[~meta[test_col].astype(bool), "molecule"].tolist()

    if nmols > len(training_pool):
        raise ValueError(f"net/nmols={nmols} but partition {partition} has only {len(training_pool)} training molecules")

    # Mathematica uses RandomSample[trainingpool, nmols].
    rng = np.random.default_rng(seed)
    train_names = list(rng.choice(training_pool, size=nmols, replace=False))

    feature_cols = ["C", "H", "N", "O"] + (["Ip_eV"] if include_ip else [])
    meta_idx = meta.set_index("molecule")
    X_train = meta_idx.loc[train_names, feature_cols].to_numpy(dtype=float)
    Y_train = np.vstack([curves[m] for m in train_names]).astype(float)
    X_test = meta_idx.loc[test_names, feature_cols].to_numpy(dtype=float)
    Y_test = np.vstack([curves[m] for m in test_names]).astype(float)

    # Official notebook: {inmin,inmax}=MinMax[input]; {outmin,outmax}=MinMax[output]
    # For the provided notebook and saved all25 weights, these are global scalar min/max.
    scaling = Scaling(
        in_min=float(np.min(X_train)),
        in_max=float(np.max(X_train)),
        out_min=float(np.min(Y_train)),
        out_max=float(np.max(Y_train)),
    )

    pack = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "energies": energies,
    }
    return meta, energies, pack, train_names, test_names, scaling


def train_one(args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    meta, energies, pack, train_names, test_names, scaling = load_data(args.include_ip, args.partition, args.net, seed)

    Xs = scale_abs(pack["X_train"], scaling.in_min, scaling.in_max)
    Ys = scale_abs(pack["Y_train"], scaling.out_min, scaling.out_max)
    Xtest_s = scale_abs(pack["X_test"], scaling.in_min, scaling.in_max)

    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    model = HarrisNet(Xs.shape[1], args.net, Ys.shape[1]).to(device=device, dtype=dtype)
    X_t = torch.tensor(Xs, dtype=dtype, device=device)
    Y_t = torch.tensor(Ys, dtype=dtype, device=device)
    Xtest_t = torch.tensor(Xtest_s, dtype=dtype, device=device)

    loss_fn = nn.MSELoss()
    losses = []

    if args.optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(1, args.epochs + 1):
            opt.zero_grad(set_to_none=True)
            pred = model(X_t)
            loss = loss_fn(pred, Y_t)
            loss.backward()
            opt.step()
            if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
                losses.append((ep, float(loss.detach().cpu())))
                if not args.quiet:
                    print(f"seed={seed} epoch={ep}/{args.epochs} loss={losses[-1][1]:.8e}")
    elif args.optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        for ep in range(1, args.epochs + 1):
            opt.zero_grad(set_to_none=True)
            pred = model(X_t)
            loss = loss_fn(pred, Y_t)
            loss.backward()
            opt.step()
            if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
                losses.append((ep, float(loss.detach().cpu())))
                if not args.quiet:
                    print(f"seed={seed} epoch={ep}/{args.epochs} loss={losses[-1][1]:.8e}")
    elif args.optimizer.lower() == "lbfgs":
        # Useful diagnostic option. Not the Mathematica default, but sometimes closer for small full-batch networks.
        opt = torch.optim.LBFGS(model.parameters(), lr=args.lr, max_iter=args.epochs, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")
        calls = {"n": 0}
        def closure():
            opt.zero_grad(set_to_none=True)
            pred = model(X_t)
            loss = loss_fn(pred, Y_t)
            loss.backward()
            calls["n"] += 1
            if calls["n"] == 1 or calls["n"] % args.log_every == 0:
                losses.append((calls["n"], float(loss.detach().cpu())))
                if not args.quiet:
                    print(f"seed={seed} lbfgs_call={calls['n']} loss={losses[-1][1]:.8e}")
            return loss
        opt.step(closure)
        with torch.no_grad():
            final_loss = loss_fn(model(X_t), Y_t)
        losses.append((calls["n"], float(final_loss.detach().cpu())))
    else:
        raise ValueError("optimizer must be adam, sgd, or lbfgs")

    with torch.no_grad():
        pred_scaled = model(Xtest_t).detach().cpu().numpy()
    pred = unscale_abs(pred_scaled, scaling.out_min, scaling.out_max)
    actual = pack["Y_test"]

    rows = []
    for i, mol in enumerate(test_names):
        denom = np.where(np.abs(actual[i]) < 1e-15, np.nan, actual[i])
        pct = np.abs((actual[i] - pred[i]) / denom) * 100.0
        rows.append({
            "seed": seed,
            "molecule": mol,
            "mape_percent": float(np.nanmean(pct)),
            "max_percent_difference": float(np.nanmax(pct)),
            "energy_at_max_error_eV": float(energies[int(np.nanargmax(pct))]),
        })

    return {
        "seed": seed,
        "train_names": train_names,
        "test_names": test_names,
        "energies": energies,
        "prediction": pred,
        "actual": actual,
        "metrics": pd.DataFrame(rows),
        "losses": pd.DataFrame(losses, columns=["step", "loss"]),
        "scaling": scaling.__dict__,
    }


def plot_mean_predictions(run_dir: Path, energies: np.ndarray, test_names: List[str], actual: np.ndarray, preds: np.ndarray, title: str):
    n = len(test_names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    for i, ax in enumerate(axes):
        ax.scatter(energies, actual[i], s=18, label="Experiment/interpolated")
        ax.plot(energies, mean[i], label="Prediction mean")
        if preds.shape[0] > 1:
            ax.fill_between(energies, mean[i] - std[i], mean[i] + std[i], alpha=0.22, label="±1 seed std" if i == 0 else None)
        pct = np.abs((actual[i] - mean[i]) / actual[i]) * 100
        ax.text(0.02, 0.83, f"{test_names[i]}\nmean max error={np.max(pct):.1f}%", transform=ax.transAxes, fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(r"$\sigma$ ($a_0^2$)")
        if i == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Electron energy (eV)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = run_dir / f"figure_{title.replace(' ', '_').replace('/', '-')}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_losses(run_dir: Path, losses_all: List[pd.DataFrame], seeds: List[int], title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    for df, seed in zip(losses_all, seeds):
        ax.plot(df["step"], df["loss"], label=f"seed {seed}", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Training step / epoch")
    ax.set_ylabel("MSE loss on scaled outputs")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if len(seeds) <= 12:
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = run_dir / "loss_curves.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def final_verdict(summary_mean: pd.DataFrame, partition: int, include_ip: bool, tolerance_pp: float):
    ref = ARTICLE_TABLE2.get((partition, include_ip), {})
    rows = []
    for _, r in summary_mean.iterrows():
        mol = r["molecule"]
        art = ref.get(mol, np.nan)
        delta = r["mean_curve_max_percent_difference"] - art if not np.isnan(art) else np.nan
        close = bool(abs(delta) <= tolerance_pp) if not np.isnan(delta) else False
        rows.append({**r.to_dict(), "article_table2_max_percent": art, "delta_vs_article_pp": delta, "close_to_article": close})
    comp = pd.DataFrame(rows)
    n_close = int(comp["close_to_article"].sum()) if len(comp) else 0
    total = len(comp)
    print("\nFINAL CHECK: ficou próximo do artigo?")
    print("=" * 100)
    if total:
        print(comp[["molecule", "mean_curve_mape_percent", "mean_curve_max_percent_difference", "article_table2_max_percent", "delta_vs_article_pp", "close_to_article"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 100)
    if n_close == total and total > 0:
        print(f"VEREDITO: SIM. Ficou próximo do artigo para {n_close}/{total} moléculas dentro de ±{tolerance_pp:g} pontos percentuais.")
    elif n_close >= max(1, total - 1):
        print(f"VEREDITO: PARCIALMENTE MUITO BOM. Ficou próximo para {n_close}/{total}; investigar as moléculas marcadas como False.")
    elif n_close >= total / 2:
        print(f"VEREDITO: PARCIAL. Ficou próximo para {n_close}/{total}; ainda há diferenças relevantes de otimização/seed.")
    else:
        print(f"VEREDITO: NÃO. Ficou próximo só para {n_close}/{total}; use o script Mathematica oficial ou ajuste otimizador/seeds.")
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=int, choices=[1, 2], default=2)
    ap.add_argument("--net", type=int, default=20, help="Number of molecules used for training: 10, 15, or 20 in the article tests.")
    ap.add_argument("--epochs", type=int, default=400000)
    ap.add_argument("--seeds", type=str, default="1", help="Comma-separated seeds, e.g. 1,2,3,4,5")
    ap.add_argument("--include-ip", action="store_true", help="Use C,H,N,O,Ip input instead of only C,H,N,O")
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "lbfgs"])
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--log-every", type=int, default=10000)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="Torch CPU threads. Keep 1 on laptops to avoid small-matrix slowdown.")
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tolerance-pp", type=float, default=10.0, help="Tolerance in percentage points vs Table 2.")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    RESULTS_DIR.mkdir(exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tag = f"partition{args.partition}_Net{args.net}{'Ip' if args.include_ip else ''}_{args.optimizer}_seeds{'-'.join(map(str,seeds))}"
    run_dir = RESULTS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {tag}")
    print(f"epochs={args.epochs}, lr={args.lr}, seeds={seeds}")

    outputs = []
    for seed in seeds:
        outputs.append(train_one(args, seed))

    energies = outputs[0]["energies"]
    test_names = outputs[0]["test_names"]
    actual = outputs[0]["actual"]
    preds = np.stack([o["prediction"] for o in outputs], axis=0)

    # Save raw arrays.
    np.savez(run_dir / "predictions_all_seeds.npz", energies=energies, test_names=np.array(test_names), actual=actual, predictions=preds, seeds=np.array(seeds))

    # Metrics per seed.
    metrics = pd.concat([o["metrics"] for o in outputs], ignore_index=True)
    metrics.to_csv(run_dir / "metrics_by_seed.csv", index=False)

    # Metrics of the mean prediction curve, matching figures 3--5 style.
    mean_pred = preds.mean(axis=0)
    rows = []
    for i, mol in enumerate(test_names):
        pct = np.abs((actual[i] - mean_pred[i]) / actual[i]) * 100
        rows.append({
            "molecule": mol,
            "mean_curve_mape_percent": float(np.mean(pct)),
            "mean_curve_max_percent_difference": float(np.max(pct)),
            "mean_curve_energy_at_max_error_eV": float(energies[int(np.argmax(pct))]),
            "across_seed_mean_of_max_percent_difference": float(metrics.loc[metrics.molecule == mol, "max_percent_difference"].mean()),
            "across_seed_std_of_max_percent_difference": float(metrics.loc[metrics.molecule == mol, "max_percent_difference"].std(ddof=0)),
        })
    summary_mean = pd.DataFrame(rows)
    summary_mean.to_csv(run_dir / "summary_mean_prediction.csv", index=False)

    # Losses.
    for o in outputs:
        o["losses"].to_csv(run_dir / f"loss_seed{o['seed']}.csv", index=False)
    plot_losses(run_dir, [o["losses"] for o in outputs], seeds, tag)

    fig_path = plot_mean_predictions(run_dir, energies, test_names, actual, preds, title=tag)
    comp = final_verdict(summary_mean, args.partition, args.include_ip, args.tolerance_pp)
    comp.to_csv(run_dir / "comparison_to_article_table2.csv", index=False)

    config = vars(args).copy()
    config["seeds"] = seeds
    config["train_names_by_seed"] = {str(o["seed"]): o["train_names"] for o in outputs}
    config["test_names"] = test_names
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved results in: {run_dir}")
    print(f"Main plot: {fig_path}")


if __name__ == "__main__":
    main()
