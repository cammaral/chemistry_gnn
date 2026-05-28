#!/usr/bin/env python3
"""
Descriptor MLP baseline.

This is not the Harris-Net reproduction. It is the first stronger classical baseline for
the GNN paper idea: molecule descriptors + electron energy -> sigma(E).

It uses the same official Figshare data and the same train/test partitions.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from common_ml import (
    RESULTS_DIR, ARTICLE_TABLE2, parse_seeds, load_table_and_curves, split_names,
    make_scalar_dataset, minmax_fit, minmax_transform, minmax_inverse,
    compute_curve_metrics, print_verdict, save_json,
)
from molecular_graphs import graph_descriptors, validate_against_metadata
from plotting_utils import plot_mean_predictions, plot_loss_curves


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64, depth: int = 3):
        super().__init__()
        layers = []
        d = n_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_descriptor_matrix(meta: pd.DataFrame, names: np.ndarray, energy: np.ndarray, use_ip: bool, use_graph_desc: bool):
    meta_idx = meta.set_index("molecule")
    rows = []
    for m, e in zip(names, energy):
        base = {
            "C": float(meta_idx.loc[m, "C"]),
            "H": float(meta_idx.loc[m, "H"]),
            "N": float(meta_idx.loc[m, "N"]),
            "O": float(meta_idx.loc[m, "O"]),
            "energy_eV": float(e),
        }
        if use_ip:
            base["Ip_eV"] = float(meta_idx.loc[m, "Ip_eV"])
        if use_graph_desc:
            base.update(graph_descriptors(m))
        rows.append(base)
    df = pd.DataFrame(rows)
    return df.to_numpy(float), list(df.columns)


def train_one(args, seed: int, meta, energies, curves, train_names, test_names):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_mols, train_e, y_train = make_scalar_dataset(train_names, energies, curves)
    test_mols, test_e, y_test = make_scalar_dataset(test_names, energies, curves)

    X_train, feature_names = build_descriptor_matrix(meta, train_mols, train_e, args.include_ip, args.graph_descriptors)
    X_test, _ = build_descriptor_matrix(meta, test_mols, test_e, args.include_ip, args.graph_descriptors)

    xmn, xmx, xden = minmax_fit(X_train)
    ymn, ymx, yden = float(y_train.min()), float(y_train.max()), float(y_train.max() - y_train.min())
    if abs(yden) < 1e-12:
        yden = 1.0
    Xs = minmax_transform(X_train, xmn, xden)
    Xtest_s = minmax_transform(X_test, xmn, xden)
    ys = minmax_transform(y_train.reshape(-1, 1), np.array([ymn]), np.array([yden])).reshape(-1)

    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    model = MLP(Xs.shape[1], hidden=args.hidden, depth=args.depth).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(Xs, dtype=dtype, device=device)
    yt = torch.tensor(ys, dtype=dtype, device=device)
    Xtt = torch.tensor(Xtest_s, dtype=dtype, device=device)
    losses = []
    for ep in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            losses.append({"seed": seed, "epoch": ep, "loss": float(loss.detach().cpu())})
            if not args.quiet:
                print(f"seed={seed} epoch={ep}/{args.epochs} loss={losses[-1]['loss']:.6e}")
    with torch.no_grad():
        pred_s = model(Xtt).detach().cpu().numpy()
    pred = minmax_inverse(pred_s, ymn, yden)
    pred_curves = pred.reshape(len(test_names), len(energies))
    return pred_curves, pd.DataFrame(losses), feature_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partition", type=int, default=2, choices=[1, 2])
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--include-ip", action="store_true")
    p.add_argument("--graph-descriptors", action="store_true")
    p.add_argument("--float64", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    meta, energies, curves = load_table_and_curves()
    validate_against_metadata(meta)
    train_names, test_names = split_names(meta, args.partition)
    y_true = np.vstack([curves[m] for m in test_names])
    seeds = parse_seeds(args.seeds)

    tag = f"descriptor_mlp_partition{args.partition}_ip{int(args.include_ip)}_graphdesc{int(args.graph_descriptors)}_seeds{'-'.join(map(str,seeds))}"
    outdir = RESULTS_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)
    preds, loss_dfs = [], []
    feature_names = None
    for s in seeds:
        pred_curves, loss_df, feature_names = train_one(args, s, meta, energies, curves, train_names, test_names)
        preds.append(pred_curves)
        loss_dfs.append(loss_df)
    pred_arr = np.stack(preds, axis=0)
    mean_pred = pred_arr.mean(axis=0)

    metrics_seed = []
    for seed, pred in zip(seeds, pred_arr):
        df = compute_curve_metrics(y_true, pred, test_names)
        df.insert(0, "seed", seed)
        metrics_seed.append(df)
    metrics_seed_df = pd.concat(metrics_seed, ignore_index=True)
    summary = compute_curve_metrics(y_true, mean_pred, test_names)
    article = ARTICLE_TABLE2.get((args.partition, args.include_ip), {})
    summary["article_table2_max_percent"] = summary["molecule"].map(article)
    summary["delta_vs_article_pp"] = summary["max_percent_difference"] - summary["article_table2_max_percent"]

    metrics_seed_df.to_csv(outdir / "metrics_by_seed.csv", index=False)
    summary.to_csv(outdir / "summary_mean_prediction.csv", index=False)
    pd.concat(loss_dfs, ignore_index=True).to_csv(outdir / "loss_all_seeds.csv", index=False)
    np.savez(outdir / "predictions_all_seeds.npz", energies=energies, test_names=np.array(test_names, dtype=object), y_true=y_true, predictions=pred_arr)
    save_json(outdir / "run_config.json", vars(args) | {"feature_names": feature_names, "train_names": train_names, "test_names": test_names})
    plot_mean_predictions(outdir, energies, test_names, y_true, pred_arr, title=tag, article_max=article)
    plot_loss_curves(outdir, loss_dfs, title=tag)
    print_verdict(summary, tag, article_mode=False)
    print(f"\nSaved results in: {outdir}")

if __name__ == "__main__":
    main()
