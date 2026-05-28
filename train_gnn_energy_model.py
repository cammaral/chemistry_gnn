#!/usr/bin/env python3
"""
Pure-PyTorch GNN model for the Harris-Nepomuceno electron-impact ionization cross sections.

Question tested here:
    Does molecular connectivity improve prediction compared with composition-only inputs?

Model:
    molecular graph -> message-passing encoder -> molecular embedding
    [embedding, normalized energy] -> MLP head -> sigma(E)

No PyTorch Geometric and no RDKit are required. Graphs are hard-coded in molecular_graphs.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common_ml import (
    RESULTS_DIR, ARTICLE_TABLE2, parse_seeds, load_table_and_curves, split_names,
    minmax_inverse, compute_curve_metrics, print_verdict, save_json,
)
from molecular_graphs import build_graphs, validate_against_metadata
from plotting_utils import plot_mean_predictions, plot_loss_curves


class SimpleGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden: int = 64, layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.self_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.neigh_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input(x))
        for ws, wn in zip(self.self_layers, self.neigh_layers):
            neigh = torch.bmm(adj_norm, h)
            h = self.act(ws(h) + wn(neigh))
            h = self.dropout(h)
            h = h * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled


class GNNRegressor(nn.Module):
    def __init__(self, node_dim: int, hidden: int, mp_layers: int, head_hidden: int, head_depth: int, dropout: float):
        super().__init__()
        self.encoder = SimpleGraphEncoder(node_dim, hidden=hidden, layers=mp_layers, dropout=dropout)
        layers = []
        d = hidden + 1  # molecular embedding + normalized energy
        for _ in range(head_depth):
            layers += [nn.Linear(d, head_hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = head_hidden
        layers += [nn.Linear(d, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x, adj_norm, mask, energy_scaled):
        emb = self.encoder(x, adj_norm, mask)
        z = torch.cat([emb, energy_scaled.unsqueeze(-1)], dim=1)
        return self.head(z).squeeze(-1)


def prepare_graph_tensors(meta: pd.DataFrame, device, dtype):
    validate_against_metadata(meta)
    graphs = build_graphs()
    names = meta["molecule"].tolist()
    arrays = {m: graphs[m].to_arrays() for m in names}
    max_n = max(x.shape[0] for x, _ in arrays.values())
    node_dim = next(iter(arrays.values()))[0].shape[1]
    X = np.zeros((len(names), max_n, node_dim), dtype=np.float32)
    A = np.zeros((len(names), max_n, max_n), dtype=np.float32)
    M = np.zeros((len(names), max_n), dtype=np.float32)
    for i, m in enumerate(names):
        x, adj = arrays[m]
        n = x.shape[0]
        X[i, :n] = x
        # self loops + symmetric degree normalization
        a = adj.copy()
        a += np.eye(n, dtype=np.float32)
        d = np.sum(a, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        a_norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
        A[i, :n, :n] = a_norm
        M[i, :n] = 1.0
    name_to_idx = {m: i for i, m in enumerate(names)}
    return (
        torch.tensor(X, dtype=dtype, device=device),
        torch.tensor(A, dtype=dtype, device=device),
        torch.tensor(M, dtype=dtype, device=device),
        name_to_idx,
        node_dim,
    )


def make_samples(mol_names: List[str], energies: np.ndarray, curves: Dict[str, np.ndarray], name_to_idx: Dict[str, int]):
    mol_idx, e_vals, y_vals = [], [], []
    for m in mol_names:
        for e, y in zip(energies, curves[m]):
            mol_idx.append(name_to_idx[m])
            e_vals.append(float(e))
            y_vals.append(float(y))
    return np.array(mol_idx, dtype=np.int64), np.array(e_vals, dtype=np.float32), np.array(y_vals, dtype=np.float32)


def loss_value(pred, target, mode: str):
    if mode == "mse":
        return torch.mean((pred - target) ** 2)
    if mode == "relative":
        return torch.mean(((pred - target) / torch.clamp(torch.abs(target), min=1e-3)) ** 2)
    raise ValueError(f"Unknown loss mode: {mode}")


def train_one(args, seed: int, meta, energies, curves, train_names, test_names):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32

    X_all, A_all, M_all, name_to_idx, node_dim = prepare_graph_tensors(meta, device, dtype)
    train_idx, train_e, train_y = make_samples(train_names, energies, curves, name_to_idx)
    test_idx, test_e, test_y = make_samples(test_names, energies, curves, name_to_idx)

    e_min, e_max = float(train_e.min()), float(train_e.max())
    e_den = max(e_max - e_min, 1e-12)
    y_min, y_max = float(train_y.min()), float(train_y.max())
    y_den = max(y_max - y_min, 1e-12)

    train_e_s = 0.05 + 0.90 * (train_e - e_min) / e_den
    test_e_s = 0.05 + 0.90 * (test_e - e_min) / e_den
    if args.target == "minmax":
        train_target = 0.05 + 0.90 * (train_y - y_min) / y_den
    elif args.target == "log":
        train_target = np.log(np.maximum(train_y, 1e-12))
    else:
        raise ValueError("target must be minmax or log")

    model = GNNRegressor(
        node_dim=node_dim,
        hidden=args.hidden,
        mp_layers=args.mp_layers,
        head_hidden=args.head_hidden,
        head_depth=args.head_depth,
        dropout=args.dropout,
    ).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ti = torch.tensor(train_idx, dtype=torch.long, device=device)
    te = torch.tensor(train_e_s, dtype=dtype, device=device)
    ty = torch.tensor(train_target, dtype=dtype, device=device)
    vi = torch.tensor(test_idx, dtype=torch.long, device=device)
    ve = torch.tensor(test_e_s, dtype=dtype, device=device)

    n = len(train_idx)
    losses = []
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        nb = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            mol_batch = ti[idx]
            pred = model(X_all[mol_batch], A_all[mol_batch], M_all[mol_batch], te[idx])
            if args.target == "minmax":
                loss = loss_value(pred, ty[idx], args.loss)
            else:
                loss = torch.mean((pred - ty[idx]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            nb += 1
        if ep == 1 or ep % args.log_every == 0 or ep == args.epochs:
            losses.append({"seed": seed, "epoch": ep, "loss": total_loss / max(nb, 1)})
            if not args.quiet:
                print(f"seed={seed} epoch={ep}/{args.epochs} loss={losses[-1]['loss']:.6e}")

    with torch.no_grad():
        pred_raw = model(X_all[vi], A_all[vi], M_all[vi], ve).detach().cpu().numpy()
    if args.target == "minmax":
        pred_y = minmax_inverse(pred_raw, y_min, y_den)
    else:
        pred_y = np.exp(pred_raw)
    if args.positive_clip:
        pred_y = np.maximum(pred_y, 0.0)
    pred_curves = pred_y.reshape(len(test_names), len(energies))
    return pred_curves, pd.DataFrame(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partition", type=int, default=2, choices=[1, 2])
    p.add_argument("--seeds", type=str, default="1,2,3,4,5")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--mp-layers", type=int, default=3)
    p.add_argument("--head-hidden", type=int, default=64)
    p.add_argument("--head-depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--target", choices=["minmax", "log"], default="minmax")
    p.add_argument("--loss", choices=["mse", "relative"], default="mse")
    p.add_argument("--positive-clip", action="store_true")
    p.add_argument("--float64", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    meta, energies, curves = load_table_and_curves()
    train_names, test_names = split_names(meta, args.partition)
    y_true = np.vstack([curves[m] for m in test_names])
    seeds = parse_seeds(args.seeds)
    tag = f"gnn_energy_partition{args.partition}_{args.target}_{args.loss}_seeds{'-'.join(map(str,seeds))}"
    outdir = RESULTS_DIR / tag
    outdir.mkdir(parents=True, exist_ok=True)

    preds, loss_dfs = [], []
    for s in seeds:
        pred_curves, loss_df = train_one(args, s, meta, energies, curves, train_names, test_names)
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
    article = ARTICLE_TABLE2.get((args.partition, False), {})
    summary["article_table2_max_percent"] = summary["molecule"].map(article)
    summary["delta_vs_article_pp"] = summary["max_percent_difference"] - summary["article_table2_max_percent"]

    metrics_seed_df.to_csv(outdir / "metrics_by_seed.csv", index=False)
    summary.to_csv(outdir / "summary_mean_prediction.csv", index=False)
    pd.concat(loss_dfs, ignore_index=True).to_csv(outdir / "loss_all_seeds.csv", index=False)
    np.savez(outdir / "predictions_all_seeds.npz", energies=energies, test_names=np.array(test_names, dtype=object), y_true=y_true, predictions=pred_arr)
    save_json(outdir / "run_config.json", vars(args) | {"train_names": train_names, "test_names": test_names})
    plot_mean_predictions(outdir, energies, test_names, y_true, pred_arr, title=tag, article_max=article)
    plot_loss_curves(outdir, loss_dfs, title=tag)
    print_verdict(summary, tag, article_mode=False)
    print(f"\nSaved results in: {outdir}")

if __name__ == "__main__":
    main()
