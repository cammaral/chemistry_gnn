#!/usr/bin/env python3
"""Convenience runner for quick experiments. Increase epochs for article-quality runs."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partition", type=int, default=2)
    p.add_argument("--seeds", type=str, default="1,2,3")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--baseline-epochs", type=int, default=20000)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    q = ["--quiet"] if args.quiet else []
    py = sys.executable
    run([py, "reproduce_mathematica_like.py", "--partition", str(args.partition), "--net", "20", "--epochs", str(args.baseline_epochs), "--seeds", args.seeds, "--optimizer", "adam", "--float64"] + q)
    run([py, "train_mlp_descriptors.py", "--partition", str(args.partition), "--epochs", str(args.epochs), "--seeds", args.seeds, "--graph-descriptors"] + q)
    run([py, "train_gnn_energy_model.py", "--partition", str(args.partition), "--epochs", str(args.epochs), "--seeds", args.seeds, "--target", "minmax", "--loss", "mse", "--positive-clip"] + q)
    run([py, "compare_all_models.py"])

if __name__ == "__main__":
    main()
