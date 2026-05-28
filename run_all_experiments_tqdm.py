#!/usr/bin/env python3
"""Run the full Harris/GNN comparison suite with tqdm progress.

This script intentionally separates: (i) article reproduction models, (ii) our descriptor/GNN models,
(iii) article-level figures and summary comparisons.

Recommended first run:
  python run_all_experiments_tqdm.py --quick

Research run:
  python run_all_experiments_tqdm.py --seeds 1,2,3,4,5,6,7,8,9,10 --harris-epochs 400000 --mlp-epochs 50000 --gnn-epochs 30000
"""
from __future__ import annotations
import argparse, subprocess, sys, shlex, time
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"


def run_cmd(cmd, log_path: Path, dry_run=False):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("DRY:", " ".join(shlex.quote(str(c)) for c in cmd))
        return 0
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(shlex.quote(str(c)) for c in cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\n\nRETURN_CODE={proc.returncode}\nELAPSED_SECONDS={time.time()-t0:.2f}\n")
    if proc.returncode != 0:
        print(f"\nFAILED: {' '.join(cmd)}")
        print(f"See log: {log_path}")
        raise SystemExit(proc.returncode)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated seeds")
    ap.add_argument("--harris-epochs", type=int, default=400000)
    ap.add_argument("--mlp-epochs", type=int, default=50000)
    ap.add_argument("--gnn-epochs", type=int, default=30000)
    ap.add_argument("--quick", action="store_true", help="Small smoke test: 2 seeds and fewer epochs.")
    ap.add_argument("--only-article", action="store_true", help="Only run Harris Net10/15/20 and article figures.")
    ap.add_argument("--skip-harris", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-gnn", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    if args.quick:
        args.seeds = "1,2"
        args.harris_epochs = min(args.harris_epochs, 20000)
        args.mlp_epochs = min(args.mlp_epochs, 5000)
        args.gnn_epochs = min(args.gnn_epochs, 5000)

    LOGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    py = sys.executable
    cmds = []

    # Always generate static reference outputs first: Table 2/Table 3, Fig2, Fig6 from official Net25 weights.
    if not args.skip_figures:
        cmds.append(("static_article_reference", [py, "article_outputs_and_figures.py", "--fig2", "--fig6"], LOGS / "static_article_reference.log"))

    if not args.skip_harris:
        for part in [1, 2]:
            for net in [10, 15, 20]:
                for ip in [False, True]:
                    name = f"harris_partition{part}_Net{net}{'Ip' if ip else ''}"
                    cmd = [py, "reproduce_mathematica_like.py", "--partition", str(part), "--net", str(net),
                           "--epochs", str(args.harris_epochs), "--seeds", args.seeds, "--float64", "--threads", str(args.threads),
                           "--quiet", "--log-every", str(max(1000, args.harris_epochs // 10))]
                    if ip:
                        cmd.append("--include-ip")
                    cmds.append((name, cmd, LOGS / f"{name}.log"))

    if not args.only_article and not args.skip_mlp:
        for part in [1, 2]:
            for ip in [False, True]:
                for graph_desc in [False, True]:
                    name = f"descriptor_mlp_partition{part}_ip{int(ip)}_graphdesc{int(graph_desc)}"
                    cmd = [py, "train_mlp_descriptors.py", "--partition", str(part), "--epochs", str(args.mlp_epochs),
                           "--seeds", args.seeds, "--threads", str(args.threads), "--quiet",
                           "--log-every", str(max(500, args.mlp_epochs // 10))]
                    if ip: cmd.append("--include-ip")
                    if graph_desc: cmd.append("--graph-descriptors")
                    cmds.append((name, cmd, LOGS / f"{name}.log"))

    if not args.only_article and not args.skip_gnn:
        for part in [1, 2]:
            for target in ["minmax", "log"]:
                for loss in (["mse", "relative"] if target == "minmax" else ["mse"]):
                    name = f"gnn_partition{part}_{target}_{loss}"
                    cmd = [py, "train_gnn_energy_model.py", "--partition", str(part), "--epochs", str(args.gnn_epochs),
                           "--seeds", args.seeds, "--target", target, "--loss", loss,
                           "--positive-clip", "--threads", str(args.threads), "--quiet",
                           "--log-every", str(max(500, args.gnn_epochs // 10))]
                    cmds.append((name, cmd, LOGS / f"{name}.log"))

    # Final article-style grids and model comparison.
    if not args.skip_figures:
        cmds.append(("article_grids_and_comparison", [py, "article_outputs_and_figures.py", "--fig345", "--compare"], LOGS / "article_grids_and_comparison.log"))
        cmds.append(("compare_all_models", [py, "compare_all_models.py"], LOGS / "compare_all_models.log"))

    print(f"Planned {len(cmds)} steps. Logs will be written to {LOGS}")
    for name, cmd, log in tqdm(cmds, desc="Full experiment suite", unit="step"):
        tqdm.write(f"Running: {name}")
        run_cmd(cmd, log, dry_run=args.dry_run)
    print("\nDone. Main outputs:")
    print("  results/article_reference/")
    print("  results/comparison_to_article_table2_all_models.csv")
    print("  results/comparison_to_article_table2_overall.csv")
    print("  results/comparison_all_models_by_molecule.csv")
    print("  results/comparison_all_models_overall.csv")

if __name__ == "__main__":
    main()
