#!/usr/bin/env python3
"""Large experiment runner for the Harris--Nepomuceno reproduction + descriptor MLP + GNN study.

This is the script to leave running overnight. It runs all implemented model families, stores logs,
skips completed runs by default, then generates a complete Markdown/CSV/PNG report.

Examples
--------
Smoke test:
    python run_research_suite_tqdm.py --profile quick --threads 1

Recommended research run:
    python run_research_suite_tqdm.py --profile full --threads 1

Large sweep:
    python run_research_suite_tqdm.py --profile huge --threads 1

Resume after interruption:
    python run_research_suite_tqdm.py --profile full --threads 1 --resume

Print commands without running:
    python run_research_suite_tqdm.py --profile full --dry-run
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


@dataclass
class Job:
    name: str
    cmd: List[str]
    log: Path
    expected_summary: Path | None = None


def comma_ints(values: str) -> list[int]:
    return [int(x.strip()) for x in values.split(",") if x.strip()]


def join_ints(values: list[int]) -> str:
    return ",".join(str(v) for v in values)


def cmd_to_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def result_dir_for_harris(part: int, net: int, ip: bool, optimizer: str, seeds: str) -> Path:
    tag = f"partition{part}_Net{net}{'Ip' if ip else ''}_{optimizer}_seeds{'-'.join(seeds.split(','))}"
    return RESULTS / tag


def result_dir_for_mlp(part: int, ip: bool, graph_desc: bool, seeds: str) -> Path:
    tag = f"descriptor_mlp_partition{part}_ip{int(ip)}_graphdesc{int(graph_desc)}_seeds{'-'.join(seeds.split(','))}"
    return RESULTS / tag


def result_dir_for_gnn(part: int, target: str, loss: str, seeds: str) -> Path:
    tag = f"gnn_energy_partition{part}_{target}_{loss}_seeds{'-'.join(seeds.split(','))}"
    return RESULTS / tag


def run_command(job: Job, dry_run: bool = False, resume: bool = True) -> str:
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if resume and job.expected_summary is not None and job.expected_summary.exists():
        return "skipped"

    if dry_run:
        print(cmd_to_str(job.cmd))
        return "dry"

    t0 = time.time()
    with open(job.log, "w", encoding="utf-8") as f:
        f.write("COMMAND: " + cmd_to_str(job.cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(job.cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0
        f.write(f"\n\nRETURN_CODE={proc.returncode}\nELAPSED_SECONDS={elapsed:.2f}\n")

    if proc.returncode != 0:
        return "failed"
    return "done"


def build_jobs(args) -> list[Job]:
    py = sys.executable
    jobs: list[Job] = []
    seeds = args.seeds
    log_every_harris = str(max(1000, args.harris_epochs // 20))
    log_every_mlp = str(max(500, args.mlp_epochs // 20))
    log_every_gnn = str(max(500, args.gnn_epochs // 20))

    # Static article outputs: article reference tables, Fig. 2, official Net25/Fig. 6.
    jobs.append(Job(
        name="article_static_fig2_fig6",
        cmd=[py, "article_outputs_and_figures.py", "--fig2", "--fig6"],
        log=LOGS / "article_static_fig2_fig6.log",
        expected_summary=RESULTS / "article_reference" / "official_net25_table3_predictions.csv",
    ))

    # Harris--Nepomuceno style models: Net10/15/20, with and without Ip, both partitions.
    if not args.skip_harris:
        for optimizer in args.harris_optimizers.split(","):
            optimizer = optimizer.strip()
            if not optimizer:
                continue
            for part in [1, 2]:
                for net in [10, 15, 20]:
                    for ip in [False, True]:
                        # For SGD/LBFGS sweeps, optionally restrict to Net20 to avoid huge cost.
                        if optimizer != "adam" and args.optimizer_sweep_net20_only and net != 20:
                            continue
                        name = f"harris_{optimizer}_partition{part}_Net{net}{'Ip' if ip else ''}"
                        outdir = result_dir_for_harris(part, net, ip, optimizer, seeds)
                        cmd = [
                            py, "reproduce_mathematica_like.py",
                            "--partition", str(part),
                            "--net", str(net),
                            "--epochs", str(args.harris_epochs if optimizer != "lbfgs" else args.lbfgs_epochs),
                            "--seeds", seeds,
                            "--optimizer", optimizer,
                            "--lr", str(args.harris_lr if optimizer != "lbfgs" else args.lbfgs_lr),
                            "--threads", str(args.threads),
                            "--log-every", log_every_harris,
                            "--quiet",
                        ]
                        if args.float64:
                            cmd.append("--float64")
                        if ip:
                            cmd.append("--include-ip")
                        jobs.append(Job(name, cmd, LOGS / f"{name}.log", outdir / "summary_mean_prediction.csv"))

    # Descriptor MLPs: formula-derived and graph-descriptor variants, with and without Ip.
    if not args.skip_mlp:
        for part in [1, 2]:
            for ip in [False, True]:
                for graph_desc in [False, True]:
                    name = f"descriptor_mlp_partition{part}_ip{int(ip)}_graphdesc{int(graph_desc)}"
                    outdir = result_dir_for_mlp(part, ip, graph_desc, seeds)
                    cmd = [
                        py, "train_mlp_descriptors.py",
                        "--partition", str(part),
                        "--epochs", str(args.mlp_epochs),
                        "--seeds", seeds,
                        "--lr", str(args.mlp_lr),
                        "--hidden", str(args.mlp_hidden),
                        "--depth", str(args.mlp_depth),
                        "--weight-decay", str(args.mlp_weight_decay),
                        "--threads", str(args.threads),
                        "--log-every", log_every_mlp,
                        "--quiet",
                    ]
                    if args.float64:
                        cmd.append("--float64")
                    if ip:
                        cmd.append("--include-ip")
                    if graph_desc:
                        cmd.append("--graph-descriptors")
                    jobs.append(Job(name, cmd, LOGS / f"{name}.log", outdir / "summary_mean_prediction.csv"))

    # GNN energy models.
    if not args.skip_gnn:
        gnn_configs: list[tuple[str, str]] = []
        # minmax target supports MSE and relative losses; log target supports MSE.
        gnn_configs.append(("minmax", "mse"))
        gnn_configs.append(("minmax", "relative"))
        gnn_configs.append(("log", "mse"))
        if args.extra_gnn_relative_log:
            # only include if you later modify train_gnn_energy_model.py to support it; currently disabled by default.
            gnn_configs.append(("log", "relative"))

        hidden_values = [int(x) for x in args.gnn_hidden_values.split(",") if x.strip()]
        mp_layer_values = [int(x) for x in args.gnn_mp_layers_values.split(",") if x.strip()]
        dropout_values = [float(x) for x in args.gnn_dropout_values.split(",") if x.strip()]

        for part in [1, 2]:
            for target, loss in gnn_configs:
                for hidden in hidden_values:
                    for mp_layers in mp_layer_values:
                        for dropout in dropout_values:
                            suffix = f"h{hidden}_mp{mp_layers}_drop{str(dropout).replace('.', 'p')}"
                            name = f"gnn_partition{part}_{target}_{loss}_{suffix}"
                            # The current GNN script names its result dir only by partition/target/loss/seeds.
                            # To avoid overwriting hyperparameter sweeps, write to unique copied dirs via --no currently unsupported.
                            # Therefore for full reproducibility we run one architecture per profile unless --allow-overwrite-sweep is used.
                            if not args.allow_overwrite_sweep and (hidden != hidden_values[0] or mp_layers != mp_layer_values[0] or dropout != dropout_values[0]):
                                continue
                            outdir = result_dir_for_gnn(part, target, loss, seeds)
                            cmd = [
                                py, "train_gnn_energy_model.py",
                                "--partition", str(part),
                                "--epochs", str(args.gnn_epochs),
                                "--seeds", seeds,
                                "--batch-size", str(args.gnn_batch_size),
                                "--lr", str(args.gnn_lr),
                                "--weight-decay", str(args.gnn_weight_decay),
                                "--hidden", str(hidden),
                                "--mp-layers", str(mp_layers),
                                "--head-hidden", str(args.gnn_head_hidden),
                                "--head-depth", str(args.gnn_head_depth),
                                "--dropout", str(dropout),
                                "--target", target,
                                "--loss", loss,
                                "--positive-clip",
                                "--threads", str(args.threads),
                                "--log-every", log_every_gnn,
                                "--quiet",
                            ]
                            if args.float64:
                                cmd.append("--float64")
                            jobs.append(Job(name, cmd, LOGS / f"{name}.log", outdir / "summary_mean_prediction.csv"))

    # Article-style summary figures and comparisons.
    jobs.append(Job(
        name="article_fig345_and_table2_comparison",
        cmd=[py, "article_outputs_and_figures.py", "--fig345", "--compare"],
        log=LOGS / "article_fig345_and_table2_comparison.log",
        expected_summary=None,
    ))
    jobs.append(Job(
        name="compare_all_models",
        cmd=[py, "compare_all_models.py"],
        log=LOGS / "compare_all_models.log",
        expected_summary=RESULTS / "comparison_all_models_overall.csv",
    ))
    jobs.append(Job(
        name="generate_final_report",
        cmd=[py, "generate_final_report.py"],
        log=LOGS / "generate_final_report.log",
        expected_summary=RESULTS / "FINAL_REPORT.md",
    ))
    return jobs


def apply_profile(args):
    if args.profile == "quick":
        args.seeds = "1,2"
        args.harris_epochs = min(args.harris_epochs or 20000, 20000)
        args.mlp_epochs = min(args.mlp_epochs or 5000, 5000)
        args.gnn_epochs = min(args.gnn_epochs or 5000, 5000)
        args.harris_optimizers = "adam"
        args.gnn_hidden_values = "64"
        args.gnn_mp_layers_values = "3"
        args.gnn_dropout_values = "0.0"
    elif args.profile == "full":
        if args.seeds == "auto":
            args.seeds = "1,2,3,4,5,6,7,8,9,10"
        args.harris_epochs = args.harris_epochs or 400000
        args.mlp_epochs = args.mlp_epochs or 80000
        args.gnn_epochs = args.gnn_epochs or 50000
        args.harris_optimizers = args.harris_optimizers or "adam,lbfgs"
        args.optimizer_sweep_net20_only = True
        args.gnn_hidden_values = args.gnn_hidden_values or "64"
        args.gnn_mp_layers_values = args.gnn_mp_layers_values or "3"
        args.gnn_dropout_values = args.gnn_dropout_values or "0.0"
    elif args.profile == "huge":
        if args.seeds == "auto":
            args.seeds = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
        args.harris_epochs = args.harris_epochs or 400000
        args.mlp_epochs = args.mlp_epochs or 120000
        args.gnn_epochs = args.gnn_epochs or 100000
        args.harris_optimizers = args.harris_optimizers or "adam,lbfgs"
        args.optimizer_sweep_net20_only = True
        args.gnn_hidden_values = args.gnn_hidden_values or "64,128"
        args.gnn_mp_layers_values = args.gnn_mp_layers_values or "2,3"
        args.gnn_dropout_values = args.gnn_dropout_values or "0.0,0.1"
    else:
        raise ValueError(args.profile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["quick", "full", "huge"], default="full")
    ap.add_argument("--seeds", default="auto", help="Comma-separated seeds or 'auto'.")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--float64", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true", default=True, help="Skip jobs whose summary files already exist.")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--skip-harris", action="store_true")
    ap.add_argument("--skip-mlp", action="store_true")
    ap.add_argument("--skip-gnn", action="store_true")

    # Defaults are filled by profile if left as None.
    ap.add_argument("--harris-epochs", type=int, default=None)
    ap.add_argument("--mlp-epochs", type=int, default=None)
    ap.add_argument("--gnn-epochs", type=int, default=None)

    ap.add_argument("--harris-optimizers", default=None, help="adam,lbfgs,sgd. Full/huge use adam,lbfgs by default.")
    ap.add_argument("--optimizer-sweep-net20-only", action="store_true", default=True)
    ap.add_argument("--harris-lr", type=float, default=1e-2)
    ap.add_argument("--lbfgs-epochs", type=int, default=25000)
    ap.add_argument("--lbfgs-lr", type=float, default=0.1)

    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--mlp-hidden", type=int, default=64)
    ap.add_argument("--mlp-depth", type=int, default=3)
    ap.add_argument("--mlp-weight-decay", type=float, default=1e-4)

    ap.add_argument("--gnn-lr", type=float, default=2e-3)
    ap.add_argument("--gnn-weight-decay", type=float, default=1e-4)
    ap.add_argument("--gnn-batch-size", type=int, default=256)
    ap.add_argument("--gnn-hidden-values", default=None)
    ap.add_argument("--gnn-mp-layers-values", default=None)
    ap.add_argument("--gnn-dropout-values", default=None)
    ap.add_argument("--gnn-head-hidden", type=int, default=64)
    ap.add_argument("--gnn-head-depth", type=int, default=2)
    ap.add_argument("--extra-gnn-relative-log", action="store_true")
    ap.add_argument("--allow-overwrite-sweep", action="store_true", help="Current GNN script output dirs do not encode hidden/mp/dropout; leave off unless you know what you are doing.")

    args = ap.parse_args()
    apply_profile(args)

    # device is currently passed only by individual scripts where supported; current runner keeps CPU-safe defaults.
    LOGS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    manifest = {
        "profile": args.profile,
        "seeds": args.seeds,
        "harris_epochs": args.harris_epochs,
        "mlp_epochs": args.mlp_epochs,
        "gnn_epochs": args.gnn_epochs,
        "n_jobs": len(jobs),
        "jobs": [{"name": j.name, "cmd": cmd_to_str(j.cmd), "log": str(j.log)} for j in jobs],
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 100)
    print("Harris/GNN research suite")
    print("=" * 100)
    print(f"Profile: {args.profile}")
    print(f"Seeds: {args.seeds}")
    print(f"Harris epochs: {args.harris_epochs}")
    print(f"MLP epochs: {args.mlp_epochs}")
    print(f"GNN epochs: {args.gnn_epochs}")
    print(f"Jobs: {len(jobs)}")
    print(f"Resume: {args.resume}")
    print(f"Logs: {LOGS}")
    print(f"Results: {RESULTS}")
    print("=" * 100)

    statuses = []
    for job in tqdm(jobs, desc="Research suite", unit="job"):
        tqdm.write(f"[{job.name}] {cmd_to_str(job.cmd)}")
        status = run_command(job, dry_run=args.dry_run, resume=args.resume)
        statuses.append({"name": job.name, "status": status, "log": str(job.log)})
        (RESULTS / "run_status.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
        if status == "failed":
            print(f"FAILED: {job.name}")
            print(f"Open log: {job.log}")
            raise SystemExit(1)

    print("\nDone. Key outputs:")
    print("  results/FINAL_REPORT.md")
    print("  results/comparison_all_models_by_molecule.csv")
    print("  results/comparison_all_models_overall.csv")
    print("  results/comparison_to_article_table2_all_models.csv")
    print("  results/comparison_to_article_table2_overall.csv")
    print("  results/article_reference/")


if __name__ == "__main__":
    main()
