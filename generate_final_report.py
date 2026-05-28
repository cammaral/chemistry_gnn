#!/usr/bin/env python3
"""Generate a complete Markdown report from all completed Harris/GNN experiment outputs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
REPORT = RESULTS / "FINAL_REPORT.md"
FIGDIR = RESULTS / "report_figures"

ARTICLE_NET20 = {
    (1, "Propanone"): 14.0,
    (1, "2-Methylpropanal"): 4.0,
    (1, "Hexan-3-one"): 7.0,
    (1, "3,3-Dimethylbutan-2-one"): 5.0,
    (1, "Methanol"): 12.0,
    (2, "Ethanal"): 13.0,
    (2, "Ethanol"): 26.0,
    (2, "Propanal"): 6.0,
    (2, "3-Methylbutan-2-one"): 13.0,
    (2, "Molecular Nitrogen"): 30.0,
}
ARTICLE_NET20IP = {
    (1, "Propanone"): 16.0,
    (1, "2-Methylpropanal"): 5.0,
    (1, "Hexan-3-one"): 7.0,
    (1, "3,3-Dimethylbutan-2-one"): 6.0,
    (1, "Methanol"): 9.0,
    (2, "Ethanal"): 14.0,
    (2, "Ethanol"): 23.0,
    (2, "Propanal"): 8.0,
    (2, "3-Methylbutan-2-one"): 12.0,
    (2, "Molecular Nitrogen"): 1940.0,
}


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "max_percent_difference" not in df.columns and "mean_curve_max_percent_difference" in df.columns:
        df["max_percent_difference"] = df["mean_curve_max_percent_difference"]
    if "mape_percent" not in df.columns and "mean_curve_mape_percent" in df.columns:
        df["mape_percent"] = df["mean_curve_mape_percent"]
    return df


def infer_partition(model: str) -> int | None:
    m = re.search(r"partition(\d+)", model)
    return int(m.group(1)) if m else None


def infer_family(model: str) -> str:
    if model.startswith("partition"):
        return "Harris-style MLP"
    if model.startswith("descriptor_mlp"):
        return "Descriptor MLP"
    if model.startswith("gnn_energy"):
        return "GNN energy"
    return "Other"


def collect_summaries() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS.glob("*/summary_mean_prediction.csv")):
        model = path.parent.name
        try:
            df = normalize_summary(pd.read_csv(path))
        except Exception:
            continue
        if not {"molecule", "max_percent_difference", "mape_percent"}.issubset(df.columns):
            continue
        part = infer_partition(model)
        df.insert(0, "model", model)
        df.insert(1, "family", infer_family(model))
        df.insert(2, "partition", part)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["article_Net20"] = out.apply(lambda r: ARTICLE_NET20.get((int(r.partition), r.molecule), np.nan) if pd.notna(r.partition) else np.nan, axis=1)
    out["article_Net20Ip"] = out.apply(lambda r: ARTICLE_NET20IP.get((int(r.partition), r.molecule), np.nan) if pd.notna(r.partition) else np.nan, axis=1)
    out["delta_vs_article_Net20"] = out["max_percent_difference"] - out["article_Net20"]
    out["beats_article_Net20"] = out["delta_vs_article_Net20"] < 0
    return out


def save_tables(all_df: pd.DataFrame):
    RESULTS.mkdir(exist_ok=True)
    all_df.to_csv(RESULTS / "report_all_model_molecule_metrics.csv", index=False)
    overall = all_df.groupby(["model", "family"], as_index=False).agg(
        n_molecules=("molecule", "count"),
        mean_mape_percent=("mape_percent", "mean"),
        mean_max_percent_difference=("max_percent_difference", "mean"),
        median_max_percent_difference=("max_percent_difference", "median"),
        worst_max_percent_difference=("max_percent_difference", "max"),
        mean_delta_vs_article_Net20=("delta_vs_article_Net20", "mean"),
        n_beats_article_Net20=("beats_article_Net20", "sum"),
    ).sort_values(["mean_max_percent_difference", "worst_max_percent_difference"])
    overall.to_csv(RESULTS / "report_overall_model_ranking.csv", index=False)

    best_by_mol = all_df.sort_values("max_percent_difference").groupby(["partition", "molecule"], as_index=False).first()
    best_by_mol.to_csv(RESULTS / "report_best_model_by_molecule.csv", index=False)

    return overall, best_by_mol


def make_plots(overall: pd.DataFrame, all_df: pd.DataFrame):
    FIGDIR.mkdir(exist_ok=True)
    if len(overall):
        top = overall.head(30).copy()
        fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(top)), 5))
        ax.bar(top["model"], top["mean_max_percent_difference"])
        ax.set_ylabel("Mean max percent error (%)")
        ax.set_title("Top models by mean per-molecule maximum percent error")
        ax.tick_params(axis="x", rotation=75)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGDIR / "top_models_mean_max_error.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if len(all_df):
        # Best model per molecule compared to article Net20.
        tmp = all_df.dropna(subset=["partition", "article_Net20"]).copy()
        if len(tmp):
            best = tmp.sort_values("max_percent_difference").groupby(["partition", "molecule"], as_index=False).first()
            labels = [f"P{int(p)} {m}" for p, m in zip(best.partition, best.molecule)]
            x = np.arange(len(best))
            fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(best)), 5))
            ax.bar(x - 0.18, best["article_Net20"], width=0.36, label="Article Net20")
            ax.bar(x + 0.18, best["max_percent_difference"], width=0.36, label="Best completed model")
            ax.set_ylabel("Max percent error (%)")
            ax.set_title("Best completed model vs article Net20 by molecule")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=55, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIGDIR / "best_vs_article_by_molecule.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def log_summary() -> pd.DataFrame:
    rows = []
    for p in sorted(LOGS.glob("*.log")):
        txt = p.read_text(errors="ignore")
        rc = None
        elapsed = None
        m = re.search(r"RETURN_CODE=([0-9-]+)", txt)
        if m:
            rc = int(m.group(1))
        m = re.search(r"ELAPSED_SECONDS=([0-9.]+)", txt)
        if m:
            elapsed = float(m.group(1))
        rows.append({"log": p.name, "return_code": rc, "elapsed_seconds": elapsed})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(RESULTS / "report_log_summary.csv", index=False)
    return df


def md_table(df: pd.DataFrame, n=20) -> str:
    if df is None or len(df) == 0:
        return "_No rows._"
    try:
        return df.head(n).round(3).to_markdown(index=False)
    except Exception:
        return "```\n" + df.head(n).round(3).to_string(index=False) + "\n```"


def main():
    RESULTS.mkdir(exist_ok=True)
    all_df = collect_summaries()
    if all_df.empty:
        REPORT.write_text("# Final report\n\nNo completed summary files were found.\n", encoding="utf-8")
        print(f"No completed summary files found. Wrote {REPORT}")
        return
    overall, best_by_mol = save_tables(all_df)
    make_plots(overall, all_df)
    logs = log_summary()

    completed = len(overall)
    failed_logs = logs[(logs.return_code.notna()) & (logs.return_code != 0)] if len(logs) else pd.DataFrame()

    # Reference-specific tables.
    article_cmp = all_df.dropna(subset=["article_Net20"]).copy()
    beating = article_cmp[article_cmp["beats_article_Net20"]].sort_values("delta_vs_article_Net20")
    near = article_cmp[article_cmp["delta_vs_article_Net20"].abs() <= 5.0].sort_values("delta_vs_article_Net20")

    lines = []
    lines.append("# Harris/GNN electron-impact ionization cross-section study — final report")
    lines.append("")
    lines.append("This report was generated automatically from the completed runs in `results/`.")
    lines.append("")
    lines.append("## What was compared")
    lines.append("")
    lines.append("- Harris-style composition MLPs: Net10, Net15, Net20; with and without ionization potential `Ip`; Adam and optional LBFGS variants.")
    lines.append("- Descriptor MLPs: scalar molecular descriptors, with/without `Ip`, with/without graph-derived descriptors.")
    lines.append("- GNN energy models: molecular graph plus electron energy, with minmax/MSE, minmax/relative, and log/MSE targets.")
    lines.append("- Article references: Table 2 maximum percent differences for Net20 and Net20Ip.")
    lines.append("")
    lines.append("## Run status")
    lines.append("")
    lines.append(f"Completed model summaries found: **{completed}**")
    lines.append(f"Log files found: **{len(logs)}**")
    lines.append(f"Failed logs: **{len(failed_logs)}**")
    if len(failed_logs):
        lines.append(md_table(failed_logs, n=20))
    lines.append("")
    lines.append("## Overall ranking")
    lines.append("")
    lines.append("Sorted by the mean, over test molecules, of the maximum percent error along each curve.")
    lines.append("")
    lines.append(md_table(overall, n=25))
    lines.append("")
    lines.append("## Best model by molecule")
    lines.append("")
    cols = ["partition", "molecule", "model", "family", "mape_percent", "max_percent_difference", "article_Net20", "delta_vs_article_Net20"]
    lines.append(md_table(best_by_mol[[c for c in cols if c in best_by_mol.columns]], n=30))
    lines.append("")
    lines.append("## Models/molecules beating article Net20")
    lines.append("")
    cols2 = ["partition", "molecule", "model", "family", "max_percent_difference", "article_Net20", "delta_vs_article_Net20"]
    lines.append(md_table(beating[[c for c in cols2 if c in beating.columns]], n=40))
    lines.append("")
    lines.append("## Models/molecules within ±5 percentage points of article Net20")
    lines.append("")
    lines.append(md_table(near[[c for c in cols2 if c in near.columns]], n=40))
    lines.append("")
    lines.append("## Figures generated for quick inspection")
    lines.append("")
    for fig in sorted(FIGDIR.glob("*.png")):
        rel = fig.relative_to(RESULTS)
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Main CSV files")
    lines.append("")
    for fname in [
        "report_all_model_molecule_metrics.csv",
        "report_overall_model_ranking.csv",
        "report_best_model_by_molecule.csv",
        "comparison_all_models_by_molecule.csv",
        "comparison_all_models_overall.csv",
        "comparison_to_article_table2_all_models.csv",
        "comparison_to_article_table2_overall.csv",
        "report_log_summary.csv",
        "run_manifest.json",
        "run_status.json",
    ]:
        if (RESULTS / fname).exists():
            lines.append(f"- `{fname}`")
    lines.append("")
    lines.append("## Notes for interpretation")
    lines.append("")
    lines.append("- The article comparison uses `max_percent_difference`, i.e. the largest absolute percent error over the 101 energy grid points.")
    lines.append("- If the PyTorch Harris Net20 remains far from the article only for `Molecular Nitrogen`, that indicates optimizer/backend sensitivity rather than a global preprocessing error.")
    lines.append("- GNN models should be judged both against the article Table 2 and against the local PyTorch Harris reproduction; these are not identical because the article used Mathematica `NetTrain`.")
    lines.append("- For paper-level claims, prefer 10–20 seeds and both partitions.")
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
