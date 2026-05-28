#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd

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


def parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_table_and_curves() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
    meta = pd.read_csv(DATA_DIR / "molecules_table1.csv")
    raw = pd.read_csv(DATA_DIR / "ionization_cross_sections_25.csv")
    energies = np.sort(raw["energy_eV"].unique()).astype(float)
    curves = {}
    for mol, g in raw.groupby("molecule"):
        gg = g.sort_values("energy_eV")
        if len(gg) != len(energies):
            raise ValueError(f"{mol}: expected {len(energies)} energies, got {len(gg)}")
        curves[mol] = gg["sigma_a0_2"].to_numpy(float)
    return meta, energies, curves


def split_names(meta: pd.DataFrame, partition: int) -> Tuple[List[str], List[str]]:
    test_col = f"partition{partition}_test"
    test = meta.loc[meta[test_col].astype(bool), "molecule"].tolist()
    train = meta.loc[~meta[test_col].astype(bool), "molecule"].tolist()
    return train, test


def make_scalar_dataset(names: List[str], energies: np.ndarray, curves: Dict[str, np.ndarray]):
    mol_names, e_vals, y_vals = [], [], []
    for m in names:
        for e, y in zip(energies, curves[m]):
            mol_names.append(m)
            e_vals.append(float(e))
            y_vals.append(float(y))
    return np.array(mol_names, dtype=object), np.array(e_vals, dtype=float), np.array(y_vals, dtype=float)


def minmax_fit(x: np.ndarray):
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    denom = mx - mn
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return mn, mx, denom


def minmax_transform(x: np.ndarray, mn: np.ndarray, denom: np.ndarray, lo=0.05, hi=0.95):
    return lo + (hi - lo) * (x - mn) / denom


def minmax_inverse(xs: np.ndarray, mn: float, denom: float, lo=0.05, hi=0.95):
    return (xs - lo) * denom / (hi - lo) + mn


def compute_curve_metrics(y_true: np.ndarray, y_pred: np.ndarray, molecules: List[str], eps: float = 1e-12) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(molecules):
        yt = y_true[i]
        yp = y_pred[i]
        pct = np.abs((yp - yt) / np.maximum(np.abs(yt), eps)) * 100.0
        rows.append({
            "molecule": m,
            "mape_percent": float(np.mean(pct)),
            "max_percent_difference": float(np.max(pct)),
            "rmse_a0_2": float(np.sqrt(np.mean((yp - yt) ** 2))),
            "mae_a0_2": float(np.mean(np.abs(yp - yt))),
            "max_abs_error_a0_2": float(np.max(np.abs(yp - yt))),
        })
    return pd.DataFrame(rows)


def verdict_table(summary: pd.DataFrame, partition: int, include_ip: bool, tolerance_pp: float = 10.0) -> pd.DataFrame:
    article = ARTICLE_TABLE2.get((partition, include_ip), {})
    out = summary.copy()
    out["article_table2_max_percent"] = out["molecule"].map(article)
    out["delta_vs_article_pp"] = out["max_percent_difference"] - out["article_table2_max_percent"]
    out["close_to_article"] = out["delta_vs_article_pp"].abs() <= tolerance_pp
    return out


def print_verdict(df: pd.DataFrame, model_name: str, article_mode: bool = True):
    print("\n" + "=" * 96)
    if article_mode:
        print(f"FINAL CHECK: {model_name} ficou próximo da Tabela 2 do artigo?")
    else:
        print(f"FINAL CHECK: resumo de desempenho para {model_name}")
    print("=" * 96)
    with pd.option_context("display.max_columns", 20, "display.width", 160):
        print(df.round(3).to_string(index=False))
    if article_mode and "close_to_article" in df:
        n = int(df["close_to_article"].sum())
        total = len(df)
        if n == total:
            print("\nVEREDITO: SIM, ficou próximo do artigo para todas as moléculas deste teste.")
        elif n >= max(1, total - 1):
            print(f"\nVEREDITO: PARCIALMENTE. Ficou próximo para {n}/{total}; investigar os casos restantes.")
        else:
            print(f"\nVEREDITO: NÃO. Ficou próximo para apenas {n}/{total}; revisar treino/otimizador/preprocessamento.")


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
