#!/usr/bin/env python3
"""
Small self-contained molecular graph library for the 25 Harris-Nepomuceno molecules.

No RDKit is required. The graphs are hand-encoded from standard valence structures.
Atoms include explicit hydrogens because the original cross sections depend strongly on
composition. Bond orders are used as edge weights in the GNN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

ATOM_TYPES = ["H", "C", "N", "O"]
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

@dataclass
class MolGraph:
    atoms: List[str]
    bonds: List[Tuple[int, int, float]]

    def to_arrays(self):
        n = len(self.atoms)
        x = np.zeros((n, len(ATOM_TYPES) + 3), dtype=np.float32)
        adj = np.zeros((n, n), dtype=np.float32)
        for i, a in enumerate(self.atoms):
            x[i, ATOM_TO_IDX[a]] = 1.0
        for i, j, order in self.bonds:
            adj[i, j] = order
            adj[j, i] = order
        deg = (adj > 0).sum(axis=1).astype(np.float32)
        bond_order_sum = adj.sum(axis=1).astype(np.float32)
        x[:, len(ATOM_TYPES)] = deg / 4.0
        x[:, len(ATOM_TYPES) + 1] = bond_order_sum / 4.0
        x[:, len(ATOM_TYPES) + 2] = np.array([atomic_number(a) for a in self.atoms], dtype=np.float32) / 8.0
        return x, adj


def atomic_number(a: str) -> int:
    return {"H": 1, "C": 6, "N": 7, "O": 8}[a]


def graph_from_heavy(heavy_atoms: List[str], heavy_bonds: List[Tuple[int, int, float]], hydrogens: List[int]) -> MolGraph:
    atoms = list(heavy_atoms)
    bonds = list(heavy_bonds)
    if len(hydrogens) != len(heavy_atoms):
        raise ValueError("hydrogens must match heavy atom count")
    for heavy_idx, n_h in enumerate(hydrogens):
        for _ in range(n_h):
            h_idx = len(atoms)
            atoms.append("H")
            bonds.append((heavy_idx, h_idx, 1.0))
    return MolGraph(atoms, bonds)


def chain_aldehyde(n_c: int) -> MolGraph:
    # CH3-(CH2)_{n-2}-CHO
    heavy = ["C"] * n_c + ["O"]
    carbonyl = n_c - 1
    o = n_c
    bonds = [(i, i + 1, 1.0) for i in range(n_c - 1)] + [(carbonyl, o, 2.0)]
    h = [3] + [2] * max(0, n_c - 2) + [1] + [0]
    return graph_from_heavy(heavy, bonds, h)


def chain_ketone(n_c: int, carbonyl_pos_1indexed: int) -> MolGraph:
    # Linear ketone with carbonyl carbon at carbonyl_pos_1indexed.
    heavy = ["C"] * n_c + ["O"]
    c = carbonyl_pos_1indexed - 1
    o = n_c
    bonds = [(i, i + 1, 1.0) for i in range(n_c - 1)] + [(c, o, 2.0)]
    h = []
    for i in range(n_c):
        if i == c:
            h.append(0)
        elif i == 0 or i == n_c - 1:
            h.append(3)
        else:
            h.append(2)
    h.append(0)
    return graph_from_heavy(heavy, bonds, h)


def ether_chain(left_c: int, right_c: int) -> MolGraph:
    # Alkyl-O-alkyl as a linear skeleton.
    heavy = ["C"] * left_c + ["O"] + ["C"] * right_c
    o = left_c
    bonds: List[Tuple[int, int, float]] = []
    for i in range(left_c - 1):
        bonds.append((i, i + 1, 1.0))
    bonds.append((left_c - 1, o, 1.0))
    bonds.append((o, o + 1, 1.0))
    for i in range(o + 1, o + right_c):
        bonds.append((i, i + 1, 1.0))
    h = []
    for i in range(left_c):
        h.append(3 if i == 0 else 2)
    h.append(0)
    for k in range(right_c):
        h.append(2 if k == 0 else (3 if k == right_c - 1 else 2))
    return graph_from_heavy(heavy, bonds, h)


def build_graphs() -> Dict[str, MolGraph]:
    g: Dict[str, MolGraph] = {}
    g["Ethanal"] = chain_aldehyde(2)
    g["Propanal"] = chain_aldehyde(3)
    g["Butanal"] = chain_aldehyde(4)
    # O=CH-CH(CH3)2
    g["2-Methylpropanal"] = graph_from_heavy(
        ["C", "O", "C", "C", "C"],
        [(0, 1, 2.0), (0, 2, 1.0), (2, 3, 1.0), (2, 4, 1.0)],
        [1, 0, 1, 3, 3],
    )
    g["Ethoxyethane"] = ether_chain(2, 2)
    g["Propoxypropane"] = ether_chain(3, 3)
    # (CH3)2CH-O-CH(CH3)2
    g["2-Isopropoxypropane"] = graph_from_heavy(
        ["O", "C", "C", "C", "C", "C", "C"],
        [(0, 1, 1.0), (0, 4, 1.0), (1, 2, 1.0), (1, 3, 1.0), (4, 5, 1.0), (4, 6, 1.0)],
        [0, 1, 3, 3, 1, 3, 3],
    )
    g["Propanone"] = chain_ketone(3, 2)
    g["Butanone"] = chain_ketone(4, 2)
    g["Pentan-2-one"] = chain_ketone(5, 2)
    g["Pentan-3-one"] = chain_ketone(5, 3)
    # CH3-CO-CH(CH3)-CH3
    g["3-Methylbutan-2-one"] = graph_from_heavy(
        ["C", "C", "O", "C", "C", "C"],
        [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0), (3, 4, 1.0), (3, 5, 1.0)],
        [3, 0, 0, 1, 3, 3],
    )
    g["Hexan-3-one"] = chain_ketone(6, 3)
    g["Hexan-2-one"] = chain_ketone(6, 2)
    # CH3-CO-C(CH3)3
    g["3,3-Dimethylbutan-2-one"] = graph_from_heavy(
        ["C", "C", "O", "C", "C", "C", "C"],
        [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0), (3, 4, 1.0), (3, 5, 1.0), (3, 6, 1.0)],
        [3, 0, 0, 0, 3, 3, 3],
    )
    # CH3-CO-CH(CH3)-CH2-CH3
    g["3-Methylpentan-2-one"] = graph_from_heavy(
        ["C", "C", "O", "C", "C", "C", "C"],
        [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 6, 1.0)],
        [3, 0, 0, 1, 2, 3, 3],
    )
    # CH3-CO-CH2-CH(CH3)-CH3
    g["4-Methylpentan-2-one"] = graph_from_heavy(
        ["C", "C", "O", "C", "C", "C", "C"],
        [(0, 1, 1.0), (1, 2, 2.0), (1, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (4, 6, 1.0)],
        [3, 0, 0, 2, 1, 3, 3],
    )
    g["Molecular Hydrogen"] = MolGraph(["H", "H"], [(0, 1, 1.0)])
    g["Molecular Nitrogen"] = MolGraph(["N", "N"], [(0, 1, 3.0)])
    g["Carbon Monoxide"] = MolGraph(["C", "O"], [(0, 1, 3.0)])
    g["Nitric Oxide"] = MolGraph(["N", "O"], [(0, 1, 2.0)])
    g["Molecular Oxygen"] = MolGraph(["O", "O"], [(0, 1, 2.0)])
    g["Methanol"] = graph_from_heavy(["C", "O"], [(0, 1, 1.0)], [3, 1])
    g["Ethanol"] = graph_from_heavy(["C", "C", "O"], [(0, 1, 1.0), (1, 2, 1.0)], [3, 2, 1])
    g["Water"] = graph_from_heavy(["O"], [], [2])
    return g


def graph_descriptors(name: str) -> Dict[str, float]:
    graph = build_graphs()[name]
    atoms = graph.atoms
    heavy = [a for a in atoms if a != "H"]
    n_atoms = len(atoms)
    n_heavy = len(heavy)
    n_bonds = len(graph.bonds)
    bond_order_sum = sum(order for _, _, order in graph.bonds)
    n_double = sum(1 for _, _, order in graph.bonds if abs(order - 2.0) < 1e-6)
    n_triple = sum(1 for _, _, order in graph.bonds if abs(order - 3.0) < 1e-6)
    n_hetero = sum(1 for a in heavy if a in {"N", "O"})
    n_h = sum(1 for a in atoms if a == "H")
    mass = sum({"H": 1.00784, "C": 12.011, "N": 14.0067, "O": 15.999}[a] for a in atoms)
    return {
        "n_atoms": float(n_atoms),
        "n_heavy": float(n_heavy),
        "n_bonds": float(n_bonds),
        "bond_order_sum": float(bond_order_sum),
        "n_double": float(n_double),
        "n_triple": float(n_triple),
        "n_hetero": float(n_hetero),
        "n_h": float(n_h),
        "mass": float(mass),
    }


def validate_against_metadata(meta):
    graphs = build_graphs()
    missing = sorted(set(meta["molecule"]) - set(graphs))
    if missing:
        raise ValueError(f"Missing graphs for: {missing}")
    for _, row in meta.iterrows():
        name = row["molecule"]
        atoms = graphs[name].atoms
        counts = {a: atoms.count(a) for a in ATOM_TYPES}
        expected = {"C": int(row["C"]), "H": int(row["H"]), "N": int(row["N"]), "O": int(row["O"])}
        got = {"C": counts["C"], "H": counts["H"], "N": counts["N"], "O": counts["O"]}
        if got != expected:
            raise ValueError(f"Formula mismatch for {name}: got {got}, expected {expected}")
    return True

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    meta = pd.read_csv(Path(__file__).resolve().parent / "data" / "molecules_table1.csv")
    validate_against_metadata(meta)
    print("All hard-coded molecular graphs match molecules_table1.csv formulas.")
