# Harris & Nepomuceno baseline + GNN project

This project keeps the previous Harris--Nepomuceno reproduction files and adds a complete first GNN pipeline.

## What is included

### Original/baseline reproduction

- `reproduce_mathematica_like.py`  
  PyTorch port of the Mathematica architecture.

- `run_official_wolfram.wl`  
  Route to run with Wolfram/Mathematica if you have `wolframscript` installed.

- `plot_wolfram_outputs.py`  
  Plots the Wolfram outputs.

- `official_mathematica/pmx_codes.zip`  
  Original Figshare ZIP that you provided.

### New GNN/ML models

- `molecular_graphs.py`  
  Self-contained hard-coded molecular graphs for all 25 molecules. No RDKit required.

- `train_mlp_descriptors.py`  
  Descriptor baseline: formula/energy plus optional simple graph descriptors.

- `train_gnn_energy_model.py`  
  Pure-PyTorch graph neural network: molecular graph + energy -> cross section.

- `compare_all_models.py`  
  Collects all model summaries and creates comparison CSV/plot.

- `run_full_pipeline.py`  
  Convenience runner for baseline + descriptor MLP + GNN.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project uses only common packages: `numpy`, `pandas`, `matplotlib`, and `torch`.
No RDKit or PyTorch Geometric is needed.

## 1. Check the molecular graphs

```bash
python molecular_graphs.py
```

Expected output:

```text
All hard-coded molecular graphs match molecules_table1.csv formulas.
```

## 2. Reproduce the Harris--Nepomuceno PyTorch baseline

Quick test:

```bash
python reproduce_mathematica_like.py --partition 2 --net 20 --epochs 20000 --seeds 1,2,3 --optimizer adam --float64
```

Article-scale run:

```bash
python reproduce_mathematica_like.py --partition 2 --net 20 --epochs 400000 --seeds 1,2,3,4,5 --optimizer adam --float64
```

This prints a final check against Table 2.

## 3. Train the descriptor MLP baseline

```bash
python train_mlp_descriptors.py \
  --partition 2 \
  --epochs 10000 \
  --seeds 1,2,3,4,5 \
  --graph-descriptors
```

This model uses scalar samples:

```text
molecule descriptors + electron energy -> sigma(E)
```

## 4. Train the GNN model

```bash
python train_gnn_energy_model.py \
  --partition 2 \
  --epochs 10000 \
  --seeds 1,2,3,4,5 \
  --target minmax \
  --loss mse \
  --positive-clip
```

A useful alternative for small targets such as N2 is relative loss:

```bash
python train_gnn_energy_model.py \
  --partition 2 \
  --epochs 10000 \
  --seeds 1,2,3,4,5 \
  --target minmax \
  --loss relative \
  --positive-clip
```

## 5. Run everything quickly

```bash
python run_full_pipeline.py --partition 2 --epochs 3000 --baseline-epochs 20000 --seeds 1,2,3
```

For serious runs, increase epochs and seeds.

## 6. Compare all models

```bash
python compare_all_models.py
```

Outputs are saved in `results/`:

- `comparison_all_models_by_molecule.csv`
- `comparison_all_models_overall.csv`
- `comparison_all_models.png`

Each model directory also contains:

- `summary_mean_prediction.csv`
- `metrics_by_seed.csv`
- `predictions_all_seeds.npz`
- `figure_mean_predictions.png`
- `loss_curves.png`

## Scientific use

The clean comparison is:

1. Harris-Net reproduction: composition only, vector output.
2. Descriptor MLP: formula + simple graph descriptors + energy.
3. GNN: molecular connectivity + energy.

The main research question is whether the graph representation improves cases where formula-only information is insufficient, especially isomers and out-of-distribution molecules such as molecular nitrogen.
