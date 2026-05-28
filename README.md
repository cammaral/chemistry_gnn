# Harris–Nepomuceno + GNN full article-comparison project

This project keeps the previous PyTorch/GNN implementation and adds scripts to compare against more of the article, not only Table 2.

## What is included

- Official Figshare interpolated datasets converted to `data/ionization_cross_sections_25.csv`.
- Reproduction-style Harris networks: Net10, Net15, Net20, with and without ionization potential.
- Descriptor MLP baselines.
- Molecular graph energy model / GNN baselines.
- Article-reference outputs:
  - Figure 2-style plot of the official interpolated experimental curves.
  - Figure 3/4/5-style plots from your reproduced runs.
  - Figure 6-style predictions using the official Figshare saved Net25 weights.
  - Table 2 and Table 3 reference CSV files.
- A general `run_all_experiments_tqdm.py` runner with tqdm.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fast smoke test

```bash
python run_all_experiments_tqdm.py --quick
```

This uses 2 seeds and reduced epochs. It is useful to test if your environment is working, but it is not a final research run.

## Research run

```bash
python run_all_experiments_tqdm.py \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --harris-epochs 400000 \
  --mlp-epochs 50000 \
  --gnn-epochs 30000
```

This runs:

- Harris Net10/15/20, partitions 1 and 2, without and with Ip.
- Descriptor MLP, partitions 1 and 2, with/without Ip and with/without extra graph descriptors.
- GNN energy models, partitions 1 and 2, minmax-MSE, minmax-relative, and log-MSE targets.
- Article-style plots and summary comparison tables.

## Main outputs

Look at:

```text
results/article_reference/
results/comparison_to_article_table2_all_models.csv
results/comparison_to_article_table2_overall.csv
results/comparison_all_models_by_molecule.csv
results/comparison_all_models_overall.csv
```

The most important metric for comparison with the paper is:

```text
max_percent_difference
```

because this is the metric reported in Table 2 of the article.

## Important note

The article used Mathematica `NetTrain` with default settings. The PyTorch implementation follows the architecture and preprocessing closely, but it is not bit-for-bit identical to Mathematica. For the most faithful run, use the included Wolfram route (`run_official_wolfram.wl`) if you have Wolfram installed.
