# Big run: Harris reproduction + Descriptor MLP + GNN comparison

This version includes two important scripts:

- `run_research_suite_tqdm.py`: runs all implemented model families with progress bars and logs.
- `generate_final_report.py`: reads every completed result and writes `results/FINAL_REPORT.md` plus summary CSVs/plots.

## Recommended full run

```bash
python run_research_suite_tqdm.py --profile full --threads 1
```

This uses:

- seeds `1,2,3,4,5,6,7,8,9,10`
- Harris-style MLP: 400000 epochs
- Descriptor MLP: 80000 epochs
- GNN: 50000 epochs
- Harris optimizers: Adam for all Net10/Net15/Net20 variants; LBFGS for Net20 variants only
- both partitions
- with and without ionization potential where implemented
- GNN minmax/MSE, minmax/relative, log/MSE

## Huge run

```bash
python run_research_suite_tqdm.py --profile huge --threads 1
```

This uses 20 seeds and longer training. It is intended to run overnight or longer.

## Resume

The runner resumes by default: if a result summary already exists, that job is skipped.

```bash
python run_research_suite_tqdm.py --profile full --threads 1 --resume
```

To force rerun everything:

```bash
python run_research_suite_tqdm.py --profile full --threads 1 --no-resume
```

## Quick test

```bash
python run_research_suite_tqdm.py --profile quick --threads 1
```

## Final outputs

After completion, open/send these files:

```text
results/FINAL_REPORT.md
results/report_overall_model_ranking.csv
results/report_best_model_by_molecule.csv
results/report_all_model_molecule_metrics.csv
results/comparison_all_models_by_molecule.csv
results/comparison_all_models_overall.csv
results/comparison_to_article_table2_all_models.csv
results/comparison_to_article_table2_overall.csv
results/report_figures/
logs/
```

The most useful file for analysis is `results/FINAL_REPORT.md`.
