# IL-10-Induced Peptide Model

This repository now supports three switchable model variants and one comparison mode.

## Model variants

- `esm_only`
  - `BaseSeq` is the only sequence input to ESM-2.
  - Modification annotations such as `PHOS`, `DEAM`, `CITR`, `ACET`, and `MCM` are kept as metadata features.
  - Two XGBoost models are trained on mean-pooled and max-pooled ESM embeddings, then stacked with a logistic meta-model.
- `handcrafted_no_fs`
  - Uses ESM2 mean/max embeddings plus AAC + DPC + aaIndex + modification metadata.
  - No feature selection.
- `handcrafted_with_fs`
  - Uses the same ESM2-plus-handcrafted feature set.
  - Applies variance filtering, aaIndex correlation filtering, and XGBoost importance filtering.

Threshold selection is learned from training out-of-fold probabilities only.

## Repository layout

- `data/`: raw CSV files
- `src/`: pipeline code
- `results/`: per-variant metrics, predictions, plots, and comparison files
- `artifacts/cache/`: cached ESM embeddings
- `artifacts/models/`: trained model bundle

## Run

Run one version:

```bash
python main.py --variant esm_only
python main.py --variant handcrafted_no_fs
python main.py --variant handcrafted_with_fs
```

Run all three and build a comparison table:

```bash
python main.py --variant all
```

Optional flags:

```bash
python main.py --variant esm_only --esm-model esm2_t12_35M_UR50D --batch-size 64 --num-folds 5
```

## Main outputs

- `results/data_audit.json`
- `results/<variant>/metrics.csv`
- `results/<variant>/fold_summary.csv`
- `results/<variant>/train_oof_predictions.csv`
- `results/<variant>/test_predictions.csv`
- `results/<variant>/roc_pr_curves.png`
- `results/model_comparison_summary.csv`
- `results/model_comparison_full_metrics.csv`
- `artifacts/models/<variant>_bundle.pkl`
