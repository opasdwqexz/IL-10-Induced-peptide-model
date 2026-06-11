from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="IL-10 peptide training pipeline with switchable ESM-only and handcrafted variants."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=root / "data" / "peptide_level_dataset_MHCII.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=root / "data" / "all_minus_benchmark_minus_mhcii.csv",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=root / "artifacts" / "cache",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "artifacts" / "models",
    )
    parser.add_argument(
        "--esm-model",
        default="esm2_t33_650M_UR50D",
    )
    parser.add_argument(
        "--variant",
        choices=["esm_only", "handcrafted_no_fs", "handcrafted_with_fs", "all"],
        default="esm_only",
        help="Choose one model variant or run all variants for comparison.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        project_root=Path(__file__).resolve().parent,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        results_dir=args.results_dir,
        cache_dir=args.cache_dir,
        model_dir=args.model_dir,
        variant=args.variant,
        esm_model=args.esm_model,
        batch_size=args.batch_size,
        n_splits=args.num_folds,
        seed=args.seed,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
