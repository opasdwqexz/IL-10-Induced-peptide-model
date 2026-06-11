from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score

from .data import load_datasets, modification_feature_columns, write_audit
from .features import (
    ESMEmbedder,
    build_esm_feature_views,
    build_handcrafted_features,
    select_handcrafted_features,
)
from .model import (
    THRESHOLD_LABELS,
    build_metric_tables,
    esm_tuning_candidates,
    save_model_bundle,
    train_esm_stacking_variant,
    train_single_xgb_variant,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANT_LABELS = {
    "esm_only": "ESM-only stacking",
    "handcrafted_no_fs": "ESM2 + AAC + DPC + aaIndex without feature selection",
    "handcrafted_with_fs": "ESM2 + AAC + DPC + aaIndex with feature selection",
}


@dataclass
class PipelineConfig:
    project_root: Path
    train_csv: Path
    test_csv: Path
    results_dir: Path
    cache_dir: Path
    model_dir: Path
    variant: str = "esm_only"
    esm_model: str = "esm2_t33_650M_UR50D"
    batch_size: int = 32
    n_splits: int = 5
    seed: int = 42


def _resolve_output_path(path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        candidate = path.with_name(f"{path.stem}_updated{path.suffix}")
        suffix_index = 2
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}_updated_{suffix_index}{path.suffix}")
            suffix_index += 1
        return candidate


def _write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    target = _resolve_output_path(path)
    df.to_csv(target, index=index)
    return target


def _write_text(content: str, path: Path) -> Path:
    target = _resolve_output_path(path)
    target.write_text(content, encoding="utf-8")
    return target


def _save_figure(fig, path: Path, **kwargs) -> Path:
    target = _resolve_output_path(path)
    fig.savefig(target, **kwargs)
    return target


def _plot_curves(train_labels: np.ndarray, train_prob: np.ndarray, test_labels: np.ndarray, test_prob: np.ndarray, output_path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    train_fpr, train_tpr, _ = roc_curve(train_labels, train_prob)
    test_fpr, test_tpr, _ = roc_curve(test_labels, test_prob)
    axes[0].plot(train_fpr, train_tpr, label=f"Train OOF (AUC={roc_auc_score(train_labels, train_prob):.3f})")
    axes[0].plot(test_fpr, test_tpr, label=f"Independent test (AUC={roc_auc_score(test_labels, test_prob):.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title(f"{title_prefix} ROC")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()

    train_precision, train_recall, _ = precision_recall_curve(train_labels, train_prob)
    test_precision, test_recall, _ = precision_recall_curve(test_labels, test_prob)
    axes[1].plot(
        train_recall,
        train_precision,
        label=f"Train OOF (AP={average_precision_score(train_labels, train_prob):.3f})",
    )
    axes[1].plot(
        test_recall,
        test_precision,
        label=f"Independent test (AP={average_precision_score(test_labels, test_prob):.3f})",
    )
    axes[1].set_title(f"{title_prefix} Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    _save_figure(fig, output_path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrices(
    train_labels: np.ndarray,
    train_prob: np.ndarray,
    test_labels: np.ndarray,
    test_prob: np.ndarray,
    threshold: float,
    threshold_label: str,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_pred = (train_prob >= threshold).astype(int)
    test_pred = (test_prob >= threshold).astype(int)
    train_cm = confusion_matrix(train_labels, train_pred, labels=[0, 1])
    test_cm = confusion_matrix(test_labels, test_pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, matrix, title in (
        (axes[0], train_cm, "Train OOF"),
        (axes[1], test_cm, "Independent test"),
    ):
        axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(f"{title}\n{threshold_label}")
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["True 0", "True 1"])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
                axis.text(col, row, f"{matrix[row, col]}", ha="center", va="center", color=color)
    fig.tight_layout()
    _save_figure(fig, output_path, dpi=150)
    plt.close(fig)

    train_df = pd.DataFrame(train_cm, index=["TrueNegativeClass", "TruePositiveClass"], columns=["Predicted0", "Predicted1"])
    test_df = pd.DataFrame(test_cm, index=["TrueNegativeClass", "TruePositiveClass"], columns=["Predicted0", "Predicted1"])
    return train_df, test_df


def _select_top_short_peptides(df: pd.DataFrame, probabilities: np.ndarray, max_length: int, top_k: int) -> tuple[pd.DataFrame, dict]:
    ranking = df.loc[:, ["Sequence", "BaseSeq", "Annotation", "FinalLabel", "SeqLength"]].copy()
    ranking["PrimaryProbability"] = probabilities
    ranking = ranking.sort_values("PrimaryProbability", ascending=False).reset_index(drop=True)

    selected_rows = []
    skipped = 0
    for _, row in ranking.iterrows():
        if int(row["SeqLength"]) > max_length:
            skipped += 1
            continue
        selected_rows.append(row.to_dict())
        if len(selected_rows) == top_k:
            break

    return pd.DataFrame(selected_rows), {
        "MaxLength": max_length,
        "RequestedTopK": top_k,
        "SelectedCount": len(selected_rows),
        "SkippedBecauseTooLong": skipped,
    }


def _compose_sequence(base_seq: str, annotation: str) -> str:
    return f"{base_seq} + {annotation}" if annotation else base_seq


def _mutate_sequence(sequence: str, position: int) -> tuple[str, str]:
    replacement = "A" if sequence[position] != "A" else "G"
    mutated = sequence[:position] + replacement + sequence[position + 1 :]
    return mutated, replacement


def _score_esm_rows(df: pd.DataFrame, embedder: ESMEmbedder, model_bundle: dict) -> np.ndarray:
    embeddings = embedder.embed_uncached(df["BaseSeq"].tolist())
    views = build_esm_feature_views(df, embeddings)
    view_names = model_bundle["view_names"]
    base_probabilities = np.column_stack(
        [
            model_bundle["base_models"][view_name].predict_proba(views[view_name])[:, 1]
            for view_name in view_names
        ]
    )
    return model_bundle["meta_model"].predict_proba(base_probabilities)[:, 1]


def _build_attribution_rows(source_row: pd.Series) -> pd.DataFrame:
    rows = []
    original_seq = str(source_row["BaseSeq"])
    annotation = str(source_row["Annotation"] or "")
    mod_columns = modification_feature_columns()
    for position in range(len(original_seq)):
        mutated_seq, mutated_residue = _mutate_sequence(original_seq, position)
        row = {column: source_row[column] for column in mod_columns}
        row.update(
            {
                "Sequence": _compose_sequence(mutated_seq, annotation),
                "BaseSeq": mutated_seq,
                "Annotation": annotation,
                "FinalLabel": int(source_row["FinalLabel"]),
                "SeqLength": len(mutated_seq),
                "Position1Based": position + 1,
                "OriginalResidue": original_seq[position],
                "MutatedResidue": mutated_residue,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _write_attribution_artifacts(
    variant_dir: Path,
    train_top_hits: pd.DataFrame,
    test_top_hits: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    embedder: ESMEmbedder,
    model_bundle: dict,
) -> None:
    heatmaps: list[tuple[str, pd.DataFrame, str, float]] = []
    summary: dict[str, dict] = {}
    for split_name, top_hits, source_df in (
        ("TrainOOF", train_top_hits, train_df),
        ("IndependentTest", test_top_hits, test_df),
    ):
        if top_hits.empty:
            continue
        anchor_hit = top_hits.iloc[0]
        source_match = source_df.loc[
            (source_df["Sequence"] == anchor_hit["Sequence"])
            & (source_df["BaseSeq"] == anchor_hit["BaseSeq"])
            & (source_df["Annotation"] == anchor_hit["Annotation"])
        ]
        anchor_row = source_match.iloc[0] if not source_match.empty else source_df.loc[source_df["BaseSeq"] == anchor_hit["BaseSeq"]].iloc[0]
        mutated_rows = _build_attribution_rows(anchor_row)
        mutated_probabilities = _score_esm_rows(mutated_rows, embedder, model_bundle)
        original_probability = float(anchor_hit["PrimaryProbability"])
        mutated_rows["OriginalProbability"] = original_probability
        mutated_rows["MutatedProbability"] = mutated_probabilities
        mutated_rows["ProbabilityDrop"] = original_probability - mutated_probabilities
        _write_csv(mutated_rows, variant_dir / f"{split_name.lower()}_attribution.csv", index=False)
        heatmaps.append(
            (
                split_name,
                mutated_rows,
                str(anchor_hit["BaseSeq"]),
                original_probability,
            )
        )
        summary[split_name] = {
            "Peptide": str(anchor_hit["BaseSeq"]),
            "OriginalProbability": original_probability,
            "Length": int(anchor_hit["SeqLength"]),
            "MaxProbabilityDrop": float(mutated_rows["ProbabilityDrop"].max()),
            "MeanProbabilityDrop": float(mutated_rows["ProbabilityDrop"].mean()),
        }

    if not heatmaps:
        return

    fig, axes = plt.subplots(len(heatmaps), 1, figsize=(14, 2.8 * len(heatmaps)))
    if len(heatmaps) == 1:
        axes = [axes]

    for axis, (split_name, mutated_rows, peptide, original_probability) in zip(axes, heatmaps):
        matrix = mutated_rows["ProbabilityDrop"].to_numpy(dtype=float).reshape(1, -1)
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm")
        axis.set_title(f"{split_name} attribution for {peptide} (score={original_probability:.3f})")
        axis.set_yticks([])
        axis.set_xticks(range(len(mutated_rows)))
        axis.set_xticklabels(
            [
                f"{row.Position1Based}\n{row.OriginalResidue}->{row.MutatedResidue}"
                for row in mutated_rows.itertuples()
            ],
            fontsize=8,
        )
        for column_index, value in enumerate(mutated_rows["ProbabilityDrop"].tolist()):
            axis.text(column_index, 0, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(image, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="Probability drop")
    fig.tight_layout()
    _save_figure(fig, variant_dir / "attribution_heatmaps.png", dpi=150)
    plt.close(fig)
    _write_text(json.dumps(summary, indent=2), variant_dir / "attribution_summary.json")


def _write_variant_predictions(
    variant_dir: Path,
    variant_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_prob: np.ndarray,
    test_prob: np.ndarray,
    train_aux: dict[str, np.ndarray],
    test_aux: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> None:
    train_predictions = train_df.loc[:, ["Sequence", "BaseSeq", "Annotation", "SeqLength", "FinalLabel"]].copy()
    train_predictions["VariantName"] = variant_name
    for column_name, values in train_aux.items():
        train_predictions[column_name] = values
    train_predictions["PrimaryProbability"] = train_prob
    _write_csv(train_predictions, variant_dir / "train_oof_predictions.csv", index=False)

    test_predictions = test_df.loc[:, ["Sequence", "BaseSeq", "Annotation", "SeqLength", "FinalLabel"]].copy()
    test_predictions["VariantName"] = variant_name
    for column_name, values in test_aux.items():
        test_predictions[column_name] = values
    test_predictions["PrimaryProbability"] = test_prob
    for threshold_key, threshold_value in thresholds.items():
        readable_label = THRESHOLD_LABELS[threshold_key].replace(" ", "")
        test_predictions[f"PredictedLabel_{readable_label}"] = (test_prob >= threshold_value).astype(int)
    _write_csv(test_predictions, variant_dir / "test_predictions.csv", index=False)


def _write_comparison(metrics_tables: list[pd.DataFrame], results_dir: Path) -> None:
    all_metrics = pd.concat(metrics_tables, ignore_index=True)
    _write_csv(all_metrics, results_dir / "model_comparison_full_metrics.csv", index=False)
    comparison = (
        all_metrics.loc[
            (all_metrics["EvaluationSplit"] == "IndependentTest")
            & (all_metrics["ThresholdKey"] == "best_mcc_on_train_oof"),
            [
                "VariantName",
                "VariantLabel",
                "ThresholdValue",
                "RocAuc",
                "PrecisionRecallAuc",
                "MatthewsCorrcoef",
                "BalancedAccuracy",
                "Precision",
                "RecallSensitivity",
                "Specificity",
                "F1Score",
                "F2Score",
            ],
        ]
        .sort_values(["MatthewsCorrcoef", "PrecisionRecallAuc"], ascending=False)
        .reset_index(drop=True)
    )
    _write_csv(comparison, results_dir / "model_comparison_summary.csv", index=False)


def _combined_esm_handcrafted_features(
    train_embeddings,
    test_embeddings,
    train_handcrafted,
    test_handcrafted,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_features = np.concatenate(
        [
            train_embeddings.mean_embeddings,
            train_embeddings.max_embeddings,
            train_handcrafted.features,
        ],
        axis=1,
    ).astype(np.float32)
    test_features = np.concatenate(
        [
            test_embeddings.mean_embeddings,
            test_embeddings.max_embeddings,
            test_handcrafted.features,
        ],
        axis=1,
    ).astype(np.float32)
    feature_names = (
        [f"ESMMean_{index:04d}" for index in range(train_embeddings.mean_embeddings.shape[1])]
        + [f"ESMMax_{index:04d}" for index in range(train_embeddings.max_embeddings.shape[1])]
        + train_handcrafted.feature_names
    )
    return train_features, test_features, feature_names


def _run_single_variant(config: PipelineConfig, bundle, variant_name: str) -> pd.DataFrame:
    variant_dir = config.results_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    y_train = bundle.train_df["FinalLabel"].to_numpy(dtype=int)
    y_test = bundle.test_df["FinalLabel"].to_numpy(dtype=int)

    run_metadata = {
        "VariantName": variant_name,
        "VariantLabel": VARIANT_LABELS[variant_name],
        "Seed": config.seed,
        "NumFolds": config.n_splits,
    }

    embedder = ESMEmbedder(model_name=config.esm_model, batch_size=config.batch_size)
    train_embeddings = embedder.embed(
        bundle.train_df["BaseSeq"].tolist(),
        config.cache_dir / "shared_train_embeddings.npz",
    )
    test_embeddings = embedder.embed(
        bundle.test_df["BaseSeq"].tolist(),
        config.cache_dir / "shared_test_embeddings.npz",
    )
    run_metadata["EmbeddingModel"] = config.esm_model
    run_metadata["EmbeddingBatchSize"] = config.batch_size
    run_metadata["EmbeddingDevice"] = embedder.device

    if variant_name == "esm_only":
        train_views = build_esm_feature_views(bundle.train_df, train_embeddings)
        test_views = build_esm_feature_views(bundle.test_df, test_embeddings)
        tuning_records = []
        candidate_results = []
        for candidate in esm_tuning_candidates():
            candidate_result = train_esm_stacking_variant(
                variant_name=variant_name,
                variant_label=VARIANT_LABELS[variant_name],
                train_views=train_views,
                test_views=test_views,
                y_train=y_train,
                seed=config.seed,
                n_splits=config.n_splits,
                candidate=candidate,
            )
            candidate_metrics, candidate_thresholds = build_metric_tables(
                variant_name=variant_name,
                variant_label=VARIANT_LABELS[variant_name],
                y_train=y_train,
                y_test=y_test,
                train_prob=candidate_result.train_probabilities,
                test_prob=candidate_result.test_probabilities,
            )
            train_row = candidate_metrics.loc[
                (candidate_metrics["EvaluationSplit"] == "TrainOOF")
                & (candidate_metrics["ThresholdKey"] == "best_mcc_on_train_oof")
            ].iloc[0]
            test_row = candidate_metrics.loc[
                (candidate_metrics["EvaluationSplit"] == "IndependentTest")
                & (candidate_metrics["ThresholdKey"] == "best_mcc_on_train_oof")
            ].iloc[0]
            tuning_records.append(
                {
                    "CandidateName": candidate.name,
                    "CandidateLabel": candidate.label,
                    "Views": ",".join(candidate.view_names),
                    "TrainOofPrecisionRecallAuc": float(train_row["PrecisionRecallAuc"]),
                    "TrainOofMatthewsCorrcoef": float(train_row["MatthewsCorrcoef"]),
                    "IndependentTestPrecisionRecallAuc": float(test_row["PrecisionRecallAuc"]),
                    "IndependentTestMatthewsCorrcoef": float(test_row["MatthewsCorrcoef"]),
                    "ChosenThresholdValue": float(candidate_thresholds["best_mcc_on_train_oof"]),
                }
            )
            candidate_results.append((candidate, candidate_result, candidate_metrics))

        tuning_summary = pd.DataFrame(tuning_records).sort_values(
            ["TrainOofPrecisionRecallAuc", "TrainOofMatthewsCorrcoef"],
            ascending=False,
        ).reset_index(drop=True)
        chosen_candidate_name = str(tuning_summary.iloc[0]["CandidateName"])
        chosen_candidate, result, _ = next(
            entry for entry in candidate_results if entry[0].name == chosen_candidate_name
        )
        _write_csv(tuning_summary, variant_dir / "tuning_summary.csv", index=False)
        run_metadata["TuningCandidates"] = tuning_records
        run_metadata["ChosenCandidate"] = {
            "CandidateName": chosen_candidate.name,
            "CandidateLabel": chosen_candidate.label,
            "Views": list(chosen_candidate.view_names),
        }
    else:
        train_handcrafted = build_handcrafted_features(
            bundle.train_df,
            config.cache_dir / "handcrafted_train_features.npz",
        )
        test_handcrafted = build_handcrafted_features(
            bundle.test_df,
            config.cache_dir / "handcrafted_test_features.npz",
        )
        train_features, test_features, feature_names = _combined_esm_handcrafted_features(
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            train_handcrafted=train_handcrafted,
            test_handcrafted=test_handcrafted,
        )
        if variant_name == "handcrafted_with_fs":
            selection = select_handcrafted_features(
                train_features=train_features,
                test_features=test_features,
                y_train=y_train,
                feature_names=feature_names,
                seed=config.seed,
            )
            train_features = selection.train_features
            test_features = selection.test_features
            feature_names = selection.feature_names
            _write_text(json.dumps(selection.selection_summary, indent=2), variant_dir / "feature_selection_summary.json")
            run_metadata["FeatureSelection"] = selection.selection_summary
        else:
            run_metadata["FeatureSelection"] = "disabled"

        result = train_single_xgb_variant(
            variant_name=variant_name,
            variant_label=VARIANT_LABELS[variant_name],
            train_features=train_features,
            test_features=test_features,
            y_train=y_train,
            seed=config.seed,
            n_splits=config.n_splits,
            feature_names=feature_names,
        )
        run_metadata["FeatureCount"] = len(feature_names)
        run_metadata["FeatureComposition"] = {
            "EsmMeanDimensions": int(train_embeddings.mean_embeddings.shape[1]),
            "EsmMaxDimensions": int(train_embeddings.max_embeddings.shape[1]),
            "HandcraftedDimensions": int(train_handcrafted.features.shape[1]),
        }

    metrics_table, thresholds = build_metric_tables(
        variant_name=result.variant_name,
        variant_label=result.variant_label,
        y_train=y_train,
        y_test=y_test,
        train_prob=result.train_probabilities,
        test_prob=result.test_probabilities,
    )
    _write_csv(metrics_table, variant_dir / "metrics.csv", index=False)
    _write_csv(result.fold_summary, variant_dir / "fold_summary.csv", index=False)
    _write_variant_predictions(
        variant_dir=variant_dir,
        variant_name=result.variant_name,
        train_df=bundle.train_df,
        test_df=bundle.test_df,
        train_prob=result.train_probabilities,
        test_prob=result.test_probabilities,
        train_aux=result.train_aux,
        test_aux=result.test_aux,
        thresholds=thresholds,
    )

    confusion_threshold_key = "best_mcc_on_train_oof"
    confusion_threshold = thresholds[confusion_threshold_key]
    train_cm_df, test_cm_df = _plot_confusion_matrices(
        train_labels=y_train,
        train_prob=result.train_probabilities,
        test_labels=y_test,
        test_prob=result.test_probabilities,
        threshold=confusion_threshold,
        threshold_label=THRESHOLD_LABELS[confusion_threshold_key],
        output_path=variant_dir / "confusion_matrices.png",
    )
    _write_csv(train_cm_df, variant_dir / "train_confusion_matrix.csv", index=True)
    _write_csv(test_cm_df, variant_dir / "test_confusion_matrix.csv", index=True)

    train_top_hits, train_top_summary = _select_top_short_peptides(
        bundle.train_df,
        result.train_probabilities,
        max_length=15,
        top_k=10,
    )
    test_top_hits, test_top_summary = _select_top_short_peptides(
        bundle.test_df,
        result.test_probabilities,
        max_length=15,
        top_k=10,
    )
    _write_csv(train_top_hits, variant_dir / "train_top10_short_peptides.csv", index=False)
    _write_csv(test_top_hits, variant_dir / "test_top10_short_peptides.csv", index=False)
    _write_text(
        json.dumps({"TrainOOF": train_top_summary, "IndependentTest": test_top_summary}, indent=2),
        variant_dir / "top_peptide_summary.json",
    )
    if variant_name == "esm_only":
        _write_attribution_artifacts(
            variant_dir=variant_dir,
            train_top_hits=train_top_hits,
            test_top_hits=test_top_hits,
            train_df=bundle.train_df,
            test_df=bundle.test_df,
            embedder=embedder,
            model_bundle=result.model_bundle,
        )

    _plot_curves(
        train_labels=y_train,
        train_prob=result.train_probabilities,
        test_labels=y_test,
        test_prob=result.test_probabilities,
        output_path=variant_dir / "roc_pr_curves.png",
        title_prefix=VARIANT_LABELS[variant_name],
    )
    save_model_bundle(result.model_bundle, thresholds, config.model_dir / f"{variant_name}_bundle.pkl")
    _write_text(json.dumps(run_metadata, indent=2), variant_dir / "run_config.json")
    return metrics_table


def run_pipeline(config: PipelineConfig) -> None:
    bundle = load_datasets(config.train_csv, config.test_csv)
    write_audit(bundle.audit, config.results_dir / "data_audit.json")

    if config.variant == "all":
        variant_names = list(VARIANT_LABELS.keys())
    else:
        variant_names = [config.variant]

    metrics_tables = []
    for variant_name in variant_names:
        metrics_tables.append(_run_single_variant(config, bundle, variant_name))

    _write_comparison(metrics_tables, config.results_dir)
