from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


THRESHOLD_LABELS = {
    "default_0p5": "Default threshold 0.5",
    "best_mcc_on_train_oof": "Best MCC threshold from train OOF",
    "best_f2_on_train_oof": "Best F2 threshold from train OOF",
}


@dataclass
class VariantRunResult:
    variant_name: str
    variant_label: str
    train_probabilities: np.ndarray
    test_probabilities: np.ndarray
    train_aux: dict[str, np.ndarray]
    test_aux: dict[str, np.ndarray]
    fold_summary: pd.DataFrame
    model_bundle: dict


@dataclass(frozen=True)
class ESMStackingCandidate:
    name: str
    label: str
    view_names: tuple[str, ...]
    param_overrides: dict[str, dict[str, float]]


def _safe_n_splits(y: np.ndarray, desired: int) -> int:
    class_counts = np.bincount(y.astype(int))
    positive_min = int(class_counts[class_counts > 0].min())
    return max(2, min(desired, positive_min))


def _metric_row(
    variant_name: str,
    variant_label: str,
    evaluation_split: str,
    threshold_key: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    sensitivity = tp / (tp + fn) if (tp + fn) else math.nan
    return {
        "VariantName": variant_name,
        "VariantLabel": variant_label,
        "EvaluationSplit": evaluation_split,
        "ThresholdKey": threshold_key,
        "ThresholdLabel": THRESHOLD_LABELS[threshold_key],
        "ThresholdValue": threshold,
        "RocAuc": roc_auc_score(y_true, y_prob),
        "PrecisionRecallAuc": average_precision_score(y_true, y_prob),
        "MatthewsCorrcoef": matthews_corrcoef(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "RecallSensitivity": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1Score": f1_score(y_true, y_pred, zero_division=0),
        "F2Score": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "TrueNegative": int(tn),
        "FalsePositive": int(fp),
        "FalseNegative": int(fn),
        "TruePositive": int(tp),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, objective: str) -> float:
    candidates = np.unique(np.clip(y_prob, 0.01, 0.99))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        y_pred = (y_prob >= threshold).astype(int)
        if objective == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif objective == "f2":
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        else:
            raise ValueError(f"Unsupported threshold objective: {objective}")
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def build_metric_tables(
    variant_name: str,
    variant_label: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_prob: np.ndarray,
    test_prob: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {
        "default_0p5": 0.5,
        "best_mcc_on_train_oof": select_threshold(y_train, train_prob, "mcc"),
        "best_f2_on_train_oof": select_threshold(y_train, train_prob, "f2"),
    }
    rows = []
    for split_name, labels, probabilities in (
        ("TrainOOF", y_train, train_prob),
        ("IndependentTest", y_test, test_prob),
    ):
        for threshold_key, threshold_value in thresholds.items():
            rows.append(
                _metric_row(
                    variant_name=variant_name,
                    variant_label=variant_label,
                    evaluation_split=split_name,
                    threshold_key=threshold_key,
                    y_true=labels,
                    y_prob=probabilities,
                    threshold=threshold_value,
                )
            )
    return pd.DataFrame(rows), thresholds


def _xgb_params(scale_pos_weight: float, max_depth: int, learning_rate: float, seed: int, with_early_stopping: bool) -> dict:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 1200,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 8.0,
        "gamma": 0.1,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": seed,
        "n_jobs": -1,
    }
    if with_early_stopping:
        params["early_stopping_rounds"] = 80
    return params


def train_single_xgb_variant(
    variant_name: str,
    variant_label: str,
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_splits: int,
    feature_names: list[str],
) -> VariantRunResult:
    oof_prob = np.zeros(len(y_train), dtype=np.float32)
    best_iterations: list[int] = []
    fold_rows: list[dict] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(train_features, y_train), start=1):
        y_fold_train = y_train[train_idx]
        scale_pos = float((y_fold_train == 0).sum() / max((y_fold_train == 1).sum(), 1))
        model = XGBClassifier(
            **_xgb_params(
                scale_pos_weight=scale_pos,
                max_depth=5,
                learning_rate=0.04,
                seed=seed,
                with_early_stopping=True,
            )
        )
        model.fit(
            train_features[train_idx],
            y_fold_train,
            eval_set=[(train_features[valid_idx], y_train[valid_idx])],
            verbose=False,
        )
        fold_prob = model.predict_proba(train_features[valid_idx])[:, 1]
        oof_prob[valid_idx] = fold_prob
        best_iteration = int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
        best_iterations.append(best_iteration)
        fold_rows.append(
            {
                "FoldIndex": fold_index,
                "BestIteration": best_iteration,
                "ValidationRocAuc": roc_auc_score(y_train[valid_idx], fold_prob),
                "ValidationPrecisionRecallAuc": average_precision_score(y_train[valid_idx], fold_prob),
            }
        )

    full_scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    final_model = XGBClassifier(
        **_xgb_params(
            scale_pos_weight=full_scale_pos,
            max_depth=5,
            learning_rate=0.04,
            seed=seed,
            with_early_stopping=False,
        )
    )
    final_model.set_params(n_estimators=int(round(np.mean(best_iterations))))
    final_model.fit(train_features, y_train, verbose=False)
    test_prob = final_model.predict_proba(test_features)[:, 1]

    return VariantRunResult(
        variant_name=variant_name,
        variant_label=variant_label,
        train_probabilities=oof_prob,
        test_probabilities=test_prob,
        train_aux={},
        test_aux={},
        fold_summary=pd.DataFrame(fold_rows),
        model_bundle={
            "model_type": "single_xgboost",
            "feature_names": feature_names,
            "model": final_model,
            "mean_best_iteration": float(np.mean(best_iterations)),
        },
    )


def _stack_base_params(scale_pos_weight: float, variant: str, seed: int, with_early_stopping: bool) -> dict:
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.6,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 8.0,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": seed,
        "n_jobs": -1,
    }
    per_view = {
        "mean": {"max_depth": 3, "gamma": 0.05},
        "max": {"max_depth": 4, "gamma": 0.15},
        "concat": {
            "max_depth": 4,
            "gamma": 0.1,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.45,
            "min_child_weight": 5,
        },
    }
    if variant not in per_view:
        raise ValueError(f"Unknown stack base variant: {variant}")
    params = dict(base_params)
    params.update(per_view[variant])
    if with_early_stopping:
        params["early_stopping_rounds"] = 80
    return params


def _fit_stack_base_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    variant: str,
    seed: int,
    param_overrides: dict[str, float] | None = None,
) -> tuple[XGBClassifier, np.ndarray, int]:
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = XGBClassifier(**_stack_base_params(scale_pos, variant, seed, with_early_stopping=True))
    if param_overrides:
        model.set_params(**param_overrides)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    valid_prob = model.predict_proba(x_valid)[:, 1]
    best_iter = int(getattr(model, "best_iteration", model.n_estimators - 1)) + 1
    return model, valid_prob, best_iter


def esm_tuning_candidates() -> list[ESMStackingCandidate]:
    return [
        ESMStackingCandidate(
            name="baseline_mean_max",
            label="Mean + max pooling stack",
            view_names=("mean", "max"),
            param_overrides={},
        ),
        ESMStackingCandidate(
            name="mean_max_concat",
            label="Mean + max + concat stack",
            view_names=("mean", "max", "concat"),
            param_overrides={},
        ),
        ESMStackingCandidate(
            name="regularized_mean_max",
            label="Regularized mean + max stack",
            view_names=("mean", "max"),
            param_overrides={
                "mean": {"learning_rate": 0.02, "max_depth": 2, "colsample_bytree": 0.45, "gamma": 0.1},
                "max": {"learning_rate": 0.02, "max_depth": 3, "colsample_bytree": 0.5, "gamma": 0.2},
            },
        ),
    ]


def _view_probability_name(view_name: str) -> str:
    mapping = {
        "mean": "MeanEmbeddingProbability",
        "max": "MaxEmbeddingProbability",
        "concat": "ConcatEmbeddingProbability",
    }
    return mapping[view_name]


def train_esm_stacking_variant(
    variant_name: str,
    variant_label: str,
    train_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    y_train: np.ndarray,
    seed: int,
    n_splits: int,
    candidate: ESMStackingCandidate | None = None,
) -> VariantRunResult:
    candidate = candidate or esm_tuning_candidates()[0]
    view_names = list(candidate.view_names)
    oof_base = np.zeros((len(y_train), len(view_names)), dtype=np.float32)
    oof_stack = np.zeros(len(y_train), dtype=np.float32)
    best_iterations = {view_name: [] for view_name in view_names}
    fold_rows: list[dict] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(train_views["mean"], y_train), start=1):
        y_fold_train = y_train[train_idx]
        outer_valid_base = np.zeros((len(valid_idx), len(view_names)), dtype=np.float32)
        inner_train_views = {name: values[train_idx] for name, values in train_views.items()}
        inner_splits = _safe_n_splits(y_fold_train, max(3, n_splits - 1))
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold_index)
        inner_oof_base = np.zeros((len(train_idx), len(view_names)), dtype=np.float32)

        for inner_train_idx, inner_valid_idx in inner_cv.split(inner_train_views["mean"], y_fold_train):
            for column_index, view_name in enumerate(view_names):
                _, inner_valid_prob, _ = _fit_stack_base_model(
                    x_train=inner_train_views[view_name][inner_train_idx],
                    y_train=y_fold_train[inner_train_idx],
                    x_valid=inner_train_views[view_name][inner_valid_idx],
                    y_valid=y_fold_train[inner_valid_idx],
                    variant=view_name,
                    seed=seed,
                    param_overrides=candidate.param_overrides.get(view_name),
                )
                inner_oof_base[inner_valid_idx, column_index] = inner_valid_prob

        for column_index, view_name in enumerate(view_names):
            _, valid_prob, best_iter = _fit_stack_base_model(
                x_train=train_views[view_name][train_idx],
                y_train=y_fold_train,
                x_valid=train_views[view_name][valid_idx],
                y_valid=y_train[valid_idx],
                variant=view_name,
                seed=seed,
                param_overrides=candidate.param_overrides.get(view_name),
            )
            oof_base[valid_idx, column_index] = valid_prob
            outer_valid_base[:, column_index] = valid_prob
            best_iterations[view_name].append(best_iter)

        meta_fold = LogisticRegression(class_weight="balanced", max_iter=4000, random_state=seed)
        meta_fold.fit(inner_oof_base, y_fold_train)
        stack_valid_prob = meta_fold.predict_proba(outer_valid_base)[:, 1]
        oof_stack[valid_idx] = stack_valid_prob

        fold_rows.append(
            {
                "FoldIndex": fold_index,
                "StackPrAuc": average_precision_score(y_train[valid_idx], stack_valid_prob),
                "StackRocAuc": roc_auc_score(y_train[valid_idx], stack_valid_prob),
                "CandidateName": candidate.name,
            }
        )
        for column_index, view_name in enumerate(view_names):
            fold_rows[-1][f"{view_name.title()}BestIteration"] = best_iterations[view_name][-1]
            fold_rows[-1][f"{view_name.title()}PrAuc"] = average_precision_score(
                y_train[valid_idx], outer_valid_base[:, column_index]
            )

    meta_model = LogisticRegression(class_weight="balanced", max_iter=4000, random_state=seed)
    meta_model.fit(oof_base, y_train)

    full_models: dict[str, XGBClassifier] = {}
    test_base = np.zeros((test_views["mean"].shape[0], len(view_names)), dtype=np.float32)
    full_scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    for column_index, view_name in enumerate(view_names):
        params = _stack_base_params(full_scale_pos, view_name, seed, with_early_stopping=False)
        params.update(candidate.param_overrides.get(view_name, {}))
        params["n_estimators"] = int(round(np.mean(best_iterations[view_name])))
        model = XGBClassifier(**params)
        model.fit(train_views[view_name], y_train, verbose=False)
        full_models[view_name] = model
        test_base[:, column_index] = model.predict_proba(test_views[view_name])[:, 1]

    test_stack = meta_model.predict_proba(test_base)[:, 1]
    return VariantRunResult(
        variant_name=variant_name,
        variant_label=variant_label,
        train_probabilities=oof_stack,
        test_probabilities=test_stack,
        train_aux={**{_view_probability_name(view_name): oof_base[:, idx] for idx, view_name in enumerate(view_names)}, "StackedProbability": oof_stack},
        test_aux={**{_view_probability_name(view_name): test_base[:, idx] for idx, view_name in enumerate(view_names)}, "StackedProbability": test_stack},
        fold_summary=pd.DataFrame(fold_rows),
        model_bundle={
            "model_type": "esm_stacking",
            "candidate_name": candidate.name,
            "candidate_label": candidate.label,
            "view_names": view_names,
            "base_models": full_models,
            "meta_model": meta_model,
            "mean_best_iterations": {view_name: float(np.mean(best_iterations[view_name])) for view_name in view_names},
        },
    )


def save_model_bundle(model_bundle: dict, thresholds: dict[str, float], path: Path) -> None:
    payload = dict(model_bundle)
    payload["thresholds"] = thresholds
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
