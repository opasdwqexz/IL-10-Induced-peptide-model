from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .data import STANDARD_AAS, dataframe_to_mod_array


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_LIST)}


@dataclass
class EmbeddingFeatures:
    mean_embeddings: np.ndarray
    max_embeddings: np.ndarray


@dataclass
class HandcraftedFeatureSet:
    features: np.ndarray
    feature_names: list[str]


@dataclass
class FeatureSelectionResult:
    train_features: np.ndarray
    test_features: np.ndarray
    feature_names: list[str]
    selection_summary: dict


class ESMEmbedder:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._alphabet = None
        self._device = None
        self._layer_index = int(model_name.split("_t")[1].split("_")[0])

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _load_model(self) -> tuple[object, object]:
        if self._model is None or self._alphabet is None:
            import esm

            model_loader = getattr(esm.pretrained, self.model_name)
            model, alphabet = model_loader()
            model = model.to(self.device).eval()
            self._model = model
            self._alphabet = alphabet
        return self._model, self._alphabet

    def _embed_sequences(self, sequences: list[str]) -> EmbeddingFeatures:
        import torch
        from tqdm import tqdm

        model, alphabet = self._load_model()
        batch_converter = alphabet.get_batch_converter()
        mean_vectors: list[np.ndarray] = []
        max_vectors: list[np.ndarray] = []

        for start in tqdm(range(0, len(sequences), self.batch_size), desc="ESM2", ncols=80):
            batch = sequences[start : start + self.batch_size]
            entries = [(f"seq_{idx}", sequence) for idx, sequence in enumerate(batch)]
            _, _, tokens = batch_converter(entries)
            tokens = tokens.to(self.device, non_blocking=self.device == "cuda")
            with torch.no_grad():
                outputs = model(tokens, repr_layers=[self._layer_index], return_contacts=False)

            reps = outputs["representations"][self._layer_index]
            for row_index, sequence in enumerate(batch):
                residue_reps = reps[row_index, 1 : len(sequence) + 1]
                mean_vectors.append(residue_reps.mean(dim=0).detach().cpu().numpy())
                max_vectors.append(residue_reps.max(dim=0).values.detach().cpu().numpy())

        return EmbeddingFeatures(
            mean_embeddings=np.asarray(mean_vectors, dtype=np.float32),
            max_embeddings=np.asarray(max_vectors, dtype=np.float32),
        )

    def embed_uncached(self, sequences: list[str]) -> EmbeddingFeatures:
        return self._embed_sequences(sequences)

    def embed(self, sequences: list[str], cache_path: Path) -> EmbeddingFeatures:
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            cached_sequences = cached["sequences"].tolist()
            cached_model_name = str(cached["model_name"].item()) if "model_name" in cached else ""
            if cached_model_name == self.model_name and list(cached_sequences) == list(sequences):
                return EmbeddingFeatures(
                    mean_embeddings=cached["mean_embeddings"].astype(np.float32),
                    max_embeddings=cached["max_embeddings"].astype(np.float32),
                )

        features = self._embed_sequences(sequences)
        np.savez_compressed(
            cache_path,
            model_name=np.asarray(self.model_name),
            sequences=np.asarray(sequences, dtype=object),
            mean_embeddings=features.mean_embeddings,
            max_embeddings=features.max_embeddings,
        )
        return features


def build_esm_feature_views(df: pd.DataFrame, embeddings: EmbeddingFeatures) -> dict[str, np.ndarray]:
    mod_array = dataframe_to_mod_array(df).to_numpy(dtype=np.float32)
    mean_view = np.concatenate([embeddings.mean_embeddings, mod_array], axis=1)
    max_view = np.concatenate([embeddings.max_embeddings, mod_array], axis=1)
    concat_view = np.concatenate([embeddings.mean_embeddings, embeddings.max_embeddings, mod_array], axis=1)
    return {
        "mean": mean_view.astype(np.float32),
        "max": max_view.astype(np.float32),
        "concat": concat_view.astype(np.float32),
    }


def _aac_vector(sequence: str) -> np.ndarray:
    length = len(sequence)
    counts = np.zeros(len(AA_LIST), dtype=np.float32)
    for aa in sequence:
        counts[AA_TO_INDEX[aa]] += 1.0
    return counts / max(length, 1)


def _dpc_vector(sequence: str) -> np.ndarray:
    vector = np.zeros(len(AA_LIST) ** 2, dtype=np.float32)
    if len(sequence) < 2:
        return vector
    for left, right in zip(sequence[:-1], sequence[1:]):
        vector[AA_TO_INDEX[left] * len(AA_LIST) + AA_TO_INDEX[right]] += 1.0
    return vector / (len(sequence) - 1)


def _aaindex_lookup() -> dict[str, dict[str, float]]:
    from propy.AAIndex import _aaindex, init
    import propy

    aaindex_path = os.path.join(os.path.dirname(propy.__file__), "aaindex")
    init(path=aaindex_path)
    lookup: dict[str, dict[str, float]] = {}
    for name, entry in _aaindex.items():
        index = getattr(entry, "index", None)
        if index is None:
            continue
        if all(aa in index and index[aa] is not None for aa in AA_LIST):
            lookup[name] = {aa: float(index[aa]) for aa in AA_LIST}
    return lookup


def build_handcrafted_features(df: pd.DataFrame, cache_path: Path) -> HandcraftedFeatureSet:
    sequences = df["BaseSeq"].tolist()
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_sequences = cached["sequences"].tolist() if "sequences" in cached else []
        if list(cached_sequences) == list(sequences):
            return HandcraftedFeatureSet(
                features=cached["features"].astype(np.float32),
                feature_names=cached["feature_names"].tolist(),
            )

    mod_frame = dataframe_to_mod_array(df)
    aac_matrix = np.vstack([_aac_vector(sequence) for sequence in sequences]).astype(np.float32)
    dpc_matrix = np.vstack([_dpc_vector(sequence) for sequence in sequences]).astype(np.float32)
    aaindex_lookup = _aaindex_lookup()
    aaindex_names = sorted(aaindex_lookup.keys())

    aaindex_rows = []
    for sequence in sequences:
        aaindex_rows.append(
            [float(np.mean([aaindex_lookup[name][aa] for aa in sequence])) for name in aaindex_names]
        )
    aaindex_matrix = np.asarray(aaindex_rows, dtype=np.float32)
    mod_matrix = mod_frame.to_numpy(dtype=np.float32)

    feature_names = (
        [f"AAC_{aa}" for aa in AA_LIST]
        + [f"DPC_{left}{right}" for left in AA_LIST for right in AA_LIST]
        + [f"AAINDEX_{name}" for name in aaindex_names]
        + list(mod_frame.columns)
    )
    features = np.concatenate([aac_matrix, dpc_matrix, aaindex_matrix, mod_matrix], axis=1).astype(np.float32)
    np.savez_compressed(
        cache_path,
        sequences=np.asarray(sequences, dtype=object),
        features=features,
        feature_names=np.asarray(feature_names, dtype=object),
    )
    return HandcraftedFeatureSet(features=features, feature_names=feature_names)


def select_handcrafted_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    seed: int,
) -> FeatureSelectionResult:
    from xgboost import XGBClassifier

    mask = np.ones(len(feature_names), dtype=bool)
    summary: dict[str, int | float] = {"initial_features": int(len(feature_names))}

    variance_selector = VarianceThreshold(threshold=1e-5)
    variance_selector.fit(train_features)
    variance_mask = variance_selector.get_support()
    mask &= variance_mask
    summary["removed_low_variance"] = int((~variance_mask).sum())

    selected_names = [name for name, keep in zip(feature_names, mask) if keep]
    selected_train = train_features[:, mask]
    selected_test = test_features[:, mask]

    aaindex_positions = [idx for idx, name in enumerate(selected_names) if name.startswith("AAINDEX_")]
    corr_keep = np.ones(selected_train.shape[1], dtype=bool)
    if len(aaindex_positions) > 1:
        aaindex_corr = np.corrcoef(selected_train[:, aaindex_positions].T)
        upper = np.triu(np.abs(aaindex_corr), k=1)
        drop_indices: set[int] = set()
        for col in range(upper.shape[1]):
            if col in drop_indices:
                continue
            correlated = np.where(upper[:col, col] > 0.95)[0]
            drop_indices.update(int(index) for index in correlated)
        for local_index in drop_indices:
            corr_keep[aaindex_positions[local_index]] = False
        summary["removed_correlated_aaindex"] = int(len(drop_indices))
    else:
        summary["removed_correlated_aaindex"] = 0

    selected_names = [name for name, keep in zip(selected_names, corr_keep) if keep]
    selected_train = selected_train[:, corr_keep]
    selected_test = selected_test[:, corr_keep]

    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    importance_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=8.0,
        scale_pos_weight=scale_pos,
        tree_method="hist",
        device="cpu",
        random_state=seed,
        n_jobs=-1,
    )
    importance_model.fit(selected_train, y_train, verbose=False)
    importances = importance_model.feature_importances_
    importance_threshold = float(importances.mean())
    importance_mask = importances >= importance_threshold
    selected_names = [name for name, keep in zip(selected_names, importance_mask) if keep]
    selected_train = selected_train[:, importance_mask]
    selected_test = selected_test[:, importance_mask]
    summary["removed_low_importance"] = int((~importance_mask).sum())
    summary["final_features"] = int(len(selected_names))
    summary["importance_threshold"] = importance_threshold

    return FeatureSelectionResult(
        train_features=selected_train.astype(np.float32),
        test_features=selected_test.astype(np.float32),
        feature_names=selected_names,
        selection_summary=summary,
    )
