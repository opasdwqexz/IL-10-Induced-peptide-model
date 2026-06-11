from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
MOD_KEYWORDS = ("PHOS", "DEAM", "CITR", "ACET", "MCM")


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    audit: dict


def _require_columns(df: pd.DataFrame, path: Path) -> None:
    missing = {"Sequence", "FinalLabel"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")


def _parse_sequence(raw_value: object) -> dict:
    raw = str(raw_value).strip().upper()
    left, has_plus, right = raw.partition("+")
    base_seq = re.sub(r"\s+", "", left)
    annotation = right.strip()
    keyword_hits = {key: int(key in raw) for key in MOD_KEYWORDS}
    mod_count = sum(raw.count(key) for key in MOD_KEYWORDS)
    is_standard = bool(base_seq) and all(char in STANDARD_AAS for char in base_seq)
    return {
        "Sequence": raw,
        "BaseSeq": base_seq,
        "HasMod": int(has_plus == "+"),
        "ModPHOS": keyword_hits["PHOS"],
        "ModDEAM": keyword_hits["DEAM"],
        "ModCITR": keyword_hits["CITR"],
        "ModACET": keyword_hits["ACET"],
        "ModMCM": keyword_hits["MCM"],
        "ModCount": mod_count,
        "Annotation": annotation,
        "IsStandardBaseSeq": int(is_standard),
        "SeqLength": len(base_seq),
        "LogLength": math.log1p(len(base_seq)),
    }


def _prepare_dataframe(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    _require_columns(df, path)
    parsed = pd.DataFrame([_parse_sequence(value) for value in df["Sequence"]])
    merged = parsed.copy()
    merged["FinalLabel"] = df["FinalLabel"].astype(int).to_numpy()
    valid = merged["IsStandardBaseSeq"].astype(bool)
    filtered = merged.loc[valid].reset_index(drop=True)

    lengths = filtered["SeqLength"]
    audit = {
        "rows": int(len(df)),
        "valid_rows": int(len(filtered)),
        "invalid_rows": int((~valid).sum()),
        "positive": int((filtered["FinalLabel"] == 1).sum()),
        "negative": int((filtered["FinalLabel"] == 0).sum()),
        "positive_ratio": float(filtered["FinalLabel"].mean()),
        "modified_rows": int(filtered["HasMod"].sum()),
        "unique_raw_sequences": int(filtered["Sequence"].nunique()),
        "unique_base_sequences": int(filtered["BaseSeq"].nunique()),
        "duplicate_raw_sequences": int(filtered["Sequence"].duplicated().sum()),
        "duplicate_base_sequences": int(filtered["BaseSeq"].duplicated().sum()),
        "length_min": int(lengths.min()),
        "length_median": float(lengths.median()),
        "length_max": int(lengths.max()),
    }
    return filtered, audit


def _overlap_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_raw = set(train_df["Sequence"])
    test_raw = set(test_df["Sequence"])
    train_base = set(train_df["BaseSeq"])
    test_base = set(test_df["BaseSeq"])
    raw_overlap = sorted(train_raw & test_raw)
    base_overlap = sorted(train_base & test_base)
    return {
        "raw_overlap_count": len(raw_overlap),
        "base_overlap_count": len(base_overlap),
        "raw_overlap_examples": raw_overlap[:10],
        "base_overlap_examples": base_overlap[:10],
    }


def load_datasets(train_path: Path, test_path: Path) -> DatasetBundle:
    train_df, train_audit = _prepare_dataframe(train_path)
    test_df, test_audit = _prepare_dataframe(test_path)
    audit = {
        "train": train_audit,
        "test": test_audit,
        "overlap": _overlap_summary(train_df, test_df),
        "mod_keywords": list(MOD_KEYWORDS),
    }
    return DatasetBundle(train_df=train_df, test_df=test_df, audit=audit)


def write_audit(audit: dict, path: Path) -> None:
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def modification_feature_columns() -> list[str]:
    return [
        "HasMod",
        "ModPHOS",
        "ModDEAM",
        "ModCITR",
        "ModACET",
        "ModMCM",
        "ModCount",
        "SeqLength",
        "LogLength",
    ]


def dataframe_to_mod_array(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, modification_feature_columns()].copy()

