"""重建三方法比較表（只寫這兩個檔，不動其他任何檔案）。

從三個 variant 既有的 results/<variant>/metrics.csv 重新彙整：
  - results/model_comparison_full_metrics.csv
  - results/model_comparison_summary.csv

彙整邏輯與 src/pipeline.py 的 _write_comparison 相同；summary 取
IndependentTest × best_mcc_on_train_oof 門檻，依 MCC、PR-AUC 由高到低排序。

僅使用 Python 標準函式庫（不需 pandas / pip）。直接以字串搬移數值，
不重新格式化浮點數，確保與原始 metrics.csv 內容一致。

用法：
    python fix_model_comparison.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VARIANTS = ["esm_only", "handcrafted_no_fs", "handcrafted_with_fs"]

SUMMARY_COLUMNS = [
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
]


def main() -> None:
    all_rows: list[dict[str, str]] = []
    header: list[str] | None = None

    for variant in VARIANTS:
        path = RESULTS / variant / "metrics.csv"
        if not path.exists():
            raise SystemExit(f"找不到 {path}，無法重建比較表。")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if header is None:
                header = list(reader.fieldnames or [])
            for row in reader:
                all_rows.append(row)

    if header is None:
        raise SystemExit("metrics.csv 沒有欄位標題。")

    # 1) full metrics：直接把三份 metrics 串接
    full_path = RESULTS / "model_comparison_full_metrics.csv"
    with full_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    # 2) summary：IndependentTest + best_mcc 門檻，依 MCC、PR-AUC 由高到低
    selected = [
        row
        for row in all_rows
        if row["EvaluationSplit"] == "IndependentTest"
        and row["ThresholdKey"] == "best_mcc_on_train_oof"
    ]
    selected.sort(
        key=lambda row: (
            -float(row["MatthewsCorrcoef"]),
            -float(row["PrecisionRecallAuc"]),
        )
    )
    summary_path = RESULTS / "model_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({column: row[column] for column in SUMMARY_COLUMNS})

    print("已重建（僅這兩個檔，其他檔案未更動）：")
    print(" -", full_path)
    print(" -", summary_path)
    print(
        f"summary 共 {len(selected)} 列，排序："
        f" {[row['VariantName'] for row in selected]}"
    )


if __name__ == "__main__":
    main()
