"""產生自包含的 Colab notebook：IL10_peptide_model_colab.ipynb

從現有的 src/*.py 與 main.py 讀取真實程式碼，組成一個可在 Colab
由上往下執行的 notebook（用 %%writefile 在 Colab 端重建套件）。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_FILES = [
    "data/peptide_level_dataset_MHCII.csv",
    "data/all_minus_benchmark_minus_mhcii.csv",
]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def read_bytes(path: str) -> bytes:
    return (ROOT / path).read_bytes()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def writefile_cell(rel_path: str) -> dict:
    body = read(rel_path)
    return code(f"%%writefile {rel_path}\n{body}")


def embed_data_cell() -> dict:
    """把 CSV 以 base64 內嵌，執行時解碼寫入 data/。"""
    lines = [
        "import base64, os\n",
        "os.makedirs('data', exist_ok=True)\n",
        "_FILES = {\n",
    ]
    for rel in DATA_FILES:
        b64 = base64.b64encode(read_bytes(rel)).decode("ascii")
        lines.append(f"    {Path(rel).name!r}: \"{b64}\",\n")
    lines += [
        "}\n",
        "for _name, _b64 in _FILES.items():\n",
        "    with open(os.path.join('data', _name), 'wb') as _f:\n",
        "        _f.write(base64.b64decode(_b64))\n",
        "print('data/ 內容：', os.listdir('data'))\n",
    ]
    return code("".join(lines))


cells: list[dict] = []

cells.append(md(
    "# IL-10 Induced Peptide Model — Colab 版\n"
    "\n"
    "**自包含 notebook**：由上往下依序執行每個 cell（或直接 Run all）即可。會在 Colab 環境重建 "
    "`src/` 套件、安裝套件、寫入內嵌資料、訓練並顯示結果。**資料已內嵌，無需上傳。**\n"
    "\n"
    "> 執行前請先到 **執行階段 / Runtime → 變更執行階段類型 → 硬體加速器：GPU**。"
    "ESM-2 (650M) 在 GPU 上快非常多。\n"
))

cells.append(md("## 0. 環境檢查"))
cells.append(code(
    "import torch\n"
    "print('PyTorch:', torch.__version__)\n"
    "print('GPU available:', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('GPU:', torch.cuda.get_device_name(0))\n"
    "else:\n"
    "    print('⚠️ 目前是 CPU runtime；ESM-2 650M 會很慢，建議改用 GPU runtime。')\n"
))

cells.append(md("## 1. 安裝套件\n\n（`torch`/`scikit-learn`/`pandas`/`numpy`/`matplotlib`/`tqdm` Colab 已內建，這裡補裝 `fair-esm` 與 `propy3`。）"))
cells.append(code(
    "!pip -q install fair-esm==2.0.0 \"propy3>=1.1.1\" \"xgboost>=2.1\"\n"
))

cells.append(md("## 2. 重建 `src/` 套件\n\n以下 cell 會把專案原始碼寫進 Colab 檔案系統（內容與原 repo 一致）。"))
cells.append(code(
    "import os\n"
    "for d in ['src', 'data', 'results', 'artifacts/cache', 'artifacts/models']:\n"
    "    os.makedirs(d, exist_ok=True)\n"
    "print('資料夾建立完成')\n"
))
cells.append(writefile_cell("src/__init__.py"))
cells.append(writefile_cell("src/data.py"))
cells.append(writefile_cell("src/features.py"))
cells.append(writefile_cell("src/model.py"))
cells.append(writefile_cell("src/pipeline.py"))
cells.append(writefile_cell("main.py"))

cells.append(md(
    "## 3. 寫入內嵌資料\n\n"
    "兩個 CSV 已用 base64 內嵌在這個 notebook（約 100 KB）。"
    "執行此格會自動解碼寫入 `data/`，**完全不需要上傳**。\n"
))
cells.append(embed_data_cell())

cells.append(md(
    "## 4. 訓練\n\n"
    "預設只跑 **`esm_only`**（效果最好的版本）。\n"
    "- 想跑三版本比較：把 `variant` 改成 `\"all\"`。\n"
    "- 想快速冒煙測試：把 `esm_model` 改成 `\"esm2_t12_35M_UR50D\"`（較小、較快）。\n"
))
cells.append(code(
    "import torch\n"
    "# 相容性修正：較新版 PyTorch 的 torch.load 預設 weights_only=True，\n"
    "# 會讓 fair-esm 載入官方權重失敗。ESM-2 權重來自官方來源，這裡明確設回 False。\n"
    "_orig_load = torch.load\n"
    "def _safe_load(*args, **kwargs):\n"
    "    kwargs['weights_only'] = False\n"
    "    return _orig_load(*args, **kwargs)\n"
    "torch.load = _safe_load\n"
    "\n"
    "from pathlib import Path\n"
    "from src.pipeline import PipelineConfig, run_pipeline\n"
    "\n"
    "config = PipelineConfig(\n"
    "    project_root=Path('.'),\n"
    "    train_csv=Path('data/peptide_level_dataset_MHCII.csv'),\n"
    "    test_csv=Path('data/all_minus_benchmark_minus_mhcii.csv'),\n"
    "    results_dir=Path('results'),\n"
    "    cache_dir=Path('artifacts/cache'),\n"
    "    model_dir=Path('artifacts/models'),\n"
    "    variant='esm_only',                 # 想比較三版本 → 'all'\n"
    "    esm_model='esm2_t33_650M_UR50D',    # 快速測試 → 'esm2_t12_35M_UR50D'\n"
    "    batch_size=32,\n"
    "    n_splits=5,\n"
    "    seed=42,\n"
    ")\n"
    "run_pipeline(config)\n"
    "print('✅ 完成，結果寫入 results/')\n"
))

cells.append(md("## 5. 檢視結果"))
cells.append(code(
    "import pandas as pd\n"
    "from IPython.display import Image, display\n"
    "\n"
    "metrics = pd.read_csv('results/esm_only/metrics.csv')\n"
    "display(metrics)\n"
    "\n"
    "for img in ['results/esm_only/roc_pr_curves.png',\n"
    "            'results/esm_only/confusion_matrices.png',\n"
    "            'results/esm_only/attribution_heatmaps.png']:\n"
    "    try:\n"
    "        display(Image(img))\n"
    "    except FileNotFoundError:\n"
    "        pass\n"
    "\n"
    "print('\\nTest 集前 10 條短胜肽：')\n"
    "display(pd.read_csv('results/esm_only/test_top10_short_peptides.csv'))\n"
))

cells.append(md("## 6. 下載結果（可選）"))
cells.append(code(
    "!zip -q -r results.zip results artifacts/models\n"
    "from google.colab import files\n"
    "files.download('results.zip')\n"
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "IL10_peptide_model_colab.ipynb"
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"已產生 {out}（{len(cells)} 個 cell）")
