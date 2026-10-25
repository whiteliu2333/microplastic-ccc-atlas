# -*- coding: utf-8 -*-
"""Train the ORIGINAL six-learner five-layer stack on the complete Excel dataset.

Usage:
  python train_export.py --excel Data2.xlsx --sheet 标准化数据

Keep Data2.xlsx local (private). Upload the generated models/ccc_stacking.joblib
with app.py and supporting files to Streamlit Community Cloud via GitHub.

Note: this trains a new final whole-data model. The paper's 5-fold OOF metrics
remain those from the separate original validation run; they are NOT recomputed
or assigned to this fitted production artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import scipy
import xgboost
import lightgbm

import stacking_core as core
from predictor import SCHEMA_VERSION, extract_stack_state, predict_ccc


def fit_bagged_gbr(matrix: np.ndarray, y: np.ndarray, params, seed: int):
    fitted = []
    for offset in core.GBR_BAG_SEEDS:
        estimator = core.make_gbr(seed + int(offset), params)
        estimator.fit(matrix, y)
        fitted.append(estimator)
    return fitted


def train_and_export(excel: Path, sheet: str, output: Path) -> None:
    if not excel.is_file():
        raise FileNotFoundError(
            f"没有找到训练 Excel：{excel}\n"
            "请先把你原来的 Data2.xlsx 放到项目目录，再运行导出命令。"
        )
    if core.TARGET_TRANSFORM != "log10":
        raise ValueError("论文网站要求保留原始 log10(CCC) 目标变换。")

    core.FILE_PATH = str(excel.resolve())
    core.SHEET_NAME = sheet
    print("1/6 读取训练数据、整理文献严格组...")
    X, y_log, y_ccc, strict_groups, meta = core.load_and_prepare_data()
    if len(np.unique(strict_groups)) < 4:
        raise ValueError("严格组数不足，无法进行训练侧 Group-CV。")
    print(f"数据：{len(X)} 条，{meta['literature_count']} 篇文献，"
          f"{meta['strict_group_count']} 个严格组，{X.shape[1]} 个原始输入字段")

    print("2/6 使用原始流程选择预处理策略（训练侧 OOF）...")
    mode, inner_cache = core.select_preprocessor_and_cache(
        X, y_log, strict_groups,
        meta["numeric_columns"],
        meta["categorical_columns"],
        seed=core.RANDOM_STATE * 10000 + 777,
    )
    print("预处理：", mode["Name"])

    candidates = {
        "RF": (core.RF_CANDIDATES, core.make_rf, 101),
        "ExtraTrees": (core.ET_CANDIDATES, core.make_extra_trees, 201),
        "LGBM": (core.LGBM_CANDIDATES, core.make_lgbm, 401),
        "XGB": (core.XGB_CANDIDATES, core.make_xgb, 501),
        "GBR": (core.GBR_CANDIDATES, core.make_gbr, 551),
        "GBR750": (core.GBR750_CANDIDATES, core.make_gbr, 601),
    }
    chosen = {}
    oof = {}
    print("3/6 为六个一级模型建立严格训练侧 OOF 预测...")
    for name in core.BASE_MODEL_NAMES:
        grid, builder, offset = candidates[name]
        details = core.evaluate_candidate_grid(
            inner_cache, y_log, grid, builder,
            seed=core.RANDOM_STATE * 10000 + offset,
        )
        chosen[name] = dict(details["Params"])
        oof[name] = np.asarray(details["OOF_Prediction"], dtype=float)
        print(f"  {name:11s}  training OOF R²(log10) = {details['OOF_R2']:.4f}")

    print("4/6 使用全体有效样本训练六个一级预测器...")
    preprocessor = core.build_preprocessor(
        meta["numeric_columns"], meta["categorical_columns"], mode
    )
    X_encoded = np.asarray(
        preprocessor.fit_transform(X, y_log), dtype=np.float32
    )
    seeds = {
        "RF": 1001,
        "ExtraTrees": 1002,
        "LGBM": 1004,
        "XGB": 1005,
        "GBR": 1005,
        "GBR750": 1006,
    }
    base_models = {}
    for name in core.BASE_MODEL_NAMES:
        if name in ("GBR", "GBR750"):
            base_models[name] = fit_bagged_gbr(
                X_encoded, y_log, chosen[name], seeds[name]
            )
        else:
            builder = candidates[name][1]
            estimator = builder(seeds[name], chosen[name])
            estimator.fit(X_encoded, y_log)
            base_models[name] = [estimator]
        print(f"  fitted: {name} ({len(base_models[name])} fitted model(s))")

    # Test the exact same prediction path that Streamlit will use. Passing
    # some full-data predictor outputs to `base_test` does not affect Meta-CV
    # selection; it is only used as a consistency probe during export.
    # The real Excel has many genuinely unreported irradiation times; those
    # missing values must remain missing during fitting (the preprocessor
    # handles them), but public website inputs require explicit valid numbers.
    # Use complete REAL experimental rows to verify the separately exported
    # inference path. No target labels are used in this consistency check.
    complete_mask = X[meta["numeric_columns"]].notna().all(axis=1)
    probes = X.loc[complete_mask].head(5).copy()
    if probes.empty:
        raise ValueError(
            "原始数据没有任何一行数值输入全部完整，无法核对用户可手动填写的推理路径。"
        )
    probes_encoded = np.asarray(preprocessor.transform(probes), dtype=np.float32)
    probe_base = np.column_stack([
        np.mean(np.column_stack([
            estimator.predict(probes_encoded)
            for estimator in base_models[name]
        ]), axis=1)
        for name in core.BASE_MODEL_NAMES
    ])
    base_oof = np.column_stack([oof[name] for name in core.BASE_MODEL_NAMES])

    print("5/6 执行完整的五层 Stacking / 重复 Group Meta-CV / Champion Guard...")
    stack = core.fit_safe_stacking(
        base_oof, probe_base, y_log, strict_groups,
        core.BASE_MODEL_NAMES,
        seed=core.RANDOM_STATE * 200000 + 999,
    )
    state = extract_stack_state(stack)
    print("  最终激活成员：", stack["Selected_Models"])
    print("  Champion Guard：", stack["Champion_Guard_Triggered"])
    print("  三层门控专家：", stack["Gate_Expert"])
    print("  第四层分区：", stack["Meta4_Mode"])
    print("  第五层状态：", state["Deploy_Meta5_Status"])

    category_options = {}
    for column in meta["categorical_columns"]:
        # Dropdowns contain ORIGINAL training categories. We never infer "Others"
        # combinations or silently substitute an untrained category.
        options = sorted({
            str(value)
            for value in X[column].dropna()
            if str(value).strip()
        }, key=lambda s: s.casefold())
        if not options:
            raise ValueError(f"训练类别字段没有有效选项：{column}")
        category_options[column] = options

    numeric_summary = {}
    for column in meta["numeric_columns"]:
        vals = pd.to_numeric(X[column], errors="coerce").dropna()
        if vals.empty:
            raise ValueError(f"训练数值字段没有有效数据：{column}")
        numeric_summary[column] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "median": float(vals.median()),
        }

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "model_id": core.STACKING_NAME,
        "target_transform": "log10",
        "target_unit": "mM",
        "preprocessor_mode": mode["Name"],
        "model_columns": list(meta["model_columns"]),
        "numeric_columns": list(meta["numeric_columns"]),
        "categorical_columns": list(meta["categorical_columns"]),
        "numeric_summary": numeric_summary,
        "category_options": category_options,
        "base_names": list(core.BASE_MODEL_NAMES),
        "preprocessor": preprocessor,
        "numeric_observed_counts": {
            column: int(X[column].notna().sum()) for column in meta["numeric_columns"]
        },
        "base_models": base_models,
        "stack_state": state,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(excel.read_bytes()).hexdigest(),
            "rows": int(len(X)),
            "literature_count": int(meta["literature_count"]),
            "strict_groups": int(meta["strict_group_count"]),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }

    print("6/6 先验证离线模型与原始 Stacking 输出完全一致，再保存...")
    predicted_ccc, predicted_log = predict_ccc(bundle, probes)
    np.testing.assert_allclose(
        predicted_log, np.asarray(stack["Test_Prediction"], dtype=float),
        rtol=1e-9, atol=1e-9, err_msg="独立推理与原 Stacking 不一致，禁止导出",
    )
    np.testing.assert_allclose(
        predicted_ccc, core.inverse_target(stack["Test_Prediction"]),
        rtol=1e-9, atol=1e-9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    public_info = {
        "model_id": bundle["model_id"],
        "target": "log10(CCC) -> CCC (mM)",
        "rows": len(X),
        "literature_count": int(meta["literature_count"]),
        "strict_groups": int(meta["strict_group_count"]),
        "input_columns": list(meta["model_columns"]),
        "base_names": list(core.BASE_MODEL_NAMES),
        "preprocessor_mode": mode["Name"],
        "selected_members": list(stack["Selected_Models"]),
        "dynamic_gate_expert": str(stack["Gate_Expert"]),
        "champion_guard_triggered": bool(stack["Champion_Guard_Triggered"]),
        "regime_calibration_enabled": bool(stack["Meta4_Enabled"]),
        "fifth_stage_enabled": bool(stack["Meta5_Enabled"]),
        "trained_utc": bundle["provenance"]["created_utc"],
        "source_sha256": bundle["provenance"]["source_sha256"],
        "note": "This is a final model trained on all valid data. "
                "Do not interpret training/Meta OOF as a fresh independent test.",
    }
    info_path = output.parent / "model_info.json"
    info_path.write_text(
        json.dumps(public_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n✅ 导出成功：", output.resolve())
    print("📄 模型描述：", info_path.resolve())
    print("✔ Stacking 推理一致性检查通过，探针 CCC (mM) =",
          np.round(predicted_ccc, 4).tolist())
    print("接下来启动： streamlit run app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the full-data microplastic CCC Stacking model")
    parser.add_argument("--excel", type=Path, default=Path("Data2.xlsx"),
                        help="原训练数据 Excel 的路径")
    parser.add_argument("--sheet", default="标准化数据",
                        help="Excel 工作表名，需与论文训练时相同")
    parser.add_argument("--output", type=Path,
                        default=Path("models") / "ccc_stacking.joblib",
                        help="模型导出路径")
    args = parser.parse_args()
    train_and_export(args.excel, args.sheet, args.output)


if __name__ == "__main__":
    main()
