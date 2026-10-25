# -*- coding: utf-8 -*-
"""Offline inference for the exact six-base-learner, five-layer Stacking design.

The training algorithm lives in stacking_core.py. The two deployment-only state
fields added to its fitted stages allow prediction without re-running Meta-CV.
No shortcut ensemble, fixed weights, or hand-picked CCC output is used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd

import stacking_core as core

SCHEMA_VERSION = 1
BUNDLE_NAME = "ccc_stacking.joblib"

# The original stack fitting function returns large diagnostic tables and training
# OOF vectors. Only this reproducible, fitted inference state is persisted.
STACK_STATE_KEYS = (
    "Deploy_Stage3_Fallback",
    "Deploy_Corrector",
    "Deploy_Corrector_Bound",
    "Deploy_Gate",
    "Deploy_Gate_Bound",
    "Deploy_Stage3_Soft_Fraction",
    "Deploy_Regime_Model",
    "Deploy_Regime_Fraction",
    "Deploy_Meta5_Status",
    "Deploy_Meta5_Bound",
    "Champion_Guard_Triggered",
    "Weights",
    "Bias",
    "Residual_Corrector",
    "Residual_Strength",
    "Gate_Expert",
    "Gate_Threshold",
    "Gate_Strength",
    "Final_Scale",
    "Final_Intercept",
    "Meta5_Blend",
    "Meta5_Scale",
    "Meta5_Intercept",
)


def extract_stack_state(fitted: Mapping) -> dict:
    return {key: fitted[key] for key in STACK_STATE_KEYS if key in fitted}


def read_bundle(path: str | Path) -> dict:
    """Only load joblib artifacts that *you* built and trust."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"找不到已训练模型文件：{artifact}")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("模型文件格式不匹配：请使用本项目的 train_export.py 重新导出。")
    if bundle.get("target_transform") != "log10":
        raise ValueError("当前论文网站仅支持原代码的 log10(CCC) 模型。")
    if list(bundle.get("base_names", ())) != list(core.BASE_MODEL_NAMES):
        raise ValueError("模型成员或顺序与原始代码不一致。")
    return bundle


def prepare_input(bundle: Mapping, frame: pd.DataFrame) -> pd.DataFrame:
    """Use the very same input column order and cleaning rules as training."""
    columns = list(bundle["model_columns"])
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("缺少输入字段：" + "、".join(missing))
    X = frame[columns].copy().reset_index(drop=True)
    for col in bundle["numeric_columns"]:
        X[col] = core.to_numeric_safe(X[col])
        if X[col].isna().any():
            raise ValueError(f"数值输入无效：{col}。请填写有限的真实数值。")
    for col in bundle["categorical_columns"]:
        if X[col].isna().any() or X[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"类别输入未填写：{col}")
        X[col] = X[col].map(core.normalize_category)
    return X


def _base_prediction_matrix(bundle: Mapping, frame: pd.DataFrame) -> np.ndarray:
    X = prepare_input(bundle, frame)
    processed = np.asarray(bundle["preprocessor"].transform(X), dtype=np.float32)
    if processed.ndim != 2 or not np.isfinite(processed).all():
        raise ValueError("预处理数据出现非法值，无法执行预测。")

    predictions = []
    for name in bundle["base_names"]:
        estimators = bundle["base_models"][name]
        if not isinstance(estimators, (list, tuple)) or not estimators:
            raise ValueError(f"{name} 的拟合模型缺失。")
        values = np.column_stack([
            np.asarray(estimator.predict(processed), dtype=float)
            for estimator in estimators
        ])
        predictions.append(np.mean(values, axis=1))
    matrix = np.column_stack(predictions)
    if not np.isfinite(matrix).all():
        raise ValueError("基础模型出现非有限预测值。")
    return matrix


def replay_meta_stack(state: Mapping, base_matrix: np.ndarray, base_names) -> np.ndarray:
    """Replay the exact fitted branch choices and corrections from core.fit_safe_stacking."""
    matrix = np.asarray(base_matrix, dtype=float)
    names = list(base_names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("基础模型的预测矩阵维度不匹配。")

    anchor = matrix[:, names.index(core.STACK_ANCHOR_NAME)]
    if bool(state["Champion_Guard_Triggered"]):
        return anchor.copy()

    # Stage 1-3 may have been rejected in favour of the GBR750 anchor.
    if bool(state["Deploy_Stage3_Fallback"]):
        stage3 = anchor.copy()
    else:
        weights = np.asarray(state["Weights"], dtype=float)
        stage2 = matrix @ weights + float(state["Bias"])
        corrector = state.get("Deploy_Corrector")
        if corrector is not None and float(state["Residual_Strength"]) > 0.0:
            features = core._build_meta2_features(matrix, names)
            correction = core._predict_meta2_corrector(corrector, features)
            bound = float(state["Deploy_Corrector_Bound"])
            stage2 = stage2 + float(state["Residual_Strength"]) * np.clip(
                correction, -bound, bound
            )

        if str(state["Gate_Expert"]) == "None":
            gated = stage2
        else:
            gate = state.get("Deploy_Gate")
            if gate is None:
                raise ValueError("模型导出时未保存已经拟合的动态门控器。")
            expert = core._get_gate_expert_prediction(
                matrix, str(state["Gate_Expert"]), names
            )
            features = core._build_gate_features(matrix, stage2, expert, names)
            probability = core._predict_gate_probability(gate, features)
            threshold = float(state["Gate_Threshold"])
            activation = np.clip(
                (probability - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0
            )
            bound = float(state["Deploy_Gate_Bound"])
            expert_delta = np.clip(expert - stage2, -bound, bound)
            gated = (
                stage2
                + float(state["Gate_Strength"]) * activation * expert_delta
            )

        stage3 = (
            float(state["Final_Intercept"])
            + float(state["Final_Scale"]) * gated
        )
        fraction = float(state["Deploy_Stage3_Soft_Fraction"])
        if fraction < 1.0:
            stage3 = anchor + fraction * (stage3 - anchor)

    # Stage 4: optional training-fitted local/residual regime offsets.
    fitted_regime = state.get("Deploy_Regime_Model")
    if fitted_regime is not None:
        stage4 = stage3 + float(state["Deploy_Regime_Fraction"]) * (
            core._predict_regime_correction(fitted_regime, matrix, stage3)
        )
    else:
        stage4 = stage3

    # Stage 5: terminal safety blending, then optional soft shrinkage.
    status = str(state["Deploy_Meta5_Status"])
    if status == "disabled":
        return np.asarray(stage4, dtype=float)
    if status not in {"stable", "soft"}:
        raise ValueError("未知第五层 Stacking 状态。")
    blend = float(state["Meta5_Blend"])
    if blend == 0.0:
        combined = anchor.copy()
    else:
        bound = float(state["Deploy_Meta5_Bound"])
        combined = anchor + blend * np.clip(stage4 - anchor, -bound, bound)
    output = (
        float(state["Meta5_Intercept"])
        + float(state["Meta5_Scale"]) * combined
    )
    if status == "soft":
        output = anchor + float(core.STACK_META5_SOFT_FRACTION) * (output - anchor)
    return np.asarray(output, dtype=float)


def predict_ccc(bundle: Mapping, input_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (CCC in mM, log10 CCC), each shaped (n_samples,)."""
    matrix = _base_prediction_matrix(bundle, input_frame)
    log_prediction = replay_meta_stack(
        bundle["stack_state"], matrix, bundle["base_names"]
    )
    if not np.isfinite(log_prediction).all():
        raise ValueError("Stacking 计算得到非有限预测值。")
    ccc = core.inverse_target(log_prediction)
    if not np.isfinite(ccc).all() or np.any(ccc <= 0):
        raise ValueError("反变换后的 CCC 非法。")
    return ccc, log_prediction


def input_warnings(bundle: Mapping, frame: pd.DataFrame) -> list[str]:
    """Advisory only: never alter or clip the requested experimental condition."""
    X = prepare_input(bundle, frame)
    notes = []
    for col, stat in bundle["numeric_summary"].items():
        value = float(X.loc[0, col])
        low, high = float(stat["min"]), float(stat["max"])
        if value < low or value > high:
            notes.append(
                f"{col}: {value:g} 超出训练范围 [{low:g}, {high:g}]，属于外推预测。"
            )
    for col, choices in bundle["category_options"].items():
        category = str(X.loc[0, col])
        if category not in choices:
            notes.append(f"{col}: “{category}” 未在训练数据中出现，属于未知类别。")
    return notes
