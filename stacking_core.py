# -*- coding: utf-8 -*-
"""
严格未知文献 CCC 预测 V19：五个公开单模型 + 内部GBR750六模型Stacking
================================================================================

主表中的五个单模型
------------------
1. Random Forest（RF）
2. ExtraTrees
3. LightGBM（LGBM）
4. XGBoost（XGB）
5. Gradient Boosting Regression（GBR，保持原150树固定参数）

最终集成模型
------------
6. Stacking_GBR750Enhanced_GroupSafe_All6

Stacking内部六个一级模型
------------------------
RF、ExtraTrees、LGBM、XGB、公开GBR和内部GBR750锚点。

关键说明
--------
1. 模型排名和主表中，公开Gradient Boosting模型统一显示为“GBR”，不带树数后缀。
2. 公开GBR保持原150树参数，并且作为Stacking内部五个伙伴模型之一。
3. GBR750只作为Stacking内部第六个一级模型、强锚点和安全回退，不单独进入主表排名。
4. RF、ExtraTrees、LGBM、XGB、公开GBR的参数保持原样；只调整Stacking成员、
   锚点命名、伙伴权重边界、门控候选和Champion Guard等集成参数。
5. 外层测试标签不参与预处理、一级模型训练、参数选择、Meta-CV、权重、门控或回退选择。
6. 程序只输出控制台，不保存任何结果文件。

依赖
----
pip install pandas numpy scipy scikit-learn openpyxl lightgbm xgboost
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("缺少 lightgbm，请执行：pip install lightgbm") from exc

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("缺少 xgboost，请执行：pip install xgboost") from exc


warnings.filterwarnings("ignore")
pd.set_option("display.width", 320)
pd.set_option("display.max_columns", 160)
pd.set_option("display.max_rows", 500)
pd.set_option("display.float_format", lambda value: f"{value:.6f}")


# =============================================================================
# 1. 用户配置
# =============================================================================

FILE_PATH = "Data2.xlsx"
SHEET_NAME = "标准化数据"

TARGET_KEYWORD = "CCC"
GROUP_KEYWORD = "文献编号"
TITLE_KEYWORD = "文献标题"

# 这是旧版约 0.8 R² 的关键口径。不要改成 raw 后再与旧结果直接比较。
TARGET_TRANSFORM = "log10"  # 可选："log10" / "raw"

# 直接复现旧版时使用 legacy_target_balanced。
# 更保守的敏感性分析可改为 group_kfold，但分数可能下降，不能与旧版直接比较。
OUTER_SPLIT_MODE = "legacy_target_balanced"  # "legacy_target_balanced" / "group_kfold"

SEARCH_PROFILE = "balanced_gbr750_internal_all6"

RANDOM_STATE = 2026
OUTER_SPLITS = 5
INNER_SPLITS = 3
META_SPLITS = 4
N_JOBS = max(1, min(8, (os.cpu_count() or 4) - 1))

GROUP_SPLIT_RESTARTS_OUTER = 140
GROUP_SPLIT_RESTARTS_INNER = 50
GROUP_SPLIT_Y_BINS = 8

AUTO_SELECT_PREPROCESSOR = True
KNN_NEIGHBORS = 7
PREPROCESSOR_CANDIDATES = [
    {"Name": "ScaledKNN_NoIF", "Imputer": "knn"},
    {"Name": "Median_NoIF", "Imputer": "median"},
]

# GBR最终拟合使用两个随机种子平均，只属于同一GBR算法的稳定化，不是元启发式优化。
GBR_BAG_SEEDS = (0, 101)

# 六模型GBR750锚定五层 Stacking。
# 第一层：GBR750高权重受限线性融合。
# 第二层：仅基于六模型严格OOF预测构造元特征，对第一层残差做强正则校准。
# 第三层：按样本动态专家门控与尺度校准。
# 第四层：按预测水平和模型分歧分区，学习强平滑的局部残差偏移。
# 第五层：在GBR750与第四层预测之间做稳定收缩、增量截断和最终仿射校准。
# 所有超参数只在当前外层训练集的重复Group Meta-CV内选择，绝不使用外层测试标签。
STACK_L2_GRID = (0.0005, 0.0015, 0.005, 0.015, 0.050, 0.150)
STACK_META_REPEATS = 6
STACK_STAGE1_SHRINK_GRID = (0.0, 0.35, 0.65, 1.0)
STACK_MAX_PARTNERS = 5
STACK_PARTNER_R2_TOL = 0.16
STACK_ABSOLUTE_MIN_R2 = 0.50
STACK_FINAL_META_WEIGHT = 0.60
STACK_BIAS_BOUND = 0.080
STACK_BIAS_L2 = 0.12
STACK_GBR_MIN_WEIGHT = 0.74
STACK_GBR_MAX_WEIGHT = 1.08

# Champion Guard：在最终输出前检查锚点与互补模型的风险。
# 只读取当前外层训练集的严格OOF和重复Group Meta-CV信息，不接触外层测试标签。
STACK_CHAMPION_GUARD_ENABLED = True
STACK_GUARD_REQUIRE_GBR_TRAIN_CHAMPION = True
STACK_GUARD_GBR_CHAMPION_TOL = 0.0020

# 旧版统一要求GBR权重>=0.86。V17分成硬下限与标准下限：
# 只有Meta证据非常强时，才允许权重落在[0.82, 0.86)。
STACK_GUARD_HARD_MIN_GBR_WEIGHT = 0.820
STACK_GUARD_STANDARD_MIN_GBR_WEIGHT = 0.860
STACK_GUARD_MAX_POSITIVE_PARTNER_WEIGHT = 0.240
STACK_GUARD_MAX_ABS_PARTNER_WEIGHT = 0.340
STACK_GUARD_COMPLEX_MIN_GBR_WEIGHT = 0.900
STACK_GUARD_INTERCEPT_EDGE = 0.095
STACK_GUARD_MIN_GAIN_AT_INTERCEPT_EDGE = 0.006

# 强证据放行：全部重复为正、最差重复仍有正增益且总增益足够。
STACK_GUARD_STRONG_GAIN = 0.0080
STACK_GUARD_STRONG_WORST_REPEAT_GAIN = 0.0030

# 当非GBR正权重较宽时，要求最差重复增益不能接近零。
STACK_GUARD_WIDE_BLEND_PARTNER_WEIGHT = 0.150
STACK_GUARD_WIDE_BLEND_MIN_WORST_REPEAT_GAIN = 0.0010

# 第二层残差校准器候选。均为强正则、小容量模型，避免在约36个训练文献上过拟合。
STACK_META2_CORRECTOR_CANDIDATES = [
    {"Name": "Ridge_a0.5", "Family": "ridge", "alpha": 0.5},
    {"Name": "Ridge_a2", "Family": "ridge", "alpha": 2.0},
    {"Name": "Ridge_a8", "Family": "ridge", "alpha": 8.0},
    {"Name": "Ridge_a25", "Family": "ridge", "alpha": 25.0},
    {
        "Name": "MetaGBR_Stump_Huber", "Family": "gbr",
        "n_estimators": 45, "learning_rate": 0.025, "max_depth": 1,
        "min_samples_leaf": 24, "subsample": 0.82, "loss": "huber",
    },
    {
        "Name": "MetaGBR_Stump_Smooth", "Family": "gbr",
        "n_estimators": 70, "learning_rate": 0.016, "max_depth": 1,
        "min_samples_leaf": 30, "subsample": 0.88, "loss": "huber",
    },
    {
        "Name": "MetaGBR_Depth2_Tiny", "Family": "gbr",
        "n_estimators": 35, "learning_rate": 0.018, "max_depth": 2,
        "min_samples_leaf": 32, "subsample": 0.82, "loss": "huber",
    },
]
STACK_META2_STRENGTH_GRID = (0.20, 0.35, 0.50, 0.70, 0.90)
STACK_META2_CLIP_MULTIPLIER_GRID = (0.40, 0.65, 0.90, 1.20)
STACK_META2_REQUIRED_GAIN = 0.00010
STACK_META2_MIN_POSITIVE_REPEAT_RATIO = 0.50
STACK_META2_MIN_POSITIVE_FOLD_RATIO = 0.50
STACK_META2_MAX_WORST_REPEAT_LOSS = 0.014
STACK_META2_MAX_WORST_FOLD_LOSS = 0.045
STACK_META2_SOFT_MIN_GAIN = 0.00002
STACK_META2_SOFT_FRACTION = 0.50
STACK_META2_SOFT_MIN_POSITIVE_REPEAT_RATIO = 0.375
STACK_META2_SOFT_MAX_WORST_REPEAT_LOSS = 0.020

# 第三层：动态专家门控 + 最终尺度校准。
# 门控只使用六模型严格OOF预测及其分歧特征，不使用文献编号或外层测试标签。
STACK_GATE_EXPERTS = ("RF", "ExtraTrees", "LGBM", "XGB", "GBR", "ConsensusMedian", "TreeMedian")
STACK_GATE_C_GRID = (0.25, 1.0, 4.0)
STACK_GATE_THRESHOLD_GRID = (0.55, 0.70)
STACK_GATE_STRENGTH_GRID = (0.20, 0.45, 0.70)
STACK_GATE_CLIP_MULTIPLIER_GRID = (0.70, 1.20)
STACK_FINAL_SCALE_GRID = (0.96, 0.99, 1.00, 1.03, 1.06, 1.10, 1.14)
STACK_FINAL_INTERCEPT_BOUND = 0.10
STACK_GATE_IMPROVEMENT_WEIGHT_CAP = 5.0
STACK_META3_REQUIRED_GAIN = 0.00012
STACK_META3_MIN_POSITIVE_REPEAT_RATIO = 0.625
STACK_META3_MIN_POSITIVE_FOLD_RATIO = 0.55
STACK_META3_MAX_WORST_REPEAT_LOSS = 0.012
STACK_META3_MAX_WORST_FOLD_LOSS = 0.045
STACK_META3_SOFT_MIN_GAIN = 0.00002
STACK_META3_SOFT_FRACTION = 0.50
STACK_META3_SOFT_MIN_POSITIVE_REPEAT_RATIO = 0.50
STACK_META3_SOFT_MAX_WORST_REPEAT_LOSS = 0.018
STACK_META3_FINAL_INTERCEPT_BLEND = 0.50

# 第四层：分区残差校准。仅使用低层严格交叉拟合预测构造分区，
# 通过强平滑和小幅收缩修正不同预测区间/分歧区间的系统偏差。
STACK_META4_MODES = ("Level", "LevelDisagreement", "LevelConsensus")
STACK_META4_LEVEL_BINS = (3, 4)
STACK_META4_SECONDARY_BINS = (1, 2)
STACK_META4_SHRINKAGE_GRID = (0.15, 0.30, 0.50)
STACK_META4_SMOOTHING_GRID = (12.0, 28.0)
STACK_META4_CLIP_MULTIPLIER_GRID = (0.45, 0.80)
STACK_META4_REQUIRED_INCREMENTAL_GAIN = 0.00008
STACK_META4_REQUIRED_TOTAL_GAIN = 0.00012
STACK_META4_MIN_POSITIVE_REPEAT_RATIO = 0.625
STACK_META4_MIN_POSITIVE_FOLD_RATIO = 0.55
STACK_META4_MAX_WORST_REPEAT_LOSS = 0.010
STACK_META4_MAX_WORST_FOLD_LOSS = 0.035
STACK_META4_SOFT_MIN_INCREMENTAL_GAIN = 0.00001
STACK_META4_SOFT_FRACTION = 0.45

# 第五层：稳定收缩融合。用GBR750作为保险锚点，对第四层增量做截断，
# 再选择融合比例和最终尺度；截距始终只在训练Meta折上估计。
STACK_META5_BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
STACK_META5_DELTA_CLIP_GRID = (0.35, 0.60, 0.90, 1.20)
STACK_META5_SCALE_GRID = (0.98, 1.00, 1.02, 1.04, 1.06)
STACK_META5_INTERCEPT_BOUND = 0.080
STACK_META5_REQUIRED_TOTAL_GAIN = 0.00012
STACK_META5_REQUIRED_INCREMENTAL_GAIN = 0.00002
STACK_META5_MIN_POSITIVE_REPEAT_RATIO = 0.75
STACK_META5_MIN_POSITIVE_FOLD_RATIO = 0.60
STACK_META5_MAX_WORST_REPEAT_LOSS = 0.007
STACK_META5_MAX_WORST_FOLD_LOSS = 0.030
STACK_META5_SOFT_MIN_TOTAL_GAIN = 0.00002
STACK_META5_SOFT_FRACTION = 0.50
STACK_META5_FINAL_INTERCEPT_BLEND = 0.50

# 非锚点模型在线性第一层中的边界。允许小幅负权重修正共同偏差，
# 但GBR750在第一层中的有效权重始终保持较高。
STACK_PARTNER_WEIGHT_BOUNDS = {
    "RF": (-0.05, 0.12),
    "ExtraTrees": (-0.08, 0.18),
    "LGBM": (-0.06, 0.12),
    "XGB": (-0.08, 0.18),
    "GBR": (-0.08, 0.18),
}

if SEARCH_PROFILE == "balanced_gbr750_internal_all6":
    INNER_SPLITS = 3
    META_SPLITS = 4
    GROUP_SPLIT_RESTARTS_OUTER = 140
    GROUP_SPLIT_RESTARTS_INNER = 50
else:
    raise ValueError("SEARCH_PROFILE 只能是 balanced_gbr750_internal_all6。")

NUMERIC_KEYWORDS = [
    "温度", "Zeta", "PH", "pH", "塑料浓度", "初始粒径", "共存物质浓度",
    "光照时间", "SSA", "比表面积", "C%", "碳含量", "H/C", "O/C", "O/H",
    "Abraham", "McGowan",
]
FORCE_CATEGORICAL_KEYWORDS = [
    "阳离子价态", "阴离子类型", "颗粒形态", "塑料表面修饰类型",
    "塑料种类", "共存物质种类",
]

# 基线模型保持不同容量层级，作为真实对照，而不是都调到同一种复杂度。
RF_CANDIDATES = [
    {
        "n_estimators": 320, "max_depth": 7, "min_samples_split": 12,
        "min_samples_leaf": 6, "max_features": 0.50, "max_samples": 0.76,
    },
]

ET_CANDIDATES = [
    {
        "n_estimators": 380, "max_depth": 9, "min_samples_split": 12,
        "min_samples_leaf": 5, "max_features": 0.55, "bootstrap": False,
    },
]

# 两种GBR角色均在运行前固定，不读取外层测试结果。
# 公开GBR：保持原150树参数，进入主表，并同时进入Stacking。
# 内部GBR750：只作为Stacking内部强锚点，不进入主表排名。
GBR_PUBLIC_PARAMS = {
    "n_estimators": 150, "learning_rate": 0.014, "max_depth": 3,
    "min_samples_split": 3, "min_samples_leaf": 2, "subsample": 0.86,
    "max_features": 1.00, "loss": "squared_error",
}
GBR750_ANCHOR_PARAMS = {
    "n_estimators": 750, "learning_rate": 0.014, "max_depth": 3,
    "min_samples_split": 3, "min_samples_leaf": 2, "subsample": 0.86,
    "max_features": 1.00, "loss": "squared_error",
}
GBR_CANDIDATES = [GBR_PUBLIC_PARAMS]
GBR750_CANDIDATES = [GBR750_ANCHOR_PARAMS]

LGBM_CANDIDATES = [
    {
        "n_estimators": 520, "learning_rate": 0.025, "num_leaves": 13,
        "max_depth": 4, "min_child_samples": 26, "subsample": 0.84,
        "colsample_bytree": 0.80, "reg_alpha": 0.15, "reg_lambda": 7.0,
    },
]

XGB_CANDIDATES = [
    {
        "n_estimators": 420, "learning_rate": 0.028, "max_depth": 3,
        "min_child_weight": 7.0, "subsample": 0.84, "colsample_bytree": 0.86,
        "reg_alpha": 0.10, "reg_lambda": 10.0, "gamma": 0.010,
    },
]

STACK_ANCHOR_NAME = "GBR750"

# Stacking内部六个一级模型：五个公开模型 + 内部GBR750锚点。
BASE_MODEL_NAMES = [
    "RF", "ExtraTrees", "LGBM", "XGB", "GBR", STACK_ANCHOR_NAME,
]
STACKING_NAME = "Stacking_GBR750Enhanced_GroupSafe_All6"

# 主表只展示五个公开单模型与最终Stacking。
PRIMARY_MODEL_NAMES = [
    "RF", "ExtraTrees", "LGBM", "XGB", "GBR", STACKING_NAME,
]
MODEL_NAMES = list(dict.fromkeys(PRIMARY_MODEL_NAMES + [STACK_ANCHOR_NAME]))


# =============================================================================
# 1.1 数据行拖累诊断配置
# =============================================================================
# 诊断原则：
# 1. 外层 OOF 大残差只表示“难预测”，不能单独证明该行有害；
# 2. 先用多种无泄漏信号筛选候选，再执行严格删一行重训；
# 3. 候选行只从训练折删除，原外层测试折完整保留，比较未知文献误差是否改善；
# 4. 输出用于回原文核查，不能在同一套 OOF 上删除后直接宣称新模型性能提高。
RUN_ROW_DIAGNOSTICS = False

# 初筛输出数量和真正执行“删一行重训”的数量。
DIAG_SCREEN_TOP_N = 45
DIAG_STRICT_RETRAIN_TOP_N = 30
DIAG_MAX_CANDIDATES_PER_LITERATURE = 4
DIAG_MAX_DETAILED_ROWS = 45

# 相似样本诊断：优先寻找其他严格组中 X 最接近的记录。
DIAG_NEIGHBOR_COUNT = 8
DIAG_NEIGHBOR_SEARCH_MULTIPLIER = 8

# 判定“删除该行后模型稳定改善”的门槛。
DIAG_MIN_POSITIVE_FOLDS = 3
DIAG_MIN_POOLED_R2_GAIN_LOG = 0.00030
DIAG_MIN_POOLED_R2_GAIN_RAW = 0.00100
DIAG_MAX_WORST_FOLD_R2_LOSS = 0.00300

# 完全相同 X 但目标明显不同，会被强制加入候选。
DIAG_SIGNATURE_CONFLICT_LOG_RANGE = 0.30

# 诊断以GBR为参考模型；它通常是log10(CCC)上的最强单模型。
DIAG_REFERENCE_MODEL = STACK_ANCHOR_NAME


# =============================================================================
# 2. 通用工具
# =============================================================================

def section(title: str) -> None:
    print("\n" + "=" * 156)
    print(title)
    print("=" * 156)


def norm_name(value) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def find_col(columns: Sequence[str], keyword: str, required: bool = True) -> Optional[str]:
    key = norm_name(keyword)
    matches = [column for column in columns if key in norm_name(column)]
    if not matches and required:
        raise KeyError(f"找不到字段关键词“{keyword}”。当前字段：{list(columns)}")
    return matches[0] if matches else None


def find_all_cols(columns: Sequence[str], keyword: str) -> List[str]:
    key = norm_name(keyword)
    return [column for column in columns if key in norm_name(column)]


def contains_any_keyword(column: str, keywords: Sequence[str]) -> bool:
    text = str(column).lower()
    return any(str(keyword).lower() in text for keyword in keywords)


def to_numeric_safe(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cleaned = (
        series.astype(str).str.strip().str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False).str.replace("%", "", regex=False)
        .replace({
            "": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan,
            "NA": np.nan, "N/A": np.nan, "-": np.nan, "--": np.nan,
            "<MISSING>": np.nan, "未报道": np.nan, "未报告": np.nan,
        })
    )
    return pd.to_numeric(cleaned, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_category(value) -> str:
    if pd.isna(value):
        return "<MISSING>"
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else "<MISSING>"


def signature_value(value) -> str:
    if pd.isna(value):
        return "<MISSING>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "<MISSING>" if np.isnan(value) else f"{float(value):.12g}"
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or float(np.var(y_true)) <= 1e-15:
        return np.nan
    return float(r2_score(y_true, y_pred))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": safe_r2(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def transform_target(y_raw: np.ndarray) -> np.ndarray:
    y_raw = np.asarray(y_raw, dtype=float)
    if TARGET_TRANSFORM == "log10":
        if np.any(y_raw <= 0):
            raise ValueError("使用 log10 目标时，CCC 必须全部大于 0。")
        return np.log10(y_raw)
    if TARGET_TRANSFORM == "raw":
        return y_raw.copy()
    raise ValueError("TARGET_TRANSFORM 只能是 log10 或 raw。")


def inverse_target(y_model: np.ndarray) -> np.ndarray:
    y_model = np.asarray(y_model, dtype=float)
    if TARGET_TRANSFORM == "log10":
        return np.power(10.0, np.clip(y_model, -12.0, 12.0))
    return y_model.copy()


def group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    _, codes = np.unique(groups, return_inverse=True)
    counts = np.bincount(codes).astype(float)
    weights = 1.0 / np.maximum(counts[codes], 1.0)
    return weights / np.mean(weights)


def normalized_cv_objective(
    y_full: np.ndarray,
    oof_prediction: np.ndarray,
    fold_records: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, float]:
    global_scale = max(float(np.std(y_full, ddof=0)), 1e-8)
    pooled_nrmse = rmse(y_full, oof_prediction) / global_scale
    fold_nrmse: List[float] = []
    fold_r2: List[float] = []
    for item in fold_records:
        scale = max(float(np.std(item["y_true"], ddof=0)), 1e-8)
        fold_nrmse.append(rmse(item["y_true"], item["y_pred"]) / scale)
        fold_r2.append(safe_r2(item["y_true"], item["y_pred"]))
    mean_fold = float(np.mean(fold_nrmse))
    sd_fold = float(np.std(fold_nrmse, ddof=1)) if len(fold_nrmse) > 1 else 0.0
    worst_fold_nrmse = float(np.max(fold_nrmse))
    valid_fold_r2 = np.asarray([value for value in fold_r2 if np.isfinite(value)], dtype=float)
    mean_fold_r2 = float(np.mean(valid_fold_r2)) if len(valid_fold_r2) else np.nan
    worst_fold_r2 = float(np.min(valid_fold_r2)) if len(valid_fold_r2) else np.nan
    objective = 0.62 * pooled_nrmse + 0.28 * mean_fold + 0.10 * sd_fold
    return {
        "Objective": float(objective),
        "OOF_R2": safe_r2(y_full, oof_prediction),
        "OOF_RMSE": rmse(y_full, oof_prediction),
        "OOF_MAE": float(mean_absolute_error(y_full, oof_prediction)),
        "Pooled_NRMSE": float(pooled_nrmse),
        "Mean_Fold_NRMSE": mean_fold,
        "SD_Fold_NRMSE": sd_fold,
        "Worst_Fold_NRMSE": worst_fold_nrmse,
        "Mean_Fold_R2": mean_fold_r2,
        "Worst_Fold_R2": worst_fold_r2,
    }

def public_details(details: Mapping) -> Dict:
    hidden = {"Params", "OOF_Prediction", "Fold_Predictions", "Vector"}
    return {key: value for key, value in details.items() if key not in hidden}


# =============================================================================
# 3. 严格组与折分
# =============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            self.parent[root_first] = root_second
        elif self.rank[root_first] > self.rank[root_second]:
            self.parent[root_second] = root_first
        else:
            self.parent[root_second] = root_first
            self.rank[root_first] += 1


def build_strict_groups(literature_ids: np.ndarray, signatures: Sequence[Tuple]) -> np.ndarray:
    uf = UnionFind(len(literature_ids))
    first_literature: Dict[object, int] = {}
    first_signature: Dict[Tuple, int] = {}
    for index, literature in enumerate(literature_ids):
        if literature in first_literature:
            uf.union(index, first_literature[literature])
        else:
            first_literature[literature] = index
    for index, signature in enumerate(signatures):
        if signature in first_signature:
            uf.union(index, first_signature[signature])
        else:
            first_signature[signature] = index
    roots = [uf.find(index) for index in range(len(literature_ids))]
    root_to_group = {root: group_id for group_id, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_group[root] for root in roots], dtype=int)


def _regression_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    unique_count = len(np.unique(y))
    q = max(2, min(int(n_bins), unique_count))
    try:
        return np.asarray(pd.qcut(y, q=q, labels=False, duplicates="drop"), dtype=int)
    except Exception:
        ranks = pd.Series(y).rank(method="average", pct=True).to_numpy()
        return np.floor(np.minimum(ranks, 0.999999) * q).astype(int)


def _assignment_score(
    fold_rows: np.ndarray,
    fold_y_sum: np.ndarray,
    fold_y_sq: np.ndarray,
    fold_bins: np.ndarray,
    fold_group_count: np.ndarray,
    global_mean: float,
    global_sd: float,
    global_bin_prop: np.ndarray,
) -> float:
    ideal_rows = max(float(np.sum(fold_rows)) / len(fold_rows), 1.0)
    size_score = float(np.mean(np.abs(fold_rows - ideal_rows) / ideal_rows))
    valid = fold_rows > 0
    means = np.full(len(fold_rows), global_mean, dtype=float)
    means[valid] = fold_y_sum[valid] / fold_rows[valid]
    mean_score = float(np.mean(np.abs(means - global_mean) / global_sd))
    variances = np.zeros(len(fold_rows), dtype=float)
    variances[valid] = fold_y_sq[valid] / fold_rows[valid] - means[valid] ** 2
    sds = np.sqrt(np.maximum(variances, 0.0))
    sd_score = float(np.mean(np.abs(sds - global_sd) / global_sd))
    bin_scores = []
    for fold in range(len(fold_rows)):
        if fold_rows[fold] <= 0:
            bin_scores.append(1.0)
        else:
            bin_scores.append(float(np.mean(np.abs(fold_bins[fold] / fold_rows[fold] - global_bin_prop))))
    ideal_groups = max(float(np.sum(fold_group_count)) / len(fold_group_count), 1.0)
    group_score = float(np.mean(np.abs(fold_group_count - ideal_groups) / ideal_groups))
    return size_score + 1.25 * mean_score + 0.30 * sd_score + 2.0 * float(np.mean(bin_scores)) + 0.10 * group_score


def target_balanced_group_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    n_restarts: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups, dtype=int)
    y = np.asarray(y, dtype=float)
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"严格组只有 {len(unique_groups)} 个，无法执行 {n_splits} 折。")
    bins = _regression_bins(y, GROUP_SPLIT_Y_BINS)
    n_actual_bins = int(np.max(bins)) + 1
    records = []
    for group_value in unique_groups:
        index = np.where(groups == group_value)[0]
        group_y = y[index]
        records.append({
            "Group": int(group_value),
            "Rows": int(len(index)),
            "Y_Sum": float(np.sum(group_y)),
            "Y_Sq": float(np.sum(group_y ** 2)),
            "Y_Mean": float(np.mean(group_y)),
            "Bins": np.bincount(bins[index], minlength=n_actual_bins).astype(float),
        })
    global_mean = float(np.mean(y))
    global_sd = max(float(np.std(y, ddof=0)), 1e-8)
    global_bin_prop = np.bincount(bins, minlength=n_actual_bins).astype(float) / len(y)
    master = np.random.default_rng(seed)
    best_score = np.inf
    best_groups: Optional[List[set]] = None
    for _ in range(int(n_restarts)):
        rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        ordered = sorted(
            records,
            key=lambda item: (-item["Rows"], -abs(item["Y_Mean"] - global_mean), rng.random()),
        )
        fold_groups = [set() for _ in range(n_splits)]
        fold_rows = np.zeros(n_splits)
        fold_y_sum = np.zeros(n_splits)
        fold_y_sq = np.zeros(n_splits)
        fold_bins = np.zeros((n_splits, n_actual_bins))
        fold_group_count = np.zeros(n_splits)
        first_order = rng.permutation(n_splits)
        for position in range(n_splits):
            item = ordered[position]
            chosen = int(first_order[position])
            fold_groups[chosen].add(item["Group"])
            fold_rows[chosen] += item["Rows"]
            fold_y_sum[chosen] += item["Y_Sum"]
            fold_y_sq[chosen] += item["Y_Sq"]
            fold_bins[chosen] += item["Bins"]
            fold_group_count[chosen] += 1
        for item in ordered[n_splits:]:
            candidate_scores = []
            for fold in range(n_splits):
                rows_new = fold_rows.copy(); rows_new[fold] += item["Rows"]
                y_sum_new = fold_y_sum.copy(); y_sum_new[fold] += item["Y_Sum"]
                y_sq_new = fold_y_sq.copy(); y_sq_new[fold] += item["Y_Sq"]
                bins_new = fold_bins.copy(); bins_new[fold] += item["Bins"]
                groups_new = fold_group_count.copy(); groups_new[fold] += 1
                candidate_scores.append(_assignment_score(
                    rows_new, y_sum_new, y_sq_new, bins_new, groups_new,
                    global_mean, global_sd, global_bin_prop,
                ))
            minimum = float(np.min(candidate_scores))
            choices = np.where(np.isclose(candidate_scores, minimum, rtol=0, atol=1e-12))[0]
            chosen = int(rng.choice(choices))
            fold_groups[chosen].add(item["Group"])
            fold_rows[chosen] += item["Rows"]
            fold_y_sum[chosen] += item["Y_Sum"]
            fold_y_sq[chosen] += item["Y_Sq"]
            fold_bins[chosen] += item["Bins"]
            fold_group_count[chosen] += 1
        score = _assignment_score(
            fold_rows, fold_y_sum, fold_y_sq, fold_bins, fold_group_count,
            global_mean, global_sd, global_bin_prop,
        )
        if score < best_score:
            best_score = score
            best_groups = [set(values) for values in fold_groups]
    if best_groups is None:
        raise RuntimeError("无法构建目标平衡严格分组折。")
    splits = []
    assignment = np.zeros(len(groups), dtype=int)
    for fold_groups in best_groups:
        test_index = np.where(np.isin(groups, list(fold_groups)))[0]
        train_index = np.where(~np.isin(groups, list(fold_groups)))[0]
        if set(groups[train_index]).intersection(set(groups[test_index])):
            raise RuntimeError("严格组泄露。")
        assignment[test_index] += 1
        splits.append((train_index, test_index))
    if not np.all(assignment == 1):
        raise RuntimeError("部分样本没有恰好进入一次测试折。")
    return splits


def make_group_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    outer: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    actual_splits = min(int(n_splits), int(pd.Series(groups).nunique()))
    if actual_splits < 2:
        raise ValueError("严格组数量不足，不能交叉验证。")
    if OUTER_SPLIT_MODE == "group_kfold" and outer:
        splitter = GroupKFold(n_splits=actual_splits)
        return list(splitter.split(np.zeros((len(y), 1)), y, groups))
    restarts = GROUP_SPLIT_RESTARTS_OUTER if outer else GROUP_SPLIT_RESTARTS_INNER
    return target_balanced_group_folds(groups, y, actual_splits, seed, restarts)


# =============================================================================
# 4. 数据读取与预处理
# =============================================================================

def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def make_numeric_imputer(strategy: str) -> SimpleImputer:
    try:
        return SimpleImputer(strategy=strategy, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy=strategy)


def make_knn_imputer() -> KNNImputer:
    try:
        return KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance", keep_empty_features=True)
    except TypeError:
        return KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance")


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    mode: Mapping,
) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        if mode["Imputer"] == "knn":
            numeric_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("imputer", make_knn_imputer()),
            ])
        else:
            numeric_pipeline = Pipeline([
                ("imputer", make_numeric_imputer("median")),
                ("scaler", StandardScaler()),
            ])
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
            ("onehot", make_onehot()),
        ])
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def safe_feature_name(value: str) -> str:
    text = re.sub(r"\s+", "_", str(value).strip())
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text)
    return text[:70]


def first_role_column(columns: Sequence[str], tokens: Sequence[str]) -> Optional[str]:
    for column in columns:
        name = norm_name(column)
        if any(norm_name(token) in name for token in tokens):
            return column
    return None


def build_physics_feature_frame(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    """只使用 X 构造确定性物理/尺度特征，不读取 y。"""
    result = frame.copy().reset_index(drop=True)
    available_numeric = [column for column in numeric_columns if column in result.columns]

    for column in available_numeric:
        values = to_numeric_safe(result[column])
        result[column] = values
        prefix = f"PHY__{safe_feature_name(column)}"
        result[f"{prefix}__missing"] = values.isna().astype(np.float32)
        result[f"{prefix}__signed_log1p"] = np.sign(values) * np.log1p(np.abs(values))
        result[f"{prefix}__signed_sqrt"] = np.sign(values) * np.sqrt(np.abs(values))

    ph_col = first_role_column(available_numeric, ("ph", "酸碱"))
    temp_col = first_role_column(available_numeric, ("温度", "temperature"))
    time_col = first_role_column(available_numeric, ("光照时间", "反应时间", "时间", "time"))
    zeta_col = first_role_column(available_numeric, ("zeta", "电位"))
    plastic_col = first_role_column(available_numeric, ("塑料浓度", "plasticconcentration"))
    size_col = first_role_column(available_numeric, ("初始粒径", "粒径", "size"))
    coexist_col = first_role_column(available_numeric, ("共存物质浓度", "coexistingconcentration"))

    if ph_col is not None:
        ph = to_numeric_safe(result[ph_col])
        result["PHY__pH_abs7"] = np.abs(ph - 7.0)
        result["PHY__pH_distance7_sq"] = (ph - 7.0) ** 2
        result["PHY__pH_acidic"] = (ph < 7.0).astype(np.float32)
        result["PHY__pH_alkaline"] = (ph > 7.0).astype(np.float32)

    if temp_col is not None:
        temperature = to_numeric_safe(result[temp_col])
        kelvin = temperature + 273.15
        result["PHY__temperature_kelvin"] = kelvin
        result["PHY__inverse_kelvin"] = 1.0 / np.maximum(kelvin, 1.0)
        result["PHY__arrhenius_proxy"] = 1000.0 / np.maximum(kelvin, 1.0)

    def add_pair(first_col: Optional[str], second_col: Optional[str], label: str) -> None:
        if first_col is None or second_col is None or first_col == second_col:
            return
        first = to_numeric_safe(result[first_col])
        second = to_numeric_safe(result[second_col])
        product = first * second
        denominator_first = np.abs(first) + 1e-8
        denominator_second = np.abs(second) + 1e-8
        result[f"PHYPAIR__{label}__product"] = product
        result[f"PHYPAIR__{label}__product_log"] = np.sign(product) * np.log1p(np.abs(product))
        result[f"PHYPAIR__{label}__ratio"] = first / denominator_second
        result[f"PHYPAIR__{label}__reverse_ratio"] = second / denominator_first
        result[f"PHYPAIR__{label}__log_ratio"] = np.log1p(np.abs(first)) - np.log1p(np.abs(second))

    add_pair(time_col, plastic_col, "time_plastic")
    add_pair(time_col, coexist_col, "time_coexist")
    add_pair(plastic_col, size_col, "plastic_size")
    add_pair(coexist_col, plastic_col, "coexist_plastic")
    add_pair(zeta_col, ph_col, "zeta_ph")
    add_pair(zeta_col, coexist_col, "zeta_coexist")
    add_pair(temp_col, time_col, "temperature_time")

    return result


def infer_frame_types(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def load_and_prepare_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Dict]:
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(FILE_PATH)

    raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    raw.columns = [re.sub(r"\s+", " ", str(column)).strip() for column in raw.columns]
    target_column = find_col(raw.columns, TARGET_KEYWORD)
    group_column = find_col(raw.columns, GROUP_KEYWORD)
    title_column = find_col(raw.columns, TITLE_KEYWORD, required=False)

    target_numeric = to_numeric_safe(raw[target_column])
    valid = target_numeric.notna()
    if TARGET_TRANSFORM == "log10":
        valid &= target_numeric > 0

    # Excel 行号按“第1行为表头、数据从第2行开始”计算。
    original_zero_based_indices = raw.index[valid].to_numpy(dtype=int)
    excel_row_numbers = original_zero_based_indices + 2

    # source_data 保留 Excel 中的原始显示值；data 用于后续类型清洗。
    source_data = raw.loc[valid].copy().reset_index(drop=True)
    data = source_data.copy()
    y_raw = target_numeric.loc[valid].to_numpy(dtype=float)
    y_model = transform_target(y_raw)

    literature_ids = []
    for row_index, value in enumerate(data[group_column]):
        if pd.isna(value) or not str(value).strip():
            literature_ids.append(f"<MISSING_{row_index}>")
        else:
            literature_ids.append(str(value).strip())
    literature_ids = np.asarray(literature_ids, dtype=object)

    excluded = set(find_all_cols(data.columns, TARGET_KEYWORD))
    excluded.add(group_column)
    if title_column is not None:
        excluded.add(title_column)
    model_columns = [column for column in data.columns if column not in excluded]
    X = data[model_columns].copy()

    numeric_columns: List[str] = []
    categorical_columns: List[str] = []
    inferred_numeric: List[str] = []
    for column in X.columns:
        if contains_any_keyword(column, FORCE_CATEGORICAL_KEYWORDS):
            X[column] = X[column].map(normalize_category)
            categorical_columns.append(column)
            continue
        if contains_any_keyword(column, NUMERIC_KEYWORDS):
            X[column] = to_numeric_safe(X[column])
            numeric_columns.append(column)
            continue
        converted = to_numeric_safe(X[column])
        numeric_ratio = converted.notna().sum() / max(X[column].notna().sum(), 1)
        if pd.api.types.is_numeric_dtype(X[column]) or numeric_ratio >= 0.95:
            X[column] = converted
            numeric_columns.append(column)
            inferred_numeric.append(column)
        else:
            X[column] = X[column].map(normalize_category)
            categorical_columns.append(column)

    signatures = [
        tuple(signature_value(value) for value in row)
        for row in X.itertuples(index=False, name=None)
    ]
    strict_groups = build_strict_groups(literature_ids, signatures)

    metadata = {
        "raw_rows": len(raw),
        "valid_rows": len(X),
        "target_column": target_column,
        "group_column": group_column,
        "title_column": title_column,
        "literature_count": int(pd.Series(literature_ids).nunique()),
        "strict_group_count": int(pd.Series(strict_groups).nunique()),
        "model_columns": model_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "inferred_numeric_columns": inferred_numeric,
        # 以下内容只用于运行结束后的数据质量审计，不进入模型特征。
        "source_data": source_data,
        "excel_row_numbers": np.asarray(excel_row_numbers, dtype=int),
        "original_zero_based_indices": np.asarray(original_zero_based_indices, dtype=int),
        "literature_ids": literature_ids.copy(),
        "signatures": list(signatures),
    }
    return X, y_model, y_raw, strict_groups, metadata


# =============================================================================
# 5. 模型、内层缓存和候选评估
# =============================================================================

def make_rf(seed: int, params: Mapping) -> RandomForestRegressor:
    return RandomForestRegressor(bootstrap=True, random_state=seed, n_jobs=N_JOBS, **params)


def make_extra_trees(seed: int, params: Mapping) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(random_state=seed, n_jobs=N_JOBS, **params)


def make_gbr(seed: int, params: Mapping) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=seed, **params)


def make_lgbm(seed: int, params: Mapping) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression", random_state=seed, n_jobs=N_JOBS, verbosity=-1,
        force_col_wise=True, subsample_freq=1, **params,
    )


def make_xgb(seed: int, params: Mapping) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=N_JOBS,
        tree_method="hist",
        verbosity=0,
        **params,
    )


@dataclass
class InnerFoldData:
    train_index: np.ndarray
    valid_index: np.ndarray
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray


def build_inner_cache(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    seed: int,
    mode: Mapping,
    precomputed_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[InnerFoldData]:
    inner_splits = (
        list(precomputed_splits)
        if precomputed_splits is not None
        else make_group_folds(groups_train, y_train, INNER_SPLITS, seed, outer=False)
    )
    cache: List[InnerFoldData] = []
    for fit_index, valid_index in inner_splits:
        fit_index = np.asarray(fit_index, dtype=int)
        valid_index = np.asarray(valid_index, dtype=int)
        if set(groups_train[fit_index]).intersection(set(groups_train[valid_index])):
            raise RuntimeError("内层严格组泄露。")
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, mode)
        X_fit = np.asarray(
            preprocessor.fit_transform(X_train.iloc[fit_index], y_train[fit_index]),
            dtype=np.float32,
        )
        X_valid = np.asarray(preprocessor.transform(X_train.iloc[valid_index]), dtype=np.float32)
        cache.append(InnerFoldData(
            train_index=fit_index,
            valid_index=valid_index,
            X_train=X_fit,
            X_valid=X_valid,
            y_train=y_train[fit_index],
            y_valid=y_train[valid_index],
        ))
    return cache

def evaluate_candidate_grid(
    inner_cache: Sequence[InnerFoldData],
    y_full: np.ndarray,
    candidates: Sequence[Mapping],
    builder: Callable[[int, Mapping], object],
    seed: int,
) -> Dict:
    best: Optional[Dict] = None
    for candidate_index, params in enumerate(candidates, start=1):
        oof = np.full(len(y_full), np.nan, dtype=float)
        fold_records = []
        for fold_index, item in enumerate(inner_cache, start=1):
            model = builder(seed + candidate_index * 10000 + fold_index * 1000, params)
            model.fit(item.X_train, item.y_train)
            prediction = np.asarray(model.predict(item.X_valid), dtype=float)
            oof[item.valid_index] = prediction
            fold_records.append({"y_true": item.y_valid, "y_pred": prediction})
        details = {
            **normalized_cv_objective(y_full, oof, fold_records),
            "Params": dict(params),
            "OOF_Prediction": oof.copy(),
        }
        if best is None or details["Objective"] < best["Objective"]:
            best = details
    if best is None:
        raise RuntimeError("候选参数为空。")
    return best


def select_preprocessor_and_cache(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    seed: int,
) -> Tuple[Mapping, List[InnerFoldData]]:
    modes = PREPROCESSOR_CANDIDATES if AUTO_SELECT_PREPROCESSOR else PREPROCESSOR_CANDIDATES[:1]
    probe_params = GBR750_CANDIDATES[0]
    rows = []
    best_mode = None
    best_cache = None
    best_objective = np.inf
    for mode_index, mode in enumerate(modes, start=1):
        cache = build_inner_cache(
            X_train, y_train, groups_train, numeric_columns, categorical_columns,
            seed + mode_index * 10000, mode,
        )
        probe = evaluate_candidate_grid(
            cache, y_train, [probe_params], make_gbr,
            seed + mode_index * 100000,
        )
        rows.append({
            "Mode": mode["Name"], "Objective": probe["Objective"],
            "OOF_R2": probe["OOF_R2"], "OOF_RMSE": probe["OOF_RMSE"],
        })
        if probe["Objective"] < best_objective:
            best_objective = probe["Objective"]
            best_mode = mode
            best_cache = cache
    print(pd.DataFrame(rows).sort_values("Objective").to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if best_mode is None or best_cache is None:
        raise RuntimeError("预处理选择失败。")
    return best_mode, best_cache


# =============================================================================
# 6. GBR锚定五层重复 Group-CV Stacking
# =============================================================================


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(mask)) < 3:
        return 0.0
    first = first[mask]
    second = second[mask]
    if float(np.std(first)) <= 1e-12 or float(np.std(second)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _stack_model_statistics(
    prediction_matrix: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    model_names: Sequence[str],
) -> pd.DataFrame:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    names = list(model_names)
    if STACK_ANCHOR_NAME not in names:
        raise ValueError("GBR750锚定Stacking要求一级模型中必须包含GBR750。")
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_prediction = prediction_matrix[:, gbr_index]
    target_residual = y - gbr_prediction
    rows = []
    for model_index, name in enumerate(names):
        fold_r2 = [
            safe_r2(y[valid_index], prediction_matrix[valid_index, model_index])
            for _, valid_index in splits
        ]
        finite = np.asarray([value for value in fold_r2 if np.isfinite(value)], dtype=float)
        pooled = safe_r2(y, prediction_matrix[:, model_index])
        mean_fold = float(np.mean(finite)) if len(finite) else -np.inf
        worst_fold = float(np.min(finite)) if len(finite) else -np.inf
        sd_fold = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        if model_index == gbr_index:
            residual_signal = 0.0
            prediction_corr = 1.0
        else:
            correction_signal = prediction_matrix[:, model_index] - gbr_prediction
            residual_signal = _safe_corr(correction_signal, target_residual)
            prediction_corr = _safe_corr(prediction_matrix[:, model_index], gbr_prediction)
        robust_score = (
            0.50 * pooled
            + 0.25 * mean_fold
            + 0.17 * worst_fold
            - 0.05 * sd_fold
            + 0.10 * abs(residual_signal)
            + 0.03 * (1.0 - abs(prediction_corr))
        )
        rows.append({
            "Model": str(name),
            "Model_Index": int(model_index),
            "Pooled_R2": float(pooled),
            "Mean_Fold_R2": mean_fold,
            "Worst_Fold_R2": worst_fold,
            "SD_Fold_R2": sd_fold,
            "Residual_Signal": float(residual_signal),
            "Corr_vs_GBR": float(prediction_corr),
            "Robust_Score": float(robust_score),
        })
    return pd.DataFrame(rows).sort_values(
        ["Robust_Score", "Pooled_R2"], ascending=[False, False]
    ).reset_index(drop=True)


def _select_residual_partners(
    prediction_matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    split_count = min(META_SPLITS, len(np.unique(groups)))
    splits = make_group_folds(groups, y, split_count, seed, outer=False)
    stats = _stack_model_statistics(prediction_matrix, y, splits, names)
    gbr_r2 = float(stats.loc[stats["Model"] == STACK_ANCHOR_NAME, "Pooled_R2"].iloc[0])

    eligible = stats[
        (stats["Model"] != STACK_ANCHOR_NAME)
        & (stats["Pooled_R2"] >= max(STACK_ABSOLUTE_MIN_R2, gbr_r2 - STACK_PARTNER_R2_TOL))
        & (stats["Worst_Fold_R2"] > -0.10)
    ].copy()
    if not eligible.empty:
        eligible["Partner_Score"] = (
            0.55 * eligible["Robust_Score"]
            + 0.23 * eligible["Residual_Signal"].abs()
            + 0.12 * (1.0 - eligible["Corr_vs_GBR"].abs())
            + 0.10 * eligible["Worst_Fold_R2"]
        )
        eligible = eligible.sort_values(
            ["Partner_Score", "Pooled_R2", "Worst_Fold_R2"],
            ascending=[False, False, False],
        )
    partner_indices = eligible["Model_Index"].astype(int).tolist()[:STACK_MAX_PARTNERS]
    if not partner_indices:
        fallback_partner = stats[stats["Model"] != STACK_ANCHOR_NAME].head(1)
        if not fallback_partner.empty:
            partner_indices = [int(fallback_partner.iloc[0]["Model_Index"])]

    selected_indices = [gbr_index] + [index for index in partner_indices if index != gbr_index]
    return {
        "Stats": stats,
        "GBR_Index": gbr_index,
        "GBR_R2": gbr_r2,
        "Partner_Indices": partner_indices,
        "Selected_Indices": selected_indices,
        "Selected_Models": [names[index] for index in selected_indices],
    }


def _weight_bounds_for_models(model_names: Sequence[str]) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for name in model_names:
        if str(name) == STACK_ANCHOR_NAME:
            bounds.append((STACK_GBR_MIN_WEIGHT, STACK_GBR_MAX_WEIGHT))
        else:
            bounds.append(tuple(STACK_PARTNER_WEIGHT_BOUNDS.get(str(name), (-0.05, 0.10))))
    return bounds


def _project_anchor_weights(candidate_weights: np.ndarray, model_names: Sequence[str]) -> np.ndarray:
    candidate_weights = np.asarray(candidate_weights, dtype=float)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    bounds = _weight_bounds_for_models(names)
    x0 = np.zeros(len(names), dtype=float)
    for index, (low, high) in enumerate(bounds):
        x0[index] = float(np.clip(candidate_weights[index], low, high))
    x0[gbr_index] += 1.0 - float(np.sum(x0))
    x0[gbr_index] = float(np.clip(x0[gbr_index], *bounds[gbr_index]))
    result = minimize(
        lambda weights: float(np.sum((weights - candidate_weights) ** 2)),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 700, "ftol": 1e-12, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        return np.asarray(result.x, dtype=float)
    fallback = np.zeros(len(names), dtype=float)
    fallback[gbr_index] = 1.0
    return fallback


def fit_gbr_anchored_weights(
    prediction_matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    l2: float,
    seed: int,
) -> Dict:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    context = _select_residual_partners(prediction_matrix, y, groups, names, seed)
    gbr_index = int(context["GBR_Index"])
    selected = list(context["Selected_Indices"])
    selected_names = [names[index] for index in selected]
    selected_bounds = _weight_bounds_for_models(selected_names)
    sample_weight = group_balanced_weights(groups)

    x0_weights = np.zeros(len(selected), dtype=float)
    gbr_position = selected.index(gbr_index)
    x0_weights[gbr_position] = 0.84
    partner_positions = [position for position, index in enumerate(selected) if index != gbr_index]
    if partner_positions:
        per_partner = 0.16 / len(partner_positions)
        for position in partner_positions:
            low, high = selected_bounds[position]
            x0_weights[position] = float(np.clip(per_partner, low, high))
    x0_weights[gbr_position] += 1.0 - float(np.sum(x0_weights))
    x0 = np.concatenate([x0_weights, [0.0]])
    bounds = selected_bounds + [(-STACK_BIAS_BOUND, STACK_BIAS_BOUND)]

    def objective(parameters: np.ndarray) -> float:
        weights = parameters[:-1]
        bias = float(parameters[-1])
        prediction = prediction_matrix[:, selected] @ weights + bias
        residual = y - prediction
        weighted_mse = float(np.average(residual ** 2, weights=sample_weight))
        partner_weights = np.delete(weights, gbr_position)
        return (
            weighted_mse
            + float(l2) * float(np.sum(partner_weights ** 2))
            + STACK_BIAS_L2 * bias * bias
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda parameters: float(np.sum(parameters[:-1]) - 1.0)}],
        options={"maxiter": 900, "ftol": 1e-12, "disp": False},
    )

    full_weights = np.zeros(len(names), dtype=float)
    if result.success and np.all(np.isfinite(result.x)):
        selected_weights = np.asarray(result.x[:-1], dtype=float)
        bias = float(np.clip(result.x[-1], -STACK_BIAS_BOUND, STACK_BIAS_BOUND))
        for selected_position, model_index in enumerate(selected):
            full_weights[model_index] = selected_weights[selected_position]
        full_weights = _project_anchor_weights(full_weights, names)
    else:
        full_weights[gbr_index] = 1.0
        bias = 0.0
    return {"Weights": full_weights, "Bias": bias, **context}


def _effective_stage1_parameters(
    learned_weights: np.ndarray,
    learned_bias: float,
    shrinkage: float,
    model_names: Sequence[str],
) -> Tuple[np.ndarray, float]:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_only = np.zeros(len(names), dtype=float)
    gbr_only[gbr_index] = 1.0
    effective_weights = gbr_only + float(shrinkage) * (np.asarray(learned_weights) - gbr_only)
    effective_weights = _project_anchor_weights(effective_weights, names)
    effective_bias = float(np.clip(float(shrinkage) * float(learned_bias), -STACK_BIAS_BOUND, STACK_BIAS_BOUND))
    return effective_weights, effective_bias


def _build_meta2_features(prediction_matrix: np.ndarray, model_names: Sequence[str]) -> np.ndarray:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr = prediction_matrix[:, gbr_index]
    differences = np.column_stack([
        prediction_matrix[:, index] - gbr
        for index in range(prediction_matrix.shape[1]) if index != gbr_index
    ])
    mean_prediction = np.mean(prediction_matrix, axis=1)
    median_prediction = np.median(prediction_matrix, axis=1)
    std_prediction = np.std(prediction_matrix, axis=1)
    min_prediction = np.min(prediction_matrix, axis=1)
    max_prediction = np.max(prediction_matrix, axis=1)
    q25 = np.quantile(prediction_matrix, 0.25, axis=1)
    q75 = np.quantile(prediction_matrix, 0.75, axis=1)
    tree_indices = [index for index, name in enumerate(names) if name in {"RF", "ExtraTrees", "LGBM", "XGB", "GBR", STACK_ANCHOR_NAME}]
    tree_matrix = prediction_matrix[:, tree_indices]
    tree_mean = np.mean(tree_matrix, axis=1)
    tree_std = np.std(tree_matrix, axis=1)
    consensus_gap = gbr - median_prediction
    magnitude_features = np.column_stack([
        gbr,
        gbr ** 2,
        np.abs(gbr - mean_prediction),
        std_prediction,
        std_prediction * np.maximum(np.abs(gbr), 0.25),
    ])
    summary = np.column_stack([
        mean_prediction, median_prediction, std_prediction,
        min_prediction, max_prediction, max_prediction - min_prediction,
        q25, q75, q75 - q25, tree_mean, tree_std, consensus_gap,
    ])
    return np.column_stack([prediction_matrix, differences, summary, magnitude_features])


def _robust_residual_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.20
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 0.03:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale < 0.03:
        scale = 0.20
    return float(scale)


def _fit_meta2_corrector(
    features: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    config: Mapping,
    seed: int,
) -> Dict:
    features = np.asarray(features, dtype=float)
    residual = np.asarray(residual, dtype=float)
    sample_weight = group_balanced_weights(np.asarray(groups))
    family = str(config["Family"])
    if family == "ridge":
        scaler = StandardScaler()
        transformed = scaler.fit_transform(features)
        estimator = Ridge(alpha=float(config["alpha"]), fit_intercept=True)
        estimator.fit(transformed, residual, sample_weight=sample_weight)
        return {"Family": family, "Scaler": scaler, "Estimator": estimator}
    if family == "gbr":
        estimator = GradientBoostingRegressor(
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            subsample=float(config["subsample"]),
            loss=str(config["loss"]),
            random_state=int(seed),
            max_features=0.75,
        )
        estimator.fit(features, residual, sample_weight=sample_weight)
        return {"Family": family, "Scaler": None, "Estimator": estimator}
    raise ValueError(f"未知二级残差校准器：{family}")


def _predict_meta2_corrector(fitted: Mapping, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    scaler = fitted.get("Scaler")
    transformed = scaler.transform(features) if scaler is not None else features
    return np.asarray(fitted["Estimator"].predict(transformed), dtype=float)


def _select_stage1_configuration(
    base_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_r2 = safe_r2(y, gbr_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))
    candidate_rows = []
    payloads: Dict[Tuple[float, float], Dict] = {}

    for l2 in STACK_L2_GRID:
        accum = {
            float(shrink): {
                "Prediction_Sum": np.zeros(len(y), dtype=float),
                "Prediction_Count": np.zeros(len(y), dtype=float),
                "Repeat_Gains": [], "Fold_Gains": [],
            }
            for shrink in STACK_STAGE1_SHRINK_GRID
        }
        all_weights: List[np.ndarray] = []
        all_biases: List[float] = []
        for repeat in range(STACK_META_REPEATS):
            repeat_seed = seed + 13007 + repeat * 7919
            splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
            repeat_predictions = {
                float(shrink): np.full(len(y), np.nan, dtype=float)
                for shrink in STACK_STAGE1_SHRINK_GRID
            }
            for fold_index, (train_index, valid_index) in enumerate(splits, start=1):
                fit = fit_gbr_anchored_weights(
                    base_oof[train_index], y[train_index], groups[train_index],
                    names, float(l2), repeat_seed + fold_index * 103,
                )
                all_weights.append(np.asarray(fit["Weights"], dtype=float))
                all_biases.append(float(fit["Bias"]))
                linear_valid = base_oof[valid_index] @ fit["Weights"] + float(fit["Bias"])
                baseline_valid = base_oof[valid_index, gbr_index]
                for shrink in STACK_STAGE1_SHRINK_GRID:
                    shrink = float(shrink)
                    prediction = baseline_valid + shrink * (linear_valid - baseline_valid)
                    repeat_predictions[shrink][valid_index] = prediction
                    accum[shrink]["Fold_Gains"].append(
                        safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                    )
            for shrink in STACK_STAGE1_SHRINK_GRID:
                shrink = float(shrink)
                repeat_prediction = repeat_predictions[shrink]
                accum[shrink]["Prediction_Sum"] += repeat_prediction
                accum[shrink]["Prediction_Count"] += np.isfinite(repeat_prediction).astype(float)
                accum[shrink]["Repeat_Gains"].append(safe_r2(y, repeat_prediction) - gbr_r2)

        average_weights = _project_anchor_weights(np.mean(np.vstack(all_weights), axis=0), names)
        average_bias = float(np.mean(all_biases))
        for shrink in STACK_STAGE1_SHRINK_GRID:
            shrink = float(shrink)
            item = accum[shrink]
            meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
            metric = metrics(y, meta_oof)
            repeat_array = np.asarray(item["Repeat_Gains"], dtype=float)
            fold_array = np.asarray(item["Fold_Gains"], dtype=float)
            gain = float(metric["R2"] - gbr_r2)
            row = {
                "L2": float(l2), "Stage1_Shrinkage": shrink,
                "R2": float(metric["R2"]), "RMSE": float(metric["RMSE"]), "MAE": float(metric["MAE"]),
                "Meta_Gain": gain,
                "Mean_Repeat_Gain": float(np.mean(repeat_array)),
                "SD_Repeat_Gain": float(np.std(repeat_array, ddof=1)) if len(repeat_array) > 1 else 0.0,
                "Positive_Repeats": int(np.sum(repeat_array > 0.0)),
                "Worst_Repeat_Gain": float(np.min(repeat_array)),
                "Positive_Folds": int(np.sum(fold_array > 0.0)),
                "Total_Folds": int(len(fold_array)),
                "Worst_Fold_Gain": float(np.min(fold_array)),
            }
            candidate_rows.append(row)
            payloads[(float(l2), shrink)] = {
                "Meta_OOF": meta_oof,
                "Average_Weights": average_weights,
                "Average_Bias": average_bias,
            }

    table = pd.DataFrame(candidate_rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.16 * table["Mean_Repeat_Gain"]
        + 0.06 * table["Worst_Repeat_Gain"] - 0.04 * table["SD_Repeat_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    key = (float(best["L2"]), float(best["Stage1_Shrinkage"]))
    return {"Table": table, "Best_Row": best, "Payload": payloads[key], "GBR_R2": gbr_r2}


def _select_meta2_configuration(
    base_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    selected_l2: float,
    stage1_shrinkage: float,
    seed: int,
) -> Dict:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_r2 = safe_r2(y, gbr_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))

    candidate_keys: List[Tuple[str, float, float]] = [("None", 0.0, 0.0)]
    for config in STACK_META2_CORRECTOR_CANDIDATES:
        for strength in STACK_META2_STRENGTH_GRID:
            for clip_multiplier in STACK_META2_CLIP_MULTIPLIER_GRID:
                candidate_keys.append((str(config["Name"]), float(strength), float(clip_multiplier)))

    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Gains": [], "Fold_Gains": [],
        }
        for key in candidate_keys
    }
    config_map = {str(config["Name"]): config for config in STACK_META2_CORRECTOR_CANDIDATES}

    for repeat in range(STACK_META_REPEATS):
        repeat_seed = seed + 77041 + repeat * 104729
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in candidate_keys}
        for fold_index, (train_index, valid_index) in enumerate(splits, start=1):
            fit = fit_gbr_anchored_weights(
                base_oof[train_index], y[train_index], groups[train_index], names,
                float(selected_l2), repeat_seed + fold_index * 211,
            )
            effective_weights, effective_bias = _effective_stage1_parameters(
                fit["Weights"], fit["Bias"], float(stage1_shrinkage), names
            )
            stage1_train = base_oof[train_index] @ effective_weights + effective_bias
            stage1_valid = base_oof[valid_index] @ effective_weights + effective_bias
            baseline_valid = base_oof[valid_index, gbr_index]
            residual_train = y[train_index] - stage1_train
            residual_scale = _robust_residual_scale(residual_train)
            train_features = _build_meta2_features(base_oof[train_index], names)
            valid_features = _build_meta2_features(base_oof[valid_index], names)

            null_key = ("None", 0.0, 0.0)
            repeat_predictions[null_key][valid_index] = stage1_valid
            accum[null_key]["Fold_Gains"].append(
                safe_r2(y[valid_index], stage1_valid) - safe_r2(y[valid_index], baseline_valid)
            )

            for corrector_index, config in enumerate(STACK_META2_CORRECTOR_CANDIDATES, start=1):
                fitted = _fit_meta2_corrector(
                    train_features, residual_train, groups[train_index], config,
                    repeat_seed + fold_index * 1009 + corrector_index * 97,
                )
                raw_correction = _predict_meta2_corrector(fitted, valid_features)
                for clip_multiplier in STACK_META2_CLIP_MULTIPLIER_GRID:
                    clip_multiplier = float(clip_multiplier)
                    bound = max(0.025, clip_multiplier * residual_scale)
                    clipped = np.clip(raw_correction, -bound, bound)
                    for strength in STACK_META2_STRENGTH_GRID:
                        strength = float(strength)
                        key = (str(config["Name"]), strength, clip_multiplier)
                        prediction = stage1_valid + strength * clipped
                        repeat_predictions[key][valid_index] = prediction
                        accum[key]["Fold_Gains"].append(
                            safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                        )

        for key in candidate_keys:
            repeat_prediction = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += repeat_prediction
            accum[key]["Prediction_Count"] += np.isfinite(repeat_prediction).astype(float)
            accum[key]["Repeat_Gains"].append(safe_r2(y, repeat_prediction) - gbr_r2)

    rows = []
    payloads = {}
    for key in candidate_keys:
        name, strength, clip_multiplier = key
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        repeat_array = np.asarray(item["Repeat_Gains"], dtype=float)
        fold_array = np.asarray(item["Fold_Gains"], dtype=float)
        gain = float(metric["R2"] - gbr_r2)
        rows.append({
            "Corrector": name, "Residual_Strength": float(strength),
            "Clip_Multiplier": float(clip_multiplier),
            "R2": float(metric["R2"]), "RMSE": float(metric["RMSE"]), "MAE": float(metric["MAE"]),
            "Meta_Gain": gain,
            "Mean_Repeat_Gain": float(np.mean(repeat_array)),
            "SD_Repeat_Gain": float(np.std(repeat_array, ddof=1)) if len(repeat_array) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(repeat_array > 0.0)),
            "Worst_Repeat_Gain": float(np.min(repeat_array)),
            "Positive_Folds": int(np.sum(fold_array > 0.0)),
            "Total_Folds": int(len(fold_array)),
            "Worst_Fold_Gain": float(np.min(fold_array)),
        })
        payloads[key] = {"Meta_OOF": meta_oof}

    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.20 * table["Mean_Repeat_Gain"]
        + 0.08 * table["Worst_Repeat_Gain"] - 0.06 * table["SD_Repeat_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (str(best["Corrector"]), float(best["Residual_Strength"]), float(best["Clip_Multiplier"]))
    return {"Table": table, "Best_Row": best, "Payload": payloads[best_key], "Config_Map": config_map}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float(np.nanmean(values))
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _get_gate_expert_prediction(
    prediction_matrix: np.ndarray,
    expert_name: str,
    model_names: Sequence[str],
) -> np.ndarray:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    names = list(model_names)
    if expert_name in names:
        return np.asarray(prediction_matrix[:, names.index(expert_name)], dtype=float)
    if expert_name == "ConsensusMedian":
        return np.asarray(np.median(prediction_matrix, axis=1), dtype=float)
    if expert_name == "TreeMedian":
        indices = [index for index, name in enumerate(names) if name in {"RF", "ExtraTrees", "LGBM", "XGB", "GBR", STACK_ANCHOR_NAME}]
        return np.asarray(np.median(prediction_matrix[:, indices], axis=1), dtype=float)
    raise ValueError(f"未知门控专家：{expert_name}")



def _make_distinct_meta_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """随机搜索目标/样本量平衡的Group折；不同seed会得到不同真实分区。"""
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    unique_groups = np.unique(groups)
    n_splits = int(min(max(2, n_splits), len(unique_groups)))
    rng = np.random.default_rng(int(seed))
    global_mean = float(np.mean(y))
    global_sd = max(float(np.std(y)), 1e-8)
    target_size = len(y) / n_splits
    target_group_count = len(unique_groups) / n_splits

    group_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    group_sizes = np.asarray([len(index) for index in group_indices], dtype=float)
    group_sums = np.asarray([float(np.sum(y[index])) for index in group_indices], dtype=float)

    best_assignment = None
    best_score = np.inf
    restarts = max(160, 20 * n_splits)
    for _ in range(restarts):
        permutation = rng.permutation(len(unique_groups))
        fold_order = rng.permutation(n_splits)
        assignment = np.empty(len(unique_groups), dtype=int)
        # 轮转分配确保每折文献数量接近，再用随机交换探索更优平衡。
        for position, group_position in enumerate(permutation):
            assignment[group_position] = int(fold_order[position % n_splits])

        for _swap in range(max(30, len(unique_groups) * 2)):
            a, b = rng.choice(len(unique_groups), size=2, replace=False)
            if assignment[a] == assignment[b]:
                continue
            current_a, current_b = assignment[a], assignment[b]
            if rng.random() < 0.55:
                assignment[a], assignment[b] = current_b, current_a

        fold_sizes = np.zeros(n_splits, dtype=float)
        fold_sums = np.zeros(n_splits, dtype=float)
        fold_group_counts = np.zeros(n_splits, dtype=float)
        for group_position, fold in enumerate(assignment):
            fold_sizes[fold] += group_sizes[group_position]
            fold_sums[fold] += group_sums[group_position]
            fold_group_counts[fold] += 1.0
        fold_means = fold_sums / np.maximum(fold_sizes, 1.0)
        size_penalty = float(np.mean(((fold_sizes - target_size) / max(target_size, 1.0)) ** 2))
        mean_penalty = float(np.mean(((fold_means - global_mean) / global_sd) ** 2))
        count_penalty = float(np.mean(((fold_group_counts - target_group_count) / max(target_group_count, 1.0)) ** 2))
        empty_penalty = 1000.0 if np.any(fold_group_counts < 1) else 0.0
        score = 0.48 * size_penalty + 0.47 * mean_penalty + 0.05 * count_penalty + empty_penalty
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("无法生成Meta Group折。")
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        validation_groups = unique_groups[best_assignment == fold]
        valid_mask = np.isin(groups, validation_groups)
        valid_index = np.flatnonzero(valid_mask)
        train_index = np.flatnonzero(~valid_mask)
        if len(valid_index) == 0 or len(train_index) == 0:
            raise RuntimeError("Meta分组器产生空训练折或验证折。")
        splits.append((train_index, valid_index))
    return splits

def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if n_bins <= 1 or len(values) == 0:
        return np.asarray([], dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.unique(np.asarray(edges, dtype=float))


def _meta4_secondary_signal(
    base_prediction_matrix: np.ndarray,
    baseline_prediction: np.ndarray,
    model_names: Sequence[str],
    mode: str,
) -> np.ndarray:
    matrix = np.asarray(base_prediction_matrix, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    if mode == "Level":
        return np.zeros(len(baseline), dtype=float)
    if mode == "LevelDisagreement":
        return np.std(matrix, axis=1)
    if mode == "LevelConsensus":
        consensus = np.median(matrix, axis=1)
        return consensus - baseline
    raise ValueError(f"未知Meta4模式：{mode}")


def _fit_regime_calibrator(
    base_prediction_matrix: np.ndarray,
    baseline_prediction: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    mode: str,
    level_bins: int,
    secondary_bins: int,
    shrinkage: float,
    smoothing: float,
    clip_multiplier: float,
) -> Dict:
    matrix = np.asarray(base_prediction_matrix, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    residual = y - baseline
    weights = group_balanced_weights(groups)
    level_edges = _quantile_edges(baseline, int(level_bins))
    secondary_signal = _meta4_secondary_signal(matrix, baseline, model_names, mode)
    effective_secondary_bins = 1 if mode == "Level" else int(secondary_bins)
    secondary_edges = _quantile_edges(secondary_signal, effective_secondary_bins)
    level_id = np.digitize(baseline, level_edges, right=False)
    secondary_id = np.digitize(secondary_signal, secondary_edges, right=False)
    cell_id = level_id * max(1, len(secondary_edges) + 1) + secondary_id
    global_offset = _weighted_mean(residual, weights)
    offsets: Dict[int, float] = {}
    for cell in np.unique(cell_id):
        mask = cell_id == cell
        weight_sum = float(np.sum(weights[mask]))
        weighted_sum = float(np.sum(weights[mask] * residual[mask]))
        offset = (weighted_sum + float(smoothing) * global_offset) / max(weight_sum + float(smoothing), 1e-12)
        offsets[int(cell)] = float(offset)
    residual_scale = _robust_residual_scale(residual)
    bound = max(0.020, float(clip_multiplier) * residual_scale)
    return {
        "Mode": str(mode),
        "Level_Edges": level_edges,
        "Secondary_Edges": secondary_edges,
        "Offsets": offsets,
        "Global_Offset": float(global_offset),
        "Shrinkage": float(shrinkage),
        "Bound": float(bound),
        "Model_Names": list(model_names),
    }


def _predict_regime_correction(
    fitted: Mapping,
    base_prediction_matrix: np.ndarray,
    baseline_prediction: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(base_prediction_matrix, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    mode = str(fitted["Mode"])
    secondary_signal = _meta4_secondary_signal(matrix, baseline, fitted["Model_Names"], mode)
    level_id = np.digitize(baseline, np.asarray(fitted["Level_Edges"], dtype=float), right=False)
    secondary_id = np.digitize(
        secondary_signal, np.asarray(fitted["Secondary_Edges"], dtype=float), right=False
    )
    cell_id = level_id * max(1, len(fitted["Secondary_Edges"]) + 1) + secondary_id
    raw = np.asarray([
        float(fitted["Offsets"].get(int(cell), fitted["Global_Offset"]))
        for cell in cell_id
    ], dtype=float)
    raw = np.clip(raw, -float(fitted["Bound"]), float(fitted["Bound"]))
    return float(fitted["Shrinkage"]) * raw


def _select_meta4_configuration(
    base_oof: np.ndarray,
    baseline_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    base_oof = np.asarray(base_oof, dtype=float)
    baseline_oof = np.asarray(baseline_oof, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    gbr_oof = base_oof[:, names.index(STACK_ANCHOR_NAME)]
    gbr_r2 = safe_r2(y, gbr_oof)
    baseline_r2 = safe_r2(y, baseline_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))

    keys: List[Tuple[str, int, int, float, float, float]] = [
        ("None", 1, 1, 0.0, 0.0, 0.0)
    ]
    for mode in STACK_META4_MODES:
        for level_bins in STACK_META4_LEVEL_BINS:
            secondary_options = (1,) if mode == "Level" else STACK_META4_SECONDARY_BINS
            for secondary_bins in secondary_options:
                for shrinkage in STACK_META4_SHRINKAGE_GRID:
                    for smoothing in STACK_META4_SMOOTHING_GRID:
                        for clip in STACK_META4_CLIP_MULTIPLIER_GRID:
                            keys.append((str(mode), int(level_bins), int(secondary_bins), float(shrinkage), float(smoothing), float(clip)))

    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Total_Gains": [], "Repeat_Incremental_Gains": [],
            "Fold_Total_Gains": [], "Fold_Incremental_Gains": [],
        }
        for key in keys
    }
    for repeat in range(STACK_META_REPEATS):
        repeat_seed = int(seed) + 270001 + repeat * 32452843
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in keys}
        for train_index, valid_index in splits:
            null_key = ("None", 1, 1, 0.0, 0.0, 0.0)
            repeat_predictions[null_key][valid_index] = baseline_oof[valid_index]
            for key in keys[1:]:
                mode, level_bins, secondary_bins, shrinkage, smoothing, clip = key
                fitted = _fit_regime_calibrator(
                    base_oof[train_index], baseline_oof[train_index], y[train_index], groups[train_index],
                    names, mode, level_bins, secondary_bins, shrinkage, smoothing, clip,
                )
                correction = _predict_regime_correction(
                    fitted, base_oof[valid_index], baseline_oof[valid_index]
                )
                repeat_predictions[key][valid_index] = baseline_oof[valid_index] + correction

            for key in keys:
                pred = repeat_predictions[key][valid_index]
                total_gain = safe_r2(y[valid_index], pred) - safe_r2(y[valid_index], gbr_oof[valid_index])
                incremental_gain = safe_r2(y[valid_index], pred) - safe_r2(y[valid_index], baseline_oof[valid_index])
                accum[key]["Fold_Total_Gains"].append(float(total_gain))
                accum[key]["Fold_Incremental_Gains"].append(float(incremental_gain))

        for key in keys:
            pred = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += pred
            accum[key]["Prediction_Count"] += np.isfinite(pred).astype(float)
            accum[key]["Repeat_Total_Gains"].append(safe_r2(y, pred) - gbr_r2)
            accum[key]["Repeat_Incremental_Gains"].append(safe_r2(y, pred) - baseline_r2)

    rows, payloads = [], {}
    for key in keys:
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        total_repeat = np.asarray(item["Repeat_Total_Gains"], dtype=float)
        incremental_repeat = np.asarray(item["Repeat_Incremental_Gains"], dtype=float)
        total_fold = np.asarray(item["Fold_Total_Gains"], dtype=float)
        incremental_fold = np.asarray(item["Fold_Incremental_Gains"], dtype=float)
        mode, level_bins, secondary_bins, shrinkage, smoothing, clip = key
        rows.append({
            "Regime_Mode": mode, "Level_Bins": level_bins, "Secondary_Bins": secondary_bins,
            "Regime_Shrinkage": shrinkage, "Regime_Smoothing": smoothing,
            "Regime_Clip_Multiplier": clip,
            "R2": metric["R2"], "RMSE": metric["RMSE"], "MAE": metric["MAE"],
            "Total_Gain": float(metric["R2"] - gbr_r2),
            "Incremental_Gain": float(metric["R2"] - baseline_r2),
            "Mean_Repeat_Total_Gain": float(np.mean(total_repeat)),
            "Mean_Repeat_Incremental_Gain": float(np.mean(incremental_repeat)),
            "SD_Repeat_Incremental_Gain": float(np.std(incremental_repeat, ddof=1)) if len(incremental_repeat) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(incremental_repeat > 0.0)),
            "Worst_Repeat_Gain": float(np.min(incremental_repeat)),
            "Positive_Folds": int(np.sum(incremental_fold > 0.0)),
            "Total_Folds": int(len(incremental_fold)),
            "Worst_Fold_Gain": float(np.min(incremental_fold)),
        })
        payloads[key] = {"Meta_OOF": meta_oof}
    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.22 * table["Mean_Repeat_Incremental_Gain"]
        + 0.10 * table["Worst_Repeat_Gain"]
        - 0.08 * table["SD_Repeat_Incremental_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (
        str(best["Regime_Mode"]), int(best["Level_Bins"]), int(best["Secondary_Bins"]),
        float(best["Regime_Shrinkage"]), float(best["Regime_Smoothing"]),
        float(best["Regime_Clip_Multiplier"]),
    )
    return {
        "Table": table, "Best_Row": best, "Payload": payloads[best_key],
        "GBR_R2": gbr_r2, "Baseline_R2": baseline_r2,
    }


def _select_meta5_configuration(
    gbr_oof: np.ndarray,
    layer4_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Dict:
    gbr_oof = np.asarray(gbr_oof, dtype=float)
    layer4_oof = np.asarray(layer4_oof, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    gbr_r2 = safe_r2(y, gbr_oof)
    layer4_r2 = safe_r2(y, layer4_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))
    keys: List[Tuple[float, float, float]] = []
    for blend in STACK_META5_BLEND_GRID:
        clips = (0.0,) if blend == 0.0 else STACK_META5_DELTA_CLIP_GRID
        for delta_clip in clips:
            for scale in STACK_META5_SCALE_GRID:
                keys.append((float(blend), float(delta_clip), float(scale)))
    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Total_Gains": [], "Repeat_Incremental_Gains": [],
            "Fold_Total_Gains": [], "Fold_Incremental_Gains": [], "Intercepts": [],
        }
        for key in keys
    }
    for repeat in range(STACK_META_REPEATS):
        repeat_seed = int(seed) + 470001 + repeat * 49979687
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in keys}
        for train_index, valid_index in splits:
            residual_scale = _robust_residual_scale(y[train_index] - gbr_oof[train_index])
            train_weights = group_balanced_weights(groups[train_index])
            for key in keys:
                blend, delta_clip, scale = key
                if blend == 0.0:
                    train_blend = gbr_oof[train_index]
                    valid_blend = gbr_oof[valid_index]
                else:
                    bound = max(0.020, float(delta_clip) * residual_scale)
                    train_delta = np.clip(layer4_oof[train_index] - gbr_oof[train_index], -bound, bound)
                    valid_delta = np.clip(layer4_oof[valid_index] - gbr_oof[valid_index], -bound, bound)
                    train_blend = gbr_oof[train_index] + float(blend) * train_delta
                    valid_blend = gbr_oof[valid_index] + float(blend) * valid_delta
                intercept = np.clip(
                    _weighted_mean(y[train_index] - float(scale) * train_blend, train_weights),
                    -STACK_META5_INTERCEPT_BOUND, STACK_META5_INTERCEPT_BOUND,
                )
                prediction = float(intercept) + float(scale) * valid_blend
                repeat_predictions[key][valid_index] = prediction
                accum[key]["Intercepts"].append(float(intercept))
                accum[key]["Fold_Total_Gains"].append(
                    safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], gbr_oof[valid_index])
                )
                accum[key]["Fold_Incremental_Gains"].append(
                    safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], layer4_oof[valid_index])
                )
        for key in keys:
            prediction = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += prediction
            accum[key]["Prediction_Count"] += np.isfinite(prediction).astype(float)
            accum[key]["Repeat_Total_Gains"].append(safe_r2(y, prediction) - gbr_r2)
            accum[key]["Repeat_Incremental_Gains"].append(safe_r2(y, prediction) - layer4_r2)

    rows, payloads = [], {}
    for key in keys:
        blend, delta_clip, scale = key
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        total_repeat = np.asarray(item["Repeat_Total_Gains"], dtype=float)
        incremental_repeat = np.asarray(item["Repeat_Incremental_Gains"], dtype=float)
        incremental_fold = np.asarray(item["Fold_Incremental_Gains"], dtype=float)
        rows.append({
            "Final_Blend": blend, "Final_Delta_Clip": delta_clip, "Final_Scale5": scale,
            "Average_Intercept5": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
            "R2": metric["R2"], "RMSE": metric["RMSE"], "MAE": metric["MAE"],
            "Total_Gain": float(metric["R2"] - gbr_r2),
            "Incremental_Gain": float(metric["R2"] - layer4_r2),
            "Mean_Repeat_Total_Gain": float(np.mean(total_repeat)),
            "Mean_Repeat_Incremental_Gain": float(np.mean(incremental_repeat)),
            "SD_Repeat_Incremental_Gain": float(np.std(incremental_repeat, ddof=1)) if len(incremental_repeat) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(total_repeat > 0.0)),
            "Worst_Repeat_Gain": float(np.min(total_repeat)),
            "Positive_Folds": int(np.sum(np.asarray(item["Fold_Total_Gains"]) > 0.0)),
            "Total_Folds": int(len(item["Fold_Total_Gains"])),
            "Worst_Fold_Gain": float(np.min(np.asarray(item["Fold_Total_Gains"]))),
        })
        payloads[key] = {
            "Meta_OOF": meta_oof,
            "Average_Intercept": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
        }
    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.25 * table["Mean_Repeat_Total_Gain"]
        + 0.12 * table["Worst_Repeat_Gain"]
        + 0.10 * table["Mean_Repeat_Incremental_Gain"]
        - 0.08 * table["SD_Repeat_Incremental_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (float(best["Final_Blend"]), float(best["Final_Delta_Clip"]), float(best["Final_Scale5"]))
    return {
        "Table": table, "Best_Row": best, "Payload": payloads[best_key],
        "GBR_R2": gbr_r2, "Layer4_R2": layer4_r2,
    }

def _build_gate_features(
    prediction_matrix: np.ndarray,
    base_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    model_names: Sequence[str],
) -> np.ndarray:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    base_prediction = np.asarray(base_prediction, dtype=float)
    expert = np.asarray(expert_prediction, dtype=float)
    gbr_index = list(model_names).index(STACK_ANCHOR_NAME)
    gbr = prediction_matrix[:, gbr_index]
    core = _build_meta2_features(prediction_matrix, model_names)
    delta = expert - base_prediction
    extra = np.column_stack([
        base_prediction,
        expert,
        delta,
        np.abs(delta),
        base_prediction ** 2,
        expert ** 2,
        base_prediction * expert,
        expert - gbr,
        base_prediction - gbr,
        np.abs(expert - gbr),
    ])
    return np.column_stack([core, extra])


def _fit_gate_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    improvement: np.ndarray,
    groups: np.ndarray,
    c_value: float,
    seed: int,
) -> Dict:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    improvement = np.asarray(improvement, dtype=float)
    groups = np.asarray(groups)
    positive_rate = float(np.mean(labels)) if len(labels) else 0.0
    unique = np.unique(labels)
    if len(unique) < 2 or np.sum(labels == 1) < 6 or np.sum(labels == 0) < 6:
        return {"Family": "constant", "Probability": positive_rate, "Scaler": None, "Estimator": None}

    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    group_weight = group_balanced_weights(groups)
    positive_improvement = np.maximum(improvement, 0.0)
    reference = float(np.median(positive_improvement[positive_improvement > 0])) if np.any(positive_improvement > 0) else 1.0
    reference = max(reference, 1e-6)
    importance = 1.0 + np.clip(
        positive_improvement / reference,
        0.0,
        STACK_GATE_IMPROVEMENT_WEIGHT_CAP,
    )
    sample_weight = group_weight * importance
    estimator = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=int(seed),
    )
    estimator.fit(transformed, labels, sample_weight=sample_weight)
    return {"Family": "logistic", "Probability": positive_rate, "Scaler": scaler, "Estimator": estimator}


def _predict_gate_probability(fitted: Mapping, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if str(fitted.get("Family")) == "constant":
        return np.full(features.shape[0], float(fitted.get("Probability", 0.0)), dtype=float)
    transformed = fitted["Scaler"].transform(features)
    return np.asarray(fitted["Estimator"].predict_proba(transformed)[:, 1], dtype=float)


def _fit_selected_stage12(
    base_train: np.ndarray,
    base_valid: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    model_names: Sequence[str],
    selected_l2: float,
    stage1_shrinkage: float,
    corrector_name: str,
    corrector_config: Optional[Mapping],
    residual_strength: float,
    clip_multiplier: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    names = list(model_names)
    fit = fit_gbr_anchored_weights(
        base_train, y_train, groups_train, names,
        float(selected_l2), int(seed),
    )
    effective_weights, effective_bias = _effective_stage1_parameters(
        fit["Weights"], fit["Bias"], float(stage1_shrinkage), names
    )
    stage1_train = base_train @ effective_weights + effective_bias
    stage1_valid = base_valid @ effective_weights + effective_bias
    if corrector_name == "None" or corrector_config is None or residual_strength <= 0.0:
        return stage1_train, stage1_valid

    residual_train = np.asarray(y_train, dtype=float) - stage1_train
    residual_scale = _robust_residual_scale(residual_train)
    fitted = _fit_meta2_corrector(
        _build_meta2_features(base_train, names),
        residual_train,
        groups_train,
        corrector_config,
        int(seed) + 4099,
    )
    correction_train = _predict_meta2_corrector(fitted, _build_meta2_features(base_train, names))
    correction_valid = _predict_meta2_corrector(fitted, _build_meta2_features(base_valid, names))
    bound = max(0.025, float(clip_multiplier) * residual_scale)
    correction_train = np.clip(correction_train, -bound, bound)
    correction_valid = np.clip(correction_valid, -bound, bound)
    return (
        stage1_train + float(residual_strength) * correction_train,
        stage1_valid + float(residual_strength) * correction_valid,
    )


def _select_meta3_configuration(
    base_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    selected_l2: float,
    stage1_shrinkage: float,
    corrector_name: str,
    corrector_config: Optional[Mapping],
    residual_strength: float,
    residual_clip_multiplier: float,
    seed: int,
) -> Dict:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = np.asarray(base_oof[:, gbr_index], dtype=float)
    gbr_r2 = safe_r2(y, gbr_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))

    candidate_keys: List[Tuple[str, float, float, float, float, float]] = []
    for scale in STACK_FINAL_SCALE_GRID:
        candidate_keys.append(("None", 0.0, 1.0, 0.0, 0.0, float(scale)))
    for expert_name in STACK_GATE_EXPERTS:
        for c_value in STACK_GATE_C_GRID:
            for threshold in STACK_GATE_THRESHOLD_GRID:
                for strength in STACK_GATE_STRENGTH_GRID:
                    for clip_multiplier in STACK_GATE_CLIP_MULTIPLIER_GRID:
                        for scale in STACK_FINAL_SCALE_GRID:
                            candidate_keys.append((
                                str(expert_name), float(c_value), float(threshold),
                                float(strength), float(clip_multiplier), float(scale),
                            ))

    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Gains": [], "Fold_Gains": [], "Intercepts": [], "Gate_Rates": [],
        }
        for key in candidate_keys
    }

    for repeat in range(STACK_META_REPEATS):
        repeat_seed = int(seed) + 190001 + repeat * 15485863
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in candidate_keys}

        for fold_index, (train_index, valid_index) in enumerate(splits, start=1):
            stage2_train, stage2_valid = _fit_selected_stage12(
                base_oof[train_index], base_oof[valid_index], y[train_index], groups[train_index],
                names, selected_l2, stage1_shrinkage, corrector_name, corrector_config,
                residual_strength, residual_clip_multiplier,
                repeat_seed + fold_index * 1009,
            )
            baseline_valid = gbr_oof[valid_index]
            train_weights = group_balanced_weights(groups[train_index])
            residual_scale = _robust_residual_scale(y[train_index] - stage2_train)

            for scale in STACK_FINAL_SCALE_GRID:
                key = ("None", 0.0, 1.0, 0.0, 0.0, float(scale))
                intercept = np.clip(
                    _weighted_mean(y[train_index] - float(scale) * stage2_train, train_weights),
                    -STACK_FINAL_INTERCEPT_BOUND,
                    STACK_FINAL_INTERCEPT_BOUND,
                )
                prediction = float(intercept) + float(scale) * stage2_valid
                repeat_predictions[key][valid_index] = prediction
                accum[key]["Intercepts"].append(float(intercept))
                accum[key]["Gate_Rates"].append(0.0)
                accum[key]["Fold_Gains"].append(
                    safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                )

            for expert_position, expert_name in enumerate(STACK_GATE_EXPERTS, start=1):
                expert_train = _get_gate_expert_prediction(base_oof[train_index], expert_name, names)
                expert_valid = _get_gate_expert_prediction(base_oof[valid_index], expert_name, names)
                base_sq_error = (y[train_index] - stage2_train) ** 2
                expert_sq_error = (y[train_index] - expert_train) ** 2
                improvement = base_sq_error - expert_sq_error
                labels = (improvement > 0.0).astype(int)
                train_features = _build_gate_features(
                    base_oof[train_index], stage2_train, expert_train, names
                )
                valid_features = _build_gate_features(
                    base_oof[valid_index], stage2_valid, expert_valid, names
                )

                for c_position, c_value in enumerate(STACK_GATE_C_GRID, start=1):
                    gate = _fit_gate_classifier(
                        train_features, labels, improvement, groups[train_index],
                        float(c_value), repeat_seed + fold_index * 100003 + expert_position * 1009 + c_position * 97,
                    )
                    probability_train = _predict_gate_probability(gate, train_features)
                    probability_valid = _predict_gate_probability(gate, valid_features)

                    for threshold in STACK_GATE_THRESHOLD_GRID:
                        denominator = max(1e-6, 1.0 - float(threshold))
                        activation_train = np.clip((probability_train - float(threshold)) / denominator, 0.0, 1.0)
                        activation_valid = np.clip((probability_valid - float(threshold)) / denominator, 0.0, 1.0)
                        for clip_multiplier in STACK_GATE_CLIP_MULTIPLIER_GRID:
                            bound = max(0.025, float(clip_multiplier) * residual_scale)
                            delta_train = np.clip(expert_train - stage2_train, -bound, bound)
                            delta_valid = np.clip(expert_valid - stage2_valid, -bound, bound)
                            for strength in STACK_GATE_STRENGTH_GRID:
                                gated_train = stage2_train + float(strength) * activation_train * delta_train
                                gated_valid = stage2_valid + float(strength) * activation_valid * delta_valid
                                gate_rate = float(np.mean(activation_valid))
                                for scale in STACK_FINAL_SCALE_GRID:
                                    key = (
                                        str(expert_name), float(c_value), float(threshold),
                                        float(strength), float(clip_multiplier), float(scale),
                                    )
                                    intercept = np.clip(
                                        _weighted_mean(y[train_index] - float(scale) * gated_train, train_weights),
                                        -STACK_FINAL_INTERCEPT_BOUND,
                                        STACK_FINAL_INTERCEPT_BOUND,
                                    )
                                    prediction = float(intercept) + float(scale) * gated_valid
                                    repeat_predictions[key][valid_index] = prediction
                                    accum[key]["Intercepts"].append(float(intercept))
                                    accum[key]["Gate_Rates"].append(gate_rate)
                                    accum[key]["Fold_Gains"].append(
                                        safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                                    )

        for key in candidate_keys:
            repeat_prediction = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += repeat_prediction
            accum[key]["Prediction_Count"] += np.isfinite(repeat_prediction).astype(float)
            accum[key]["Repeat_Gains"].append(safe_r2(y, repeat_prediction) - gbr_r2)

    rows: List[Dict] = []
    payloads: Dict[Tuple[str, float, float, float, float, float], Dict] = {}
    for key in candidate_keys:
        expert_name, c_value, threshold, strength, clip_multiplier, scale = key
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        repeat_array = np.asarray(item["Repeat_Gains"], dtype=float)
        fold_array = np.asarray(item["Fold_Gains"], dtype=float)
        gain = float(metric["R2"] - gbr_r2)
        rows.append({
            "Gate_Expert": expert_name,
            "Gate_C": float(c_value),
            "Gate_Threshold": float(threshold),
            "Gate_Strength": float(strength),
            "Gate_Clip_Multiplier": float(clip_multiplier),
            "Final_Scale": float(scale),
            "Average_Intercept": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
            "Average_Gate_Rate": float(np.mean(item["Gate_Rates"])) if item["Gate_Rates"] else 0.0,
            "R2": float(metric["R2"]), "RMSE": float(metric["RMSE"]), "MAE": float(metric["MAE"]),
            "Meta_Gain": gain,
            "Mean_Repeat_Gain": float(np.mean(repeat_array)),
            "SD_Repeat_Gain": float(np.std(repeat_array, ddof=1)) if len(repeat_array) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(repeat_array > 0.0)),
            "Worst_Repeat_Gain": float(np.min(repeat_array)),
            "Positive_Folds": int(np.sum(fold_array > 0.0)),
            "Total_Folds": int(len(fold_array)),
            "Worst_Fold_Gain": float(np.min(fold_array)),
        })
        payloads[key] = {
            "Meta_OOF": meta_oof,
            "Average_Intercept": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
        }

    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"]
        + 0.24 * table["Mean_Repeat_Gain"]
        + 0.10 * table["Worst_Repeat_Gain"]
        - 0.07 * table["SD_Repeat_Gain"]
        + 0.00015 * table["Positive_Repeats"]
    )
    table = table.sort_values(
        ["Robust_Select_Score", "R2", "RMSE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (
        str(best["Gate_Expert"]), float(best["Gate_C"]), float(best["Gate_Threshold"]),
        float(best["Gate_Strength"]), float(best["Gate_Clip_Multiplier"]), float(best["Final_Scale"]),
    )
    return {"Table": table, "Best_Row": best, "Payload": payloads[best_key], "GBR_R2": gbr_r2}


def _fit_meta3_stacking(
    base_oof: np.ndarray,
    base_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    """前三层：GBR锚定线性层、残差校准层、动态专家门控与尺度校准层。"""
    base_oof = np.asarray(base_oof, dtype=float)
    base_test = np.asarray(base_test, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    if base_oof.shape[1] != len(names):
        raise ValueError("base_oof列数与model_names不一致。")
    if STACK_ANCHOR_NAME not in names:
        raise ValueError("Stacking必须包含内部GBR750锚点。")
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_test = base_test[:, gbr_index]

    stage1 = _select_stage1_configuration(base_oof, y, groups, names, seed)
    stage1_best = stage1["Best_Row"]
    selected_l2 = float(stage1_best["L2"])
    stage1_shrinkage = float(stage1_best["Stage1_Shrinkage"])

    meta2 = _select_meta2_configuration(
        base_oof, y, groups, names, selected_l2, stage1_shrinkage, seed + 500003
    )
    meta2_best = meta2["Best_Row"]
    corrector_name = str(meta2_best["Corrector"])
    selected_residual_strength = float(meta2_best["Residual_Strength"])
    selected_clip_multiplier = float(meta2_best["Clip_Multiplier"])
    meta2_gain = float(meta2_best["Meta_Gain"])

    meta2_min_positive_repeats = int(np.ceil(STACK_META_REPEATS * STACK_META2_MIN_POSITIVE_REPEAT_RATIO))
    meta2_min_positive_folds = int(np.ceil(float(meta2_best["Total_Folds"]) * STACK_META2_MIN_POSITIVE_FOLD_RATIO))
    meta2_stable = (
        corrector_name != "None"
        and meta2_gain >= STACK_META2_REQUIRED_GAIN
        and int(meta2_best["Positive_Repeats"]) >= meta2_min_positive_repeats
        and int(meta2_best["Positive_Folds"]) >= meta2_min_positive_folds
        and float(meta2_best["Worst_Repeat_Gain"]) >= -STACK_META2_MAX_WORST_REPEAT_LOSS
        and float(meta2_best["Worst_Fold_Gain"]) >= -STACK_META2_MAX_WORST_FOLD_LOSS
    )
    meta2_soft_min_positive = int(np.ceil(STACK_META_REPEATS * STACK_META2_SOFT_MIN_POSITIVE_REPEAT_RATIO))
    meta2_soft = (
        corrector_name != "None"
        and meta2_gain >= STACK_META2_SOFT_MIN_GAIN
        and int(meta2_best["Positive_Repeats"]) >= meta2_soft_min_positive
        and float(meta2_best["Worst_Repeat_Gain"]) >= -STACK_META2_SOFT_MAX_WORST_REPEAT_LOSS
    )
    if meta2_stable:
        used_corrector_name = corrector_name
        used_residual_strength = selected_residual_strength
    elif meta2_soft:
        used_corrector_name = corrector_name
        used_residual_strength = selected_residual_strength * STACK_META2_SOFT_FRACTION
    else:
        used_corrector_name = "None"
        used_residual_strength = 0.0
    used_corrector_config = meta2["Config_Map"].get(used_corrector_name)

    meta3 = _select_meta3_configuration(
        base_oof, y, groups, names,
        selected_l2, stage1_shrinkage,
        used_corrector_name, used_corrector_config,
        used_residual_strength, selected_clip_multiplier,
        seed + 900001,
    )
    best = meta3["Best_Row"]
    meta3_gain = float(best["Meta_Gain"])
    min_positive_repeats = int(np.ceil(STACK_META_REPEATS * STACK_META3_MIN_POSITIVE_REPEAT_RATIO))
    min_positive_folds = int(np.ceil(float(best["Total_Folds"]) * STACK_META3_MIN_POSITIVE_FOLD_RATIO))
    stable = (
        meta3_gain >= STACK_META3_REQUIRED_GAIN
        and int(best["Positive_Repeats"]) >= min_positive_repeats
        and int(best["Positive_Folds"]) >= min_positive_folds
        and float(best["Worst_Repeat_Gain"]) >= -STACK_META3_MAX_WORST_REPEAT_LOSS
        and float(best["Worst_Fold_Gain"]) >= -STACK_META3_MAX_WORST_FOLD_LOSS
    )
    soft_min_positive = int(np.ceil(STACK_META_REPEATS * STACK_META3_SOFT_MIN_POSITIVE_REPEAT_RATIO))
    soft_eligible = (
        meta3_gain >= STACK_META3_SOFT_MIN_GAIN
        and int(best["Positive_Repeats"]) >= soft_min_positive
        and float(best["Worst_Repeat_Gain"]) >= -STACK_META3_SOFT_MAX_WORST_REPEAT_LOSS
    )

    full_fit = fit_gbr_anchored_weights(base_oof, y, groups, names, selected_l2, seed + 99991)
    average_weights = np.asarray(stage1["Payload"]["Average_Weights"], dtype=float)
    blended_weights = (
        (1.0 - STACK_FINAL_META_WEIGHT) * np.asarray(full_fit["Weights"], dtype=float)
        + STACK_FINAL_META_WEIGHT * average_weights
    )
    blended_weights = _project_anchor_weights(blended_weights, names)
    blended_bias = (
        (1.0 - STACK_FINAL_META_WEIGHT) * float(full_fit["Bias"])
        + STACK_FINAL_META_WEIGHT * float(stage1["Payload"]["Average_Bias"])
    )
    effective_weights, effective_bias = _effective_stage1_parameters(
        blended_weights, blended_bias, stage1_shrinkage, names
    )
    stage1_train = base_oof @ effective_weights + effective_bias
    stage1_test = base_test @ effective_weights + effective_bias

    fitted_corrector = None  # deployment-only: fitted second-layer calibrator
    correction_bound = 0.0
    if used_corrector_name != "None" and used_corrector_config is not None and used_residual_strength > 0.0:
        residual_train = y - stage1_train
        residual_scale = _robust_residual_scale(residual_train)
        fitted_corrector = _fit_meta2_corrector(
            _build_meta2_features(base_oof, names), residual_train, groups,
            used_corrector_config, seed + 700001,
        )
        train_correction = _predict_meta2_corrector(fitted_corrector, _build_meta2_features(base_oof, names))
        test_correction = _predict_meta2_corrector(fitted_corrector, _build_meta2_features(base_test, names))
        correction_bound = max(0.025, selected_clip_multiplier * residual_scale)
        train_correction = np.clip(train_correction, -correction_bound, correction_bound)
        test_correction = np.clip(test_correction, -correction_bound, correction_bound)
        stage2_train = stage1_train + used_residual_strength * train_correction
        stage2_test = stage1_test + used_residual_strength * test_correction
        stage2_crossfit = np.asarray(meta2["Payload"]["Meta_OOF"], dtype=float)
        if meta2_soft and not meta2_stable:
            stage2_crossfit = gbr_oof + STACK_META2_SOFT_FRACTION * (stage2_crossfit - gbr_oof)
    else:
        stage2_train = stage1_train
        stage2_test = stage1_test
        stage2_crossfit = np.asarray(stage1["Payload"]["Meta_OOF"], dtype=float)

    if not stable and not soft_eligible:
        fallback_weights = np.zeros(len(names), dtype=float)
        fallback_weights[gbr_index] = 1.0
        return {
            "Safety_Fallback": True,
            "Deploy_Stage3_Fallback": True,
            "Soft_Shrinkage": False,
            "Fallback_Model": STACK_ANCHOR_NAME,
            "Selected_L2": selected_l2,
            "Stage1_Shrinkage": stage1_shrinkage,
            "Weights": fallback_weights,
            "Bias": 0.0,
            "Selected_Models": [STACK_ANCHOR_NAME],
            "Residual_Corrector": used_corrector_name,
            "Residual_Strength": float(used_residual_strength),
            "Residual_Clip_Multiplier": float(selected_clip_multiplier),
            "Gate_Expert": "None",
            "Gate_C": 0.0,
            "Gate_Threshold": 1.0,
            "Gate_Strength": 0.0,
            "Gate_Clip_Multiplier": 0.0,
            "Final_Scale": 1.0,
            "Final_Intercept": 0.0,
            "Train_OOF_Prediction": gbr_oof.copy(),
            "Test_Prediction": gbr_test.copy(),
            "Candidate_Table": meta2["Table"],
            "Stage1_Candidate_Table": stage1["Table"],
            "Meta3_Candidate_Table": meta3["Table"],
            "Meta_Gain": meta3_gain,
            "Meta3_Gain": meta3_gain,
            "Meta2_Gain": meta2_gain,
            "Positive_Repeats": int(best["Positive_Repeats"]),
            "Worst_Repeat_Gain": float(best["Worst_Repeat_Gain"]),
            "GBR750_Weight": 1.0,
            "Meta2_Enabled": used_corrector_name != "None",
            "Meta3_Enabled": False,
        }

    expert_name = str(best["Gate_Expert"])
    gate_c = float(best["Gate_C"])
    gate_threshold = float(best["Gate_Threshold"])
    gate_strength = float(best["Gate_Strength"])
    gate_clip_multiplier = float(best["Gate_Clip_Multiplier"])
    final_scale = float(best["Final_Scale"])
    average_intercept = float(meta3["Payload"]["Average_Intercept"])

    gate = None  # deployment-only: fitted third-layer expert gate
    gate_bound = 0.0
    if expert_name == "None":
        gated_test = stage2_test.copy()
        gated_oof_for_intercept = stage2_crossfit.copy()
    else:
        expert_oof = _get_gate_expert_prediction(base_oof, expert_name, names)
        expert_test = _get_gate_expert_prediction(base_test, expert_name, names)
        improvement = (y - stage2_crossfit) ** 2 - (y - expert_oof) ** 2
        labels = (improvement > 0.0).astype(int)
        train_features = _build_gate_features(base_oof, stage2_crossfit, expert_oof, names)
        test_features = _build_gate_features(base_test, stage2_test, expert_test, names)
        gate = _fit_gate_classifier(
            train_features, labels, improvement, groups, gate_c, seed + 1200001
        )
        probability_train = _predict_gate_probability(gate, train_features)
        probability_test = _predict_gate_probability(gate, test_features)
        denominator = max(1e-6, 1.0 - gate_threshold)
        activation_train = np.clip((probability_train - gate_threshold) / denominator, 0.0, 1.0)
        activation_test = np.clip((probability_test - gate_threshold) / denominator, 0.0, 1.0)
        residual_scale = _robust_residual_scale(y - stage2_crossfit)
        gate_bound = max(0.025, gate_clip_multiplier * residual_scale)
        train_delta = np.clip(expert_oof - stage2_crossfit, -gate_bound, gate_bound)
        test_delta = np.clip(expert_test - stage2_test, -gate_bound, gate_bound)
        gated_oof_for_intercept = stage2_crossfit + gate_strength * activation_train * train_delta
        gated_test = stage2_test + gate_strength * activation_test * test_delta

    full_intercept = np.clip(
        _weighted_mean(y - final_scale * gated_oof_for_intercept, group_balanced_weights(groups)),
        -STACK_FINAL_INTERCEPT_BOUND,
        STACK_FINAL_INTERCEPT_BOUND,
    )
    final_intercept = (
        (1.0 - STACK_META3_FINAL_INTERCEPT_BLEND) * average_intercept
        + STACK_META3_FINAL_INTERCEPT_BLEND * float(full_intercept)
    )
    test_prediction = final_intercept + final_scale * gated_test
    meta_prediction = np.asarray(meta3["Payload"]["Meta_OOF"], dtype=float)

    if not stable:
        test_prediction = gbr_test + STACK_META3_SOFT_FRACTION * (test_prediction - gbr_test)
        meta_prediction = gbr_oof + STACK_META3_SOFT_FRACTION * (meta_prediction - gbr_oof)

    selected_models = [
        name for name, weight in zip(names, effective_weights) if abs(float(weight)) > 1e-6
    ]
    if used_corrector_name != "None":
        selected_models.append(f"Meta2:{used_corrector_name}")
    if expert_name != "None":
        selected_models.append(f"Gate:{expert_name}")
    if abs(final_scale - 1.0) > 1e-8 or abs(final_intercept) > 1e-8:
        selected_models.append("FinalCalibration")

    return {
        "Safety_Fallback": False,
        "Deploy_Stage3_Fallback": False,
        "Deploy_Corrector": fitted_corrector,
        "Deploy_Corrector_Bound": float(correction_bound),
        "Deploy_Gate": gate,
        "Deploy_Gate_Bound": float(gate_bound),
        "Deploy_Stage3_Soft_Fraction": float(1.0 if stable else STACK_META3_SOFT_FRACTION),
        "Soft_Shrinkage": bool(not stable),
        "Fallback_Model": None,
        "Selected_L2": selected_l2,
        "Stage1_Shrinkage": stage1_shrinkage,
        "Weights": effective_weights,
        "Bias": effective_bias,
        "Selected_Models": selected_models,
        "Residual_Corrector": used_corrector_name,
        "Residual_Strength": float(used_residual_strength),
        "Residual_Clip_Multiplier": float(selected_clip_multiplier),
        "Gate_Expert": expert_name,
        "Gate_C": gate_c,
        "Gate_Threshold": gate_threshold,
        "Gate_Strength": gate_strength,
        "Gate_Clip_Multiplier": gate_clip_multiplier,
        "Final_Scale": final_scale,
        "Final_Intercept": float(final_intercept),
        "Train_OOF_Prediction": meta_prediction,
        "Test_Prediction": test_prediction,
        "Candidate_Table": meta2["Table"],
        "Stage1_Candidate_Table": stage1["Table"],
        "Meta3_Candidate_Table": meta3["Table"],
        "Meta_Gain": meta3_gain,
        "Meta3_Gain": meta3_gain,
        "Meta2_Gain": meta2_gain,
        "Positive_Repeats": int(best["Positive_Repeats"]),
        "Worst_Repeat_Gain": float(best["Worst_Repeat_Gain"]),
        "GBR750_Weight": float(effective_weights[gbr_index]),
        "Meta2_Enabled": used_corrector_name != "None",
        "Meta3_Enabled": True,
    }



def _evaluate_champion_guard(
    base_oof: np.ndarray,
    y: np.ndarray,
    model_names: Sequence[str],
    stage3: Mapping,
    final_gain: float,
    final_positive_repeats: int,
    final_worst_repeat_gain: float,
) -> Dict:
    """只基于外层训练侧严格OOF证据评估GBR750增强型Stacking风险。"""
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    base_r2 = {
        name: float(safe_r2(y, np.asarray(base_oof[:, index], dtype=float)))
        for index, name in enumerate(names)
    }
    best_base_model = max(base_r2, key=base_r2.get)
    best_base_r2 = float(base_r2[best_base_model])
    gbr_r2 = float(base_r2[STACK_ANCHOR_NAME])
    gbr_gap_to_best = float(best_base_r2 - gbr_r2)

    default_weights = np.eye(1, len(names), gbr_index).ravel()
    weights = np.asarray(stage3.get("Weights", default_weights), dtype=float)
    partner_weights = np.delete(weights, gbr_index)
    positive_partner_weight = float(np.sum(np.clip(partner_weights, 0.0, None)))
    absolute_partner_weight = float(np.sum(np.abs(partner_weights)))
    gbr_weight = float(weights[gbr_index])
    meta2_enabled = bool(stage3.get("Meta2_Enabled", False))
    gate_expert = str(stage3.get("Gate_Expert", "None"))
    gate_enabled = gate_expert != "None"
    intercept = float(stage3.get("Final_Intercept", 0.0))

    strong_evidence = bool(
        final_gain >= STACK_GUARD_STRONG_GAIN
        and int(final_positive_repeats) >= int(STACK_META_REPEATS)
        and float(final_worst_repeat_gain) >= STACK_GUARD_STRONG_WORST_REPEAT_GAIN
    )

    reasons: List[str] = []
    if (
        STACK_GUARD_REQUIRE_GBR_TRAIN_CHAMPION
        and best_base_model != STACK_ANCHOR_NAME
        and gbr_gap_to_best > STACK_GUARD_GBR_CHAMPION_TOL
        and not strong_evidence
    ):
        reasons.append(
            f"GBR750并非训练OOF冠军：{best_base_model}高出{gbr_gap_to_best:.6f}"
        )

    if gbr_weight < STACK_GUARD_HARD_MIN_GBR_WEIGHT:
        reasons.append(
            f"GBR750权重低于硬下限：{gbr_weight:.4f} < "
            f"{STACK_GUARD_HARD_MIN_GBR_WEIGHT:.4f}"
        )
    elif (
        gbr_weight < STACK_GUARD_STANDARD_MIN_GBR_WEIGHT
        and not strong_evidence
    ):
        reasons.append(
            f"GBR750权重低于标准下限且强证据不足：{gbr_weight:.4f} < "
            f"{STACK_GUARD_STANDARD_MIN_GBR_WEIGHT:.4f}"
        )

    if positive_partner_weight > STACK_GUARD_MAX_POSITIVE_PARTNER_WEIGHT:
        reasons.append(
            f"非锚点正权重过大：{positive_partner_weight:.4f} > "
            f"{STACK_GUARD_MAX_POSITIVE_PARTNER_WEIGHT:.4f}"
        )
    if absolute_partner_weight > STACK_GUARD_MAX_ABS_PARTNER_WEIGHT:
        reasons.append(
            f"非锚点绝对权重过大：{absolute_partner_weight:.4f} > "
            f"{STACK_GUARD_MAX_ABS_PARTNER_WEIGHT:.4f}"
        )

    if (
        meta2_enabled and gate_enabled
        and gbr_weight < STACK_GUARD_COMPLEX_MIN_GBR_WEIGHT
        and not strong_evidence
    ):
        reasons.append("残差校准与门控同时启用，但GBR750锚点或强证据不足")


    # 宽融合若最差重复增益接近零，说明对文献划分非常敏感。
    if (
        positive_partner_weight > STACK_GUARD_WIDE_BLEND_PARTNER_WEIGHT
        and final_worst_repeat_gain < STACK_GUARD_WIDE_BLEND_MIN_WORST_REPEAT_GAIN
    ):
        reasons.append(
            f"宽融合的最差重复增益过弱：{final_worst_repeat_gain:.6f} < "
            f"{STACK_GUARD_WIDE_BLEND_MIN_WORST_REPEAT_GAIN:.6f}"
        )

    if (
        abs(intercept) >= STACK_GUARD_INTERCEPT_EDGE
        and gbr_weight < STACK_GUARD_COMPLEX_MIN_GBR_WEIGHT
        and final_gain < STACK_GUARD_MIN_GAIN_AT_INTERCEPT_EDGE
        and not strong_evidence
    ):
        reasons.append(
            f"最终截距触边且Meta增益不足：intercept={intercept:.4f}, "
            f"gain={final_gain:.6f}"
        )

    return {
        "Triggered": bool(STACK_CHAMPION_GUARD_ENABLED and len(reasons) > 0),
        "Reasons": reasons,
        "Strong_Evidence_Override": strong_evidence,
        "Base_R2": base_r2,
        "Best_Base_Model": best_base_model,
        "Best_Base_R2": best_base_r2,
        "GBR_Base_R2": gbr_r2,
        "GBR_Gap_To_Best": gbr_gap_to_best,
        "GBR750_Weight_PreGuard": gbr_weight,
        "Positive_Partner_Weight": positive_partner_weight,
        "Absolute_Partner_Weight": absolute_partner_weight,
        "Meta2_Enabled_PreGuard": meta2_enabled,
        "Gate_Enabled_PreGuard": gate_enabled,
        "Final_Positive_Repeats": int(final_positive_repeats),
        "Final_Worst_Repeat_Gain": float(final_worst_repeat_gain),
    }


def fit_safe_stacking(
    base_oof: np.ndarray,
    base_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    """五层Stacking：前三层模型 + 分区残差层 + 稳定收缩融合层。"""
    base_oof = np.asarray(base_oof, dtype=float)
    base_test = np.asarray(base_test, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_test = base_test[:, gbr_index]

    stage3 = _fit_meta3_stacking(base_oof, base_test, y, groups, names, int(seed))
    stage3_oof = np.asarray(stage3["Train_OOF_Prediction"], dtype=float)
    stage3_test = np.asarray(stage3["Test_Prediction"], dtype=float)

    meta4 = _select_meta4_configuration(
        base_oof, stage3_oof, y, groups, names, int(seed) + 1500001
    )
    best4 = meta4["Best_Row"]
    mode4 = str(best4["Regime_Mode"])
    meta4_stable = (
        mode4 != "None"
        and float(best4["Incremental_Gain"]) >= STACK_META4_REQUIRED_INCREMENTAL_GAIN
        and float(best4["Total_Gain"]) >= STACK_META4_REQUIRED_TOTAL_GAIN
        and int(best4["Positive_Repeats"]) >= int(np.ceil(STACK_META_REPEATS * STACK_META4_MIN_POSITIVE_REPEAT_RATIO))
        and int(best4["Positive_Folds"]) >= int(np.ceil(float(best4["Total_Folds"]) * STACK_META4_MIN_POSITIVE_FOLD_RATIO))
        and float(best4["Worst_Repeat_Gain"]) >= -STACK_META4_MAX_WORST_REPEAT_LOSS
        and float(best4["Worst_Fold_Gain"]) >= -STACK_META4_MAX_WORST_FOLD_LOSS
    )
    meta4_soft = (
        mode4 != "None"
        and float(best4["Incremental_Gain"]) >= STACK_META4_SOFT_MIN_INCREMENTAL_GAIN
        and float(best4["Total_Gain"]) > 0.0
    )

    fitted4 = None  # deployment-only: fitted fourth-layer correction
    soft_fraction4 = 0.0
    if meta4_stable or meta4_soft:
        fitted4 = _fit_regime_calibrator(
            base_oof, stage3_oof, y, groups, names,
            mode4, int(best4["Level_Bins"]), int(best4["Secondary_Bins"]),
            float(best4["Regime_Shrinkage"]), float(best4["Regime_Smoothing"]),
            float(best4["Regime_Clip_Multiplier"]),
        )
        correction_train4 = _predict_regime_correction(fitted4, base_oof, stage3_oof)
        correction_test4 = _predict_regime_correction(fitted4, base_test, stage3_test)
        soft_fraction4 = 1.0 if meta4_stable else STACK_META4_SOFT_FRACTION
        layer4_test = stage3_test + soft_fraction4 * correction_test4
        layer4_oof = np.asarray(meta4["Payload"]["Meta_OOF"], dtype=float)
        if not meta4_stable:
            layer4_oof = stage3_oof + STACK_META4_SOFT_FRACTION * (layer4_oof - stage3_oof)
    else:
        mode4 = "None"
        layer4_oof = stage3_oof.copy()
        layer4_test = stage3_test.copy()

    meta5 = _select_meta5_configuration(
        gbr_oof, layer4_oof, y, groups, int(seed) + 2500001
    )
    best5 = meta5["Best_Row"]
    meta5_stable = (
        float(best5["Total_Gain"]) >= STACK_META5_REQUIRED_TOTAL_GAIN
        and float(best5["Incremental_Gain"]) >= STACK_META5_REQUIRED_INCREMENTAL_GAIN
        and int(best5["Positive_Repeats"]) >= int(np.ceil(STACK_META_REPEATS * STACK_META5_MIN_POSITIVE_REPEAT_RATIO))
        and int(best5["Positive_Folds"]) >= int(np.ceil(float(best5["Total_Folds"]) * STACK_META5_MIN_POSITIVE_FOLD_RATIO))
        and float(best5["Worst_Repeat_Gain"]) >= -STACK_META5_MAX_WORST_REPEAT_LOSS
        and float(best5["Worst_Fold_Gain"]) >= -STACK_META5_MAX_WORST_FOLD_LOSS
    )
    meta5_soft = (
        float(best5["Total_Gain"]) >= STACK_META5_SOFT_MIN_TOTAL_GAIN
        and float(best5["Incremental_Gain"]) > 0.0
    )

    blend5 = float(best5["Final_Blend"])
    clip5 = float(best5["Final_Delta_Clip"])
    scale5 = float(best5["Final_Scale5"])
    residual_scale = _robust_residual_scale(y - gbr_oof)
    bound5 = 0.0
    if blend5 == 0.0:
        train_blend5 = gbr_oof.copy()
        test_blend5 = gbr_test.copy()
    else:
        bound5 = max(0.020, clip5 * residual_scale)
        train_delta5 = np.clip(layer4_oof - gbr_oof, -bound5, bound5)
        test_delta5 = np.clip(layer4_test - gbr_test, -bound5, bound5)
        train_blend5 = gbr_oof + blend5 * train_delta5
        test_blend5 = gbr_test + blend5 * test_delta5
    full_intercept5 = np.clip(
        _weighted_mean(y - scale5 * train_blend5, group_balanced_weights(groups)),
        -STACK_META5_INTERCEPT_BOUND, STACK_META5_INTERCEPT_BOUND,
    )
    average_intercept5 = float(meta5["Payload"]["Average_Intercept"])
    intercept5 = (
        (1.0 - STACK_META5_FINAL_INTERCEPT_BLEND) * average_intercept5
        + STACK_META5_FINAL_INTERCEPT_BLEND * float(full_intercept5)
    )
    final_test = intercept5 + scale5 * test_blend5
    final_oof = np.asarray(meta5["Payload"]["Meta_OOF"], dtype=float)

    if not meta5_stable and meta5_soft:
        final_test = gbr_test + STACK_META5_SOFT_FRACTION * (final_test - gbr_test)
        final_oof = gbr_oof + STACK_META5_SOFT_FRACTION * (final_oof - gbr_oof)
    elif not meta5_stable and not meta5_soft:
        # 第五层无可靠增益时，保留通过严格Meta验证的低层结果；若低层也未启用则等于GBR。
        final_test = layer4_test
        final_oof = layer4_oof
        blend5, clip5, scale5, intercept5 = 1.0, 0.0, 1.0, 0.0

    selected_models = list(stage3.get("Selected_Models", []))
    if mode4 != "None":
        selected_models.append(f"Meta4:{mode4}")
    if meta5_stable or meta5_soft:
        selected_models.append("Meta5:StabilityBlend")
    selected_models = list(dict.fromkeys(selected_models))

    pre_guard_gain = float(safe_r2(y, final_oof) - safe_r2(y, gbr_oof))
    guard = _evaluate_champion_guard(
        base_oof, y, names, stage3, pre_guard_gain,
        int(best5["Positive_Repeats"]),
        float(best5["Worst_Repeat_Gain"]),
    )
    guard_triggered = bool(guard["Triggered"])

    pre_guard_selected_models = list(selected_models)
    pre_guard_weights = np.asarray(stage3.get("Weights"), dtype=float).copy()
    pre_guard_bias = float(stage3.get("Bias", 0.0))
    if guard_triggered:
        final_oof = gbr_oof.copy()
        final_test = gbr_test.copy()
        selected_models = [STACK_ANCHOR_NAME]
        final_weights = np.zeros(len(names), dtype=float)
        final_weights[gbr_index] = 1.0
        final_bias = 0.0
        final_gain = 0.0
    else:
        final_weights = pre_guard_weights
        final_bias = pre_guard_bias
        final_gain = pre_guard_gain

    is_final_gbr = bool(np.allclose(final_oof, gbr_oof, atol=1e-12))
    result = dict(stage3)
    result.update({
        "Safety_Fallback": bool(is_final_gbr),
        "Soft_Shrinkage": bool((not guard_triggered) and (stage3.get("Soft_Shrinkage", False) or (meta4_soft and not meta4_stable) or (meta5_soft and not meta5_stable))),
        "Fallback_Model": STACK_ANCHOR_NAME if is_final_gbr else None,
        "Selected_Models": selected_models,
        "Weights": final_weights,
        "Bias": final_bias,
        "GBR750_Weight": float(final_weights[gbr_index]),
        "Train_OOF_Prediction": final_oof,
        "Test_Prediction": final_test,
        "Champion_Guard_Triggered": guard_triggered,
        "Champion_Guard_Reasons": "；".join(guard["Reasons"]) if guard["Reasons"] else "None",
        "Champion_Guard_Strong_Evidence_Override": guard["Strong_Evidence_Override"],
        "Best_Base_Model_TrainOOF": guard["Best_Base_Model"],
        "GBR_Base_R2_TrainOOF": guard["GBR_Base_R2"],
        "GBR_Gap_To_Best_TrainOOF": guard["GBR_Gap_To_Best"],
        "Positive_Partner_Weight_PreGuard": guard["Positive_Partner_Weight"],
        "Absolute_Partner_Weight_PreGuard": guard["Absolute_Partner_Weight"],
        "PreGuard_Meta_Gain": pre_guard_gain,
        "PreGuard_Selected_Models": pre_guard_selected_models,
        "PreGuard_Weights": pre_guard_weights,
        "PreGuard_Bias": pre_guard_bias,
        "Deploy_Regime_Model": fitted4,
        "Deploy_Regime_Fraction": float(soft_fraction4),
        "Deploy_Meta5_Status": "stable" if meta5_stable else ("soft" if meta5_soft else "disabled"),
        "Deploy_Meta5_Bound": float(bound5),
        "Meta4_Mode": mode4,
        "Meta4_Level_Bins": int(best4["Level_Bins"]),
        "Meta4_Secondary_Bins": int(best4["Secondary_Bins"]),
        "Meta4_Shrinkage": float(best4["Regime_Shrinkage"]),
        "Meta4_Smoothing": float(best4["Regime_Smoothing"]),
        "Meta4_Clip_Multiplier": float(best4["Regime_Clip_Multiplier"]),
        "Meta4_Gain": float(best4["Total_Gain"]),
        "Meta4_Incremental_Gain": float(best4["Incremental_Gain"]),
        "Meta4_Candidate_Table": meta4["Table"],
        "Meta5_Blend": float(blend5),
        "Meta5_Delta_Clip": float(clip5),
        "Meta5_Scale": float(scale5),
        "Meta5_Intercept": float(intercept5),
        "Meta5_Gain": float(best5["Total_Gain"]),
        "Meta5_Incremental_Gain": float(best5["Incremental_Gain"]),
        "Meta5_Candidate_Table": meta5["Table"],
        "Meta_Gain": final_gain,
        "Positive_Repeats": int(best5["Positive_Repeats"]),
        "Worst_Repeat_Gain": float(best5["Worst_Repeat_Gain"]),
        "Meta4_Enabled": mode4 != "None",
        "Meta5_Enabled": bool(meta5_stable or meta5_soft),
    })
    return result


# =============================================================================
# 8. 外层完整模型训练
# =============================================================================

def fit_predict_model(
    builder: Callable[[int, Mapping], object],
    params: Mapping,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = builder(seed, params)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def fit_predict_gbr_bagged(
    params: Mapping,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    predictions = []
    for offset in GBR_BAG_SEEDS:
        model = make_gbr(seed + int(offset), params)
        model.fit(X_train, y_train)
        predictions.append(np.asarray(model.predict(X_test), dtype=float))
    return np.mean(np.column_stack(predictions), axis=1)



# =============================================================================
# 9. 数据行拖累诊断
# =============================================================================

def robust_location_scale(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return median, scale


def percentile_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True, na_option="bottom").fillna(0.0).to_numpy(dtype=float)


def compute_numeric_outlier_signals(
    X: pd.DataFrame,
    numeric_columns: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X)
    maximum_z = np.zeros(n, dtype=float)
    extreme_count = np.zeros(n, dtype=int)
    for column in numeric_columns:
        values = to_numeric_safe(X[column]).to_numpy(dtype=float)
        center, scale = robust_location_scale(values)
        z = np.abs((values - center) / scale)
        z[~np.isfinite(z)] = 0.0
        maximum_z = np.maximum(maximum_z, z)
        extreme_count += (z >= 4.5).astype(int)
    return maximum_z, extreme_count


def compute_category_rarity_signal(
    X: pd.DataFrame,
    categorical_columns: Sequence[str],
) -> np.ndarray:
    n = len(X)
    rarity = np.zeros(n, dtype=float)
    for column in categorical_columns:
        values = X[column].map(normalize_category)
        frequencies = values.value_counts(dropna=False)
        probabilities = values.map(frequencies).to_numpy(dtype=float) / max(n, 1)
        current = -np.log10(np.maximum(probabilities, 1.0 / max(n, 1)))
        rarity = np.maximum(rarity, current)
    return rarity


def compute_within_literature_target_z(
    y: np.ndarray,
    literature_ids: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    literature_ids = np.asarray(literature_ids, dtype=object)
    output = np.zeros(len(y), dtype=float)
    _, global_scale = robust_location_scale(y)
    for literature in pd.unique(literature_ids):
        index = np.where(literature_ids == literature)[0]
        values = y[index]
        if len(index) <= 1:
            output[index] = 0.0
            continue
        center, local_scale = robust_location_scale(values)
        # 小文献或文献内目标完全相同的情况下，引入少量全局尺度，避免无限大。
        effective_scale = max(local_scale, 0.08 * global_scale)
        output[index] = np.abs((values - center) / effective_scale)
    return output


def compute_signature_conflict_signals(
    signatures: Sequence[Tuple],
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    count = np.ones(len(y), dtype=int)
    target_range = np.zeros(len(y), dtype=float)
    target_deviation = np.zeros(len(y), dtype=float)
    mapping: Dict[Tuple, List[int]] = {}
    for index, signature in enumerate(signatures):
        mapping.setdefault(signature, []).append(index)
    for indices in mapping.values():
        idx = np.asarray(indices, dtype=int)
        values = y[idx]
        count[idx] = len(idx)
        target_range[idx] = float(np.max(values) - np.min(values)) if len(idx) > 1 else 0.0
        target_deviation[idx] = np.abs(values - float(np.median(values)))
    return count, target_range, target_deviation


def compute_cross_group_neighbor_signals(
    X: pd.DataFrame,
    y: np.ndarray,
    strict_groups: np.ndarray,
    metadata: Mapping,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    mode = {"Name": "Diagnostic_Median", "Imputer": "median"}
    preprocessor = build_preprocessor(
        metadata["numeric_columns"],
        metadata["categorical_columns"],
        mode,
    )
    matrix = np.asarray(preprocessor.fit_transform(X), dtype=np.float32)
    search_count = min(
        n,
        max(DIAG_NEIGHBOR_COUNT * DIAG_NEIGHBOR_SEARCH_MULTIPLIER + 1, 35),
    )
    neighbors = NearestNeighbors(n_neighbors=search_count, metric="euclidean", n_jobs=N_JOBS)
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix, return_distance=True)

    local_median = np.full(n, np.nan, dtype=float)
    target_gap = np.full(n, np.nan, dtype=float)
    mean_distance = np.full(n, np.nan, dtype=float)
    neighbor_count = np.zeros(n, dtype=int)
    for row in range(n):
        selected_rows: List[int] = []
        selected_distances: List[float] = []
        for distance, candidate in zip(distances[row], indices[row]):
            candidate = int(candidate)
            if candidate == row:
                continue
            if strict_groups[candidate] == strict_groups[row]:
                continue
            selected_rows.append(candidate)
            selected_distances.append(float(distance))
            if len(selected_rows) >= DIAG_NEIGHBOR_COUNT:
                break
        if not selected_rows:
            continue
        neighbor_values = np.asarray(y[selected_rows], dtype=float)
        local_median[row] = float(np.median(neighbor_values))
        target_gap[row] = abs(float(y[row]) - local_median[row])
        mean_distance[row] = float(np.mean(selected_distances))
        neighbor_count[row] = len(selected_rows)
    return local_median, target_gap, mean_distance, neighbor_count


def build_row_diagnostic_table(
    X: pd.DataFrame,
    y: np.ndarray,
    y_raw: np.ndarray,
    strict_groups: np.ndarray,
    oof_predictions: Mapping[str, np.ndarray],
    metadata: Mapping,
) -> pd.DataFrame:
    literature_ids = np.asarray(metadata["literature_ids"], dtype=object)
    literature_counts = pd.Series(literature_ids).value_counts()
    source_data = metadata["source_data"]

    available_consensus_models = [
        name for name in ("GBR", "XGB", "ExtraTrees", "RF")
        if name in oof_predictions
    ]
    available_base_models = [name for name in BASE_MODEL_NAMES if name in oof_predictions]
    consensus_matrix = np.column_stack([oof_predictions[name] for name in available_consensus_models])
    all_model_matrix = np.column_stack([oof_predictions[name] for name in available_base_models])
    consensus_prediction = np.median(consensus_matrix, axis=1)
    model_disagreement = np.std(all_model_matrix, axis=1, ddof=0)
    prediction_spread = np.max(all_model_matrix, axis=1) - np.min(all_model_matrix, axis=1)

    reference_prediction = np.asarray(oof_predictions[DIAG_REFERENCE_MODEL], dtype=float)
    reference_raw_prediction = inverse_target(reference_prediction)
    reference_abs_error_log = np.abs(y - reference_prediction)
    reference_abs_error_raw = np.abs(y_raw - reference_raw_prediction)
    consensus_abs_error_log = np.abs(y - consensus_prediction)

    numeric_max_z, numeric_extreme_count = compute_numeric_outlier_signals(
        X, metadata["numeric_columns"]
    )
    category_rarity = compute_category_rarity_signal(X, metadata["categorical_columns"])
    within_literature_z = compute_within_literature_target_z(y, literature_ids)
    signature_count, signature_range, signature_deviation = compute_signature_conflict_signals(
        metadata["signatures"], y
    )
    neighbor_median, neighbor_gap, neighbor_distance, neighbor_count = compute_cross_group_neighbor_signals(
        X, y, strict_groups, metadata
    )

    # 多信号初筛分数。它只用于决定哪些行进入昂贵的删一行重训，不直接判定删除。
    screen_score = (
        0.27 * percentile_rank(reference_abs_error_log)
        + 0.17 * percentile_rank(consensus_abs_error_log)
        + 0.16 * percentile_rank(np.nan_to_num(neighbor_gap, nan=0.0))
        + 0.10 * percentile_rank(within_literature_z)
        + 0.08 * percentile_rank(numeric_max_z)
        + 0.04 * percentile_rank(category_rarity)
        + 0.08 * percentile_rank(model_disagreement)
        + 0.10 * percentile_rank(signature_deviation + signature_range)
    )

    table = pd.DataFrame({
        "Data_Index_0Based": np.arange(len(X), dtype=int),
        "Valid_Data_Row_1Based": np.arange(1, len(X) + 1, dtype=int),
        "Excel_Row": np.asarray(metadata["excel_row_numbers"], dtype=int),
        "Literature_ID": literature_ids,
        "Strict_Group": np.asarray(strict_groups, dtype=int),
        "Literature_Row_Count": [int(literature_counts[value]) for value in literature_ids],
        "Target_Log": np.asarray(y, dtype=float),
        "Target_Raw": np.asarray(y_raw, dtype=float),
        "GBR_OOF_Pred_Log": reference_prediction,
        "GBR_OOF_Pred_Raw": reference_raw_prediction,
        "GBR_Signed_Error_Log": reference_prediction - y,
        "GBR_Abs_Error_Log": reference_abs_error_log,
        "GBR_Abs_Error_Raw": reference_abs_error_raw,
        "Consensus_Pred_Log": consensus_prediction,
        "Consensus_Abs_Error_Log": consensus_abs_error_log,
        "Model_Disagreement_SD": model_disagreement,
        "Model_Prediction_Spread": prediction_spread,
        "Within_Literature_Target_Z": within_literature_z,
        "Numeric_Max_Robust_Z": numeric_max_z,
        "Numeric_Extreme_Count": numeric_extreme_count,
        "Category_Rarity": category_rarity,
        "Neighbor_Target_Median_Log": neighbor_median,
        "Neighbor_Target_Gap_Log": neighbor_gap,
        "Neighbor_Mean_Distance": neighbor_distance,
        "Neighbor_Count": neighbor_count,
        "Same_X_Count": signature_count,
        "Same_X_Target_Range_Log": signature_range,
        "Same_X_Target_Deviation_Log": signature_deviation,
        "Screen_Score": screen_score,
    })

    title_column = metadata.get("title_column")
    if title_column is not None and title_column in source_data.columns:
        table.insert(4, "Literature_Title", source_data[title_column].astype(str).to_numpy())

    thresholds = {
        "reference": float(np.quantile(reference_abs_error_log, 0.90)),
        "consensus": float(np.quantile(consensus_abs_error_log, 0.90)),
        "neighbor": float(np.nanquantile(neighbor_gap, 0.90)) if np.isfinite(neighbor_gap).any() else np.inf,
        "disagreement": float(np.quantile(model_disagreement, 0.90)),
        "feature": max(4.5, float(np.quantile(numeric_max_z, 0.95))),
    }
    reasons = []
    for row in table.itertuples(index=False):
        current = []
        if row.GBR_Abs_Error_Log >= thresholds["reference"]:
            current.append("GBR严格OOF大残差")
        if row.Consensus_Abs_Error_Log >= thresholds["consensus"]:
            current.append("多模型一致大残差")
        if np.isfinite(row.Neighbor_Target_Gap_Log) and row.Neighbor_Target_Gap_Log >= thresholds["neighbor"]:
            current.append("相似X对应目标不一致")
        if row.Within_Literature_Target_Z >= 3.5:
            current.append("文献内目标异常")
        if row.Numeric_Max_Robust_Z >= thresholds["feature"]:
            current.append("数值特征极端")
        if row.Model_Disagreement_SD >= thresholds["disagreement"]:
            current.append("模型分歧大")
        if row.Same_X_Count > 1 and row.Same_X_Target_Range_Log >= DIAG_SIGNATURE_CONFLICT_LOG_RANGE:
            current.append("完全相同X但CCC冲突")
        if not current:
            current.append("综合异常分数较高")
        reasons.append("；".join(current))
    table["Suspicion_Reasons"] = reasons
    return table


def select_rows_for_strict_retraining(diagnostic_table: pd.DataFrame) -> List[int]:
    ordered = diagnostic_table.sort_values(
        ["Screen_Score", "GBR_Abs_Error_Log"], ascending=[False, False]
    )
    selected: List[int] = []
    per_literature: Dict[object, int] = {}

    # 完全相同 X 但目标冲突的行优先进入复核。
    forced = ordered[
        (ordered["Same_X_Count"] > 1)
        & (ordered["Same_X_Target_Range_Log"] >= DIAG_SIGNATURE_CONFLICT_LOG_RANGE)
    ]
    candidates = pd.concat([forced, ordered], axis=0).drop_duplicates("Data_Index_0Based")

    for row in candidates.itertuples(index=False):
        literature = row.Literature_ID
        # 单条文献可以核查，但不能进入“建议删除”影响测试。
        if int(row.Literature_Row_Count) <= 1:
            continue
        used = per_literature.get(literature, 0)
        if used >= DIAG_MAX_CANDIDATES_PER_LITERATURE:
            continue
        selected.append(int(row.Data_Index_0Based))
        per_literature[literature] = used + 1
        if len(selected) >= DIAG_STRICT_RETRAIN_TOP_N:
            break
    return selected


def evaluate_single_row_removal_influence(
    candidate_index: int,
    X: pd.DataFrame,
    y: np.ndarray,
    y_raw: np.ndarray,
    metadata: Mapping,
    outer_fold_artifacts: Sequence[Mapping],
) -> Dict[str, float]:
    baseline_predictions = []
    removed_predictions = []
    true_values = []
    fold_delta_log = []
    fold_delta_raw = []

    for artifact in outer_fold_artifacts:
        train_index = np.asarray(artifact["Train_Index"], dtype=int)
        test_index = np.asarray(artifact["Test_Index"], dtype=int)
        if candidate_index not in set(train_index.tolist()):
            # 该行所在严格组在此折是测试组，它没有参与训练，因此这一折不能用于训练影响判断。
            continue

        reduced_train_index = train_index[train_index != candidate_index]
        preprocessor = build_preprocessor(
            metadata["numeric_columns"],
            metadata["categorical_columns"],
            artifact["Mode"],
        )
        X_train = np.asarray(
            preprocessor.fit_transform(X.iloc[reduced_train_index], y[reduced_train_index]),
            dtype=np.float32,
        )
        X_test = np.asarray(preprocessor.transform(X.iloc[test_index]), dtype=np.float32)
        removed_prediction = fit_predict_gbr_bagged(
            artifact["GBR_Params"],
            X_train,
            y[reduced_train_index],
            X_test,
            int(artifact["GBR_Seed"]),
        )
        baseline_prediction = np.asarray(artifact["GBR_Test_Prediction"], dtype=float)

        baseline_log_metric = metrics(y[test_index], baseline_prediction)
        removed_log_metric = metrics(y[test_index], removed_prediction)
        fold_delta_log.append(removed_log_metric["R2"] - baseline_log_metric["R2"])

        baseline_raw_metric = metrics(y_raw[test_index], inverse_target(baseline_prediction))
        removed_raw_metric = metrics(y_raw[test_index], inverse_target(removed_prediction))
        fold_delta_raw.append(removed_raw_metric["R2"] - baseline_raw_metric["R2"])

        baseline_predictions.append(baseline_prediction)
        removed_predictions.append(removed_prediction)
        true_values.append(y[test_index])

    if not true_values:
        return {
            "Affected_Folds": 0,
            "Affected_Test_N": 0,
            "Pooled_Delta_R2_Log": np.nan,
            "Pooled_Delta_R2_Raw": np.nan,
            "Pooled_RMSE_Reduction_Log": np.nan,
            "Pooled_RMSE_Reduction_Raw": np.nan,
            "Mean_Fold_Delta_R2_Log": np.nan,
            "Mean_Fold_Delta_R2_Raw": np.nan,
            "Positive_Folds_Log": 0,
            "Positive_Folds_Raw": 0,
            "Worst_Fold_Delta_R2_Log": np.nan,
            "Worst_Fold_Delta_R2_Raw": np.nan,
        }

    y_concat = np.concatenate(true_values)
    baseline_concat = np.concatenate(baseline_predictions)
    removed_concat = np.concatenate(removed_predictions)
    baseline_log = metrics(y_concat, baseline_concat)
    removed_log = metrics(y_concat, removed_concat)

    y_raw_concat = inverse_target(y_concat)
    baseline_raw_prediction = inverse_target(baseline_concat)
    removed_raw_prediction = inverse_target(removed_concat)
    baseline_raw = metrics(y_raw_concat, baseline_raw_prediction)
    removed_raw = metrics(y_raw_concat, removed_raw_prediction)

    fold_delta_log_array = np.asarray(fold_delta_log, dtype=float)
    fold_delta_raw_array = np.asarray(fold_delta_raw, dtype=float)
    return {
        "Affected_Folds": int(len(fold_delta_log_array)),
        "Affected_Test_N": int(len(y_concat)),
        "Pooled_Delta_R2_Log": float(removed_log["R2"] - baseline_log["R2"]),
        "Pooled_Delta_R2_Raw": float(removed_raw["R2"] - baseline_raw["R2"]),
        "Pooled_RMSE_Reduction_Log": float(baseline_log["RMSE"] - removed_log["RMSE"]),
        "Pooled_RMSE_Reduction_Raw": float(baseline_raw["RMSE"] - removed_raw["RMSE"]),
        "Mean_Fold_Delta_R2_Log": float(np.mean(fold_delta_log_array)),
        "Mean_Fold_Delta_R2_Raw": float(np.mean(fold_delta_raw_array)),
        "Positive_Folds_Log": int(np.sum(fold_delta_log_array > 0.0)),
        "Positive_Folds_Raw": int(np.sum(fold_delta_raw_array > 0.0)),
        "Worst_Fold_Delta_R2_Log": float(np.min(fold_delta_log_array)),
        "Worst_Fold_Delta_R2_Raw": float(np.min(fold_delta_raw_array)),
    }


def apply_removal_recommendations(diagnostic_table: pd.DataFrame) -> pd.DataFrame:
    table = diagnostic_table.copy()
    table["Verified_Log_Training_Harm"] = (
        (table["Pooled_Delta_R2_Log"] >= DIAG_MIN_POOLED_R2_GAIN_LOG)
        & (table["Positive_Folds_Log"] >= DIAG_MIN_POSITIVE_FOLDS)
        & (table["Worst_Fold_Delta_R2_Log"] >= -DIAG_MAX_WORST_FOLD_R2_LOSS)
    )
    table["Verified_Raw_Training_Harm"] = (
        (table["Pooled_Delta_R2_Raw"] >= DIAG_MIN_POOLED_R2_GAIN_RAW)
        & (table["Positive_Folds_Raw"] >= DIAG_MIN_POSITIVE_FOLDS)
        & (table["Worst_Fold_Delta_R2_Raw"] >= -DIAG_MAX_WORST_FOLD_R2_LOSS)
    )

    # 排序分数只用于安排人工核查顺序。
    log_gain_rank = percentile_rank(np.nan_to_num(table["Pooled_Delta_R2_Log"], nan=-1.0))
    raw_gain_rank = percentile_rank(np.nan_to_num(table["Pooled_Delta_R2_Raw"], nan=-1.0))
    table["Removal_Priority_Score"] = (
        0.55 * log_gain_rank
        + 0.20 * raw_gain_rank
        + 0.25 * percentile_rank(table["Screen_Score"].to_numpy(dtype=float))
    )

    table["Recommended_Temporary_Exclusion"] = False
    table["Suggested_Action"] = "重点核查；暂无稳定训练拖累证据，不建议直接删除"
    table.loc[table["Literature_Row_Count"] <= 1, "Suggested_Action"] = (
        "只核查，禁止删除：该文献仅1条数据"
    )
    table.loc[table["Verified_Raw_Training_Harm"], "Suggested_Action"] = (
        "高优先级核查：主要拖累原始CCC尺度；确认录入/单位错误后可暂时剔除复跑"
    )
    table.loc[table["Verified_Log_Training_Harm"], "Suggested_Action"] = (
        "高优先级核查：稳定拖累log模型；确认原文错误后可暂时剔除复跑"
    )
    both = table["Verified_Log_Training_Harm"] & table["Verified_Raw_Training_Harm"]
    table.loc[both, "Suggested_Action"] = (
        "最高优先级核查：同时拖累log与原始CCC；确认错误后可暂时剔除复跑"
    )

    # 强制每篇文献至少保留一条。这里只生成建议，不实际修改数据。
    recommended_count: Dict[object, int] = {}
    order = table.sort_values(
        ["Verified_Log_Training_Harm", "Verified_Raw_Training_Harm", "Removal_Priority_Score"],
        ascending=[False, False, False],
    ).index
    for index in order:
        row = table.loc[index]
        verified = bool(row["Verified_Log_Training_Harm"] or row["Verified_Raw_Training_Harm"])
        if not verified:
            continue
        literature = row["Literature_ID"]
        capacity = max(int(row["Literature_Row_Count"]) - 1, 0)
        used = recommended_count.get(literature, 0)
        if used < capacity:
            table.at[index, "Recommended_Temporary_Exclusion"] = True
            recommended_count[literature] = used + 1
        else:
            table.at[index, "Suggested_Action"] = (
                str(table.at[index, "Suggested_Action"])
                + "；但受每篇文献至少保留1条约束，本行不进入自动暂时剔除建议"
            )
    return table


def print_detailed_source_rows(
    rows: pd.DataFrame,
    metadata: Mapping,
    oof_predictions: Mapping[str, np.ndarray],
) -> None:
    source_data = metadata["source_data"]
    prediction_names = [
        name for name in ("GBR", "XGB", "ExtraTrees", "RF", "LGBM", STACKING_NAME)
        if name in oof_predictions
    ]
    ordered = rows.sort_values(
        ["Recommended_Temporary_Exclusion", "Verified_Log_Training_Harm", "Removal_Priority_Score"],
        ascending=[False, False, False],
    ).head(DIAG_MAX_DETAILED_ROWS)

    for position, row in enumerate(ordered.itertuples(index=False), start=1):
        data_index = int(row.Data_Index_0Based)
        print("\n" + "-" * 156)
        print(
            f"核查记录 {position}/{len(ordered)} | Excel行号={row.Excel_Row} | "
            f"有效数据行={row.Valid_Data_Row_1Based} | 文献={row.Literature_ID} | 严格组={row.Strict_Group}"
        )
        print("疑点：", row.Suspicion_Reasons)
        print("建议：", row.Suggested_Action)
        print("是否进入暂时剔除建议：", bool(row.Recommended_Temporary_Exclusion))
        print(
            "诊断指标："
            f" Screen={row.Screen_Score:.4f}, GBR_abs_error_log={row.GBR_Abs_Error_Log:.4f},"
            f" Neighbor_gap_log={row.Neighbor_Target_Gap_Log:.4f},"
            f" Within_lit_Z={row.Within_Literature_Target_Z:.3f},"
            f" Numeric_max_Z={row.Numeric_Max_Robust_Z:.3f},"
            f" Same_X_count={row.Same_X_Count}, Same_X_range_log={row.Same_X_Target_Range_Log:.4f}"
        )
        print(
            "严格删一行重训影响："
            f" ΔR2_log={row.Pooled_Delta_R2_Log:+.6f},"
            f" ΔR2_raw={row.Pooled_Delta_R2_Raw:+.6f},"
            f" log改善折={int(row.Positive_Folds_Log)}/{int(row.Affected_Folds)},"
            f" 最差折ΔR2_log={row.Worst_Fold_Delta_R2_Log:+.6f}"
        )
        print("各模型严格外层OOF预测：")
        for name in prediction_names:
            prediction_log = float(oof_predictions[name][data_index])
            prediction_raw = float(inverse_target(np.asarray([prediction_log]))[0])
            print(f"  {name}: log={prediction_log:.6f}, raw={prediction_raw:.6f}")
        print("原始Excel该行全部字段：")
        source_row = source_data.iloc[data_index]
        for column, value in source_row.items():
            display_value = "<MISSING>" if pd.isna(value) else value
            print(f"  {column}: {display_value}")


def run_row_diagnostics(
    X: pd.DataFrame,
    y: np.ndarray,
    y_raw: np.ndarray,
    strict_groups: np.ndarray,
    oof_predictions: Mapping[str, np.ndarray],
    metadata: Mapping,
    outer_fold_artifacts: Sequence[Mapping],
) -> pd.DataFrame:
    section("8. 数据行异常初筛：严格OOF误差、相似样本、文献内异常和重复X冲突")
    print("说明：大残差行只是待核查候选；只有删一行重训在多个未知文献折上稳定改善，才认定为训练拖累。")
    diagnostic_table = build_row_diagnostic_table(
        X, y, y_raw, strict_groups, oof_predictions, metadata
    )
    screen_table = diagnostic_table.sort_values(
        ["Screen_Score", "GBR_Abs_Error_Log"], ascending=[False, False]
    ).head(DIAG_SCREEN_TOP_N)
    screen_columns = [
        "Excel_Row", "Valid_Data_Row_1Based", "Literature_ID", "Literature_Row_Count",
        "Target_Raw", "Target_Log", "GBR_OOF_Pred_Raw", "GBR_Abs_Error_Log",
        "Consensus_Abs_Error_Log", "Neighbor_Target_Gap_Log", "Within_Literature_Target_Z",
        "Numeric_Max_Robust_Z", "Same_X_Count", "Same_X_Target_Range_Log",
        "Model_Disagreement_SD", "Screen_Score", "Suspicion_Reasons",
    ]
    print(screen_table[screen_columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    section("9. 严格删一行重训：验证哪些行真正拖累未知文献预测")
    candidate_indices = select_rows_for_strict_retraining(diagnostic_table)
    influence_rows = []
    for position, candidate_index in enumerate(candidate_indices, start=1):
        row = diagnostic_table.loc[
            diagnostic_table["Data_Index_0Based"] == candidate_index
        ].iloc[0]
        print(
            f"  影响复核 {position}/{len(candidate_indices)}：Excel行={int(row['Excel_Row'])}，"
            f"文献={row['Literature_ID']}，初筛分数={row['Screen_Score']:.4f}"
        )
        influence = evaluate_single_row_removal_influence(
            candidate_index, X, y, y_raw, metadata, outer_fold_artifacts
        )
        influence_rows.append({"Data_Index_0Based": candidate_index, **influence})

    if influence_rows:
        influence_frame = pd.DataFrame(influence_rows)
    else:
        influence_frame = pd.DataFrame(columns=[
            "Data_Index_0Based", "Affected_Folds", "Affected_Test_N",
            "Pooled_Delta_R2_Log", "Pooled_Delta_R2_Raw",
            "Pooled_RMSE_Reduction_Log", "Pooled_RMSE_Reduction_Raw",
            "Mean_Fold_Delta_R2_Log", "Mean_Fold_Delta_R2_Raw",
            "Positive_Folds_Log", "Positive_Folds_Raw",
            "Worst_Fold_Delta_R2_Log", "Worst_Fold_Delta_R2_Raw",
        ])
    diagnostic_table = diagnostic_table.merge(
        influence_frame, on="Data_Index_0Based", how="left"
    )
    integer_columns = [
        "Affected_Folds", "Affected_Test_N", "Positive_Folds_Log", "Positive_Folds_Raw"
    ]
    for column in integer_columns:
        diagnostic_table[column] = diagnostic_table[column].fillna(0).astype(int)
    diagnostic_table = apply_removal_recommendations(diagnostic_table)

    tested = diagnostic_table[diagnostic_table["Affected_Folds"] > 0].sort_values(
        ["Verified_Log_Training_Harm", "Pooled_Delta_R2_Log", "Screen_Score"],
        ascending=[False, False, False],
    )
    impact_columns = [
        "Excel_Row", "Valid_Data_Row_1Based", "Literature_ID", "Literature_Row_Count",
        "Target_Raw", "GBR_OOF_Pred_Raw", "GBR_Abs_Error_Log", "Screen_Score",
        "Affected_Folds", "Pooled_Delta_R2_Log", "Mean_Fold_Delta_R2_Log",
        "Positive_Folds_Log", "Worst_Fold_Delta_R2_Log", "Pooled_RMSE_Reduction_Log",
        "Pooled_Delta_R2_Raw", "Positive_Folds_Raw", "Verified_Log_Training_Harm",
        "Verified_Raw_Training_Harm", "Recommended_Temporary_Exclusion", "Suspicion_Reasons",
    ]
    print(tested[impact_columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    recommended = diagnostic_table[diagnostic_table["Recommended_Temporary_Exclusion"]].copy()
    print("\n暂时剔除建议汇总（仅供核查后复跑，不会自动修改数据）：")
    if recommended.empty:
        print("  没有任何数据行同时满足稳定训练拖累门槛和每篇文献至少保留1条约束。")
    else:
        recommendation_columns = [
            "Excel_Row", "Valid_Data_Row_1Based", "Literature_ID", "Literature_Row_Count",
            "Target_Raw", "GBR_OOF_Pred_Raw", "Pooled_Delta_R2_Log",
            "Pooled_Delta_R2_Raw", "Positive_Folds_Log", "Worst_Fold_Delta_R2_Log",
            "Suspicion_Reasons", "Suggested_Action",
        ]
        print(recommended[recommendation_columns].to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        remain_check = (
            diagnostic_table.groupby("Literature_ID")["Literature_Row_Count"].first()
            - recommended.groupby("Literature_ID").size().reindex(
                diagnostic_table["Literature_ID"].unique(), fill_value=0
            )
        )
        if (remain_check < 1).any():
            raise RuntimeError("暂时剔除建议违反每篇文献至少保留1条约束。")
        print(f"  建议行数={len(recommended)}；已验证所有文献删除后仍至少保留1条。")

    section("10. 待核查行完整数据：Excel行号、目标、预测和原始全部字段")
    # 输出初筛前列和所有经过严格重训复核的行；包含单条文献，但单条文献永不建议删除。
    detail_indices = set(screen_table["Data_Index_0Based"].astype(int).tolist())
    detail_indices.update(candidate_indices)
    detail_rows = diagnostic_table[
        diagnostic_table["Data_Index_0Based"].isin(sorted(detail_indices))
    ].copy()
    print_detailed_source_rows(detail_rows, metadata, oof_predictions)

    print("\n重要使用规则：")
    print("1. 本模块只定位需回原文核查的行，不会自动删除或改写任何数据。")
    print("2. 先检查单位、正负号、小数点、pH/温度条件、CCC定义和文献表格转录。")
    print("3. 只有确认原始录入错误后才修正；真实极端值不能因模型不喜欢而删除。")
    print("4. 修正数据后必须从头重新生成外层折并重新评估，不能在同一OOF结果上宣称提升。")
    return diagnostic_table


# =============================================================================
# 12. 主程序
# =============================================================================

def main() -> None:
    start_time = time.time()
    X, y, y_raw, strict_groups, metadata = load_and_prepare_data()

    section("1. 数据、目标尺度和严格防泄露设置")
    print("运行配置：", SEARCH_PROFILE)
    print("外层折分模式：", OUTER_SPLIT_MODE)
    print(
        "主建模目标：",
        f"{TARGET_TRANSFORM}({metadata['target_column']})"
        if TARGET_TRANSFORM != "raw"
        else metadata["target_column"],
    )
    print(
        "外层折数 / 内层折数 / Meta折数：",
        OUTER_SPLITS, "/", INNER_SPLITS, "/", META_SPLITS,
    )
    print("原始行数 / 有效样本：", metadata["raw_rows"], "/", metadata["valid_rows"])
    print("文献数量 / 严格组数量：", metadata["literature_count"], "/", metadata["strict_group_count"])
    print("模型输入列：", metadata["model_columns"])
    print("数值列：", metadata["numeric_columns"])
    print("类别列：", metadata["categorical_columns"])
    print("Stacking内部六个一级模型：", BASE_MODEL_NAMES)
    print("Stacking内部GBR750锚点参数：", GBR750_ANCHOR_PARAMS)
    print("主表GBR单模型参数：", {"GBR": GBR_PUBLIC_PARAMS})
    print("Champion Guard五层Stacking模型：", STACKING_NAME)
    print("主表只排名五个公开单模型和最终Stacking；GBR750只在Stacking内部使用，不单独排名。")
    print("说明：外层测试标签不进入预处理、参数选择、模型训练、Meta-CV或Stacking权重学习。")
    print("探索目标：Stacking争取接近或超过0.83；实际结果只由严格外层OOF决定。")
    print("说明：程序只输出控制台，不保存任何结果文件。")

    outer_splits = make_group_folds(
        strict_groups, y, OUTER_SPLITS, RANDOM_STATE, outer=True
    )

    section("2. 外层五折目标分布检查")
    distribution_rows = []
    for fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        distribution_rows.append({
            "Fold": fold,
            "Train_N": len(train_index),
            "Test_N": len(test_index),
            "Train_Groups": len(np.unique(strict_groups[train_index])),
            "Test_Groups": len(np.unique(strict_groups[test_index])),
            "Train_LogY_Mean": float(np.mean(y[train_index])),
            "Test_LogY_Mean": float(np.mean(y[test_index])),
            "Train_RawY_Mean": float(np.mean(y_raw[train_index])),
            "Test_RawY_Mean": float(np.mean(y_raw[test_index])),
        })
    print(pd.DataFrame(distribution_rows).to_string(
        index=False, float_format=lambda value: f"{value:.5f}"
    ))

    oof_predictions = {
        name: np.full(len(y), np.nan, dtype=float)
        for name in MODEL_NAMES
    }
    fold_records: List[Dict] = []
    parameter_records: List[Dict] = []
    stacking_records: List[Dict] = []
    outer_fold_artifacts: List[Dict] = []

    section("3. 外层严格五折")
    for outer_fold, (train_index, test_index) in enumerate(outer_splits, start=1):
        fold_start = time.time()
        train_index = np.asarray(train_index, dtype=int)
        test_index = np.asarray(test_index, dtype=int)

        X_train_raw = X.iloc[train_index].reset_index(drop=True)
        X_test_raw = X.iloc[test_index].reset_index(drop=True)
        y_train = y[train_index]
        y_test = y[test_index]
        raw_test = y_raw[test_index]
        groups_train = strict_groups[train_index]
        groups_test = strict_groups[test_index]

        if set(groups_train).intersection(set(groups_test)):
            raise RuntimeError("发生外层严格组泄露。")

        print("\n" + "-" * 156)
        print(
            f"外层折 {outer_fold}/{OUTER_SPLITS}：训练={len(train_index)}，"
            f"测试={len(test_index)}，训练组={len(np.unique(groups_train))}，"
            f"测试组={len(np.unique(groups_test))}"
        )

        # 五个公开单模型与Stacking内部GBR750均使用相同的原始13列输入。
        print(f"  特征视图：Raw={X_train_raw.shape[1]}列")

        print("  选择原始视图预处理模式（GBR探针）……")
        mode, raw_inner_cache = select_preprocessor_and_cache(
            X_train_raw,
            y_train,
            groups_train,
            metadata["numeric_columns"],
            metadata["categorical_columns"],
            RANDOM_STATE * 10000 + outer_fold * 1000,
        )
        print("  选中：", mode["Name"])

        model_details: Dict[str, Dict] = {}
        print("  RF 内层选择……")
        model_details["RF"] = evaluate_candidate_grid(
            raw_inner_cache, y_train, RF_CANDIDATES, make_rf,
            outer_fold * 10000 + 101,
        )
        print("   ", public_details(model_details["RF"]))

        print("  ExtraTrees 内层选择……")
        model_details["ExtraTrees"] = evaluate_candidate_grid(
            raw_inner_cache, y_train, ET_CANDIDATES, make_extra_trees,
            outer_fold * 10000 + 201,
        )
        print("   ", public_details(model_details["ExtraTrees"]))

        print("  LGBM 内层选择……")
        model_details["LGBM"] = evaluate_candidate_grid(
            raw_inner_cache, y_train, LGBM_CANDIDATES, make_lgbm,
            outer_fold * 10000 + 401,
        )
        print("   ", public_details(model_details["LGBM"]))

        print("  XGBoost 内层选择……")
        model_details["XGB"] = evaluate_candidate_grid(
            raw_inner_cache, y_train, XGB_CANDIDATES, make_xgb,
            outer_fold * 10000 + 501,
        )
        print("   ", public_details(model_details["XGB"]))

        print("  GBR 固定参数严格OOF……")
        model_details["GBR"] = evaluate_candidate_grid(
            raw_inner_cache, y_train, GBR_CANDIDATES, make_gbr,
            outer_fold * 10000 + 551,
        )
        print("   ", public_details(model_details["GBR"]),
              "参数=", model_details["GBR"]["Params"])

        print("  [Stacking内部] GBR750锚点固定参数严格OOF……")
        model_details[STACK_ANCHOR_NAME] = evaluate_candidate_grid(
            raw_inner_cache, y_train, GBR750_CANDIDATES, make_gbr,
            outer_fold * 10000 + 601,
        )
        print("   ", public_details(model_details[STACK_ANCHOR_NAME]),
              "参数=", model_details[STACK_ANCHOR_NAME]["Params"])

        train_oof = {
            name: np.asarray(model_details[name]["OOF_Prediction"], dtype=float)
            for name in BASE_MODEL_NAMES
        }
        raw_preprocessor = build_preprocessor(
            metadata["numeric_columns"],
            metadata["categorical_columns"],
            mode,
        )
        X_raw_full_train = np.asarray(
            raw_preprocessor.fit_transform(X_train_raw, y_train),
            dtype=np.float32,
        )
        X_raw_full_test = np.asarray(
            raw_preprocessor.transform(X_test_raw), dtype=np.float32
        )

        test_prediction = {
            "RF": fit_predict_model(
                make_rf, model_details["RF"]["Params"],
                X_raw_full_train, y_train, X_raw_full_test,
                outer_fold * 1000 + 1,
            ),
            "ExtraTrees": fit_predict_model(
                make_extra_trees, model_details["ExtraTrees"]["Params"],
                X_raw_full_train, y_train, X_raw_full_test,
                outer_fold * 1000 + 2,
            ),
            "LGBM": fit_predict_model(
                make_lgbm, model_details["LGBM"]["Params"],
                X_raw_full_train, y_train, X_raw_full_test,
                outer_fold * 1000 + 4,
            ),
            "XGB": fit_predict_model(
                make_xgb, model_details["XGB"]["Params"],
                X_raw_full_train, y_train, X_raw_full_test,
                outer_fold * 1000 + 5,
            ),
            "GBR": fit_predict_gbr_bagged(
                model_details["GBR"]["Params"],
                X_raw_full_train, y_train, X_raw_full_test,
                outer_fold * 1000 + 5,
            ),
            STACK_ANCHOR_NAME: fit_predict_gbr_bagged(
                model_details[STACK_ANCHOR_NAME]["Params"],
                X_raw_full_train, y_train, X_raw_full_test,
                outer_fold * 1000 + 6,
            ),
        }

        base_oof = np.column_stack([
            train_oof[name] for name in BASE_MODEL_NAMES
        ])
        base_test = np.column_stack([
            test_prediction[name] for name in BASE_MODEL_NAMES
        ])
        stack = fit_safe_stacking(
            base_oof,
            base_test,
            y_train,
            groups_train,
            BASE_MODEL_NAMES,
            RANDOM_STATE * 200000 + outer_fold * 1000,
        )
        train_oof[STACKING_NAME] = stack["Train_OOF_Prediction"]
        test_prediction[STACKING_NAME] = stack["Test_Prediction"]

        print("  Stacking Safety Fallback：", stack["Safety_Fallback"])
        print("  Champion Guard触发：", stack["Champion_Guard_Triggered"])
        print("  Champion Guard强证据放行：", stack["Champion_Guard_Strong_Evidence_Override"])
        print("  Champion Guard原因：", stack["Champion_Guard_Reasons"])
        print("  训练OOF最佳一级模型 / GBR750差距：", (
            stack["Best_Base_Model_TrainOOF"],
            round(float(stack["GBR_Gap_To_Best_TrainOOF"]), 6),
        ))
        print("  Guard前非锚点正权重 / 绝对权重：", (
            round(float(stack["Positive_Partner_Weight_PreGuard"]), 4),
            round(float(stack["Absolute_Partner_Weight_PreGuard"]), 4),
        ))
        print("  Stacking 五层软收缩：", stack["Soft_Shrinkage"])
        print("  Stacking 回退模型：", stack["Fallback_Model"])
        print("  Stacking候选成员：", stack["Selected_Models"])
        print("  Stacking第一层 L2：", stack["Selected_L2"])
        print("  Stacking第一层收缩：", round(float(stack["Stage1_Shrinkage"]), 4))
        print("  Stacking第一层偏置：", round(float(stack["Bias"]), 6))
        print("  Stacking内部GBR750有效权重：", round(float(stack["GBR750_Weight"]), 4))
        print("  Stacking二级残差校准器：", stack["Residual_Corrector"])
        print("  Stacking二级校准强度：", round(float(stack["Residual_Strength"]), 4))
        print("  Stacking二级截断倍数：", round(float(stack["Residual_Clip_Multiplier"]), 4))
        print("  Stacking Meta2 R2增益：", round(float(stack["Meta2_Gain"]), 6))
        print("  Stacking三层门控专家：", stack["Gate_Expert"])
        print("  Stacking门控 C / 阈值 / 强度：", (
            round(float(stack["Gate_C"]), 4),
            round(float(stack["Gate_Threshold"]), 4),
            round(float(stack["Gate_Strength"]), 4),
        ))
        print("  Stacking门控截断倍数：", round(float(stack["Gate_Clip_Multiplier"]), 4))
        print("  Stacking最终尺度 / 截距：", (
            round(float(stack["Final_Scale"]), 4),
            round(float(stack["Final_Intercept"]), 6),
        ))
        print("  Stacking Meta3低层增益：", round(float(stack.get("Meta3_Gain", stack.get("Meta_Gain", 0.0))), 6))
        print("  Stacking第四层分区模式：", stack["Meta4_Mode"])
        print("  Stacking第四层 bins / 收缩 / 平滑 / 截断：", (
            stack["Meta4_Level_Bins"], stack["Meta4_Secondary_Bins"],
            round(float(stack["Meta4_Shrinkage"]), 4), round(float(stack["Meta4_Smoothing"]), 4),
            round(float(stack["Meta4_Clip_Multiplier"]), 4),
        ))
        print("  Stacking Meta4总增益 / 增量增益：", (
            round(float(stack["Meta4_Gain"]), 6), round(float(stack["Meta4_Incremental_Gain"]), 6)
        ))
        print("  Stacking第五层 blend / delta截断 / scale / intercept：", (
            round(float(stack["Meta5_Blend"]), 4), round(float(stack["Meta5_Delta_Clip"]), 4),
            round(float(stack["Meta5_Scale"]), 4), round(float(stack["Meta5_Intercept"]), 6),
        ))
        print("  Stacking Meta5总增益 / 增量增益：", (
            round(float(stack["Meta5_Gain"]), 6), round(float(stack["Meta5_Incremental_Gain"]), 6)
        ))
        print("  Stacking最终严格Meta增益：", round(float(stack["Meta_Gain"]), 6))
        print("  Stacking正增益重复次数：", f"{stack['Positive_Repeats']}/{STACK_META_REPEATS}")
        print("  Stacking最差重复增益：", round(float(stack["Worst_Repeat_Gain"]), 6))
        print("  Stacking第一层候选审计：")
        stage1_columns = [
            "L2", "Stage1_Shrinkage", "R2", "Meta_Gain",
            "Positive_Repeats", "Worst_Repeat_Gain",
        ]
        print(stack["Stage1_Candidate_Table"][stage1_columns].head(6).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))
        print("  Stacking第二层候选审计：")
        audit_columns = [
            "Corrector", "Residual_Strength", "Clip_Multiplier", "R2", "Meta_Gain",
            "Positive_Repeats", "Worst_Repeat_Gain", "Positive_Folds", "Total_Folds",
        ]
        print(stack["Candidate_Table"][audit_columns].head(8).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))
        print("  Stacking第三层门控与尺度候选审计：")
        gate_columns = [
            "Gate_Expert", "Gate_C", "Gate_Threshold", "Gate_Strength",
            "Gate_Clip_Multiplier", "Final_Scale", "Average_Gate_Rate",
            "R2", "Meta_Gain", "Positive_Repeats", "Worst_Repeat_Gain",
        ]
        print(stack["Meta3_Candidate_Table"][gate_columns].head(10).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))
        print("  Stacking第四层分区残差候选审计：")
        meta4_columns = [
            "Regime_Mode", "Level_Bins", "Secondary_Bins", "Regime_Shrinkage",
            "Regime_Smoothing", "Regime_Clip_Multiplier", "R2", "Total_Gain",
            "Incremental_Gain", "Positive_Repeats", "Worst_Repeat_Gain",
        ]
        print(stack["Meta4_Candidate_Table"][meta4_columns].head(10).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))
        print("  Stacking第五层稳定收缩候选审计：")
        meta5_columns = [
            "Final_Blend", "Final_Delta_Clip", "Final_Scale5", "Average_Intercept5",
            "R2", "Total_Gain", "Incremental_Gain", "Positive_Repeats", "Worst_Repeat_Gain",
        ]
        print(stack["Meta5_Candidate_Table"][meta5_columns].head(10).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))
        print("  Stacking 第一层基础权重：", {
            name: round(float(weight), 4)
            for name, weight in zip(BASE_MODEL_NAMES, stack["Weights"])
        })

        stacking_records.append({
            "Outer_Fold": outer_fold,
            "Safety_Fallback": stack["Safety_Fallback"],
            "Champion_Guard_Triggered": stack["Champion_Guard_Triggered"],
            "Champion_Guard_Strong_Evidence_Override": stack["Champion_Guard_Strong_Evidence_Override"],
            "Champion_Guard_Reasons": stack["Champion_Guard_Reasons"],
            "Best_Base_Model_TrainOOF": stack["Best_Base_Model_TrainOOF"],
            "GBR_Gap_To_Best_TrainOOF": stack["GBR_Gap_To_Best_TrainOOF"],
            "Positive_Partner_Weight_PreGuard": stack["Positive_Partner_Weight_PreGuard"],
            "Absolute_Partner_Weight_PreGuard": stack["Absolute_Partner_Weight_PreGuard"],
            "PreGuard_Meta_Gain": stack["PreGuard_Meta_Gain"],
            "Soft_Shrinkage": stack["Soft_Shrinkage"],
            "Fallback_Model": stack["Fallback_Model"],
            "Selected_Models": ", ".join(stack["Selected_Models"]),
            "L2": stack["Selected_L2"],
            "Stage1_Shrinkage": stack["Stage1_Shrinkage"],
            "Bias": stack["Bias"],
            "GBR750_Effective_Weight": stack["GBR750_Weight"],
            "Residual_Corrector": stack["Residual_Corrector"],
            "Residual_Strength": stack["Residual_Strength"],
            "Residual_Clip_Multiplier": stack["Residual_Clip_Multiplier"],
            "Meta2_Gain": stack["Meta2_Gain"],
            "Gate_Expert": stack["Gate_Expert"],
            "Gate_C": stack["Gate_C"],
            "Gate_Threshold": stack["Gate_Threshold"],
            "Gate_Strength": stack["Gate_Strength"],
            "Gate_Clip_Multiplier": stack["Gate_Clip_Multiplier"],
            "Final_Scale": stack["Final_Scale"],
            "Final_Intercept": stack["Final_Intercept"],
            "Meta3_Gain": stack.get("Meta3_Gain", np.nan),
            "Meta4_Mode": stack["Meta4_Mode"],
            "Meta4_Level_Bins": stack["Meta4_Level_Bins"],
            "Meta4_Secondary_Bins": stack["Meta4_Secondary_Bins"],
            "Meta4_Shrinkage": stack["Meta4_Shrinkage"],
            "Meta4_Smoothing": stack["Meta4_Smoothing"],
            "Meta4_Clip_Multiplier": stack["Meta4_Clip_Multiplier"],
            "Meta4_Gain": stack["Meta4_Gain"],
            "Meta4_Incremental_Gain": stack["Meta4_Incremental_Gain"],
            "Meta5_Blend": stack["Meta5_Blend"],
            "Meta5_Delta_Clip": stack["Meta5_Delta_Clip"],
            "Meta5_Scale": stack["Meta5_Scale"],
            "Meta5_Intercept": stack["Meta5_Intercept"],
            "Meta5_Gain": stack["Meta5_Gain"],
            "Meta5_Incremental_Gain": stack["Meta5_Incremental_Gain"],
            "Final_Meta_Gain": stack["Meta_Gain"],
            "Positive_Repeats": stack["Positive_Repeats"],
            "Worst_Repeat_Gain": stack["Worst_Repeat_Gain"],
            **{
                f"Weight_{name}": weight
                for name, weight in zip(BASE_MODEL_NAMES, stack["Weights"])
            },
        })

        for name in BASE_MODEL_NAMES:
            parameter_records.append({
                "Outer_Fold": outer_fold,
                "Model": name,
                "Inner_Objective": model_details[name]["Objective"],
                "Inner_OOF_R2": model_details[name]["OOF_R2"],
                "Inner_OOF_RMSE": model_details[name]["OOF_RMSE"],
                "Params": str(model_details[name]["Params"]),
            })

        # 保存最小必要信息，供严格删一行重训诊断使用。
        outer_fold_artifacts.append({
            "Outer_Fold": outer_fold,
            "Train_Index": train_index.copy(),
            "Test_Index": test_index.copy(),
            "Mode": dict(mode),
            "GBR_Params": dict(model_details[STACK_ANCHOR_NAME]["Params"]),
            "GBR_Seed": int(outer_fold * 1000 + 6),
            "GBR_Test_Prediction": np.asarray(
                test_prediction[STACK_ANCHOR_NAME], dtype=float
            ).copy(),
        })

        rows = []
        for name in PRIMARY_MODEL_NAMES:
            train_metric = metrics(y_train, train_oof[name])
            test_metric = metrics(y_test, test_prediction[name])
            raw_metric = metrics(
                raw_test, inverse_target(test_prediction[name])
            )
            record = {
                "Fold": outer_fold,
                "Model": name,
                "Train_OOF_R2_Log": train_metric["R2"],
                "Test_R2_Log": test_metric["R2"],
                "Test_RMSE_Log": test_metric["RMSE"],
                "Test_MAE_Log": test_metric["MAE"],
                "Test_R2_Raw": raw_metric["R2"],
                "Test_RMSE_Raw": raw_metric["RMSE"],
                "Test_MAE_Raw": raw_metric["MAE"],
            }
            rows.append(record)
            fold_records.append(record)
            oof_predictions[name][test_index] = test_prediction[name]

        # GBR750只作为Stacking内部输入保存，不进入主表排名。
        oof_predictions[STACK_ANCHOR_NAME][test_index] = test_prediction[STACK_ANCHOR_NAME]

        fold_table = pd.DataFrame(rows)
        print(fold_table.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ))

        best_row = fold_table.sort_values(
            ["Test_R2_Log", "Test_RMSE_Log"], ascending=[False, True]
        ).iloc[0]
        print(
            f"\n  本折第一名（log10目标）：{best_row['Model']}，"
            f"R2={best_row['Test_R2_Log']:.4f}"
        )
        print(f"  本折耗时：{time.time() - fold_start:.1f} 秒")

    fold_frame = pd.DataFrame(fold_records)

    section("4. 五折均值与标准差（主表模型，主指标为log10 CCC）")
    summary_rows = []
    for name in PRIMARY_MODEL_NAMES:
        subset = fold_frame[fold_frame["Model"] == name]
        summary_rows.append({
            "Model": name,
            "Mean_Train_OOF_R2_Log": subset["Train_OOF_R2_Log"].mean(),
            "Mean_Test_R2_Log": subset["Test_R2_Log"].mean(),
            "SD_Test_R2_Log": subset["Test_R2_Log"].std(ddof=1),
            "Worst_Test_R2_Log": subset["Test_R2_Log"].min(),
            "Mean_Test_RMSE_Log": subset["Test_RMSE_Log"].mean(),
            "Mean_Test_MAE_Log": subset["Test_MAE_Log"].mean(),
            "Mean_Test_R2_Raw": subset["Test_R2_Raw"].mean(),
            "Mean_Test_RMSE_Raw": subset["Test_RMSE_Raw"].mean(),
            "Mean_Test_MAE_Raw": subset["Test_MAE_Raw"].mean(),
        })
    summary_frame = pd.DataFrame(summary_rows).sort_values(
        "Mean_Test_R2_Log", ascending=False
    )
    print(summary_frame.to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))

    def build_overall_table(model_names: Sequence[str]) -> pd.DataFrame:
        rows = []
        for model_name in model_names:
            log_metric = metrics(y, oof_predictions[model_name])
            raw_metric = metrics(y_raw, inverse_target(oof_predictions[model_name]))
            rows.append({
                "Model": model_name,
                "R2_Log": log_metric["R2"],
                "RMSE_Log": log_metric["RMSE"],
                "MAE_Log": log_metric["MAE"],
                "R2_Raw": raw_metric["R2"],
                "RMSE_Raw": raw_metric["RMSE"],
                "MAE_Raw": raw_metric["MAE"],
            })
        table = pd.DataFrame(rows).sort_values(
            ["R2_Log", "RMSE_Log"], ascending=[False, True]
        ).reset_index(drop=True)
        table.insert(0, "Rank", np.arange(1, len(table) + 1))
        table["Gap_to_Previous_R2_Log"] = (
            table["R2_Log"].shift(1) - table["R2_Log"]
        ).fillna(0.0)
        return table

    section("5. 主实验表：五个单模型 vs 内部GBR750六模型Stacking")
    primary_overall = build_overall_table(PRIMARY_MODEL_NAMES)
    print(primary_overall.to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))

    section("6. 主表模型严格外层 OOF 预测相关性")
    prediction_frame = pd.DataFrame({
        name: oof_predictions[name] for name in PRIMARY_MODEL_NAMES
    })
    print(prediction_frame.corr().to_string(
        float_format=lambda value: f"{value:.3f}"
    ))

    section("7. 每折模型参数与 Stacking 权重")
    print("五个公开单模型与Stacking内部GBR750每折严格OOF配置：")
    print(pd.DataFrame(parameter_records).to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print("\nStacking每折权重：")
    print(pd.DataFrame(stacking_records).to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))

    if RUN_ROW_DIAGNOSTICS:
        run_row_diagnostics(
            X=X,
            y=y,
            y_raw=y_raw,
            strict_groups=strict_groups,
            oof_predictions=oof_predictions,
            metadata=metadata,
            outer_fold_artifacts=outer_fold_artifacts,
        )

    section("11. 方法说明")
    print("1. 主表五个单模型为RF、ExtraTrees、LGBM、XGBoost和GBR；模型排名中统一显示GBR。")
    print("2. 公开GBR保持原150树固定参数，既单独进入主表，也作为Stacking内部伙伴模型。")
    print("3. Stacking内部六个一级模型为RF、ExtraTrees、LGBM、XGBoost、GBR和GBR750。")
    print("4. GBR750只作为Stacking内部强锚点和安全回退，不进入主表、不单独排名。")
    print("5. Stacking从GBR750严格Group-OOF锚点出发，利用其余五个模型做线性补偿、残差校准、动态门控、分区修正与稳定收缩。")
    print("6. 除Stacking内部成员、锚点命名、伙伴权重、门控与Champion Guard外，五个公开单模型参数保持原样。")
    print("7. 所有Stacking选择只读取当前外层训练侧重复Group Meta-CV；外层测试标签不参与模型、权重、门控或Guard选择。")
    print("8. 同一文献或完全相同输入签名不会跨训练/测试折；程序只输出控制台，不自动修改数据或保存结果文件。")
    print(f"\n总运行时间：{time.time() - start_time:.1f} 秒")
    print("=" * 156)


if __name__ == "__main__":
    main()    