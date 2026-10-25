# -*- coding: utf-8 -*-
"""Post-training smoke test with explicit NON-EXPERIMENTAL sanity inputs.

Usage:
    python smoke_test.py

The sample is constructed from training medians and first observed categories
solely to confirm the real exported model is callable. NOT a validation score.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from predictor import read_bundle, predict_ccc

bundle_path = Path(__file__).resolve().parent / "models" / "ccc_stacking.joblib"
bundle = read_bundle(bundle_path)
row = {}
for col in bundle["model_columns"]:
    if col in bundle["numeric_columns"]:
        row[col] = float(bundle["numeric_summary"][col]["median"])
    else:
        row[col] = bundle["category_options"][col][0]
ccc, log_ccc = predict_ccc(bundle, pd.DataFrame([row]))
assert np.isfinite(ccc[0]) and ccc[0] > 0
assert np.isclose(np.log10(ccc[0]), log_ccc[0], rtol=1e-8, atol=1e-8)
print("PASS — exported Stacking predictor is callable.")
print("SANITY-CHECK ONLY — synthetic combination of medians and first categories.")
print(f"CCC={ccc[0]:.6g} mM; log10(CCC)={log_ccc[0]:.6f}")
print("Do not report this artificial input or prediction as an experimental result.")
