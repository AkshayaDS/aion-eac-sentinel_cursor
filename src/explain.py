from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def _load_artifacts() -> dict[str, Any]:
    return {
        "eac_model": joblib.load(MODELS_DIR / "eac_model.pkl"),
        "preprocessor": joblib.load(MODELS_DIR / "preprocessor.pkl"),
        "feature_columns": joblib.load(MODELS_DIR / "feature_columns.pkl"),
        "encoded_feature_names": joblib.load(MODELS_DIR / "encoded_feature_names.pkl"),
    }


def explain_eac_prediction(input_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    artifacts = _load_artifacts()
    eac_model = artifacts["eac_model"]
    preprocessor = artifacts["preprocessor"]
    feature_columns = artifacts["feature_columns"]
    encoded_feature_names = artifacts["encoded_feature_names"]

    transformed = preprocessor.transform(input_df[feature_columns])
    # Use XGBoost native SHAP contributions to avoid SHAP/XGBoost base_score parsing issues
    booster = eac_model.get_booster()
    dmat = xgb.DMatrix(transformed)
    contribs = booster.predict(dmat, pred_contribs=True)  # (n_rows, n_features + 1 bias)
    row_contribs = contribs[0]
    row_shap = row_contribs[:-1]  # drop bias term

    explain_df = pd.DataFrame(
        {"feature": encoded_feature_names, "shap_value": row_shap}
    )
    explain_df["impact_abs"] = explain_df["shap_value"].abs()
    explain_df = explain_df.sort_values("impact_abs", ascending=False).head(top_n)
    return explain_df
