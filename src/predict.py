from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.business_rules import (
    calculate_overrun_pct,
    calculate_traditional_eac,
    calculate_vac,
    get_recommendation,
    get_risk_band,
)
from src.data_generator import save_dataset
from src.train import DATA_PATH, train_models


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def _load_artifacts(rebuild_on_failure: bool = True) -> dict[str, Any]:
    try:
        return {
            "eac_model": joblib.load(MODELS_DIR / "eac_model.pkl"),
            "risk_model": joblib.load(MODELS_DIR / "risk_model.pkl"),
            "preprocessor": joblib.load(MODELS_DIR / "preprocessor.pkl"),
            "feature_columns": joblib.load(MODELS_DIR / "feature_columns.pkl"),
        }
    except Exception:
        # Hosted environments may fail to unpickle locally-trained artifacts
        # (version/module-path mismatch). Rebuild once and retry.
        if not rebuild_on_failure:
            raise
        save_dataset(DATA_PATH, n_rows=700)
        train_models()
        return _load_artifacts(rebuild_on_failure=False)


def prepare_features(input_data: dict[str, Any]) -> pd.DataFrame:
    data = dict(input_data)
    ac = float(data["AC"])
    ev = float(data["EV"])
    pv = float(data["PV"])

    data["CPI"] = ev / ac if ac else 1.0
    data["SPI"] = ev / pv if pv else 1.0
    data["cost_variance"] = ev - ac
    data["schedule_variance"] = ev - pv

    return pd.DataFrame([data])


def predict_program_risk(input_data: dict[str, Any]) -> dict[str, Any]:
    artifacts = _load_artifacts()
    eac_model = artifacts["eac_model"]
    risk_model = artifacts["risk_model"]
    preprocessor = artifacts["preprocessor"]
    feature_columns = artifacts["feature_columns"]

    input_df = prepare_features(input_data)
    model_input = input_df[feature_columns]
    transformed = preprocessor.transform(model_input)

    ai_eac = float(eac_model.predict(transformed)[0])
    risk_probability = float(risk_model.predict_proba(transformed)[0][1])

    bac = float(input_data["BAC"])
    ac = float(input_data["AC"])
    ev = float(input_data["EV"])
    cpi = float(input_df["CPI"].iloc[0])

    traditional_eac = calculate_traditional_eac(ac=ac, bac=bac, ev=ev, cpi=cpi)
    overrun_pct = calculate_overrun_pct(bac=bac, ai_eac=ai_eac)
    vac = calculate_vac(bac=bac, ai_eac=ai_eac)
    risk_band = get_risk_band(risk_probability=risk_probability, overrun_pct=overrun_pct)
    recommendation = get_recommendation(risk_band)

    return {
        "input_df": input_df,
        "traditional_eac": traditional_eac,
        "ai_eac": ai_eac,
        "risk_probability": risk_probability,
        "risk_band": risk_band,
        "overrun_pct": overrun_pct,
        "vac": vac,
        "recommendation": recommendation,
    }
