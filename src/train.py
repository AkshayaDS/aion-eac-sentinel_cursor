from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data_generator import save_dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_program_data.csv"
MODELS_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "BAC",
    "AC",
    "EV",
    "PV",
    "CPI",
    "SPI",
    "cost_variance",
    "schedule_variance",
    "subcontractor_delay_days",
    "change_orders_count",
    "material_cost_inflation_pct",
    "supplier_risk_score",
    "months_remaining",
    "historical_rebaseline_count",
]
CATEGORICAL_FEATURES = ["program_type", "contract_type", "program_phase"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        save_dataset(DATA_PATH, n_rows=500)
    return pd.read_csv(DATA_PATH)


def train_models() -> dict:
    df = load_training_data()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if "overrun_flag" not in df.columns:
        df["overrun_flag"] = (df["final_eac"] > df["BAC"] * 1.02).astype(int)

    X = df[FEATURE_COLUMNS].copy()
    y_reg = df["final_eac"].copy()
    y_clf = df["overrun_flag"].copy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=42,
        stratify=y_clf,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    reg_model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    reg_model.fit(X_train_t, y_reg_train)

    clf_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf_model.fit(X_train_t, y_clf_train)

    reg_pred = reg_model.predict(X_test_t)
    clf_pred = clf_model.predict(X_test_t)
    clf_prob = clf_model.predict_proba(X_test_t)[:, 1]

    metrics = {
        "reg_mae": mean_absolute_error(y_reg_test, reg_pred),
        "reg_r2": r2_score(y_reg_test, reg_pred),
        "clf_accuracy": accuracy_score(y_clf_test, clf_pred),
        "clf_roc_auc": roc_auc_score(y_clf_test, clf_prob),
    }

    feature_names = preprocessor.get_feature_names_out().tolist()

    joblib.dump(reg_model, MODELS_DIR / "eac_model.pkl")
    joblib.dump(clf_model, MODELS_DIR / "risk_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    joblib.dump(FEATURE_COLUMNS, MODELS_DIR / "feature_columns.pkl")
    joblib.dump(feature_names, MODELS_DIR / "encoded_feature_names.pkl")
    joblib.dump(
        {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODELS_DIR / "model_metadata.pkl",
    )

    return metrics


if __name__ == "__main__":
    train_metrics = train_models()
    print("Training complete.")
    print({k: round(v, 4) for k, v in train_metrics.items()})
