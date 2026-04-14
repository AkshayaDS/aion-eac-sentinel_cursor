# AION EAC Sentinel

AI-powered defense program cost overrun predictor for early EAC risk detection.

## What this app does

- Predicts **AI-based Final EAC** using `XGBoost Regressor`
- Predicts **overrun probability** using `Logistic Regression`
- Compares **Traditional EVMS EAC vs AI EAC**
- Generates **risk band** (Low / Medium / High), **VAC**, and recommended actions
- Explains drivers with **SHAP** top feature impacts
- Shows a **portfolio-level risk table** for prioritization

## Project structure

- `app.py` - Streamlit dashboard UI
- `requirements.txt` - Python dependencies
- `data/synthetic_program_data.csv` - generated synthetic dataset
- `src/data_generator.py` - synthetic defense program data generator
- `src/train.py` - model training and artifact persistence
- `src/predict.py` - prediction pipeline for single program intake
- `src/explain.py` - SHAP explainability for EAC model output
- `src/business_rules.py` - EVMS, risk, VAC, recommendation logic
- `models/` - persisted ML artifacts

## Local run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models (also creates synthetic dataset if missing):

```bash
python -m src.train
```

3. Start Streamlit app:

```bash
streamlit run app.py
```

## Notes

- Synthetic data follows defense-finance style EVMS patterns (BAC, AC, EV, PV, CPI, SPI, delays, inflation, supplier risk, rebaseline count).
- The dashboard framing aligns to enterprise defense finance use-cases from your provided reference content.
