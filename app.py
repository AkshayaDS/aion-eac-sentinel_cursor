from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.business_rules import generate_executive_summary
from src.data_generator import save_dataset
from src.explain import explain_eac_prediction
from src.predict import predict_program_risk
from src.train import train_models


st.set_page_config(page_title="AION EAC Sentinel", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_program_data.csv"
MODELS_DIR = ROOT / "models"
DEFAULT_PAYLOAD = {
    "program_name": "NGAP Phase 2",
    "program_type": "Missile",
    "contract_type": "Cost Plus",
    "program_phase": "Build",
    "BAC": 120_000_000.0,
    "AC": 68_000_000.0,
    "EV": 60_000_000.0,
    "PV": 65_000_000.0,
    "subcontractor_delay_days": 35,
    "change_orders_count": 5,
    "material_cost_inflation_pct": 7.0,
    "supplier_risk_score": 62.0,
    "months_remaining": 14,
    "historical_rebaseline_count": 1,
}


def ensure_pipeline_ready() -> None:
    required = [
        MODELS_DIR / "eac_model.pkl",
        MODELS_DIR / "risk_model.pkl",
        MODELS_DIR / "preprocessor.pkl",
        MODELS_DIR / "feature_columns.pkl",
        MODELS_DIR / "encoded_feature_names.pkl",
    ]
    if all(path.exists() for path in required):
        return
    save_dataset(DATA_PATH, n_rows=500)
    train_models()


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def status_badge(risk: str) -> str:
    return "🔴 Escalate" if risk == "High" else "🟡 Watchlist" if risk == "Medium" else "🟢 Healthy"


def as_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


def as_int(value: object, default: int) -> int:
    return int(round(as_float(value, float(default))))


def sanitize_payload(raw: dict) -> dict:
    safe = dict(DEFAULT_PAYLOAD)
    safe.update(raw)
    safe["BAC"] = as_float(safe.get("BAC"), DEFAULT_PAYLOAD["BAC"])
    safe["AC"] = as_float(safe.get("AC"), DEFAULT_PAYLOAD["AC"])
    safe["EV"] = as_float(safe.get("EV"), DEFAULT_PAYLOAD["EV"])
    safe["PV"] = as_float(safe.get("PV"), DEFAULT_PAYLOAD["PV"])
    safe["subcontractor_delay_days"] = as_int(
        safe.get("subcontractor_delay_days"), DEFAULT_PAYLOAD["subcontractor_delay_days"]
    )
    safe["change_orders_count"] = as_int(
        safe.get("change_orders_count"), DEFAULT_PAYLOAD["change_orders_count"]
    )
    safe["material_cost_inflation_pct"] = as_float(
        safe.get("material_cost_inflation_pct"), DEFAULT_PAYLOAD["material_cost_inflation_pct"]
    )
    safe["supplier_risk_score"] = as_float(
        safe.get("supplier_risk_score"), DEFAULT_PAYLOAD["supplier_risk_score"]
    )
    safe["months_remaining"] = as_int(
        safe.get("months_remaining"), DEFAULT_PAYLOAD["months_remaining"]
    )
    safe["historical_rebaseline_count"] = as_int(
        safe.get("historical_rebaseline_count"), DEFAULT_PAYLOAD["historical_rebaseline_count"]
    )
    return safe


def do_prediction(payload: dict) -> None:
    pred = predict_program_risk(payload)
    explain_df = explain_eac_prediction(pred["input_df"], top_n=10)
    summary = generate_executive_summary(
        payload["program_name"], pred["risk_band"], pred["ai_eac"], payload["BAC"], ", ".join(explain_df["feature"].head(3).tolist())
    )
    st.session_state["latest_payload"] = payload
    st.session_state["latest_result"] = pred
    st.session_state["latest_explain_df"] = explain_df
    st.session_state["latest_summary"] = summary
    st.session_state["draft_payload"] = payload

def bootstrap_prediction_state() -> None:
    if "draft_payload" not in st.session_state:
        st.session_state["draft_payload"] = sanitize_payload(dict(DEFAULT_PAYLOAD))
    if "latest_result" in st.session_state:
        return
    payload = sanitize_payload(st.session_state["draft_payload"])
    do_prediction(payload)


def inject_theme(theme_mode: str) -> None:
    if theme_mode == "Dark":
        bg = "#06122a"
        panel = "#0d1e3b"
        accent = "#00c0f3"
        muted = "#b7c8e5"
        text = "#e9f1ff"
        border = "rgba(138,166,205,0.28)"
        hero_bg = "linear-gradient(100deg, #153273 0%, #006eb3 55%, #0093c4 100%)"
    else:
        bg = "#eef3fb"
        panel = "#ffffff"
        accent = "#0047bb"
        muted = "#334a68"
        text = "#0a1f44"
        border = "rgba(15,54,118,0.16)"
        hero_bg = "linear-gradient(100deg, #184cc2 0%, #008cd6 55%, #00bae3 100%)"
    st.markdown(
        f"""
        <style>
        .stApp, .stApp > header, .main {{
            background-color: {bg} !important;
            color: {text} !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {panel} !important;
            border-right: 1px solid {border} !important;
        }}
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText {{
            color: {text} !important;
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            color: {text} !important;
        }}
        .stTabs [aria-selected="true"] {{
            border-bottom: 2px solid {accent} !important;
            font-weight: 700 !important;
        }}
        input, textarea {{
            color: {text} !important;
            background-color: {panel} !important;
        }}
        div[data-baseweb="select"] > div {{
            color: {text} !important;
            background-color: {panel} !important;
            border-color: {border} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {text} !important;
            font-size: clamp(1rem, 2vw, 1.4rem) !important;
        }}
        [data-testid="stMetricValue"] > div {{
            font-size: clamp(1rem, 2vw, 1.4rem) !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {muted} !important;
        }}
        .stButton > button {{
            background: {accent} !important;
            color: {"#041226" if theme_mode == "Dark" else "#ffffff"} !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }}
        .stButton > button:hover {{
            opacity: 0.92;
            transform: translateY(-1px);
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid {border};
            border-radius: 14px;
            padding: 1rem;
            background: {panel};
        }}
        .hero-card {{
            border: 1px solid {border};
            border-radius: 14px;
            padding: 16px 18px;
            background: {hero_bg};
            margin-bottom: 12px;
        }}
        .hero-title {{
            font-size: clamp(1.2rem, 2vw, 1.5rem);
            color: #ffffff;
            font-weight: 700;
        }}
        .hero-sub {{
            color: #f0f4f8;
            font-size: clamp(0.95rem, 1.3vw, 1.1rem);
            margin-top: 8px;
            line-height: 1.5;
        }}
        .moving-wrap {{
            overflow: hidden;
            width: 100%;
            border: 1px solid {border};
            border-radius: 14px;
            background: {panel};
            padding: 8px 0;
            margin: 8px 0 16px 0;
        }}
        .moving-track {{
            display: flex;
            width: max-content;
            animation: slide 28s linear infinite;
        }}
        .moving-card {{
            min-width: 260px;
            margin: 0 10px;
            padding: 10px 14px;
            border-radius: 12px;
            border: 1px solid {border};
            background: {"#0a2750" if theme_mode == "Dark" else "#f6f9ff"};
        }}
        .moving-label {{
            font-size: 0.82rem;
            color: {muted};
            margin-bottom: 4px;
        }}
        .moving-value {{
            font-size: 1.05rem;
            font-weight: 700;
            color: {accent};
        }}
        @keyframes slide {{
            0% {{ transform: translateX(0); }}
            100% {{ transform: translateX(-50%); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_sample_programs() -> dict[str, dict]:
    return {
        "NGAP Phase 2 (High Risk)": {**DEFAULT_PAYLOAD},
        "Radar Modernization (Medium Risk)": {
            **DEFAULT_PAYLOAD,
            "program_name": "Radar Modernization",
            "program_type": "Radar",
            "contract_type": "Fixed Price",
            "program_phase": "Test",
            "BAC": 80_000_000.0,
            "AC": 49_000_000.0,
            "EV": 46_000_000.0,
            "PV": 48_000_000.0,
            "subcontractor_delay_days": 18,
            "change_orders_count": 4,
            "material_cost_inflation_pct": 4.0,
            "supplier_risk_score": 45.0,
            "months_remaining": 12,
            "historical_rebaseline_count": 1,
        },
        "Sustainment Alpha (Low Risk)": {
            **DEFAULT_PAYLOAD,
            "program_name": "Sustainment Alpha",
            "program_type": "Sustainment",
            "contract_type": "Cost Plus",
            "program_phase": "Deployment",
            "BAC": 40_000_000.0,
            "AC": 21_000_000.0,
            "EV": 22_000_000.0,
            "PV": 21_500_000.0,
            "subcontractor_delay_days": 5,
            "change_orders_count": 1,
            "material_cost_inflation_pct": 2.3,
            "supplier_risk_score": 18.0,
            "months_remaining": 9,
            "historical_rebaseline_count": 0,
        },
    }


def portfolio_predictions(limit: int = 20) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH).head(limit).copy()
    out_rows = []
    for _, row in df.iterrows():
        payload = {
            "program_name": row["program_name"],
            "program_type": row["program_type"],
            "contract_type": row["contract_type"],
            "program_phase": row["program_phase"],
            "BAC": float(row["BAC"]),
            "AC": float(row["AC"]),
            "EV": float(row["EV"]),
            "PV": float(row["PV"]),
            "subcontractor_delay_days": int(row["subcontractor_delay_days"]),
            "change_orders_count": int(row["change_orders_count"]),
            "material_cost_inflation_pct": float(row["material_cost_inflation_pct"]),
            "supplier_risk_score": float(row["supplier_risk_score"]),
            "months_remaining": int(row["months_remaining"]),
            "historical_rebaseline_count": int(row["historical_rebaseline_count"]),
        }
        pred = predict_program_risk(payload)
        cpi = payload["EV"] / payload["AC"] if payload["AC"] else 1.0
        spi = payload["EV"] / payload["PV"] if payload["PV"] else 1.0
        out_rows.append(
            {
                "Program Name": payload["program_name"],
                "Program Type": payload["program_type"],
                "Contract Type": payload["contract_type"],
                "Phase": payload["program_phase"],
                "BAC": payload["BAC"],
                "AI EAC": pred["ai_eac"],
                "Traditional EAC": pred["traditional_eac"],
                "Overrun %": pred["overrun_pct"],
                "Risk": pred["risk_band"],
                "Risk Probability": pred["risk_probability"],
                "CPI": cpi,
                "SPI": spi,
                "Delay": payload["subcontractor_delay_days"],
                "Inflation %": payload["material_cost_inflation_pct"],
                "Recommended Action": pred["recommendation"],
                "Status Badge": status_badge(pred["risk_band"]),
            }
        )
    return pd.DataFrame(out_rows)


def show_alert_banner(portfolio_df: pd.DataFrame) -> None:
    escalations = portfolio_df[(portfolio_df["Risk"] == "High") | (portfolio_df["Overrun %"] > 5)]
    if len(escalations) > 0:
        st.warning(
            f"⚠️ {len(escalations)} programs exceed 5% projected overrun and require finance escalation this week."
        )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">HCLTech AI Force-aligned Decision Platform</div>
            <div class="hero-sub">
                Transforming defense program finance with AI-powered Estimate-at-Completion intelligence,
                overrun risk detection and actionable portfolio insights for enterprise leadership.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_moving_cards(portfolio_df: pd.DataFrame) -> None:
    total_programs = len(portfolio_df)
    high_risk = int((portfolio_df["Risk"] == "High").sum())
    exposure = portfolio_df["AI EAC"].sum() - portfolio_df["BAC"].sum()
    avg_cpi = portfolio_df["CPI"].mean()
    avg_spi = portfolio_df["SPI"].mean()
    cards = [
        ("Programs Monitored", f"{total_programs}"),
        ("High Risk Programs", f"{high_risk}"),
        ("Portfolio Exposure", money(exposure)),
        ("Average CPI", f"{avg_cpi:.2f}"),
        ("Average SPI", f"{avg_spi:.2f}"),
        ("Escalation Candidates", f"{int((portfolio_df['Overrun %'] > 5).sum())}"),
    ]
    card_html = "".join(
        [
            f'<div class="moving-card"><div class="moving-label">{label}</div><div class="moving-value">{value}</div></div>'
            for label, value in cards + cards
        ]
    )
    st.markdown(
        f'<div class="moving-wrap"><div class="moving-track">{card_html}</div></div>',
        unsafe_allow_html=True,
    )


def page_executive_overview(portfolio_df: pd.DataFrame) -> None:
    st.subheader("1) Executive Overview Dashboard")
    show_alert_banner(portfolio_df)

    total_programs = len(portfolio_df)
    high_risk = int((portfolio_df["Risk"] == "High").sum())
    total_bac = portfolio_df["BAC"].sum()
    total_ai_eac = portfolio_df["AI EAC"].sum()
    exposure = total_ai_eac - total_bac
    avg_cpi = portfolio_df["CPI"].mean()
    escalations = int((portfolio_df["Status Badge"] == "🔴 Escalate").sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Programs Monitored", total_programs)
    c2.metric("High Risk Programs", high_risk)
    c3.metric("Portfolio BAC", money(total_bac))
    c4.metric("Predicted Portfolio EAC", money(total_ai_eac))
    c5.metric("Overrun Exposure", money(exposure))
    c6.metric("Escalations Recommended", escalations)
    st.caption(f"Average CPI: {avg_cpi:.2f} | Average SPI: {portfolio_df['SPI'].mean():.2f}")

    a, b = st.columns(2)
    with a:
        risk_dist = portfolio_df["Risk"].value_counts().reset_index()
        risk_dist.columns = ["Risk", "Count"]
        fig = px.pie(risk_dist, values="Count", names="Risk", hole=0.55, title="Portfolio Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        top_risk = portfolio_df.sort_values("Overrun %", ascending=False).head(10)
        fig = px.bar(top_risk, x="Overrun %", y="Program Name", orientation="h", title="Top 10 At-Risk Programs", color="Risk")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    c, d = st.columns(2)
    with c:
        compare = pd.DataFrame(
            {
                "Metric": ["Total BAC", "Total Traditional EAC", "Total AI EAC"],
                "Value": [total_bac, portfolio_df["Traditional EAC"].sum(), total_ai_eac],
            }
        )
        fig = px.bar(compare, x="Metric", y="Value", text="Value", title="BAC vs Traditional EAC vs AI EAC")
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with d:
        fig = px.scatter(
            portfolio_df,
            x="CPI",
            y="SPI",
            size="BAC",
            color="Risk",
            hover_name="Program Name",
            title="Program Health Matrix (CPI vs SPI)",
        )
        st.plotly_chart(fig, use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    view_high = b1.button("View High Risk Programs", key="overview_view_high")
    b2.download_button("Export Executive Snapshot", data=portfolio_df.to_csv(index=False), file_name="executive_snapshot.csv", mime="text/csv")
    open_scenario = b3.button("Open Scenario Simulator", key="overview_open_scenario")
    generate_brief = b4.button("Generate Brief", key="overview_generate_brief")
    if view_high:
        st.dataframe(
            portfolio_df[portfolio_df["Risk"] == "High"].sort_values("Overrun %", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    if open_scenario:
        st.info("Open the `Scenario Simulator` tab to run live what-if analysis for the current selected program.")
    if generate_brief:
        st.success(st.session_state.get("latest_summary", "Executive brief is ready after a prediction run."))


def page_program_intake() -> None:
    st.subheader("2) Program Intake & Prediction")
    samples = get_sample_programs()

    left, right = st.columns([1.2, 1])
    with left:
        sample_choice = st.selectbox("Quick Select Sample", list(samples.keys()))
        if st.button("Load Sample Program"):
            st.session_state["draft_payload"] = sanitize_payload(dict(samples[sample_choice]))
            do_prediction(st.session_state["draft_payload"])
            st.rerun()
        if st.button("Reset Inputs"):
            st.session_state["draft_payload"] = sanitize_payload(dict(DEFAULT_PAYLOAD))
            do_prediction(st.session_state["draft_payload"])
            st.rerun()
        if "draft_payload" not in st.session_state:
            st.session_state["draft_payload"] = sanitize_payload(dict(samples[sample_choice]))
        draft = sanitize_payload(st.session_state["draft_payload"])

        st.markdown("**Section A: Program Info**")
        c1, c2 = st.columns(2)
        with c1:
            program_name = st.text_input("Program Name", value=draft["program_name"])
            program_type = st.selectbox("Program Type", ["Missile", "Radar", "Avionics", "ISR", "Sustainment"], index=["Missile", "Radar", "Avionics", "ISR", "Sustainment"].index(draft["program_type"]))
        with c2:
            contract_type = st.selectbox("Contract Type", ["Cost Plus", "Fixed Price"], index=["Cost Plus", "Fixed Price"].index(draft["contract_type"]))
            program_phase = st.selectbox("Program Phase", ["Design", "Build", "Test", "Deployment"], index=["Design", "Build", "Test", "Deployment"].index(draft["program_phase"]))

        st.markdown("**Section B: Financial Inputs**")
        f1, f2, f3, f4 = st.columns(4)
        bac = f1.number_input("BAC", min_value=1_000_000.0, value=float(draft["BAC"]))
        ac = f2.number_input("AC", min_value=100_000.0, value=float(draft["AC"]))
        ev = f3.number_input("EV", min_value=100_000.0, value=float(draft["EV"]))
        pv = f4.number_input("PV", min_value=100_000.0, value=float(draft["PV"]))

        st.markdown("**Section C: Operational Inputs**")
        o1, o2, o3 = st.columns(3)
        delay = o1.slider("Supplier Delay (days)", 0, 180, int(draft["subcontractor_delay_days"]))
        changes = o2.number_input("Change Orders Count", min_value=0, max_value=40, value=int(draft["change_orders_count"]))
        inflation = o3.slider("Material Inflation %", 0.0, 25.0, float(draft["material_cost_inflation_pct"]))
        p1, p2, p3 = st.columns(3)
        supplier_risk = p1.slider("Supplier Risk Score", 0.0, 100.0, float(draft["supplier_risk_score"]))
        months = p2.slider("Months Remaining", 1, 60, int(draft["months_remaining"]))
        rebaseline = p3.number_input("Rebaseline Count", min_value=0, max_value=10, value=int(draft["historical_rebaseline_count"]))

        save_btn = st.button("Save This Program")

        payload = {
            "program_name": program_name,
            "program_type": program_type,
            "contract_type": contract_type,
            "program_phase": program_phase,
            "BAC": bac,
            "AC": ac,
            "EV": ev,
            "PV": pv,
            "subcontractor_delay_days": delay,
            "change_orders_count": changes,
            "material_cost_inflation_pct": inflation,
            "supplier_risk_score": supplier_risk,
            "months_remaining": months,
            "historical_rebaseline_count": rebaseline,
        }
        st.session_state["draft_payload"] = sanitize_payload(payload)
        
        # Realtime predictions (without needing Predict EAC button)
        do_prediction(st.session_state["draft_payload"])

        if save_btn:
            st.session_state.setdefault("saved_programs", []).append(payload.copy())
            st.success(f"Saved program: {payload['program_name']}")

    with right:
        st.markdown("### Program Quick Actions")
        if st.button("Load NGAP Phase 2"):
            st.session_state["draft_payload"] = sanitize_payload(dict(samples["NGAP Phase 2 (High Risk)"]))
            do_prediction(st.session_state["draft_payload"])
            st.rerun()
        if st.button("Load Radar Modernization"):
            st.session_state["draft_payload"] = sanitize_payload(dict(samples["Radar Modernization (Medium Risk)"]))
            do_prediction(st.session_state["draft_payload"])
            st.rerun()
        if st.button("Load Sustainment Alpha"):
            st.session_state["draft_payload"] = sanitize_payload(dict(samples["Sustainment Alpha (Low Risk)"]))
            do_prediction(st.session_state["draft_payload"])
            st.rerun()


def require_latest() -> bool:
    if "latest_result" not in st.session_state:
        st.info("Run prediction from Program Intake first.")
        return False
    return True


def page_prediction_analysis() -> None:
    st.subheader("3) Prediction Analysis & EVMS Comparison")
    if not require_latest():
        return
    payload = st.session_state["latest_payload"]
    result = st.session_state["latest_result"]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("BAC", money(payload["BAC"]))
    c2.metric("Traditional EVMS EAC", money(result["traditional_eac"]))
    c3.metric("AI Predicted EAC", money(result["ai_eac"]))
    c4.metric("Overrun %", pct(result["overrun_pct"]))
    c5.metric("VAC", money(result["vac"]))
    c6.metric("Risk Level", status_badge(result["risk_band"]))

    tabs = st.tabs(["Financial View", "Schedule View", "Risk View", "AI View"])

    with tabs[0]:
        bars = pd.DataFrame(
            {"Metric": ["BAC", "Traditional EAC", "AI EAC"], "Value": [payload["BAC"], result["traditional_eac"], result["ai_eac"]]}
        )
        st.plotly_chart(px.bar(bars, x="Metric", y="Value", text="Value", title="Traditional EAC vs AI EAC vs BAC"), use_container_width=True)

        explain_df = st.session_state["latest_explain_df"]
        total_overrun = result["ai_eac"] - payload["BAC"]
        
        abs_sum = explain_df["impact_abs"].sum()
        if abs_sum != 0:
            scale_factor = total_overrun / abs_sum
            contributions = explain_df["impact_abs"] * scale_factor * (explain_df["shap_value"] / explain_df["impact_abs"].replace(0, 1))
        else:
            contributions = explain_df["shap_value"] * 0

        top_c = contributions.head(4)
        remainder = total_overrun - top_c.sum()

        x_labels = ["BAC"] + explain_df["feature"].head(4).tolist() + ["Other Factors", "Final AI EAC"]
        y_values = [payload["BAC"]] + top_c.tolist() + [remainder, result["ai_eac"]]
        measures = ["absolute"] + (["relative"] * 5) + ["total"]

        wf = go.Figure(
            go.Waterfall(
                x=x_labels,
                y=y_values,
                measure=measures,
            )
        )
        wf.update_layout(title="Dynamic Overrun Waterfall")
        st.plotly_chart(wf, use_container_width=True)

    with tabs[1]:
        months = list(range(1, payload["months_remaining"] + 1))
        
        # Real-time pacing values depending on actual values. Starts with Actual Cost instead of zeroing.
        planned = [payload["AC"] + (payload["BAC"] - payload["AC"]) * (i / max(1, payload["months_remaining"])) for i in months]
        traditional = [payload["AC"] + (result["traditional_eac"] - payload["AC"]) * (i / max(1, payload["months_remaining"])) for i in months]
        ai = [payload["AC"] + (result["ai_eac"] - payload["AC"]) * (i / max(1, payload["months_remaining"])) for i in months]
        
        trend_df = pd.DataFrame({"Month": months, "Planned Curve": planned, "Traditional Curve": traditional, "AI Projection": ai})
        st.plotly_chart(px.line(trend_df, x="Month", y=["Planned Curve", "Traditional Curve", "AI Projection"], title="Financial Burn Down Curve"), use_container_width=True)

    with tabs[2]:
        zone = "green" if result["overrun_pct"] < 2 else "yellow" if result["overrun_pct"] < 5 else "red"
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=result["overrun_pct"], title={"text": "Overrun % Gauge"}, gauge={"axis": {"range": [-5, 20]}, "bar": {"color": zone}, "steps": [{"range": [-5, 2], "color": "#d1f3da"}, {"range": [2, 5], "color": "#fff4bf"}, {"range": [5, 20], "color": "#f7c0c0"}]}))
        st.plotly_chart(gauge, use_container_width=True)
        st.success(f"Suggested action: {result['recommendation']}")

    with tabs[3]:
        drill = st.button("Drill into Risk Drivers", key="ai_drill_risk")
        run_what_if = st.button("Run What-If Scenario", key="ai_run_what_if")
        compare_hist = st.button("Compare to Historical Baseline", key="ai_compare_hist")
        escalate = st.button("Flag for Escalation", key="ai_flag_escalation")
        if drill:
            st.info("Risk driver detail prepared. Open `Risk Explainability` tab for top SHAP contributors.")
        if run_what_if:
            st.info("What-if flow available in `Scenario Simulator` tab.")
        if compare_hist:
            st.success(
                f"Baseline check: Traditional EAC {money(result['traditional_eac'])} vs AI EAC {money(result['ai_eac'])}."
            )
        if escalate:
            st.error("Escalation flag set for finance review within 7 days.")


def page_explainability() -> None:
    st.subheader("4) Risk Driver Explainability (SHAP)")
    if not require_latest():
        return
    result = st.session_state["latest_result"]
    explain_df = st.session_state["latest_explain_df"].copy().head(10)
    st.info(f"Prediction summary: Program is **{result['risk_band'].upper()} RISK** due to performance inefficiency and supply chain stress.")

    mode = st.toggle("Business View / Data Science View", value=True)
    if mode:
        st.caption("Business View enabled: plain-language driver interpretation.")
    else:
        st.caption("Data Science View enabled: raw SHAP contribution details.")

    left, right = st.columns(2)
    with left:
        fig = px.bar(explain_df.sort_values("impact_abs"), x="shap_value", y="feature", orientation="h", title="SHAP Feature Importance", color="shap_value", color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        explain_df["direction"] = explain_df["shap_value"].apply(lambda x: "Risk Up" if x > 0 else "Risk Down")
        fig = px.bar(explain_df, x="shap_value", y="feature", color="direction", orientation="h", title="Positive vs Negative Contributors")
        st.plotly_chart(fig, use_container_width=True)

    cards = explain_df.head(5)
    c1, c2, c3, c4, c5 = st.columns(5)
    cols = [c1, c2, c3, c4, c5]
    for i, (_, row) in enumerate(cards.iterrows()):
        cols[i % 5].metric(row["feature"], "High" if abs(row["shap_value"]) > cards["shap_value"].abs().median() else "Medium")

    with st.expander("Technical Details"):
        st.dataframe(explain_df[["feature", "shap_value", "impact_abs"]], use_container_width=True, hide_index=True)

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Explain in Business Language"):
        st.info("Business language explanation: the key drivers are supply chain stress and past performance inefficiency.")
    if b2.button("Show Technical View"):
        st.info("Technical view is available via the toggle switch above.")
    if b3.button("Compare with Traditional EVMS"):
        st.info(f"Traditional EVMS EAC: ${result['traditional_eac']:,.0f} vs AI EAC: ${result['ai_eac']:,.0f}")
    if b4.button("Export Risk Rationale"):
        st.success("Risk rationale prepared for export.")


def page_portfolio(portfolio_df: pd.DataFrame) -> None:
    st.subheader("5) Portfolio Command Center")
    show_alert_banner(portfolio_df)

    c1, c2, c3, c4 = st.columns(4)
    type_filter = c1.selectbox("Program Type", ["All"] + sorted(portfolio_df["Program Type"].unique().tolist()))
    contract_filter = c2.selectbox("Contract Type", ["All"] + sorted(portfolio_df["Contract Type"].unique().tolist()))
    risk_filter = c3.multiselect("Risk", options=["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    phase_filter = c4.selectbox("Phase", ["All"] + sorted(portfolio_df["Phase"].unique().tolist()))

    f = portfolio_df.copy()
    if type_filter != "All":
        f = f[f["Program Type"] == type_filter]
    if contract_filter != "All":
        f = f[f["Contract Type"] == contract_filter]
    if phase_filter != "All":
        f = f[f["Phase"] == phase_filter]
    f = f[f["Risk"].isin(risk_filter)]

    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Sort by Highest Risk"):
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        f["order"] = f["Risk"].map(risk_order)
        f = f.sort_values(by=["order", "Overrun %"], ascending=[True, False]).drop(columns=["order"])
    if b2.button("Sort by Largest Dollar Exposure"):
        f = f.assign(Exposure=f["AI EAC"] - f["BAC"]).sort_values(by="Exposure", ascending=False)
    if b3.button("Show Only Escalations"):
        f = f[f["Risk"] == "High"]
    b4.download_button("Export Portfolio CSV", data=f.to_csv(index=False), file_name="portfolio_view.csv", mime="text/csv")

    st.dataframe(f, use_container_width=True, hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        heat_cols = ["CPI", "SPI", "Delay", "Inflation %", "Risk Probability"]
        heat_df = f[["Program Name"] + heat_cols].set_index("Program Name")
        st.plotly_chart(px.imshow(heat_df, aspect="auto", title="Portfolio Risk Heatmap"), use_container_width=True)
    with g2:
        st.plotly_chart(px.scatter(f, x="BAC", y="Overrun %", size="Delay", color="Risk", hover_name="Program Name", title="Risk vs BAC Bubble Chart"), use_container_width=True)

    risk_type = f.groupby(["Program Type", "Risk"]).size().reset_index(name="Count")
    st.plotly_chart(px.bar(risk_type, x="Program Type", y="Count", color="Risk", barmode="stack", title="Risk by Program Type"), use_container_width=True)


def page_scenario_simulator() -> None:
    st.subheader("6) What-If Scenario Simulator")
    if not require_latest():
        return
    base_payload = st.session_state["latest_payload"].copy()
    base_result = st.session_state["latest_result"]

    l, r = st.columns([1, 1.3])
    with l:
        st.markdown("### Scenario Controls")
        cpi_target = st.slider("Adjust CPI", 0.7, 1.2, float(base_payload["EV"] / base_payload["AC"] if base_payload["AC"] else 1.0), step=0.01)
        spi_target = st.slider("Adjust SPI", 0.7, 1.2, float(base_payload["EV"] / base_payload["PV"] if base_payload["PV"] else 1.0), step=0.01)
        delay_delta = st.slider("Reduce/Increase Supplier Delay (days)", -40, 40, 0)
        infl_delta = st.slider("Inflation Change (%)", -5.0, 8.0, 0.0, step=0.1)
        change_delta = st.slider("Change Order Adjustment", -8, 10, 0)
        months_delta = st.slider("Months Remaining Adjustment", -6, 12, 0)
        if st.button("Reset Scenario"):
            st.rerun()

    with r:
        scenario = base_payload.copy()
        scenario["subcontractor_delay_days"] = max(0, scenario["subcontractor_delay_days"] + delay_delta)
        scenario["material_cost_inflation_pct"] = max(0.0, scenario["material_cost_inflation_pct"] + infl_delta)
        scenario["change_orders_count"] = max(0, scenario["change_orders_count"] + change_delta)
        scenario["months_remaining"] = max(1, scenario["months_remaining"] + months_delta)
        target_ev_cpi = cpi_target * scenario["AC"]
        target_ev_spi = spi_target * scenario["PV"]
        scenario["EV"] = max(100_000.0, (target_ev_cpi + target_ev_spi) / 2.0)

        scenario_result = predict_program_risk(scenario)
        st.markdown("### Baseline vs Scenario")
        comp = pd.DataFrame(
            {
                "Case": ["Baseline AI EAC", "Scenario AI EAC"],
                "Value": [base_result["ai_eac"], scenario_result["ai_eac"]],
            }
        )
        st.plotly_chart(px.bar(comp, x="Case", y="Value", text="Value", title="Baseline vs Scenario EAC"), use_container_width=True)

        impact = scenario_result["ai_eac"] - base_result["ai_eac"]
        st.metric("EAC Change", money(impact), delta=pct((impact / base_payload["BAC"]) * 100 if base_payload["BAC"] else 0.0))
        st.write(f"Baseline Risk: {status_badge(base_result['risk_band'])} -> Scenario Risk: {status_badge(scenario_result['risk_band'])}")

        tornado = pd.DataFrame(
            {
                "Factor": ["CPI", "SPI", "Delay", "Inflation", "Change Orders", "Supplier Risk"],
                "Impact Score": [
                    abs(1.0 - cpi_target) * 10,
                    abs(1.0 - spi_target) * 8,
                    abs(delay_delta) * 0.4,
                    abs(infl_delta) * 1.2,
                    abs(change_delta) * 0.9,
                    scenario["supplier_risk_score"] * 0.03,
                ],
            }
        ).sort_values("Impact Score")
        st.plotly_chart(px.bar(tornado, x="Impact Score", y="Factor", orientation="h", title="Sensitivity Tornado Chart"), use_container_width=True)


def page_executive_brief() -> None:
    st.subheader("7) Executive Report / Decision Brief")
    if not require_latest():
        return
    payload = st.session_state["latest_payload"]
    result = st.session_state["latest_result"]
    explain_df = st.session_state["latest_explain_df"]
    summary = st.session_state["latest_summary"]

    top3 = ", ".join(explain_df["feature"].head(3).tolist())
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Executive Summary Card</div>
            <div class="hero-sub">
                Program: {payload['program_name']} | Risk: {result['risk_band'].upper()} |
                Predicted EAC: {money(result['ai_eac'])} | Overrun: {pct(result['overrun_pct'])} |
                Top Drivers: {top3} | Escalation Window: 7 days
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        mini = pd.DataFrame({"Metric": ["BAC", "Traditional", "AI"], "Value": [payload["BAC"], result["traditional_eac"], result["ai_eac"]]})
        st.plotly_chart(px.bar(mini, x="Metric", y="Value", title="Mini BAC vs EAC"), use_container_width=True)
    with g2:
        top = explain_df.head(5)
        st.plotly_chart(px.bar(top, x="shap_value", y="feature", orientation="h", title="Mini SHAP Summary"), use_container_width=True)
    with g3:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=result["overrun_pct"], title={"text": "Mini Risk Gauge"}, gauge={"axis": {"range": [-5, 20]}}))
        st.plotly_chart(gauge, use_container_width=True)

    st.write(summary)
    b1, b2, b3, b4 = st.columns(4)
    brief_text = f"{summary}\nRecommended action: {result['recommendation']}"
    b1.download_button("Download Summary", data=brief_text, file_name="executive_brief.txt", mime="text/plain")
    if b2.button("Copy Executive Note"):
        st.toast("Executive Note Copied!", icon="📋")
    if b3.button("Share to Leadership"):
        st.toast("Email draft shared to leadership.", icon="✉️")
    if b4.button("Mark Escalation Initiated"):
        st.toast("Escalation sequence initiated.", icon="🚀")


def main() -> None:
    ensure_pipeline_ready()
    bootstrap_prediction_state()
    with st.sidebar:
        st.header("Global Settings")
        theme_mode = st.selectbox("Theme Mode", ["Dark", "Light"], index=0)
        st.divider()
        st.caption("Portfolio Filters")
        sidebar_type = st.selectbox(
            "Filter Portfolio by Type",
            ["All", "Missile", "Radar", "Avionics", "ISR", "Sustainment"],
        )
        sidebar_contract = st.selectbox(
            "Filter Portfolio by Contract", ["All", "Cost Plus", "Fixed Price"]
        )
        st.divider()
        if st.button("Retrain Models"):
            save_dataset(DATA_PATH, n_rows=700)
            metrics = train_models()
            st.success(f"Retrained | R2={metrics['reg_r2']:.3f} | AUC={metrics['clf_roc_auc']:.3f}")

    inject_theme(theme_mode)

    import plotly.io as pio
    pio.templates.default = "plotly_dark" if theme_mode == "Dark" else "plotly_white"
    orig_plotly_chart = st.plotly_chart
    def custom_plotly_chart(fig, **kwargs):
        if "theme" not in kwargs:
            kwargs["theme"] = None
        orig_plotly_chart(fig, **kwargs)
    st.plotly_chart = custom_plotly_chart

    render_header()
    portfolio_df = portfolio_predictions(limit=20)
    if sidebar_type != "All":
        portfolio_df = portfolio_df[portfolio_df["Program Type"] == sidebar_type]
    if sidebar_contract != "All":
        portfolio_df = portfolio_df[portfolio_df["Contract Type"] == sidebar_contract]
    render_moving_cards(portfolio_df)

    nav_tabs = st.tabs(
        [
            "📊 Executive Overview",
            "📥 Program Intake",
            "📈 Prediction Analysis",
            "🧠 Risk Explainability",
            "📋 Portfolio View",
            "🧪 Scenario Simulator",
            "📝 Executive Brief",
        ]
    )
    with nav_tabs[0]:
        page_executive_overview(portfolio_df)
    with nav_tabs[1]:
        page_program_intake()
    with nav_tabs[2]:
        page_prediction_analysis()
    with nav_tabs[3]:
        page_explainability()
    with nav_tabs[4]:
        page_portfolio(portfolio_df)
    with nav_tabs[5]:
        page_scenario_simulator()
    with nav_tabs[6]:
        page_executive_brief()


if __name__ == "__main__":
    main()
