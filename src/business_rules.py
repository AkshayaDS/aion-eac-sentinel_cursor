from __future__ import annotations

from dataclasses import dataclass


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b


def calculate_traditional_eac(ac: float, bac: float, ev: float, cpi: float) -> float:
    return ac + safe_divide(bac - ev, max(cpi, 1e-6), default=0.0)


def calculate_vac(bac: float, ai_eac: float) -> float:
    return bac - ai_eac


def calculate_overrun_pct(bac: float, ai_eac: float) -> float:
    return safe_divide(ai_eac - bac, bac, default=0.0) * 100.0


def get_risk_band(risk_probability: float, overrun_pct: float) -> str:
    if risk_probability >= 0.70 or overrun_pct > 5:
        return "High"
    if risk_probability >= 0.35 or overrun_pct >= 2:
        return "Medium"
    return "Low"


def get_recommendation(risk_band: str) -> str:
    if risk_band == "High":
        return (
            "Escalate to Program Finance, trigger rebaseline review, and validate supplier recovery "
            "plan within 7 days."
        )
    if risk_band == "Medium":
        return "Monitor weekly, review change-order controls, and assess procurement exposure."
    return "Continue standard monitoring and maintain current controls."


def generate_executive_summary(
    program_name: str,
    risk_band: str,
    ai_eac: float,
    bac: float,
    top_driver_text: str,
) -> str:
    overrun_pct = calculate_overrun_pct(bac, ai_eac)
    return (
        f"Program {program_name} is forecast at ${ai_eac:,.0f} vs BAC ${bac:,.0f} "
        f"({overrun_pct:+.1f}%). Risk level is {risk_band.upper()}. "
        f"Primary drivers: {top_driver_text}."
    )


@dataclass
class DecisionOutput:
    traditional_eac: float
    ai_eac: float
    overrun_pct: float
    vac: float
    risk_probability: float
    risk_band: str
    recommendation: str
