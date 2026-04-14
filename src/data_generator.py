from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROGRAM_TYPES = ["Missile", "Radar", "Avionics", "ISR", "Sustainment"]
CONTRACT_TYPES = ["Cost Plus", "Fixed Price"]
PROGRAM_PHASES = ["Design", "Build", "Test", "Deployment"]


def generate_synthetic_data(n_rows: int = 500, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    rows = []
    for i in range(n_rows):
        program_type = rng.choice(PROGRAM_TYPES)
        contract_type = rng.choice(CONTRACT_TYPES, p=[0.62, 0.38])
        program_phase = rng.choice(PROGRAM_PHASES, p=[0.25, 0.35, 0.25, 0.15])

        bac = float(rng.uniform(20_000_000, 500_000_000))
        completion_ratio = float(rng.uniform(0.25, 0.9))
        ac = bac * completion_ratio * rng.uniform(0.92, 1.08)
        ev = ac * rng.uniform(0.78, 1.12)
        pv = ac * rng.uniform(0.82, 1.08)

        subcontractor_delay_days = int(rng.integers(0, 121))
        change_orders_count = int(rng.integers(0, 16))
        material_cost_inflation_pct = float(rng.uniform(1, 12))
        supplier_risk_score = float(rng.uniform(0, 100))
        months_remaining = int(rng.integers(3, 37))
        historical_rebaseline_count = int(rng.integers(0, 5))

        cpi = ev / ac if ac > 0 else 1.0
        spi = ev / pv if pv > 0 else 1.0
        cost_variance = ev - ac
        schedule_variance = ev - pv

        traditional_eac = ac + (bac - ev) / max(cpi, 0.35)

        risk_pressure = (
            max(0, (1.0 - cpi)) * 0.35
            + max(0, (1.0 - spi)) * 0.22
            + (subcontractor_delay_days / 120.0) * 0.13
            + (change_orders_count / 15.0) * 0.10
            + (material_cost_inflation_pct / 12.0) * 0.10
            + (supplier_risk_score / 100.0) * 0.07
            + (historical_rebaseline_count / 4.0) * 0.03
        )
        contract_penalty = 0.015 if contract_type == "Fixed Price" else 0.005
        type_penalty = 0.01 if program_type in {"Missile", "ISR"} else 0.0

        final_eac = traditional_eac * (1 + risk_pressure * 0.28 + contract_penalty + type_penalty)
        final_eac += rng.normal(0, bac * 0.015)
        final_eac = max(final_eac, bac * 0.75)

        overrun_flag = int(final_eac > bac * 1.02)

        rows.append(
            {
                "program_id": f"PRG-{1000 + i}",
                "program_name": f"{program_type} Program {i + 1}",
                "program_type": program_type,
                "contract_type": contract_type,
                "program_phase": program_phase,
                "BAC": bac,
                "AC": ac,
                "EV": ev,
                "PV": pv,
                "CPI": cpi,
                "SPI": spi,
                "cost_variance": cost_variance,
                "schedule_variance": schedule_variance,
                "subcontractor_delay_days": subcontractor_delay_days,
                "change_orders_count": change_orders_count,
                "material_cost_inflation_pct": material_cost_inflation_pct,
                "supplier_risk_score": supplier_risk_score,
                "months_remaining": months_remaining,
                "historical_rebaseline_count": historical_rebaseline_count,
                "final_eac": final_eac,
                "overrun_flag": overrun_flag,
            }
        )

    df = pd.DataFrame(rows)
    df["risk_band"] = pd.cut(
        (df["final_eac"] - df["BAC"]) / df["BAC"] * 100,
        bins=[-999, 2, 5, 999],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return df


def save_dataset(output_path: Path | None = None, n_rows: int = 500) -> Path:
    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_program_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(n_rows=n_rows)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    file_path = save_dataset()
    print(f"Synthetic dataset generated at: {file_path}")
