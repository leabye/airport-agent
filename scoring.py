"""
Deterministic investment-opportunity scoring.

The agent is for a firm that funds US airport modernization. The score is a
proxy for: "if we add terminal / flight capacity here, where is the unmet
demand large enough that extra throughput is most likely to be used?"

It is NOT an NPV model (no construction cost, no fares, no local politics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_WEIGHTS = {
    "demand": 0.30,
    "congestion": 0.30,
    "unmet_demand": 0.25,
    "long_haul": 0.15,
}


@dataclass
class ScoringResult:
    table: pd.DataFrame
    weights: dict
    warnings: list = field(default_factory=list)


def _min_max_normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def validate_weights(weights: dict) -> tuple[dict, list]:
    warnings = []
    clean = {k: max(0.0, float(v)) for k, v in (weights or {}).items() if k in DEFAULT_WEIGHTS}
    for missing in set(DEFAULT_WEIGHTS) - set(clean):
        clean[missing] = 0.0
        warnings.append(f"Weight for '{missing}' not provided; defaulted to 0.")
    total = sum(clean.values())
    if total <= 0:
        warnings.append("All weights were zero or invalid; falling back to default weights.")
        return dict(DEFAULT_WEIGHTS), warnings
    if abs(total - 1.0) > 1e-6:
        clean = {k: v / total for k, v in clean.items()}
        warnings.append(f"Weights summed to {total:.2f}, not 1.0; rescaled proportionally.")
    return clean, warnings


def format_reason(row: pd.Series | dict, lang: str = "en") -> str:
    """One-language rationale from KPI columns (English, or Hebrew if lang='he')."""
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    n = int(get("n_routes", 0) or 0)
    delay = float(get("avg_delay_min") or 0)
    live = get("live_aircraft")
    pct = float(get("long_haul_pct", 0) or 0)
    delay_txt = get("delay_reason") or ""
    he = lang == "he"

    if n >= 50:
        bits = [f"רשת יעדים צפופה ({n} יעדים ייחודיים)" if he else f"dense route network ({n} unique destinations)"]
    elif n >= 15:
        bits = [f"רשת בינונית ({n} יעדים)" if he else f"moderate network ({n} destinations)"]
    else:
        bits = [f"רשת דלילה ({n} יעדים)" if he else f"thin network ({n} destinations)"]

    if get("faa_delay_flag") or delay > 0:
        why = delay_txt or ("תוכנית עיכובי FAA" if he else "FAA delay program")
        bits.append(
            f"עומס פעיל ({delay:.0f} דק׳: {why})" if he else f"active congestion ({delay:.0f} min delay: {why})"
        )
    else:
        bits.append("אין תוכנית עיכובי FAA כרגע" if he else "no FAA delay program right now")

    if live is not None and pd.notna(live):
        bits.append(f"{int(live)} מטוסים חיים בקרבת מקום" if he else f"{int(live)} live aircraft nearby")

    bits.append(
        f"{pct:.1f}% מהנתיבים המפורסמים הם טיסות ארוכות טווח (≥3000 ק״מ)"
        if he
        else f"{pct:.1f}% of published routes are long-haul (≥3000 km)"
    )
    return "; ".join(bits)


def _deterministic_reason(row: pd.Series) -> str:
    return format_reason(row, "en")


def score_airports(
    kpi_table: pd.DataFrame,
    weights: dict | None = None,
    top_n: int | None = 10,
    extra_warnings: list | None = None,
    lang: str = "en",
) -> ScoringResult:
    weights, warnings = validate_weights(weights or DEFAULT_WEIGHTS)
    if extra_warnings:
        warnings = list(extra_warnings) + warnings

    if kpi_table is None or kpi_table.empty:
        warnings.append(
            "אף שדה תעופה לא התאים למסננים." if lang == "he" else "No airports matched the given filters."
        )
        empty = kpi_table if kpi_table is not None else pd.DataFrame()
        return ScoringResult(table=empty, weights=weights, warnings=warnings)

    df = kpi_table.copy()
    df["demand_norm"] = _min_max_normalize(df["n_routes"])
    df["congestion_raw"] = (
        pd.to_numeric(df["avg_delay_min"], errors="coerce").fillna(0)
        + df["faa_delay_flag"].astype(float) * 20
        + pd.to_numeric(df["live_aircraft"], errors="coerce").fillna(0)
    )
    df["congestion_norm"] = _min_max_normalize(df["congestion_raw"])
    df["unmet_demand_norm"] = df["demand_norm"] * df["congestion_norm"]
    df["long_haul_norm"] = _min_max_normalize(df["long_haul_pct"])

    df["investment_score"] = (100 * (
        weights["demand"] * df["demand_norm"]
        + weights["congestion"] * df["congestion_norm"]
        + weights["unmet_demand"] * df["unmet_demand_norm"]
        + weights["long_haul"] * df["long_haul_norm"]
    )).round(2)
    df["reason"] = df.apply(lambda r: format_reason(r, lang), axis=1)
    df = df.sort_values("investment_score", ascending=False).reset_index(drop=True)
    df.index += 1

    df["unmet_demand_index"] = (df["unmet_demand_norm"] * 100).round(1)
    cols = [
        "name", "iata", "city", "n_routes", "n_long_haul", "long_haul_pct",
        "avg_delay_min", "faa_delay_flag", "delay_reason", "live_aircraft",
        "unmet_demand_index", "investment_score", "reason",
    ]
    result = df[[c for c in cols if c in df.columns]]
    if top_n is not None:
        result = result.head(top_n)
        if len(df) < top_n:
            warnings.append(
                f"רק {len(df)} שדות התאימו למסננים (נתבקש top {top_n})."
                if lang == "he"
                else f"Only {len(df)} airports matched the filters (requested top {top_n})."
            )

    if lang == "he":
        warnings.append("הציון הוא דירוג יחסי 0–100 בתוך ההשוואה הזו, לא תחזית רווח בדולרים.")
        warnings.append("long_haul_pct סופר נתיבים מפורסמים ייחודיים, לא מושבים ולא תדירות שבועית.")
    else:
        warnings.append(
            "Score is a relative 0–100 ranking inside this comparison set, not a dollar profit forecast."
        )
        warnings.append(
            "long_haul_pct counts unique published routes, not seats or weekly frequencies."
        )
    return ScoringResult(table=result, weights=weights, warnings=warnings)


def compare_airports(
    kpi_table: pd.DataFrame,
    weights: dict | None = None,
    extra_warnings: list | None = None,
    lang: str = "en",
) -> ScoringResult:
    """Keep every requested airport, still attach the same score for a side-by-side view."""
    return score_airports(
        kpi_table, weights=weights, top_n=None, extra_warnings=extra_warnings, lang=lang
    )
