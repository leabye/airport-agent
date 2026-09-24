"""
Public aviation data layer.

REST APIs (JSON, no key) — this is the assignment's "public APIs" layer:
  1. Airport Gap   GET /api/airports/{IATA}
  2. FAA ASWS      delay board + per-airport status
  3. OpenSky       live aircraft in a bbox

Bundled local files in data/ (shipped with the source, not fetched at runtime):
US airport catalog, IATA coordinates, and a route list for demand / long-haul.
Those files are datasets, not APIs.

Scoring lives in scoring.py.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from regions import REGION_BBOX

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATIC_TTL = 24 * 3600
LIVE_TTL = 5 * 60

AIRPORT_GAP = "https://airportgap.com/api/airports/{code}"
FAA_STATUS = "https://external-api.faa.gov/asws/api/airport/status/{code}"
FAA_DELAYS = "https://external-api.faa.gov/asws/api/airport/delays"
FAA_NAS_XML = "https://nasstatus.faa.gov/api/airport-status-information"
OPENSKY_STATES = "https://opensky-network.org/api/states/all"

# IATA-style long-haul cutoff used for the Anchorage-style questions.
LONG_HAUL_KM = 3000
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "airport-investment-agent/1.0 (course project)"})


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def _cached_text(url: str, filename: str, ttl: int) -> str:
    path = _cache_path(filename)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl:
        with open(path, encoding="utf-8") as f:
            return f.read()
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return resp.text


def parse_delay_to_minutes(text: str | None) -> float:
    """Turn FAA phrases like '1 hour and 21 minutes' or '40 minutes' into minutes."""
    if not text:
        return 0.0
    raw = text.lower().strip()
    hours = 0.0
    minutes = 0.0
    hm = re.search(r"(\d+(?:\.\d+)?)\s*hour", raw)
    mm = re.search(r"(\d+(?:\.\d+)?)\s*min", raw)
    if hm:
        hours = float(hm.group(1))
    if mm:
        minutes = float(mm.group(1))
    if hours == 0 and minutes == 0:
        only = re.search(r"(\d+(?:\.\d+)?)", raw)
        if only:
            minutes = float(only.group(1))
    return hours * 60 + minutes


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Scalar wrapper used by tests; route KPIs use the vectorized path."""
    return float(_haversine_np(np.array([lat1]), np.array([lon1]), np.array([lat2]), np.array([lon2]))[0])


def _haversine_np(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@lru_cache(maxsize=1)
def _airports_cached() -> pd.DataFrame:
    """Worldwide IATA coordinates from the bundled data/ folder (not a live Git clone)."""
    coords = pd.read_csv(os.path.join(DATA_DIR, "airport_coords.csv"))
    us = pd.read_csv(os.path.join(DATA_DIR, "us_airports.csv"))
    extra = us[["iata", "name", "city", "country", "icao", "type", "country_iso"]]
    df = coords.merge(extra, on="iata", how="left")
    df["iata"] = df["iata"].astype(str).str.upper()
    df["country_iso"] = df["country_iso"].fillna("")
    df["country"] = df["country"].where(df["country"].notna(), df["country_iso"])
    df["country"] = np.where(df["country_iso"] == "US", "United States", df["country"])
    return df


def load_airports() -> pd.DataFrame:
    return _airports_cached().copy()


def load_us_airports() -> pd.DataFrame:
    """Bundled US commercial fields (IATA + scheduled medium/large)."""
    us = pd.read_csv(os.path.join(DATA_DIR, "us_airports.csv"))
    us["iata"] = us["iata"].astype(str).str.upper()
    cols = ["name", "city", "country", "iata", "icao", "lat", "lon", "type"]
    return us[[c for c in cols if c in us.columns]].drop_duplicates("iata")


def fetch_airport_gap(iata: str) -> dict | None:
    """REST lookup for one airport (Airport Gap — student-friendly public API)."""
    code = iata.upper()
    path = _cache_path(f"airportgap_{code}.json")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < STATIC_TTL:
        with open(path, encoding="utf-8") as f:
            cached = json.load(f)
        return cached or None
    try:
        resp = SESSION.get(AIRPORT_GAP.format(code=code), timeout=15)
        if resp.status_code == 404:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(None, f)
            return None
        resp.raise_for_status()
        payload = resp.json().get("data") or {}
        attrs = payload.get("attributes") or {}
        info = {
            "iata": (attrs.get("iata") or code).upper(),
            "icao": attrs.get("icao"),
            "name": attrs.get("name"),
            "city": attrs.get("city"),
            "country": attrs.get("country"),
            "lat": float(attrs["latitude"]) if attrs.get("latitude") not in (None, "") else None,
            "lon": float(attrs["longitude"]) if attrs.get("longitude") not in (None, "") else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f)
        return info
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return None


def enrich_named_airports(table: pd.DataFrame, iata_codes: list[str], warnings: list[str]) -> pd.DataFrame:
    """Overwrite identity/coords for named IATA codes from Airport Gap when available."""
    if not iata_codes:
        return table
    for code in iata_codes:
        info = fetch_airport_gap(code)
        if not info:
            warnings.append(f"Airport Gap had no record for {code}; using bundled US catalog.")
            continue
        idx = table.index[table["iata"] == code]
        if len(idx) == 0:
            # Named US IATA missing from the commercial filter — append a row so compares still work.
            if info.get("country") not in ("United States", "US") and info.get("country"):
                warnings.append(f"{code} from Airport Gap is '{info.get('country')}'; mandate is US-only.")
                continue
            row = {c: None for c in table.columns}
            row.update({
                "name": info.get("name"),
                "city": info.get("city"),
                "country": "United States",
                "iata": code,
                "icao": info.get("icao"),
                "lat": info.get("lat"),
                "lon": info.get("lon"),
                "type": "large_airport",
            })
            table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
            warnings.append(f"Added {code} from Airport Gap; it was missing from the bundled US catalog.")
            continue
        i = idx[0]
        for field in ("name", "city", "icao", "lat", "lon"):
            if info.get(field) not in (None, ""):
                table.at[i, field] = info[field]
    return table


@lru_cache(maxsize=1)
def _routes_cached() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "routes.dat")
    cols = [
        "airline", "airline_id", "src_iata", "src_id", "dst_iata", "dst_id",
        "codeshare", "stops", "equipment",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    df["src_iata"] = df["src_iata"].astype(str).str.upper()
    df["dst_iata"] = df["dst_iata"].astype(str).str.upper()
    return df


def load_routes() -> pd.DataFrame:
    return _routes_cached().copy()


def build_route_kpis(us_airports: pd.DataFrame) -> pd.DataFrame:
    """
    Unique origin→destination pairs (not airline duplicates / codeshares).
    long_haul_pct is a ROUTE share, not a flight-frequency share — we do not
    have seats or weekly frequencies in this public dump.
    """
    coords = _airports_cached().drop_duplicates("iata").set_index("iata")[["lat", "lon"]]
    routes = load_routes()
    us_iata = set(us_airports["iata"])
    routes = routes[routes["src_iata"].isin(us_iata)].drop_duplicates(["src_iata", "dst_iata"])

    routes = routes.merge(coords, left_on="src_iata", right_index=True, how="left")
    routes = routes.merge(
        coords, left_on="dst_iata", right_index=True, how="left", suffixes=("_src", "_dst")
    )
    routes = routes.dropna(subset=["lat_src", "lon_src", "lat_dst", "lon_dst"])
    routes["distance_km"] = _haversine_np(
        routes["lat_src"].to_numpy(),
        routes["lon_src"].to_numpy(),
        routes["lat_dst"].to_numpy(),
        routes["lon_dst"].to_numpy(),
    )
    routes["is_long_haul"] = routes["distance_km"] >= LONG_HAUL_KM

    grouped = routes.groupby("src_iata").agg(
        n_routes=("dst_iata", "nunique"),
        n_long_haul=("is_long_haul", "sum"),
        mean_distance_km=("distance_km", "mean"),
        max_distance_km=("distance_km", "max"),
    ).reset_index()
    grouped["long_haul_pct"] = (grouped["n_long_haul"] / grouped["n_routes"] * 100).fillna(0)
    grouped = grouped.rename(columns={"src_iata": "iata"})
    return grouped


def fetch_faa_delay_board() -> dict:
    """One call covering current US ground delays / stops / arrival delays."""
    path = _cache_path("faa_delays.json")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < LIVE_TTL:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    resp = SESSION.get(FAA_DELAYS, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data


def fetch_faa_airport_status(iata: str) -> dict | None:
    code = iata.upper()
    try:
        resp = SESSION.get(FAA_STATUS.format(code=code), timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _passenger_relevant_reason(reason: str) -> bool:
    """Drop GA/transient NOTAMs that are not passenger-throughput constraints."""
    r = (reason or "").lower()
    if not r:
        return False
    if "non sked" in r or "transient ga" in r:
        return False
    return True


def flatten_faa_board(board: dict) -> dict[str, dict]:
    """Map IATA -> {avg_delay_min, delay_reason, delay_types} from the bulk board."""
    out: dict[str, dict] = {}

    def add(code: str, minutes: float, reason: str, kind: str):
        code = (code or "").upper()
        if not code:
            return
        row = out.setdefault(code, {"avg_delay_min": 0.0, "delay_reason": "", "delay_types": []})
        row["avg_delay_min"] = max(row["avg_delay_min"], minutes)
        row["delay_types"].append(kind)
        if reason:
            row["delay_reason"] = reason

    for item in (board.get("GroundDelays") or {}).get("groundDelay") or []:
        add(item.get("airport", ""), parse_delay_to_minutes(item.get("avgTime")), item.get("reason", ""), "ground_delay")
    for item in (board.get("GroundStops") or {}).get("groundStop") or []:
        add(item.get("airport", ""), 90.0, item.get("reason", ""), "ground_stop")
    for item in (board.get("ArriveDepartDelays") or {}).get("arriveDepart") or []:
        mins = max(
            parse_delay_to_minutes(item.get("minDelay")),
            parse_delay_to_minutes(item.get("maxDelay")),
            parse_delay_to_minutes(item.get("avgDelay")),
        )
        add(item.get("airport", ""), mins, item.get("reason", ""), "arrive_depart")
    return out


def fetch_nas_reasons() -> dict[str, str]:
    """Optional second FAA feed (XML) with human-readable delay reasons."""
    try:
        text = _cached_text(FAA_NAS_XML, "faa_nas.xml", LIVE_TTL)
        root = ET.fromstring(text)
        reasons = {}
        for node in root.iter():
            tag = node.tag.lower()
            if tag.endswith("ground_delay") or tag.endswith("ground_stop") or tag.endswith("delay"):
                arpt = None
                reason = None
                for child in list(node):
                    ctag = child.tag.lower()
                    if ctag.endswith("arpt") or ctag.endswith("airport"):
                        arpt = (child.text or "").strip().upper()
                    if ctag.endswith("reason"):
                        reason = (child.text or "").strip()
                if arpt and reason:
                    reasons[arpt] = reason
        return reasons
    except Exception:
        return {}


def fetch_opensky_count(lat: float, lon: float, delta: float = 0.18) -> int | None:
    """Count live transponders in a ~20 km box around the airport."""
    params = {
        "lamin": lat - delta,
        "lamax": lat + delta,
        "lomin": lon - delta,
        "lomax": lon + delta,
    }
    try:
        resp = SESSION.get(OPENSKY_STATES, params=params, timeout=20)
        if resp.status_code in (401, 429, 503):
            return None
        resp.raise_for_status()
        states = resp.json().get("states") or []
        return len(states)
    except requests.RequestException:
        return None


def build_airport_kpi_table(
    region: str | None = None,
    iata_codes: list[str] | None = None,
    enrich_live: bool = True,
    live_limit: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Assemble the KPI table for US commercial airports (IATA + scheduled service).

    Live FAA/OpenSky fields are filled for requested airports plus the busiest
    remaining airports up to `live_limit` (rate-limit friendly).
    """
    warnings: list[str] = []
    us = load_us_airports()
    if region:
        if region not in REGION_BBOX:
            warnings.append(f"Unknown region '{region}'; using all US airports.")
        else:
            lat_min, lat_max, lon_min, lon_max = REGION_BBOX[region]
            us = us[us["lat"].between(lat_min, lat_max) & us["lon"].between(lon_min, lon_max)]

    if iata_codes:
        wanted = {c.upper() for c in iata_codes}
        us = us[us["iata"].isin(wanted)]
        us = enrich_named_airports(us, list(wanted), warnings)

    if us.empty:
        warnings.append("No US airports matched the region/IATA filters.")
        return us, warnings

    kpis = build_route_kpis(us)
    table = us.merge(kpis, on="iata", how="left")
    for col, default in [
        ("n_routes", 0),
        ("n_long_haul", 0),
        ("long_haul_pct", 0.0),
        ("mean_distance_km", 0.0),
        ("max_distance_km", 0.0),
    ]:
        table[col] = table[col].fillna(default)

    # Drop fields with no scheduled-route signal — they cannot support the demand KPIs.
    table = table[table["n_routes"] > 0].copy()

    table["avg_delay_min"] = 0.0
    table["delay_reason"] = ""
    table["faa_delay_flag"] = False
    table["faa_supported"] = False
    table["live_aircraft"] = pd.NA
    table["weather"] = ""

    if table.empty:
        warnings.append("No US airports with published routes matched the filters.")
        return table, warnings

    if not enrich_live:
        return table.reset_index(drop=True), warnings

    board_map: dict[str, dict] = {}
    nas_reasons: dict[str, str] = {}
    try:
        board_map = flatten_faa_board(fetch_faa_delay_board())
        nas_reasons = fetch_nas_reasons()
    except Exception as exc:
        warnings.append(f"FAA delay board unavailable ({exc}). Congestion uses OpenSky only where possible.")

    requested = [c.upper() for c in (iata_codes or [])]
    busiest = table.sort_values("n_routes", ascending=False)["iata"].tolist()
    cap = max(live_limit, len(requested))
    enrich_codes = list(dict.fromkeys(requested + busiest))[:cap]

    for code in enrich_codes:
        idx = table.index[table["iata"] == code]
        if len(idx) == 0:
            continue
        i = idx[0]
        if code in board_map:
            table.at[i, "avg_delay_min"] = board_map[code]["avg_delay_min"]
            table.at[i, "delay_reason"] = board_map[code]["delay_reason"] or nas_reasons.get(code, "")
            table.at[i, "faa_delay_flag"] = True
            table.at[i, "faa_supported"] = True
        status = fetch_faa_airport_status(code) if code in requested else None
        if status:
            table.at[i, "faa_supported"] = True
            reason = table.at[i, "delay_reason"]
            statuses = status.get("Status") or []
            if statuses:
                cand = statuses[0].get("Reason") or statuses[0].get("Type") or ""
                if _passenger_relevant_reason(cand):
                    reason = cand or reason
                    if table.at[i, "avg_delay_min"] == 0:
                        table.at[i, "avg_delay_min"] = parse_delay_to_minutes(statuses[0].get("AvgDelay"))
            delay_flag = bool(status.get("Delay")) and _passenger_relevant_reason(reason)
            table.at[i, "faa_delay_flag"] = bool(table.at[i, "faa_delay_flag"] or delay_flag)
            if delay_flag and not table.at[i, "delay_reason"]:
                table.at[i, "delay_reason"] = reason
            elif _passenger_relevant_reason(reason):
                table.at[i, "delay_reason"] = reason
            else:
                if not _passenger_relevant_reason(str(table.at[i, "delay_reason"])):
                    table.at[i, "delay_reason"] = ""
                    table.at[i, "faa_delay_flag"] = False
            weather = status.get("Weather") or {}
            temp = weather.get("Temp") or weather.get("Weather") or ""
            table.at[i, "weather"] = str(temp)
        row = table.loc[i]
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            warnings.append(f"No coordinates for {code}; skipped OpenSky.")
            continue
        count = fetch_opensky_count(float(row["lat"]), float(row["lon"]))
        if count is None:
            warnings.append(f"OpenSky live count unavailable for {code}.")
        else:
            table.at[i, "live_aircraft"] = count

    missing_live = table["live_aircraft"].isna().sum()
    if missing_live:
        warnings.append(
            f"Live aircraft counts filled for {int(table['live_aircraft'].notna().sum())} airports; "
            f"{int(missing_live)} others have route demand only. Treat congestion as uncertain there."
        )
    if not table["faa_supported"].any():
        warnings.append(
            "FAA status covers major US airports only. Smaller fields have no official delay flag."
        )

    return table.reset_index(drop=True), warnings
