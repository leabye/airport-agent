"""US region filters used by the investment-screening agent."""

from __future__ import annotations

# Bounding boxes: (lat_min, lat_max, lon_min, lon_max)
REGION_BBOX = {
    "new_england": (41.00, 47.50, -73.55, -66.88),
    "northeast": (38.80, 47.50, -80.55, -66.88),
    "california": (32.50, 42.05, -124.50, -114.10),
    "west_coast": (32.50, 49.00, -125.00, -116.50),
    "texas": (25.80, 36.55, -106.70, -93.50),
    "florida": (24.40, 31.05, -87.65, -79.90),
    "alaska": (51.20, 71.50, -173.00, -129.90),
    "us": (18.90, 71.50, -179.00, -66.88),
}

REGION_ALIASES = {
    "new england": "new_england",
    "new-england": "new_england",
    "northeast": "northeast",
    "north east": "northeast",
    "california": "california",
    "west coast": "west_coast",
    "pacific": "west_coast",
    "texas": "texas",
    "florida": "florida",
    "alaska": "alaska",
    "united states": "us",
    "usa": "us",
    "u.s.": "us",
    "u.s.a": "us",
    "the us": "us",
}

# Common spoken names -> IATA. Used by the heuristic parser and as a hint to the LLM.
AIRPORT_ALIASES = {
    "la": "LAX",
    "lax": "LAX",
    "los angeles": "LAX",
    "santa ana": "SNA",
    "orange county": "SNA",
    "john wayne": "SNA",
    "sna": "SNA",
    "sfo": "SFO",
    "san francisco": "SFO",
    "anchorage": "ANC",
    "anc": "ANC",
    "boston": "BOS",
    "logan": "BOS",
    "providence": "PVD",
    "pvd": "PVD",
    "hartford": "BDL",
    "bradley": "BDL",
    "bdl": "BDL",
    "manchester": "MHT",
    "mht": "MHT",
    "pwm": "PWM",
    "burlington": "BTV",
    "btv": "BTV",
    "jfk": "JFK",
    "lga": "LGA",
    "laguardia": "LGA",
    "ewr": "EWR",
    "newark": "EWR",
    "ord": "ORD",
    "ohare": "ORD",
    "dfw": "DFW",
    "sea": "SEA",
    "seattle": "SEA",
    "mia": "MIA",
    "miami": "MIA",
    "atl": "ATL",
    "atlanta": "ATL",
}



