"""Unit tests for delay parsing, long-haul math, and investment ranking."""

import pandas as pd

from data_fetcher import flatten_faa_board, haversine_km, parse_delay_to_minutes
from llm_interface import heuristic_parse
from scoring import DEFAULT_WEIGHTS, score_airports, validate_weights


def make_table():
    return pd.DataFrame([
        {
            "name": "Busy Delayed Hub", "iata": "AAA", "city": "Alpha",
            "n_routes": 120, "n_long_haul": 20, "long_haul_pct": 16.7,
            "avg_delay_min": 40, "faa_delay_flag": True, "delay_reason": "low ceilings",
            "live_aircraft": 25,
        },
        {
            "name": "Quiet Regional", "iata": "BBB", "city": "Beta",
            "n_routes": 8, "n_long_haul": 0, "long_haul_pct": 0.0,
            "avg_delay_min": 0, "faa_delay_flag": False, "delay_reason": "",
            "live_aircraft": 2,
        },
        {
            "name": "Long-haul Gateway", "iata": "CCC", "city": "Gamma",
            "n_routes": 40, "n_long_haul": 25, "long_haul_pct": 62.5,
            "avg_delay_min": 10, "faa_delay_flag": True, "delay_reason": "volume",
            "live_aircraft": 12,
        },
    ])


def test_delay_parser():
    assert parse_delay_to_minutes("40 minutes") == 40
    assert parse_delay_to_minutes("1 hour and 21 minutes") == 81
    assert parse_delay_to_minutes("") == 0


def test_haversine_sfo_nrt_is_long_haul():
    # SFO–NRT is well over 3000 km
    km = haversine_km(37.6189, -122.375, 35.7647, 140.3864)
    assert km > 8000


def test_ranking_prefers_constrained_busy_hub():
    result = score_airports(make_table(), top_n=3)
    assert result.table.iloc[0]["iata"] == "AAA"


def test_determinism():
    table = make_table()
    a = score_airports(table, top_n=3)
    b = score_airports(table, top_n=3)
    assert a.table["investment_score"].tolist() == b.table["investment_score"].tolist()


def test_long_haul_weight_can_promote_gateway():
    weights = {"demand": 0.0, "congestion": 0.0, "unmet_demand": 0.0, "long_haul": 1.0}
    result = score_airports(make_table(), weights=weights, top_n=3)
    assert result.table.iloc[0]["iata"] == "CCC"


def test_validate_weights_rescales():
    clean, warnings = validate_weights({"demand": 1, "congestion": 1, "unmet_demand": 0, "long_haul": 0})
    assert abs(sum(clean.values()) - 1) < 1e-9
    assert any("rescaled" in w for w in warnings)


def test_validate_weights_fallback():
    clean, warnings = validate_weights({"demand": 0, "congestion": 0, "unmet_demand": 0, "long_haul": 0})
    assert clean == DEFAULT_WEIGHTS
    assert any("default" in w.lower() for w in warnings)


def test_flatten_faa_board_picks_worst_delay():
    board = {
        "GroundDelays": {"groundDelay": [{"airport": "SFO", "avgTime": "45 minutes", "reason": "WX"}]},
        "GroundStops": {"groundStop": []},
        "ArriveDepartDelays": {"arriveDepart": [{"airport": "SFO", "minDelay": "10 minutes", "maxDelay": "80 minutes", "avgDelay": "30 minutes"}]},
    }
    out = flatten_faa_board(board)
    assert out["SFO"]["avg_delay_min"] == 80
    assert "ground_delay" in out["SFO"]["delay_types"]
    assert "arrive_depart" in out["SFO"]["delay_types"]


def test_empty_table():
    result = score_airports(pd.DataFrame())
    assert result.table.empty
    assert any("no airports" in w.lower() for w in result.warnings)


def test_heuristic_new_england():
    d = heuristic_parse("Which airports in New England are strong candidates for terminal expansion?")
    assert d["intent"] == "rank_expansion"
    assert d["region"] == "new_england"
    assert d["airports"] == []


def test_heuristic_lax_sna():
    d = heuristic_parse("Compare LA and Santa Ana airport congestion levels.")
    assert d["intent"] == "compare"
    assert set(d["airports"]) == {"LAX", "SNA"}


def test_heuristic_anchorage_long_haul():
    d = heuristic_parse("What is the percentage of long haul flights out of Anchorage airport?")
    assert d["intent"] == "airport_metric"
    assert d["focus_metric"] == "long_haul_pct"
    assert "ANC" in d["airports"]


def test_heuristic_sfo_unmet():
    d = heuristic_parse("What is the unmet flight demand in SFO airport and why?")
    assert d["intent"] == "airport_metric"
    assert d["focus_metric"] == "unmet_demand"
    assert "SFO" in d["airports"]


def test_heuristic_hello_is_help():
    d = heuristic_parse("hi")
    assert d["intent"] == "help"


def test_heuristic_pourquoi_is_followup():
    from llm_interface import looks_like_followup
    assert looks_like_followup("pourquoi")
    assert looks_like_followup("why the first airport")
    d = heuristic_parse("pourquoi la reponse")
    assert d["intent"] == "followup"


def test_heuristic_which_is_help():
    from llm_interface import question_has_us_scope
    d = heuristic_parse("which")
    assert d["intent"] == "help"
    assert not question_has_us_scope("which")


def test_heuristic_no_airport_no_region_is_help():
    d = heuristic_parse("what is the weather like")
    assert d["intent"] == "help"


def test_heuristic_new_england_still_ranks():
    d = heuristic_parse("Which airports in New England are strong candidates for terminal expansion?")
    assert d["intent"] == "rank_expansion"
    assert d["region"] == "new_england"


def test_reason_stays_one_language():
    from scoring import format_reason
    row = make_table().iloc[0]
    en = format_reason(row, "en")
    he = format_reason(row, "he")
    assert "dense route network" in en
    assert "רשת יעדים צפופה" in he
    assert "dense route network" not in he


def test_followup_hebrew_is_not_mixed():
    from llm_interface import general_chat_reply, reply_lang
    assert reply_lang("למה בחרת בזה") == "he"
    assert reply_lang("why the first") == "en"
    rows = score_airports(make_table(), top_n=1).table.to_dict(orient="records")
    text = general_chat_reply("למה בחרת בזה", rows, [], lang="he")
    assert "בראש" in text
    assert "רשת יעדים צפופה" in text
    assert "dense route network" not in text


def test_help_hebrew_vs_english():
    from llm_interface import help_reply
    assert "only" in help_reply("hello").lower()
    assert "רק" in help_reply("מה אתה יכול")
