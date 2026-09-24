"""
LLM boundary: intent parsing + narration. Never computes scores or airport KPIs.

Primary model: Anthropic Claude if ANTHROPIC_API_KEY is set.
If the key is missing, a local heuristic parser + templated explanation still
runs so the deterministic pipeline can be demoed without an LLM.
"""

from __future__ import annotations

import json
import os
import re

from functools import lru_cache

from i18n import localize_warnings
from regions import AIRPORT_ALIASES, REGION_ALIASES
from scoring import DEFAULT_WEIGHTS, format_reason

MODEL = "claude-sonnet-4-6"


def reply_lang(user_message: str) -> str:
    """English by default. Hebrew script in the question → answer in Hebrew."""
    if re.search(r"[\u0590-\u05FF]", user_message or ""):
        return "he"
    return "en"


def looks_like_followup(user_message: str) -> bool:
    """True for 'why / pourquoi' about the last ranking, not a new screen."""
    q = (user_message or "").lower().strip()
    if not q:
        return False
    hints = (
        "למה",
        "מדוע",
        "הסבר",
        "why did",
        "why is the",
        "why the",
        "why that",
        "why this",
        "why him",
        "explain the",
        "explain that",
        "the first",
        "le premier",
        "la première",
        "la premiere",
        "the score",
        "le score",
        "tell me more",
        "plus de détail",
        "plus de detail",
        "what does the score",
        "c'est quoi le",
        "comment ça",
        "comment ca",
    )
    if any(h in q for h in hints):
        return True
    if re.match(r"^(pourquoi|why|explain|למה|מדוע)\b", q):
        return True
    return False


def question_has_us_scope(user_message: str) -> bool:
    """True if the text names a US airport alias or a US region/country."""
    q = (user_message or "").lower()
    for phrase in AIRPORT_ALIASES:
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return True
    for phrase in REGION_ALIASES:
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return True
    return False


def _has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def get_client():
    from anthropic import Anthropic
    return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


PARSE_SYSTEM_PROMPT = """You convert questions from a US airport-modernization investment analyst
into JSON. You NEVER invent airport statistics.

Return ONLY valid JSON (no markdown) matching:
{
  "intent": "rank_expansion" | "compare" | "airport_metric" | "followup" | "help",
  "region": "new_england" | "northeast" | "california" | "west_coast" | "texas" | "florida" | "alaska" | "us" | null,
  "airports": ["IATA", "..."],
  "focus_metric": "expansion" | "congestion" | "long_haul_pct" | "unmet_demand" | "general",
  "top_n": 10,
  "weights": {
    "demand": 0.30,
    "congestion": 0.30,
    "unmet_demand": 0.25,
    "long_haul": 0.15
  },
  "assumptions": ["short strings"]
}

Rules:
- This firm only screens US airports. If the user names a foreign city, say so in assumptions.
- "LA" / "Los Angeles" -> LAX. "Santa Ana" / "Orange County" / "John Wayne" -> SNA.
- "New England" -> region new_england, intent rank_expansion, focus_metric expansion.
- Compare questions must list both IATA codes.
- Long-haul percentage -> airport_metric + long_haul_pct + that airport.
- Unmet demand / why is SFO constrained -> airport_metric + unmet_demand.
- Follow-ups about a previous table ("why the first one?", "what does the score mean?") -> followup.
- Greetings, "what can you do?", or any question with no US airport and no US region -> help.
- Do NOT default to ranking all US airports when the question is vague.
- "Best airport" with no region -> help (scope reminder), not a national screen.
- If no preference on weights, use the defaults above and note that as an assumption.
- Prefer IATA codes in airports[]. Empty list means "all airports in region".
"""


def heuristic_parse(user_message: str) -> dict:
    """Deterministic backup when no LLM key is configured."""
    q = user_message.lower()
    assumptions = ["Parsed without an LLM (no ANTHROPIC_API_KEY); used keyword rules."]
    intent = "rank_expansion"
    focus = "expansion"
    region = None
    if "compar" in q or " vs " in q or " versus " in q or "השווה" in user_message or "השוואה" in user_message:
        intent = "compare"
        focus = "congestion" if ("congest" in q or "עומס" in user_message) else "expansion"
    elif "long haul" in q or "long-haul" in q or "טווח" in user_message:
        intent = "airport_metric"
        focus = "long_haul_pct"
    elif "unmet" in q or "ביקוש" in user_message or ("sfo" in q and "why" in q):
        intent = "airport_metric"
        focus = "unmet_demand"
    elif looks_like_followup(user_message):
        intent = "followup"
        focus = "general"
    elif re.search(r"^\s*(hi|hey|hello|bonjour|salut|coucou|thanks|merci)\b", q) and not any(
        k in q for k in ("best", "rank", "airport", "congestion", "expansion", "sfo", "lax")
    ):
        intent = "help"
        focus = "general"
    if "congest" in q or "עומס" in user_message:
        focus = "congestion"

    for phrase, code in sorted(REGION_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            region = code
            break

    airports = []
    for phrase, iata in sorted(AIRPORT_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            if iata not in airports:
                airports.append(iata)

    words = re.findall(r"[a-z0-9]+", q)
    if intent == "compare" and len(airports) < 2:
        intent = "help"
        focus = "general"
        assumptions.append("Compare requested but fewer than two US airports named; showed scope help.")
    elif intent == "airport_metric" and not airports:
        intent = "help"
        focus = "general"
        assumptions.append("Metric requested but no US airport named; showed scope help.")
    elif (
        intent == "rank_expansion"
        and not airports
        and region is None
    ):
        intent = "help"
        focus = "general"
        assumptions.append("No US airport or region in the question; stayed in scope instead of ranking all US.")

    if region is None and intent == "rank_expansion":
        region = "us"
        assumptions.append("No region named; defaulted to all US commercial airports with published routes.")

    top_n = 10
    m = re.search(r"top\s+(\d+)", q)
    if m:
        top_n = int(m.group(1))

    weights = dict(DEFAULT_WEIGHTS)
    if focus == "congestion":
        weights = {"demand": 0.15, "congestion": 0.50, "unmet_demand": 0.25, "long_haul": 0.10}
        assumptions.append("Tilted weights toward congestion because the question asked about congestion.")
    if focus == "long_haul_pct":
        weights = {"demand": 0.10, "congestion": 0.10, "unmet_demand": 0.10, "long_haul": 0.70}
        assumptions.append("Tilted weights toward long-haul share.")
    if focus == "unmet_demand":
        weights = {"demand": 0.20, "congestion": 0.25, "unmet_demand": 0.45, "long_haul": 0.10}
        assumptions.append("Tilted weights toward unmet-demand proxy.")

    return {
        "intent": intent,
        "region": region,
        "airports": airports,
        "focus_metric": focus,
        "top_n": top_n,
        "weights": weights,
        "assumptions": assumptions,
    }


def parse_intent(user_message: str, conversation_history: list[dict] | None = None) -> dict:
    if not _has_anthropic():
        return heuristic_parse(user_message)

    history = (conversation_history or [])[-6:]
    messages = history + [{"role": "user", "content": user_message}]
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=600,
        system=PARSE_SYSTEM_PROMPT,
        messages=messages,
    )
    text = "".join(block.text for block in resp.content if block.type == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.I | re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        fallback = heuristic_parse(user_message)
        fallback["assumptions"].append("LLM JSON parse failed; used heuristic parser.")
        return fallback
    data.setdefault("intent", "rank_expansion")
    data.setdefault("region", None)
    data.setdefault("airports", [])
    data.setdefault("focus_metric", "expansion")
    data.setdefault("top_n", 10)
    data.setdefault("weights", dict(DEFAULT_WEIGHTS))
    data.setdefault("assumptions", [])
    return data


EXPLAIN_SYSTEM_PROMPT = """You advise a US airport-modernization investment firm.
You receive a user question, deterministic scores, and pipeline warnings.

Rules:
- Only use numbers in the provided table. Do not invent passenger counts, fares, or costs.
- Answer the user's actual question (expansion candidates, congestion compare, long-haul %, unmet demand).
- Explain the PROXY: unmet demand ≈ route demand × live congestion (FAA delays + nearby aircraft).
- State uncertainty plainly: route counts ≠ flight frequencies; FAA covers major airports; OpenSky is a snapshot.
- Keep it tight: lead answer, 2–4 sentences of reasoning, then point at the table.
- If comparing two airports, say which is more congested right now and why the data says that.
"""


def _row_reason(row: dict, lang: str) -> str:
    return format_reason(row, lang)


def _template_explain(
    user_message: str,
    weights: dict,
    table_records: list[dict],
    warnings: list[str],
    lang: str = "en",
) -> str:
    he = lang == "he"
    notes = localize_warnings(warnings, lang)
    if not table_records:
        empty = "אף שדה לא התאים; אין דירוג." if he else "No airports matched those filters, so I cannot rank an investment case."
        return empty + " " + " ".join(notes)
    q = user_message.lower()
    lines = []
    long_haul_q = "long haul" in q or "long-haul" in q or "טווח" in user_message
    unmet_q = "unmet" in q or "ביקוש" in user_message
    if long_haul_q:
        row = table_records[0]
        if he:
            lines.append(
                f"**נתח טיסות ארוכות טווח מ-{row.get('iata')}:** {row.get('long_haul_pct', 0):.1f}% "
                f"מהנתיבים הייחודיים מ-{row.get('name')} הם ≥3000 ק״מ "
                f"({int(row.get('n_long_haul') or 0)} מתוך {int(row.get('n_routes') or 0)} יעדים). "
                "זה נתח נתיבים, לא מושבים או תדירות."
            )
        else:
            lines.append(
                f"**{row.get('iata')} long-haul share:** {row.get('long_haul_pct', 0):.1f}% of unique published "
                f"routes from {row.get('name')} are ≥3000 km ({int(row.get('n_long_haul') or 0)} of "
                f"{int(row.get('n_routes') or 0)} destinations). This is a route-count share, not seats or frequencies."
            )
        lines.append("")
    elif ("congest" in q or "עומס" in user_message) and len(table_records) >= 2:
        a, b = table_records[0], table_records[1]
        if he:
            lines.append(
                f"**השוואת עומס:** ל-{a.get('iata')} יש {a.get('avg_delay_min', 0):.0f} דק׳ עיכוב FAA "
                f"ו-{a.get('live_aircraft', 'n/a')} מטוסים חיים מול {b.get('iata')} "
                f"{b.get('avg_delay_min', 0):.0f} דק׳ / {b.get('live_aircraft', 'n/a')} מטוסים. "
                f"ציונים: {a.get('iata')} {float(a.get('investment_score') or 0):.1f} מול "
                f"{b.get('iata')} {float(b.get('investment_score') or 0):.1f}."
            )
        else:
            lines.append(
                f"**Congestion compare:** {a.get('iata')} has {a.get('avg_delay_min', 0):.0f} min FAA delay "
                f"and {a.get('live_aircraft', 'n/a')} live aircraft nearby vs {b.get('iata')} "
                f"{b.get('avg_delay_min', 0):.0f} min / {b.get('live_aircraft', 'n/a')} aircraft. "
                f"Investment scores: {a.get('iata')} {float(a.get('investment_score') or 0):.1f} vs "
                f"{b.get('iata')} {float(b.get('investment_score') or 0):.1f}."
            )
        lines.append("")
    elif unmet_q:
        row = table_records[0]
        delay = row.get("delay_reason") or (
            "אין סיבת עיכוב FAA כרגע" if he else "no FAA delay reason right now"
        )
        if he:
            lines.append(
                f"**קירוב ביקוש שלא נענה ב-{row.get('iata')}:** מדד {row.get('unmet_demand_index', 'n/a')} "
                f"(ביקוש × עומס אחרי min-max בקבוצה הזו). אין נתון על נוסעים שנדחו; "
                f"הקירוב הוא רשת צפופה פלוס אילוץ חי ({delay})."
            )
        else:
            lines.append(
                f"**Unmet-demand proxy at {row.get('iata')}:** index {row.get('unmet_demand_index', 'n/a')} "
                f"(demand × congestion after min-max in this set). We do not observe turned-away passengers; "
                f"the proxy is a dense route network plus live constraint ({delay})."
            )
        lines.append("")
    else:
        top = table_records[0]
        if he:
            lines.append(
                f"**מועמד ראשון: {top.get('name')} ({top.get('iata')})**, "
                f"ציון {float(top.get('investment_score') or 0):.1f}. "
                "הציון משלב ביקוש נתיבים, עומס חי, קירוב לביקוש שלא נענה ונתח טיסות ארוכות טווח — "
                "זה לא תחרות פופולריות ולא רווח בדולרים."
            )
        else:
            lines.append(
                f"**Top candidate: {top.get('name')} ({top.get('iata')})**, "
                f"investment score {float(top.get('investment_score') or 0):.1f}. "
                "The score combines route demand, live congestion, an unmet-demand proxy, and long-haul share — "
                "it is not a popularity contest or a dollar profit."
            )
        lines.append("")
    for i, row in enumerate(table_records[:5], 1):
        reason = _row_reason(row, lang)
        lines.append(
            f"{i}. {row.get('name')} ({row.get('iata')}): score {float(row.get('investment_score') or 0):.1f} — {reason}"
        )
    lines.append("")
    label = "הנחות / אי-ודאות: " if he else "Assumptions / uncertainty: "
    lines.append(label + " | ".join(notes[:6]))
    return "\n".join(lines)


def explain_results(
    user_message: str,
    weights: dict,
    table_records: list[dict],
    warnings: list[str],
    conversation_history: list[dict] | None = None,
    lang: str | None = None,
) -> str:
    lang = lang or reply_lang(user_message)
    notes = localize_warnings(warnings, lang)
    if not _has_anthropic():
        return _template_explain(user_message, weights, table_records, notes, lang=lang)

    lang_rule = (
        "Write the entire answer in Hebrew. Do not mix English sentence fragments except IATA codes and airport names."
        if lang == "he"
        else "Write the entire answer in English."
    )
    context = {
        "user_question": user_message,
        "weights_used": weights,
        "ranked_airports": table_records,
        "pipeline_warnings": notes,
    }
    history = conversation_history or []
    messages = history + [{
        "role": "user",
        "content": f"Explain this result:\n{json.dumps(context, indent=2, default=str)}",
    }]
    resp = get_client().messages.create(
        model=MODEL,
        max_tokens=700,
        system=EXPLAIN_SYSTEM_PROMPT + "\n" + lang_rule,
        messages=messages,
    )
    return "".join(block.text for block in resp.content if block.type == "text").strip()


HELP_TEXT_EN = """This agent is **only** for advising on **US airport expansion / modernization** (extra flight and passenger capacity).

It does not answer general questions, other countries, or topics outside that mandate.

Ask something that names a **US airport** or a **US region**, for example:
- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?
"""

HELP_TEXT_HE = """הסוכן הזה מיועד **רק** לייעוץ על **הרחבה / מודרניזציה של שדות תעופה בארה״ב** (קיבולת טיסות ונוסעים).

הוא לא עונה על שאלות כלליות, מדינות אחרות, או נושאים מחוץ למנדט.

שאלי משהו שכולל **שדה בארה״ב** או **אזור בארה״ב**, לדוגמה:
- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?
"""


def help_reply(user_message: str = "", lang: str | None = None) -> str:
    lang = lang or reply_lang(user_message)
    return HELP_TEXT_HE if lang == "he" else HELP_TEXT_EN


def general_chat_reply(
    user_message: str,
    last_table_records: list[dict] | None,
    warnings: list[str],
    conversation_history: list[dict] | None = None,
    lang: str | None = None,
) -> str:
    lang = lang or reply_lang(user_message)
    notes = localize_warnings(warnings, lang)
    if not _has_anthropic():
        rows = last_table_records or []
        if not rows:
            return help_reply(user_message, lang=lang)
        top = rows[0]
        score = float(top.get("investment_score") or 0)
        reason = format_reason(top, lang)
        if lang == "he":
            return (
                f"**{top.get('name')} ({top.get('iata')})** בראש עם ציון "
                f"{score:.1f}. {reason}\n\n"
                "הציון הוא סכום משוקלל (ביקוש, עומס, קירוב לביקוש שלא נענה, טיסות ארוכות טווח) "
                "בתוך **הטבלה הזו**, לא רווח בדולרים. "
                "שאלי שאלה חדשה עם שדה או אזור בארה״ב כדי לחשב מחדש."
            )
        return (
            f"**{top.get('name')} ({top.get('iata')})** ranks first with investment score "
            f"{score:.1f}. {reason}\n\n"
            "That score is demand + congestion + unmet-demand proxy + long-haul share **in this comparison set**, "
            "not a dollar profit. Ask a new US airport or region question to refresh the table."
        )
    lang_rule = (
        "Reply entirely in Hebrew except IATA codes and official airport names."
        if lang == "he"
        else "Reply entirely in English."
    )
    system = (
        "You are embedded in an airport investment-screening tool. Answer follow-ups using ONLY "
        "the last ranking table and warnings. Do not invent new airport statistics. "
        "If the user asks for a new comparison you cannot answer from the table, say they should "
        "ask a fresh ranking/compare question. " + lang_rule
    )
    payload = {
        "user_question": user_message,
        "last_table": last_table_records or [],
        "warnings": notes,
    }
    history = conversation_history or []
    messages = history + [{"role": "user", "content": json.dumps(payload, default=str)}]
    resp = get_client().messages.create(model=MODEL, max_tokens=500, system=system, messages=messages)
    return "".join(block.text for block in resp.content if block.type == "text").strip()
