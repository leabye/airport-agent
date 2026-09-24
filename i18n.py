"""Hebrew copies of pipeline warnings. UI and default answers stay English."""

WARNING_HE = {
    "Score is a relative 0–100 ranking inside this comparison set, not a dollar profit forecast.": (
        "הציון הוא דירוג יחסי 0–100 בתוך ההשוואה הזו, לא תחזית רווח בדולרים."
    ),
    "long_haul_pct counts unique published routes, not seats or weekly frequencies.": (
        "long_haul_pct סופר נתיבים מפורסמים ייחודיים, לא מושבים ולא תדירות שבועית."
    ),
    "No airports matched the given filters.": "אף שדה תעופה לא התאים למסננים.",
    "Parsed without an LLM (no ANTHROPIC_API_KEY); used keyword rules.": (
        "ניתוח בלי LLM (אין ANTHROPIC_API_KEY); נעשה שימוש בכללי מילות מפתח."
    ),
    "Tilted weights toward congestion because the question asked about congestion.": (
        "המשקלות הוטו לעומס כי השאלה עסקה בעומס."
    ),
    "Tilted weights toward long-haul share.": "המשקלות הוטו לנתח הטיסות ארוכות הטווח.",
    "Tilted weights toward unmet-demand proxy.": "המשקלות הוטו לקירוב הביקוש שלא נענה.",
    "Follow-up with no previous ranking; showed scope help.": (
        "שאלת המשך בלי דירוג קודם; הוצגה תזכורת היקף."
    ),
    "No region named; defaulted to all US commercial airports with published routes.": (
        "לא צוין אזור; ברירת מחדל: שדות תעופה מסחריים בארה״ב עם נתיבים מפורסמים."
    ),
    "No US airport or region in the question; stayed in scope instead of ranking all US.": (
        "אין שדה או אזור בארה״ב בשאלה; נשארנו בהיקף במקום לדרג את כל ארה״ב."
    ),
    "Compare requested but fewer than two US airports named; showed scope help.": (
        "נתבקשה השוואה בלי שני שדות בארה״ב; הוצגה תזכורת היקף."
    ),
    "Metric requested but no US airport named; showed scope help.": (
        "נתבקש מדד בלי שדה בארה״ב; הוצגה תזכורת היקף."
    ),
}


def localize_warnings(warnings: list, lang: str) -> list:
    if lang != "he":
        return list(warnings or [])
    return [WARNING_HE.get(w, w) for w in (warnings or [])]
