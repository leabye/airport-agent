# US Airport Modernization Agent

Conversational agent that ranks **US** airports for modernization investment
using a **deterministic** score (demand × congestion proxy) plus an LLM only
for intent parsing and explanation.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional; keyword parser works without it
streamlit run app.py
```

Try the assignment prompts:

- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?

## Layout

| File | Role |
|---|---|
| `data_fetcher.py` | Airport Gap + FAA + OpenSky (REST). Local `data/` catalog + routes. No scoring. |
| `scoring.py` | Deterministic investment score. Unit-tested. |
| `llm_interface.py` | Claude (or heuristic fallback). Parse + explain only. |
| `app.py` | Streamlit chat (+ optional browser voice). |
| `regions.py` | US region boxes and airport aliases. |
| `DESIGN.md` | Scoring, tradeoffs, where AI is used. |

## Tests (no API key)

```bash
pytest test_scoring.py -v
```
