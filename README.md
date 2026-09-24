# US Airport Modernization Agent

This agent screens **US airports** for a firm that funds terminal / flight-capacity
modernization. It answers in a chat (optional microphone next to Send).

The ranking is a **deterministic score**, not an LLM guess. The model may parse the
question and explain the table. It does **not** invent airport KPIs.

Live public APIs: [Airport Gap](https://airportgap.com),
[FAA ASWS](https://nasstatus.faa.gov/), [OpenSky](https://opensky-network.org).
Route demand and long-haul % come from bundled files in `data/` (datasets, not APIs).

```
investment_score = 100 × (
    0.30·demand + 0.30·congestion + 0.25·unmet_demand + 0.15·long_haul
)
```

Scores are min-max **inside the current comparison**. This is **not** a dollar NPV
(no construction cost, fares, or politics). `long_haul_pct` counts unique published
routes, not seats or weekly frequencies.

Scope: **US only**. Vague questions with no US airport or region get a reminder
instead of ranking the whole country. Follow-ups such as “why the first airport?”
reuse the last table.

Full method: [`DESIGN.md`](DESIGN.md).

## Run

Needs Python 3.10+. Opens http://localhost:8501

**Mac** (Terminal):

```bash
cd airport-agent-main
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

**Windows** (PowerShell):

```powershell
cd airport-agent-main
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

`ANTHROPIC_API_KEY` is **optional**. Without it, a keyword parser still ranks and
explains.

Try:

- Which airports in New England are strong candidates for terminal expansion?
- Compare LA and Santa Ana airport congestion levels.
- What is the percentage of long haul flights out of Anchorage airport?
- What is the unmet flight demand in SFO airport and why?

Then: `why the first airport?` · `which` (should stay in scope, not rank all US).

## Layout

| File | Role |
|---|---|
| `data_fetcher.py` | Airport Gap + FAA + OpenSky (REST). Local `data/` catalog + routes. |
| `scoring.py` | Deterministic investment score. |
| `llm_interface.py` | Parse + explain only. |
| `app.py` | Streamlit chat + voice. |
| `regions.py` | US region boxes and airport aliases. |
| `DESIGN.md` | Scoring, data, where AI is used. |

```bash

python3 -m pytest test_scoring.py -v```
