"""
Streamlit chat UI for the US airport modernization investment agent.

Pipeline: parse intent → public APIs + deterministic score → LLM narration.
Follow-ups reuse the last table instead of silently re-scoring the whole US.
"""

from __future__ import annotations

import hashlib

import streamlit as st

import data_fetcher
import i18n
import llm_interface
import scoring


def transcribe_recording(raw: bytes, mime: str | None = None) -> str | None:
    """Turn a Chrome/Streamlit mic clip into text. Writes a temp file so WAV/WebM both work."""
    try:
        import speech_recognition as sr
    except ImportError:
        return None
    import os
    import subprocess
    import tempfile

    mime = (mime or "").lower()
    suffix = ".wav"
    if "webm" in mime:
        suffix = ".webm"
    elif "ogg" in mime:
        suffix = ".ogg"
    elif "mpeg" in mime or "mp3" in mime:
        suffix = ".mp3"

    src_path = None
    wav_path = None
    primary = "en-US"
    fallback = "fr-FR"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src:
            src.write(raw)
            src_path = src.name
        wav_path = src_path
        if suffix != ".wav":
            wav_path = src_path + ".wav"
            conv = subprocess.run(
                ["ffmpeg", "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path],
                capture_output=True,
                timeout=20,
            )
            if conv.returncode != 0:
                return None
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language=primary)
        except sr.UnknownValueError:
            try:
                return recognizer.recognize_google(audio, language="he-IL")
            except sr.UnknownValueError:
                return recognizer.recognize_google(audio, language=fallback)
    except Exception:
        return None
    finally:
        for path in (src_path, wav_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


st.set_page_config(
    page_title="Airport Investment Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, rgba(125, 211, 252, 0.16), transparent 55%),
            radial-gradient(900px 420px at 100% 0%, rgba(245, 193, 108, 0.10), transparent 50%),
            #08111C;
    }
    .block-container { padding-top: 1.4rem; max-width: 1120px; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stDecoration"] { display: none; }
    .hero {
        border: 1px solid rgba(125, 211, 252, 0.18);
        background: linear-gradient(135deg, rgba(18, 32, 51, 0.92), rgba(8, 17, 28, 0.65));
        border-radius: 22px;
        padding: 1.55rem 1.7rem 1.35rem;
        margin-bottom: 0.35rem;
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
    }
    .hero .kicker {
        font-family: "DM Sans", sans-serif;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-size: 0.72rem;
        color: #F5C16C;
        margin: 0 0 0.45rem;
    }
    .hero h1 {
        font-family: "Fraunces", serif;
        font-size: 2.05rem;
        font-weight: 600;
        line-height: 1.15;
        margin: 0 0 0.45rem;
        color: #F4FAFF;
    }
    .hero p {
        margin: 0;
        color: #B7C9D6;
        font-size: 1.02rem;
        max-width: 42rem;
    }
    .stChatMessage { border-radius: 16px; }
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_table" not in st.session_state:
    st.session_state.last_table = None
if "last_warnings" not in st.session_state:
    st.session_state.last_warnings = []
if "pending_voice" not in st.session_state:
    st.session_state.pending_voice = None
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

with st.sidebar:
    st.markdown("**Live APIs**")
    st.caption("Airport Gap · FAA ASWS · OpenSky")
    st.markdown("**Score weights**")
    c1, c2 = st.columns(2)
    c1.metric("Demand", "30%")
    c2.metric("Congestion", "30%")
    c3, c4 = st.columns(2)
    c3.metric("Unmet", "25%")
    c4.metric("Long-haul", "15%")
    with st.expander("Scope & uncertainty", expanded=False):
        st.markdown(
            """
- **Mandate:** rank US airports using extra flight / passenger capacity — not a full financial model.
- **Score:** `100 × (0.30 demand + 0.30 congestion + 0.25 unmet + 0.15 long-haul)` after min-max **in this comparison**.
- **Data:** REST APIs plus bundled `data/` files (catalog + routes). Files are not APIs.
- **Limits:** routes ≠ seats; FAA covers major airports; OpenSky is a snapshot.
- **LLM:** parse and explain only. It does **not** compute the score.
            """
        )

st.markdown(
    """
<div class="hero">
  <p class="kicker">US modernization screen</p>
  <h1>Where extra airport capacity is most likely to be used</h1>
  <p>Deterministic ranking from Airport Gap, FAA, and OpenSky. The model explains the table — it does not invent the score.</p>
</div>
""",
    unsafe_allow_html=True,
)

EXAMPLE_PROMPTS = [
    ("New England", "Which airports in New England are strong candidates for terminal expansion?"),
    ("LA vs Santa Ana", "Compare LA and Santa Ana congestion levels."),
    ("Anchorage long-haul", "What is the percentage of long-haul flights out of Anchorage?"),
    ("SFO unmet demand", "What is the unmet flight demand at SFO and why?"),
]

st.markdown("**Try a question**")
cols = st.columns(len(EXAMPLE_PROMPTS))
for col, (label, prompt) in zip(cols, EXAMPLE_PROMPTS):
    if col.button(label, use_container_width=True):
        st.session_state.pending_voice = prompt
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("table") is not None:
            st.dataframe(msg["table"], width="stretch")

prompt = st.chat_input(
    "Type a question about US airports…",
    accept_audio=True,
    audio_sample_rate=16000,
)

user_input = None
if st.session_state.pending_voice:
    user_input = st.session_state.pending_voice
    st.session_state.pending_voice = None
elif prompt is not None:
    audio = prompt.audio
    typed = (prompt.text or "").strip()
    if audio is not None:
        raw = audio.getvalue()
        digest = hashlib.sha1(raw).hexdigest()
        if digest != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = digest
            with st.spinner("Transcribing your question…"):
                spoken = transcribe_recording(raw, getattr(audio, "type", None))
            if spoken:
                user_input = spoken
            else:
                st.warning("Could not transcribe that recording. Allow the mic, speak clearly, or type.")
    elif typed:
        user_input = typed

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        lang = llm_interface.reply_lang(user_input)
        with st.spinner("Interpreting the question..."):
            intent = llm_interface.parse_intent(user_input, history)

        assumptions = intent.get("assumptions") or []
        q_intent = intent.get("intent") or "rank_expansion"
        airports = [a.upper() for a in (intent.get("airports") or []) if a]
        region = intent.get("region")
        top_n = intent.get("top_n") or 10
        weights = intent.get("weights") or scoring.DEFAULT_WEIGHTS

        if llm_interface.looks_like_followup(user_input) and st.session_state.last_table is not None:
            q_intent = "followup"
        elif q_intent == "followup" and st.session_state.last_table is None:
            q_intent = "help"
            assumptions.append("Follow-up with no previous ranking; showed scope help.")
        elif q_intent != "followup" and not llm_interface.question_has_us_scope(user_input):
            q_intent = "help"

        if q_intent == "help":
            explanation = llm_interface.help_reply(user_input, lang=lang)
            st.markdown(explanation)
            st.session_state.messages.append({"role": "assistant", "content": explanation})
        elif q_intent == "followup" and st.session_state.last_table is not None:
            with st.spinner("Answering from the last ranking..."):
                explanation = llm_interface.general_chat_reply(
                    user_input,
                    st.session_state.last_table.to_dict(orient="records"),
                    st.session_state.last_warnings,
                    history,
                    lang=lang,
                )
            st.markdown(explanation)
            st.session_state.messages.append({"role": "assistant", "content": explanation})
        else:
            with st.spinner("Calling Airport Gap + FAA + OpenSky and scoring..."):
                try:
                    iata = airports or None
                    if q_intent in ("compare", "airport_metric") and not iata:
                        raise ValueError("Need at least one US IATA code for this question.")
                    if q_intent == "rank_expansion" and not airports:
                        iata = None
                    live_limit = 12 if q_intent == "rank_expansion" else 8
                    kpi_table, data_warnings = data_fetcher.build_airport_kpi_table(
                        region=None if iata else region,
                        iata_codes=iata,
                        enrich_live=True,
                        live_limit=live_limit,
                    )
                    data_warnings = i18n.localize_warnings(data_warnings, lang)
                    assumptions = i18n.localize_warnings(assumptions, lang)
                    if q_intent == "compare":
                        result = scoring.compare_airports(
                            kpi_table, weights=weights, extra_warnings=data_warnings, lang=lang
                        )
                    else:
                        result = scoring.score_airports(
                            kpi_table, weights=weights, top_n=top_n, extra_warnings=data_warnings, lang=lang
                        )
                except Exception as e:
                    st.error(f"Data or scoring error: {e}")
                    st.stop()

            all_warnings = assumptions + result.warnings
            with st.spinner("Writing the investment rationale..."):
                explanation = llm_interface.explain_results(
                    user_message=user_input,
                    weights=result.weights,
                    table_records=result.table.to_dict(orient="records"),
                    warnings=all_warnings,
                    conversation_history=history,
                    lang=lang,
                )

            st.markdown(explanation)
            if all_warnings:
                st.info("**Assumptions & uncertainty**\n\n" + "\n".join(f"- {w}" for w in all_warnings))
            st.dataframe(result.table, width="stretch")

            st.session_state.last_table = result.table
            st.session_state.last_warnings = all_warnings
            st.session_state.messages.append({
                "role": "assistant",
                "content": explanation,
                "table": result.table,
            })
