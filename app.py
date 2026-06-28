"""
app.py  —  ARENA  |  Streamlit Community Cloud entry point

Deploy:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io → New app → select this repo → app.py
    3. Leave deployment secrets empty for public use
    4. Each user pastes their own Anthropic API key in the sidebar

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
    Then paste your Anthropic API key into the sidebar for that session.
"""

import hashlib
import io
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import ReliabilityAgent

LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024
MAX_UPLOAD_ROWS = 25_000
MAX_UPLOAD_COLUMNS = 80
MAX_SESSION_TURNS = 10
MAX_PROMPT_CHARS = 2_000

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ARENA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Tighten chat message spacing */
    .stChatMessage { padding: 0.5rem 0; }
    /* Style the quick-prompt buttons */
    div[data-testid="column"] .stButton button {
        font-size: 0.8rem;
        padding: 0.35rem 0.6rem;
        border-radius: 20px;
        border: 1px solid #d0d0d0;
        background: #f7f7fb;
        color: #333;
        text-align: left;
        white-space: normal;
        height: auto;
        line-height: 1.3;
    }
    div[data-testid="column"] .stButton button:hover {
        background: #ededff;
        border-color: #aaa;
    }
    /* Metric card tweaks */
    [data-testid="metric-container"] { background: #f7f8fc; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Sample datasets bundled with the repo ─────────────────────
DATA_DIR = Path(__file__).parent / "data"
SAMPLE_DATASETS = {
    "Battery cells — wear-out, 4 stress groups":      "battery_cell_events.csv",
    "Bearing wear — speed × load stress":             "bearing_wear_events.csv",
    "Semiconductor burn-in — infant mortality":        "semiconductor_burnin_events.csv",
    "LED lumen degradation — L70 threshold":          "led_events.csv",
    "Field warranty returns — mixed fleet":            "field_warranty_returns.csv",
}

QUICK_PROMPTS = [
    "Summarise the dataset and overall failure rate",
    "Fit a Weibull model and interpret the beta value",
    "Show B10 and B50 life for each group",
    "Run a Kaplan-Meier breakdown by group",
    "Which factors most accelerate failure?",
    "What are the top design risks based on this data?",
]


def _api_key_fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _format_columns(columns: list[str]) -> str:
    return ", ".join(f"`{column}`" for column in columns) if columns else "—"


def _validate_dataset_limits(df: pd.DataFrame) -> str | None:
    if df.empty:
        return "The dataset is empty."
    if len(df) > MAX_UPLOAD_ROWS:
        return f"Dataset exceeds the {MAX_UPLOAD_ROWS:,}-row limit."
    if len(df.columns) > MAX_UPLOAD_COLUMNS:
        return f"Dataset exceeds the {MAX_UPLOAD_COLUMNS}-column limit."
    return None


def _load_sample_csv(csv_path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        LOGGER.exception("Failed to load sample dataset: %s", csv_path)
        return None, "Could not read the bundled sample dataset."

    validation_error = _validate_dataset_limits(df)
    if validation_error:
        return None, validation_error
    return df, None


def _load_uploaded_csv(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
        max_megabytes = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        return None, f"Upload exceeds the {max_megabytes} MB limit."

    try:
        raw_bytes = uploaded_file.getvalue()
        df = pd.read_csv(io.BytesIO(raw_bytes), nrows=MAX_UPLOAD_ROWS + 1)
    except Exception:
        LOGGER.exception("Failed to parse uploaded CSV: %s", uploaded_file.name)
        return None, "Could not read the uploaded CSV. Use a valid comma-separated file."

    validation_error = _validate_dataset_limits(df)
    if validation_error:
        return None, validation_error
    return df, None


def _session_turn_count() -> int:
    return sum(1 for message in st.session_state.get("messages", []) if message["role"] == "user")


def _submit_prompt(prompt: str) -> bool:
    prompt = prompt.strip()
    if not prompt:
        return False

    if _session_turn_count() >= MAX_SESSION_TURNS:
        st.warning(
            f"Session question limit reached ({MAX_SESSION_TURNS}). "
            "Clear the conversation to continue."
        )
        return False

    if len(prompt) > MAX_PROMPT_CHARS:
        st.warning(f"Prompt too long. Keep questions under {MAX_PROMPT_CHARS:,} characters.")
        return False

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                reply = st.session_state.agent.chat(prompt)
            except Exception:
                LOGGER.exception("Agent chat failed")
                reply = (
                    "Analysis failed. Check your API key, dataset, and usage limits, "
                    "then try again."
                )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    return True

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 ARENA")
    st.caption("Agentic Reliability Engineering & Analysis")
    st.divider()

    st.markdown("**Session API key**")
    st.caption(
        "This app never auto-uses deployment secrets or environment keys. "
        "Each session requires a user-supplied Anthropic API key."
    )
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-...",
        help="Your key is used only within your session and never stored or logged.",
    )

    st.divider()

    # Dataset source
    st.markdown("**Load a dataset**")
    source = st.radio("Source", ["Sample datasets", "Upload your own CSV"],
                      label_visibility="collapsed")

    df: pd.DataFrame | None = None
    dataset_label = None

    if source == "Sample datasets":
        choice = st.selectbox(
            "Choose a sample",
            ["— pick one —"] + list(SAMPLE_DATASETS.keys()),
            label_visibility="collapsed",
        )
        if choice != "— pick one —":
            csv_path = DATA_DIR / SAMPLE_DATASETS[choice]
            if csv_path.exists():
                df, load_error = _load_sample_csv(csv_path)
                if load_error:
                    st.warning(load_error)
                else:
                    dataset_label = choice
            else:
                st.warning(f"File not found: {csv_path.name}")
    else:
        st.caption(
            "Uploaded CSVs may be processed by Anthropic to answer your questions. "
            "Do not upload secrets, personal data, or regulated data."
        )
        st.caption(
            f"Upload limits: {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB, "
            f"{MAX_UPLOAD_ROWS:,} rows, {MAX_UPLOAD_COLUMNS} columns."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                     label_visibility="collapsed")
        if uploaded:
            df, load_error = _load_uploaded_csv(uploaded)
            if load_error:
                st.warning(load_error)
            else:
                dataset_label = uploaded.name

    st.divider()

    # Schema info (shown once agent is loaded)
    if "agent" in st.session_state:
        schema = st.session_state.agent.schema
        st.markdown("**Detected schema**")
        st.markdown(
            f"- **Rows**: {schema['n_rows']:,}\n"
            f"- **Time**: `{schema.get('time_col') or '—'}`\n"
            f"- **Event**: `{schema.get('event_col') or '—'}`\n"
            f"- **Group**: `{schema.get('group_col') or '—'}`"
        )
        with st.expander("All columns"):
            for col, dtype in schema["dtypes"].items():
                st.markdown(f"- `{col}` — {dtype}")
        with st.expander("Model-visible analysis columns"):
            st.markdown(f"- **Grouping**: {_format_columns(schema.get('safe_group_columns', []))}")
            st.markdown(f"- **Numeric covariates**: {_format_columns(schema.get('safe_covariate_columns', []))}")
            st.markdown(f"- **DOE factors**: {_format_columns(schema.get('binary_factor_columns', []))}")
            blocked_count = len(schema.get("blocked_columns", []))
            st.caption(
                f"{blocked_count} sensitive-looking column(s) are blocked from model-driven analysis."
            )
        st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        if "agent" in st.session_state:
            st.session_state.agent.reset()
        st.session_state.messages = []
        st.session_state.pop("_queued", None)
        st.rerun()

# ── Initialise / swap agent ────────────────────────────────────
if df is not None and api_key:
    api_key_fingerprint = _api_key_fingerprint(api_key)
    if (
        "agent" not in st.session_state
        or st.session_state.get("_dataset_label") != dataset_label
        or st.session_state.get("_api_key_fingerprint") != api_key_fingerprint
    ):
        try:
            with st.spinner("Loading dataset…"):
                st.session_state.agent = ReliabilityAgent(df, api_key=api_key)
                st.session_state._dataset_label = dataset_label
                st.session_state._api_key_fingerprint = api_key_fingerprint
                st.session_state.messages = []
        except Exception:
            LOGGER.exception("Could not initialise agent")
            st.error("Could not initialise the agent. Check your API key and dataset format.")
            st.stop()

# ── Main area ──────────────────────────────────────────────────
st.markdown("# 📊 ARENA")

# ── Gate: no dataset ──
if df is None:
    st.info("👈  Pick a sample dataset or upload a CSV from the sidebar to get started.")

    # Show a preview of what's available
    st.markdown("### Available sample datasets")
    cols = st.columns(3)
    icons = ["🔋", "⚙️", "💾", "💡", "🚗"]
    descs = [
        "140 battery cells across 4 temperature × C-rate stress conditions. "
        "Tests wear-out Weibull fitting and AFT stress modelling.",
        "80 bearings across 3 speeds and 3 loads. "
        "Strong wear-out regime with vibration monitoring signals.",
        "200 chips across temperature and voltage burn-in stress. "
        "Infant mortality regime (β < 1).",
        "60 LEDs tracked to L70 lumen threshold. "
        "Gradual photonic degradation curves.",
        "500 fleet units with usage profiles, regions, and failure modes. "
        "Multi-mode warranty analysis.",
    ]
    for i, (name, desc) in enumerate(zip(SAMPLE_DATASETS.keys(), descs)):
        with cols[i % 3]:
            st.markdown(
                f"<div style='padding:12px;border:1px solid #e0e0e0;border-radius:10px;"
                f"margin-bottom:10px;background:#fafafa'>"
                f"<div style='font-size:1.4rem'>{icons[i]}</div>"
                f"<div style='font-weight:600;margin:4px 0 4px'>{name.split('—')[0].strip()}</div>"
                f"<div style='font-size:0.82rem;color:#555'>{desc}</div></div>",
                unsafe_allow_html=True,
            )
    st.stop()

# ── Gate: no API key ──
if not api_key:
    st.warning("🔑  Enter your own Anthropic API key in the sidebar to start chatting.")
    st.stop()

# ── Gate: agent not ready ──
if "agent" not in st.session_state:
    st.warning("Dataset loaded but agent not ready. Check your API key.")
    st.stop()

# ── Dataset metrics strip ──────────────────────────────────────
schema  = st.session_state.agent.schema
ev_col  = schema.get("event_col")
t_col   = schema.get("time_col")

m1, m2, m3, m4 = st.columns(4)
n       = schema["n_rows"]
n_ev    = int(df[ev_col].sum()) if ev_col and ev_col in df.columns else 0
fr_pct  = round(100 * n_ev / n, 1) if n else 0
t_range = f"{df[t_col].min():.0f} – {df[t_col].max():.0f}" if t_col and t_col in df.columns else "—"

m1.metric("Dataset",    dataset_label.split("—")[0].strip() if dataset_label else "—")
m2.metric("Rows",       f"{n:,}")
m3.metric("Failure rate", f"{fr_pct}%")
m4.metric("Time range", t_range)

st.divider()

# ── Quick-start prompts (shown only before first message) ──────
session_limit_reached = _session_turn_count() >= MAX_SESSION_TURNS
st.caption(
    f"Session usage: {_session_turn_count()}/{MAX_SESSION_TURNS} questions. "
    f"Prompt limit: {MAX_PROMPT_CHARS:,} characters."
)

if not st.session_state.get("messages"):
    st.markdown("**Suggested questions — click to send:**")
    cols = st.columns(3)
    for i, prompt in enumerate(QUICK_PROMPTS):
        with cols[i % 3]:
            if st.button(prompt, key=f"qp_{i}", use_container_width=True,
                         disabled=session_limit_reached):
                st.session_state._queued = prompt
                st.rerun()

# ── Chat history ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle queued quick-prompt ─────────────────────────────────
if st.session_state.get("_queued"):
    prompt = st.session_state.pop("_queued")
    if _submit_prompt(prompt):
        st.rerun()

# ── Live chat input ────────────────────────────────────────────
if session_limit_reached:
    st.info("Session question limit reached. Clear the conversation to continue.")
elif user_input := st.chat_input("Ask a reliability question…"):
    _submit_prompt(user_input)
