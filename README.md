# ARENA — Agentic Reliability Engineering & Analysis

An agentic AI assistant for reliability engineering — Weibull fitting, Kaplan-Meier survival curves, AFT stress modelling, DOE analysis, and B-life estimation.

---

## What it does

Upload a failure-event CSV, enter your Anthropic API key, and ask reliability questions in plain English. The agent selects and runs the appropriate statistical analysis, then explains the results.

**Example questions you can ask:**
- "Fit a Weibull distribution and give me B10 life."
- "Plot a Kaplan-Meier survival curve grouped by test condition."
- "Which stress factors significantly affect time-to-failure?"
- "Run a DOE analysis and rank the main effects."
- "What is the B5 life at 85 °C using the Arrhenius model?"

---

## Deploy to Streamlit Community Cloud (free, no install)

1. **Fork or push this repo to GitHub**

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub

3. Click **New app** → select your repo → set the main file to `app.py` → click **Deploy**

4. Leave **Advanced settings → Secrets** empty for public deployments. This app intentionally does **not** auto-use deployment secrets or environment keys.

5. Each user enters their own Anthropic API key in the sidebar for that session.

6. Done — your app is live at `https://YOUR-APP-NAME.streamlit.app`

---

## Run locally

**Requirements:** Python 3.10+, an [Anthropic API key](https://console.anthropic.com/)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Paste your Anthropic API key into the sidebar when the app opens.

---

## Security defaults

- The app never auto-uses `ANTHROPIC_API_KEY` from deployment secrets or environment variables.
- Uploaded CSVs are capped at 5 MB, 25,000 rows, and 80 columns.
- Sessions are capped at 10 user questions, with a 2,000-character prompt limit.
- Only safe analysis columns are exposed to the model; sensitive-looking and high-cardinality columns are blocked from model-driven grouping and breakdowns.
- Uploaded CSVs may be processed by Anthropic during analysis. Do not upload secrets, personal data, or regulated data.

---

## File structure

```
.
├── app.py                  ← Streamlit entry point (Community Cloud looks for this)
├── agent.py                ← Agentic tool-use loop (Claude + reliability tools)
├── reliability_engine.py   ← Pure statistical functions (Weibull, KM, AFT, DOE)
├── requirements.txt        ← Dependencies
├── data/                   ← Bundled sample datasets
│   ├── battery_cell_events.csv
│   ├── battery_degradation_curves.csv
│   ├── bearing_wear_events.csv
│   ├── semiconductor_burnin_events.csv
│   ├── led_events.csv
│   ├── led_lumen_degradation_curves.csv
│   └── field_warranty_returns.csv
└── .gitignore
```

All three Python files live at the **repo root** — this is required for Streamlit Community Cloud to find them correctly.

---

## Who pays for the API?

| Setup | Who pays |
|---|---|
| Each user enters a key in the sidebar | That user |
| No key entered | No API calls are made |

This repo is configured for bring-your-own-key usage. It does not support silent shared-key billing by default.

---

## Sample datasets included

| Dataset | Domain | Failure regime |
|---|---|---|
| Battery cell events | Energy storage | Wear-out (β ≈ 2.1–2.5) |
| Battery degradation curves | Energy storage | Capacity fade over cycles |
| Bearing wear events | Rotating machinery | Wear-out (β = 2.8) |
| Semiconductor burn-in | Electronics | Infant mortality (β = 0.65) |
| LED lumen events | Photonics | L70 degradation threshold |
| LED lumen degradation curves | Photonics | Lumen maintenance over time |
| Field warranty returns | Fleet / warranty | Mixed usage profiles |

---

## Automated security checks

GitHub Actions runs `pip-audit` and `bandit` on pushes to `main` and on pull requests via [.github/workflows/security.yml](.github/workflows/security.yml).

---

## Architecture & design

See [ARCHITECTURE.md](ARCHITECTURE.md) for statistical foundations, agent design, session lifecycle, privacy model, and architectural trade-offs.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
