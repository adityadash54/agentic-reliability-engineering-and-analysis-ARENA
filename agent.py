"""
agent.py  —  ARENA agentic tool loop
Imports reliability_engine from the same directory (repo root).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import numpy as np
import pandas as pd
import anthropic

from reliability_engine import (
    fit_weibull_mle,
    fit_weibull_by_group,
    kaplan_meier,
    fit_aft,
    aft_predict,
    fit_ols_effects,
    detect_schema,
    bx_life,
    interpret_beta,
)

# ── Tool definitions ───────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "fit_weibull",
        "description": (
            "Fit a two-parameter Weibull model by MLE to right-censored "
            "time-to-event data. Can fit overall or per-group. Returns beta, "
            "eta, B10, B50, and an interpretation of the failure regime."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to stratify by. Omit for overall fit.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "kaplan_meier",
        "description": (
            "Compute Kaplan-Meier survival estimates. Returns survival "
            "probabilities at key time points and median survival time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to stratify by. Omit for overall curve.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "fit_aft_model",
        "description": (
            "Fit an accelerated failure time (Weibull AFT) model with "
            "covariates to quantify how stress factors accelerate failure. "
            "Returns coefficients showing the log-scale effect of each factor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "covariate_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of numeric covariate column names.",
                }
            },
            "required": ["covariate_cols"],
        },
    },
    {
        "name": "doe_analysis",
        "description": (
            "Run a DOE / OLS main-effects analysis on a response variable "
            "using binary-coded factor columns. Returns estimated effects and p-values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "response_col": {
                    "type": "string",
                    "description": "Numeric column to use as the response.",
                },
                "factor_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Binary 0/1 coded factor columns.",
                },
            },
            "required": ["response_col", "factor_cols"],
        },
    },
    {
        "name": "bx_life_table",
        "description": (
            "Compute B-life values (B5, B10, B20, B50) from a fitted Weibull "
            "model for each stress group."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Column to stratify by.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "dataset_summary",
        "description": (
            "Return a quick summary of the loaded dataset: shape, failure rate, "
            "censoring rate, time range, and safe analysis columns available."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "failure_mode_breakdown",
        "description": (
            "If a failure_mode or categorical column exists, return counts, "
            "rates, and mean time-to-event by mode."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode_col": {
                    "type": "string",
                    "description": "Name of the failure mode / category column.",
                }
            },
            "required": ["mode_col"],
        },
    },
]

LOGGER = logging.getLogger(__name__)

SENSITIVE_COLUMN_TOKENS = {
    "account",
    "address",
    "birth",
    "client",
    "customer",
    "dob",
    "email",
    "id",
    "identifier",
    "name",
    "owner",
    "person",
    "phone",
    "ssn",
    "tax",
    "user",
}
MAX_GROUP_CATEGORICAL_CARDINALITY = 25
MAX_GROUP_NUMERIC_CARDINALITY = 12

# ── JSON serialisation helper ──────────────────────────────────

def _safe_json(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_safe_json(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records"))
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    return obj


def _looks_sensitive(column_name: str) -> bool:
    normalized = column_name.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    if any(token in SENSITIVE_COLUMN_TOKENS for token in tokens):
        return True
    return any(fragment in normalized for fragment in ("email", "phone", "address", "account"))


def _is_binary_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    try:
        unique_values = {float(value) for value in pd.unique(non_null)}
    except (TypeError, ValueError):
        return False
    return unique_values.issubset({0.0, 1.0})


def _format_allowed_columns(columns: list[str]) -> str:
    return ", ".join(columns) if columns else "none"


def _validate_allowed_column(column_name: Optional[str], allowed: list[str], label: str) -> Optional[dict]:
    if not column_name:
        return {"error": f"{label} is required for this analysis."}
    if column_name not in allowed:
        if not allowed:
            return {"error": f"No safe {label.lower()} columns are available for this dataset."}
        return {
            "error": (
                f"{label} '{column_name}' is blocked. "
                f"Allowed columns: {_format_allowed_columns(allowed)}."
            )
        }
    return None


def _validate_allowed_columns(column_names: list[str], allowed: list[str], label: str) -> Optional[dict]:
    invalid = [column_name for column_name in column_names if column_name not in allowed]
    if not invalid:
        return None
    if not allowed:
        return {"error": f"No safe {label.lower()} columns are available for this dataset."}
    return {
        "error": (
            f"Blocked or unavailable {label.lower()}: {invalid}. "
            f"Allowed columns: {_format_allowed_columns(allowed)}."
        )
    }


def _build_analysis_metadata(df: pd.DataFrame, schema: dict) -> dict:
    time_col = schema.get("time_col")
    event_col = schema.get("event_col")

    blocked_columns: list[str] = []
    safe_group_columns: list[str] = []
    safe_breakdown_columns: list[str] = []
    safe_numeric_columns: list[str] = []
    safe_covariate_columns: list[str] = []
    binary_factor_columns: list[str] = []

    for column_name in df.columns:
        series = df[column_name]
        if _looks_sensitive(column_name):
            blocked_columns.append(column_name)
            continue

        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_outcome_indicator = column_name == event_col or "censor" in column_name.lower()

        if is_numeric and not is_outcome_indicator:
            safe_numeric_columns.append(column_name)
            if column_name != time_col:
                safe_covariate_columns.append(column_name)
                if _is_binary_series(series):
                    binary_factor_columns.append(column_name)

        if column_name in {time_col, event_col}:
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        nunique = int(non_null.nunique(dropna=True))
        if nunique < 2:
            continue

        if is_numeric:
            if nunique <= MAX_GROUP_NUMERIC_CARDINALITY:
                safe_group_columns.append(column_name)
            continue

        if nunique <= MAX_GROUP_CATEGORICAL_CARDINALITY:
            safe_group_columns.append(column_name)
            safe_breakdown_columns.append(column_name)

    prompt_group_col = schema.get("group_col")
    if prompt_group_col not in safe_group_columns:
        prompt_group_col = safe_group_columns[0] if safe_group_columns else None

    prompt_time_col = time_col if time_col and not _looks_sensitive(time_col) else None
    prompt_event_col = event_col if event_col and not _looks_sensitive(event_col) else None

    return {
        "n_columns": len(df.columns),
        "blocked_columns": blocked_columns,
        "safe_group_columns": safe_group_columns,
        "safe_breakdown_columns": safe_breakdown_columns,
        "safe_numeric_columns": safe_numeric_columns,
        "safe_covariate_columns": safe_covariate_columns,
        "binary_factor_columns": binary_factor_columns,
        "prompt_group_col": prompt_group_col,
        "prompt_time_col": prompt_time_col,
        "prompt_event_col": prompt_event_col,
    }


def _build_agent_schema(df: pd.DataFrame) -> dict:
    schema = detect_schema(df)
    schema.update(_build_analysis_metadata(df, schema))
    return schema

# ── Tool executor ──────────────────────────────────────────────

def execute_tool(name: str, inputs: dict, df: pd.DataFrame, schema: dict) -> dict:
    time_col  = schema.get("time_col")
    event_col = schema.get("event_col")
    safe_group_columns = schema.get("safe_group_columns", [])
    safe_breakdown_columns = schema.get("safe_breakdown_columns", [])
    safe_covariate_columns = schema.get("safe_covariate_columns", [])
    binary_factor_columns = schema.get("binary_factor_columns", [])
    safe_numeric_columns = schema.get("safe_numeric_columns", [])

    try:
        if name == "dataset_summary":
            ec = event_col or next((c for c in df.columns if any(k in c.lower() for k in ("event","fail","return","claim"))), None)
            tc = time_col  or next((c for c in df.columns if any(k in c.lower() for k in ("time","cycle","hour"))), None)
            n  = len(df)
            nev = int(df[ec].sum()) if ec else 0
            return {
                "n_rows": n,
                "n_columns": len(df.columns),
                "n_failures": nev,
                "n_censored": n - nev,
                "failure_rate_pct": round(100 * nev / n, 1),
                "time_range": [float(df[tc].min()), float(df[tc].max())] if tc else None,
                "time_col": schema.get("prompt_time_col"),
                "event_col": schema.get("prompt_event_col"),
                "preferred_group_col": schema.get("prompt_group_col"),
                "safe_group_columns": safe_group_columns,
                "safe_numeric_columns": safe_numeric_columns,
                "safe_covariate_columns": safe_covariate_columns,
                "binary_factor_columns": binary_factor_columns,
                "blocked_column_count": len(schema.get("blocked_columns", [])),
            }

        if name == "fit_weibull":
            if not time_col or not event_col:
                return {"error": "Could not detect time/event columns."}
            gc = inputs.get("group_col")
            if gc:
                validation_error = _validate_allowed_column(gc, safe_group_columns, "Grouping column")
                if validation_error:
                    return validation_error
                return {"weibull_by_group": _safe_json(fit_weibull_by_group(df, time_col, event_col, gc))}
            return {"weibull_overall": _safe_json(fit_weibull_mle(df[time_col].values, df[event_col].values))}

        if name == "kaplan_meier":
            if not time_col or not event_col:
                return {"error": "Could not detect time/event columns."}
            gc = inputs.get("group_col")
            if gc:
                validation_error = _validate_allowed_column(gc, safe_group_columns, "Grouping column")
                if validation_error:
                    return validation_error
                out = {}
                for grp, g in df.groupby(gc):
                    km   = kaplan_meier(g[time_col].values, g[event_col].values)
                    ta, sa = km["time"].values, km["survival"].values
                    out[str(grp)] = {
                        "median_survival": float(ta[np.searchsorted(-sa, -0.5)]) if (sa < 0.5).any() else None,
                        "final_survival":  round(float(sa[-1]), 4),
                    }
                return {"kaplan_meier_by_group": out}
            km = kaplan_meier(df[time_col].values, df[event_col].values)
            return {"kaplan_meier_summary": _safe_json(km.tail(10))}

        if name == "fit_aft_model":
            if not time_col or not event_col:
                return {"error": "Could not detect time/event columns."}
            cov_cols = inputs.get("covariate_cols", [])
            if not cov_cols:
                return {"error": "At least one safe numeric covariate is required."}
            validation_error = _validate_allowed_columns(cov_cols, safe_covariate_columns, "Covariate")
            if validation_error:
                return validation_error
            return {"aft_model": _safe_json(fit_aft(df, time_col, event_col, cov_cols))}

        if name == "doe_analysis":
            resp  = inputs.get("response_col")
            facts = inputs.get("factor_cols", [])
            response_error = _validate_allowed_column(resp, safe_numeric_columns, "Response column")
            if response_error:
                return response_error
            factor_error = _validate_allowed_columns(facts, binary_factor_columns, "Factor")
            if factor_error:
                return factor_error
            return {"doe_effects": _safe_json(fit_ols_effects(df, resp, facts))}

        if name == "bx_life_table":
            gc    = inputs.get("group_col")
            if gc:
                validation_error = _validate_allowed_column(gc, safe_group_columns, "Grouping column")
                if validation_error:
                    return validation_error
            pairs = list(df.groupby(gc)) if gc else [("overall", df)]
            rows  = []
            for grp, g in pairs:
                fit = fit_weibull_mle(g[time_col].values, g[event_col].values)
                if fit["success"] and np.isfinite(fit["shape_beta"]):
                    b, e = fit["shape_beta"], fit["scale_eta"]
                    rows.append({
                        "group": str(grp), "beta": round(b, 3),
                        "B5":  round(bx_life(0.05, b, e), 1),
                        "B10": round(bx_life(0.10, b, e), 1),
                        "B20": round(bx_life(0.20, b, e), 1),
                        "B50": round(bx_life(0.50, b, e), 1),
                        "interpretation": interpret_beta(b),
                    })
            return {"bx_life_table": rows}

        if name == "failure_mode_breakdown":
            mc = inputs.get("mode_col")
            validation_error = _validate_allowed_column(mc, safe_breakdown_columns, "Breakdown column")
            if validation_error:
                return validation_error
            rows = []
            for mode, g in df.groupby(mc):
                row = {"mode": str(mode), "count": len(g),
                       "pct": round(100 * len(g) / len(df), 1)}
                if time_col:
                    row["mean_time"] = round(float(g[time_col].mean()), 1)
                rows.append(row)
            return {"failure_mode_breakdown": sorted(rows, key=lambda x: -x["count"])}

    except Exception:
        LOGGER.exception("Tool execution failed: %s", name)
        return {"error": "Tool execution failed. Try a simpler request or smaller dataset."}

    return {"error": f"Unknown tool: {name}"}

# ── System prompt ──────────────────────────────────────────────

def _build_system_prompt(schema: dict) -> str:
    safe_group_columns = _format_allowed_columns(schema.get("safe_group_columns", []))
    safe_covariate_columns = _format_allowed_columns(schema.get("safe_covariate_columns", []))
    binary_factor_columns = _format_allowed_columns(schema.get("binary_factor_columns", []))
    return (
        f"You are an expert reliability engineer and data scientist.\n"
        f"Dataset: {schema['n_rows']:,} rows, {schema.get('n_columns', len(schema['columns'])):,} columns.\n"
        f"Detected analysis targets — time: {schema.get('prompt_time_col') or 'not detected'}, "
        f"event: {schema.get('prompt_event_col') or 'not detected'}, "
        f"group: {schema.get('prompt_group_col') or 'not available'}.\n"
        f"Safe grouping columns: {safe_group_columns}.\n"
        f"Safe numeric covariates: {safe_covariate_columns}.\n"
        f"Safe DOE factor columns: {binary_factor_columns}.\n"
        "Sensitive-looking or high-cardinality columns are blocked from model-driven analysis.\n\n"
        "Use the available tools to answer questions with real computed numbers. "
        "Interpret results in plain engineering language. Highlight key numbers "
        "(beta, B10, p-values, failure rates). Give actionable recommendations. "
        "Be concise — 3-6 sentences unless more detail is requested. "
        "Never invent statistics; always base answers on tool output."
    )

# ── Agent class ────────────────────────────────────────────────

class ReliabilityAgent:
    def __init__(self, df: pd.DataFrame, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-4-6", max_tool_rounds: int = 3,
                 max_history_turns: int = 8):
        self.df      = df
        self.schema  = _build_agent_schema(df)
        self.model   = model
        self.max_tool_rounds = max_tool_rounds
        self.max_history_turns = max_history_turns
        self.history: list[dict] = []

        if not api_key:
            raise ValueError("Anthropic API key required.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system = _build_system_prompt(self.schema)

    def _trim_history(self):
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        max_pending_messages = max(1, self.max_history_turns * 2 - 1)
        messages = list(self.history[-max_pending_messages:])

        for _ in range(self.max_tool_rounds):
            try:
                response = self.client.messages.create(
                    model=self.model, max_tokens=2048,
                    system=self.system, tools=TOOLS, messages=messages,
                )
            except Exception:
                LOGGER.exception("Anthropic API request failed")
                raise
            tool_calls  = [b for b in response.content if b.type == "tool_use"]
            text_blocks = [b for b in response.content if b.type == "text"]

            if response.stop_reason == "end_turn" or not tool_calls:
                reply = "\n".join(b.text for b in text_blocks).strip()
                self.history.append({"role": "assistant", "content": reply})
                self._trim_history()
                return reply

            messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {"type": "tool_result", "tool_use_id": tc.id,
                 "content": json.dumps(_safe_json(execute_tool(tc.name, tc.input, self.df, self.schema)))}
                for tc in tool_calls
            ]
            messages.append({"role": "user", "content": tool_results})

        fallback = "Max analysis rounds reached — please rephrase or ask a more specific question."
        self.history.append({"role": "assistant", "content": fallback})
        self._trim_history()
        return fallback

    def reset(self):
        self.history = []

    def load_new_dataset(self, df: pd.DataFrame):
        self.df     = df
        self.schema = _build_agent_schema(df)
        self.system = _build_system_prompt(self.schema)
        self.reset()
