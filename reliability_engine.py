"""
reliability_engine.py
─────────────────────
Core statistical engine for Weibull MLE fitting, Kaplan-Meier survival
analysis, AFT covariate modelling, DOE effects, and B-life estimation.

All functions are pure (no side effects) and return plain dicts or
DataFrames so they can be called from the agent, a notebook, or a
REST endpoint without modification.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Weibull helpers
# ──────────────────────────────────────────────

def weibull_survival(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """S(t) = exp(-(t/eta)^beta)"""
    t = np.asarray(t, dtype=float)
    return np.exp(-((t / eta) ** beta))


def weibull_failure_prob(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """F(t) = 1 - S(t)"""
    return 1.0 - weibull_survival(t, beta, eta)


def weibull_hazard(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """h(t) = (beta/eta)*(t/eta)^(beta-1)"""
    t = np.clip(np.asarray(t, dtype=float), 1e-9, None)
    return (beta / eta) * ((t / eta) ** (beta - 1.0))


def bx_life(unreliability: float, beta: float, eta: float) -> float:
    """Return Bx life for a given unreliability fraction (0 < x < 1)."""
    return eta * (-np.log(1.0 - unreliability)) ** (1.0 / beta)


def interpret_beta(beta: float) -> str:
    if not np.isfinite(beta):
        return "not estimable"
    if beta < 0.95:
        return "decreasing hazard — infant mortality / early-life pattern"
    if beta <= 1.05:
        return "approximately constant hazard — random failure pattern"
    return "increasing hazard — wear-out pattern"


# ──────────────────────────────────────────────
# Weibull MLE fit
# ──────────────────────────────────────────────

def fit_weibull_mle(
    times: np.ndarray,
    events: np.ndarray,
) -> dict:
    """
    Fit a two-parameter Weibull model by maximum likelihood to
    right-censored data.

    Parameters
    ----------
    times  : array of observed times (failure or censoring).
    events : array of 0/1 integers (1 = observed failure).

    Returns
    -------
    dict with keys: shape_beta, scale_eta, B10_cycles, B50_cycles,
                    n, n_events, nll, success, message,
                    beta_interpretation.
    """
    times  = np.clip(np.asarray(times,  dtype=float), 1e-9, None)
    events = np.asarray(events, dtype=int)
    n_events = int(events.sum())

    if n_events == 0:
        return dict(shape_beta=np.nan, scale_eta=np.nan,
                    B10=np.nan, B50=np.nan,
                    n=len(times), n_events=0, nll=np.nan,
                    success=False, beta_interpretation="not estimable",
                    message="No observed failures — cannot fit Weibull.")

    def neg_log_likelihood(log_params):
        beta, eta = np.exp(log_params)
        z = times / eta
        ll = events * (np.log(beta) - np.log(eta) + (beta - 1.0) * np.log(z) - z**beta) \
             + (1 - events) * (-z**beta)
        return -float(np.sum(ll))

    x0 = np.log([1.5, np.median(times)])
    res = minimize(neg_log_likelihood, x0=x0, method="BFGS")
    if not res.success:
        res = minimize(neg_log_likelihood, x0=x0, method="Nelder-Mead")

    beta, eta = np.exp(res.x)
    return dict(
        shape_beta=float(beta),
        scale_eta=float(eta),
        B10=float(bx_life(0.10, beta, eta)),
        B50=float(bx_life(0.50, beta, eta)),
        n=len(times),
        n_events=n_events,
        nll=float(res.fun),
        success=bool(res.success),
        beta_interpretation=interpret_beta(beta),
        message=str(res.message),
    )


def fit_weibull_by_group(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Fit Weibull models for each unique value in group_col.
    Returns a tidy DataFrame with one row per group.
    """
    rows = []
    for grp, g in df.groupby(group_col):
        fit = fit_weibull_mle(g[time_col].values, g[event_col].values)
        fit[group_col] = grp
        rows.append(fit)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Kaplan-Meier
# ──────────────────────────────────────────────

def kaplan_meier(
    times: np.ndarray,
    events: np.ndarray,
) -> pd.DataFrame:
    """
    Compute the Kaplan-Meier survival table for right-censored data.

    Returns a DataFrame with columns:
        time, n_at_risk, n_events, n_censored, survival, failure_prob
    """
    df = pd.DataFrame({"time": np.asarray(times, float),
                       "event": np.asarray(events, int)})
    event_times = np.sort(df.loc[df["event"] == 1, "time"].unique())
    S = 1.0
    rows = [{"time": 0.0, "n_at_risk": len(df),
             "n_events": 0, "n_censored": 0,
             "survival": S, "failure_prob": 0.0}]
    for t in event_times:
        n_risk    = int((df["time"] >= t).sum())
        n_ev      = int(((df["time"] == t) & (df["event"] == 1)).sum())
        n_cens    = int(((df["time"] == t) & (df["event"] == 0)).sum())
        if n_risk > 0:
            S *= 1.0 - n_ev / n_risk
        rows.append({"time": float(t), "n_at_risk": n_risk,
                     "n_events": n_ev, "n_censored": n_cens,
                     "survival": float(S), "failure_prob": float(1 - S)})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# AFT (Weibull accelerated failure time) model
# ──────────────────────────────────────────────

def fit_aft(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariate_cols: list[str],
) -> dict:
    """
    Fit a Weibull AFT model:
        log(eta_i) = mu + sum_j(gamma_j * x_ij)

    Returns dict with keys:
        mu, coefficients (dict covariate->value), shape_beta,
        log_likelihood, covariate_cols.
    """
    times  = np.clip(df[time_col].values.astype(float), 1e-9, None)
    events = df[event_col].values.astype(int)
    X      = df[covariate_cols].values.astype(float)
    n      = len(times)

    def neg_ll(params):
        log_beta = params[0]
        gamma    = params[1:]          # intercept + covariate weights
        beta     = np.exp(log_beta)
        log_eta  = gamma[0] + X @ gamma[1:]
        eta      = np.exp(log_eta)
        z        = times / eta
        ll = events * (np.log(beta) - log_eta + (beta - 1)*np.log(z) - z**beta) \
             + (1 - events) * (-z**beta)
        return -float(np.sum(ll))

    x0 = np.zeros(2 + len(covariate_cols))
    x0[0] = np.log(1.5)
    x0[1] = np.log(np.median(times))
    res = minimize(neg_ll, x0=x0, method="BFGS")
    if not res.success:
        res = minimize(neg_ll, x0=x0, method="Nelder-Mead")

    beta = float(np.exp(res.x[0]))
    mu   = float(res.x[1])
    coef = {c: float(v) for c, v in zip(covariate_cols, res.x[2:])}

    return dict(
        mu=mu,
        shape_beta=beta,
        beta_interpretation=interpret_beta(beta),
        coefficients=coef,
        log_likelihood=float(-res.fun),
        covariate_cols=covariate_cols,
        success=bool(res.success),
    )


def aft_predict(
    aft_result: dict,
    t_values: np.ndarray,
    covariate_values: dict,
) -> np.ndarray:
    """
    Predict failure probability F(t) for a covariate combination.

    Parameters
    ----------
    aft_result       : output of fit_aft()
    t_values         : array of time points to evaluate at
    covariate_values : dict mapping covariate name -> scalar value
    """
    log_eta = aft_result["mu"] + sum(
        aft_result["coefficients"][c] * covariate_values[c]
        for c in aft_result["covariate_cols"]
    )
    eta  = np.exp(log_eta)
    beta = aft_result["shape_beta"]
    return weibull_failure_prob(np.asarray(t_values, float), beta, eta)


# ──────────────────────────────────────────────
# DOE / OLS effects
# ──────────────────────────────────────────────

def fit_ols_effects(
    df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
) -> pd.DataFrame:
    """
    Fit a linear main-effects + interaction model by OLS.
    factor_cols should be 0/1 coded binary variables.

    Returns a DataFrame with columns:
        term, estimate, std_error, t_stat, p_value.
    """
    y  = df[response_col].to_numpy(float)
    Xm = df[factor_cols].to_numpy(float)
    # build intercept + all pairwise interactions
    X  = np.column_stack([np.ones(len(df)), Xm])
    names = ["Intercept"] + list(factor_cols)
    if len(factor_cols) == 2:
        interaction = Xm[:, 0] * Xm[:, 1]
        X = np.column_stack([X, interaction])
        names.append(f"{factor_cols[0]} x {factor_cols[1]}")

    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid    = y - X @ beta_hat
    n, p     = X.shape
    resid_var = float((resid @ resid) / (n - p))
    cov      = resid_var * np.linalg.inv(X.T @ X)
    se       = np.sqrt(np.diag(cov))
    t_stat   = beta_hat / se
    p_val    = 2.0 * (1.0 - student_t.cdf(np.abs(t_stat), df=n - p))
    return pd.DataFrame({
        "term": names,
        "estimate": beta_hat,
        "std_error": se,
        "t_stat": t_stat,
        "p_value": p_val,
    })


# ──────────────────────────────────────────────
# Dataset auto-detection
# ──────────────────────────────────────────────

def detect_schema(df: pd.DataFrame) -> dict:
    """
    Heuristically detect time, event, and covariate columns
    from an uploaded DataFrame.
    """
    cols = [c.lower() for c in df.columns]

    time_candidates = [c for c in df.columns
                       if any(k in c.lower() for k in
                              ("time", "cycle", "hour", "age", "duration", "life"))]
    event_candidates = [c for c in df.columns
                        if any(k in c.lower() for k in
                               ("event", "fail", "return", "defect", "fault"))]
    group_candidates = [c for c in df.columns
                        if any(k in c.lower() for k in
                               ("group", "stress", "condition", "profile",
                                "region", "lot", "mode"))]

    return dict(
        time_col=time_candidates[0] if time_candidates else None,
        event_col=event_candidates[0] if event_candidates else None,
        group_col=group_candidates[0] if group_candidates else None,
        n_rows=len(df),
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
    )
