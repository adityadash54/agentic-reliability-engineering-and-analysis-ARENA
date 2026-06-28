# ARENA — Architecture & Design
## Agentic Reliability Engineering & Analysis

This agentic reliability engineering analysis platform is built with a Streamlit
front end, a pure-Python statistical engine, and a Claude language model
acting as the analytical reasoning layer through a structured tool-use loop.

---

## Table of contents

1. [Project overview and motivation](#project-overview-and-motivation)
2. [Statistical and algorithmic foundations](#statistical-and-algorithmic-foundations)
3. [System architecture](#system-architecture)
4. [Agent design and tool-use workflow](#agent-design-and-tool-use-workflow)
5. [Data flow and session lifecycle](#data-flow-and-session-lifecycle)
6. [Schema detection and column safety](#schema-detection-and-column-safety)
7. [Privacy and security controls](#privacy-and-security-controls)
8. [User interface design](#user-interface-design)
9. [Deployment approach](#deployment-approach)
10. [Trade-offs and design decisions](#trade-offs-and-design-decisions)
11. [Extensibility and future directions](#extensibility-and-future-directions)

---

## Project overview and motivation

Reliability engineering is a quantitative discipline concerned with the
probability that a physical system or component will perform its intended
function over a specified period under defined conditions. It involves
working with time-to-event datasets in which each observation is either a
confirmed failure or a censored observation, which is when the unit survived
the monitoring window without failing. The statistical machinery for
processing such data is mature but historically siloed inside desktop
tools like Minitab, JMP, or Reliasoft Weibull++, which are expensive,
require installation, and produce results that an engineer must interpret
manually.

The motivation for this project is to lower that access barrier by pairing
a correct, dependency-light Python statistics engine directly with a large
language model acting as an expert interpreter. The result is a conversational
interface where an engineer can load a dataset, ask natural-language questions
such as "what is the B10 life for each stress group," and receive a
statistically grounded answer with engineering interpretation — without
having to write code or understand the numerical output format.

The target audience is reliability engineers, quality engineers, and data
scientists working in hardware-adjacent domains: automotive, aerospace,
consumer electronics, energy storage, semiconductor manufacturing, and
rotating machinery. The five bundled sample datasets — battery cell wear-out,
bearing fatigue, semiconductor burn-in, LED lumen degradation, and fleet
warranty returns — represent canonical failure regimes found in those
domains.

---

## Statistical and algorithmic foundations

The statistical foundation rests on four well-established methods for
reliability and survival analysis. Each was chosen for its interpretability
in an engineering context and its tractability under scipy-based numerical
optimization.

### Weibull maximum likelihood estimation

The two-parameter Weibull distribution is the workhorse of reliability
analysis. Its survival function is S(t) = exp(-(t/eta)^beta), where beta
is the shape parameter and eta is the scale parameter (the characteristic
life). Beta is the primary diagnostic quantity: a value below roughly 0.95
indicates infant mortality or early-life failures caused by manufacturing
defects; a value near 1.0 indicates a constant, memory-less hazard consistent
with random external shocks; and a value above roughly 1.05 indicates an
increasing hazard, meaning the part is wearing out over time.

Maximum likelihood estimation for right-censored data involves maximizing
the log-likelihood, which has separate terms for observed failures and for
censored observations. For a failure at time t, the contribution is the
log of the Weibull probability density function. For a censored observation,
the contribution is the log of the survival function at that time — because
all we know is the unit was still alive at censoring. The optimization is
performed in log-parameter space (log beta and log eta) to enforce positivity
without requiring constrained optimization. BFGS is used as the primary
optimizer for its fast convergence on smooth likelihoods, with Nelder-Mead
as a fallback for ill-conditioned problems such as very small samples or
near-complete censoring.

The choice to implement MLE from scratch using scipy rather than a
reliability-specific library like lifelines or reliability reflects a
deliberate dependency minimization decision discussed in the architecture
section below. The correctness of the implementation is anchored by the
standard decomposition of the Weibull log-likelihood under right censoring,
which is textbook material in Meeker and Escobar.

### Kaplan-Meier survival estimation

Kaplan-Meier is a non-parametric estimator of the survival function. It
makes no distributional assumptions and is therefore a useful complement
to the parametric Weibull fit — if the KM curve deviates substantially
from the fitted Weibull, the Weibull assumption may not hold for that
dataset. The estimator multiplies conditional survival probabilities at
each observed failure time: S(t) = product over all t_i <= t of
(1 - d_i / n_i), where d_i is the number of failures at time t_i and
n_i is the number of units at risk just before t_i. The implementation
handles ties and censoring correctly.

The agent exposes KM primarily as a stratified exploratory tool — it is
most useful for comparing survival curves across stress groups before
committing to a parametric model. The summary returned to the language model
includes median survival time per group and the final (rightmost) survival
probability, which together give a compact but informative picture of
group-level differences.

### Accelerated failure time model

The Weibull AFT model extends the single-population Weibull fit by allowing
the scale parameter eta to be a function of covariates. Specifically, the
log scale is modeled as a linear combination of covariates: log(eta_i) =
mu + sum_j(gamma_j * x_ij). A positive coefficient gamma_j means that
increasing the corresponding covariate increases the characteristic life —
the stress factor is protective. A negative coefficient means the factor
accelerates failure. The shape parameter beta is shared across all
observations, which is a simplifying assumption that can be relaxed in
more complex implementations.

This model is appropriate for designed experiments under stress, such as
accelerated life testing at elevated temperature or voltage, or observational
datasets where failure times co-vary with continuous stress measurements.
The implementation fits the full AFT log-likelihood jointly over beta and
all covariate weights using BFGS with a Nelder-Mead fallback, consistent
with the single-population Weibull fit.

### Design of experiments main-effects analysis

For datasets that include binary-coded experimental factors — for example,
a factor taking the value 0 for ambient temperature and 1 for elevated
temperature — the platform supports an OLS main-effects analysis. This is
the classical 2^k factorial screening approach where each factor effect
is estimated as a regression coefficient. The implementation uses ordinary
least squares via NumPy's lstsq, computes standard errors from the residual
variance and the covariance matrix of the design, and reports t-statistics
and two-tailed p-values using the Student-t distribution.

For exactly two factors, a pairwise interaction term is automatically
included. For more than two factors, only main effects are estimated, which
is appropriate for screening designs where interactions are presumed small
relative to main effects. The response variable can be any continuous numeric
column — typical choices are time to failure, a degradation signal level,
or a quality characteristic.

### B-life estimation

B-life (or Bx life) is the standard reliability metric: B10 is the time
by which 10 percent of the population is expected to have failed. It is
derived analytically from the Weibull parameters as
t_Bx = eta * (-log(1 - x))^(1/beta). The platform computes B5, B10, B20,
and B50 for each stress group or for the overall population, providing a
table that is immediately actionable for warranty planning and design
margin analysis.

---

## System architecture

The system has three Python modules: the statistical engine, the agent
layer, and the Streamlit application layer. The separation is deliberate
and follows a pure-function discipline that keeps each layer independently
testable and reusable.

```
┌─────────────────────────────────────────────┐
│              app.py  (Streamlit UI)          │
│  session state · data validation · security  │
└────────────────────┬────────────────────────┘
                     │ uses
┌────────────────────▼────────────────────────┐
│             agent.py  (Agent layer)          │
│  tool definitions · tool-use loop · history  │
└──────────┬─────────────────────┬────────────┘
           │ calls               │ calls
┌──────────▼──────────┐   ┌──────▼───────────┐
│ reliability_engine  │   │  Anthropic API   │
│   (pure functions)  │   │  (Claude model)  │
└─────────────────────┘   └──────────────────┘
```

The statistical engine contains only pure functions that accept arrays and
DataFrames and return dicts or DataFrames. It has no knowledge of the UI,
the agent, or the language model. This means the engine can be called from
a Jupyter notebook, a REST endpoint, or a test harness without modification.
The only dependencies are NumPy, pandas, and SciPy — a minimal and stable
stack.

The agent layer wraps the statistical engine in a set of tool definitions
that the language model can call, implements the tool-use conversation loop,
and manages conversation history. It depends on the Anthropic Python client
and on the statistical engine.

The Streamlit application layer owns the user interface, session state, data
loading and validation, and security controls. It depends on the agent layer
and on pandas. It has no direct dependency on the statistical engine or the
Anthropic client — it communicates with the model exclusively through the
agent object.

This three-layer separation means a developer can replace Streamlit with a
different front end, replace the language model with a different provider,
or extend the statistical engine, without touching the other layers.

### Dependency minimization

There are five package requirements: NumPy, pandas, SciPy, Anthropic,
and Streamlit. This is a deliberate constraint. A heavier statistics
library like lifelines provides more functionality — log-rank tests,
confidence intervals on KM curves, Cox regression — but at the cost of
a larger dependency graph, slower cold starts on cloud deployments, and
potential version conflicts with the other packages.

The chosen stack boots quickly on Streamlit Community Cloud and on minimal
cloud instances. The trade-off is that confidence intervals are not
computed for Weibull estimates, and the KM estimator does not produce
Greenwood standard errors. The intended use case is conversational exploratory
analysis; statistical completeness can be layered on incrementally as needed.

---

## Agent design and tool-use workflow

The agent implements the standard tool-use loop pattern supported by the
Anthropic messages API. On each user turn, the agent sends the conversation
history plus a system prompt to the language model. The model may respond
with a text answer, with one or more tool calls, or with a mix of both.
If tool calls are present, the agent executes each tool locally, serializes
the result to JSON, and appends a tool-result message to the conversation.
The loop continues until the model returns a response with stop reason
"end_turn" and no tool calls, or until a configurable maximum of tool rounds
is reached.

The maximum tool rounds parameter defaults to three. This limits the cost
and latency of a single user turn. In practice, most reliability questions
are answered in one or two tool calls — a dataset summary followed by a
Weibull fit, for example. The three-round limit is rarely hit, but it
prevents runaway loops if the model misinterprets a question and issues
redundant tool calls.

The agent maintains a rolling conversation history capped at eight turns
(sixteen messages, counting both user and assistant sides). Older messages
are dropped when the limit is exceeded. This bounds the prompt size and
token cost across a long session without requiring the user to manually
manage context. The trade-off is that the model may lose context about
earlier questions in a long session, but for the ten-question session limit
enforced at the UI layer, this rarely matters in practice.

### System prompt design

The system prompt is built dynamically from the dataset schema at agent
initialization. It tells the model its role, the dataset dimensions, the
detected time and event columns, and — critically — the exact set of
columns that are safe for grouping, for numeric covariate analysis, and
for DOE factor analysis. This information is injected at construction time
rather than discovered through tool calls, which saves at least one round
trip per session.

The system prompt also instructs the model to base all quantitative
statements on tool output, never to invent statistics, and to be concise
(three to six sentences unless more detail is requested). This phrasing
is important: without the "never invent" constraint, language models will
sometimes hallucinate plausible-sounding but incorrect numerical results
when they cannot call a tool.

The prompt exposes safe column lists in plain text so the model can
construct valid tool arguments without trial and error. If the model
attempts to use a blocked column, the tool executor returns a structured
error response naming the allowed alternatives, which the model can then
use to self-correct in the next round.

### Tool definitions

Seven tools are defined. Each tool definition includes a name, a natural-
language description, and an input schema in JSON Schema format. The
description is the primary signal the model uses to decide which tool to
call; it should be precise about what the tool computes and what it returns.

| Tool | Purpose |
|---|---|
| `dataset_summary` | Row count, failure rate, time range, column overview |
| `fit_weibull` | MLE Weibull fit — overall or stratified by group |
| `kaplan_meier` | Non-parametric survival curve — overall or by group |
| `fit_aft` | Weibull AFT model with continuous stress covariates |
| `doe_analysis` | OLS main-effects analysis for binary experimental factors |
| `blife_table` | B5 / B10 / B20 / B50 estimates per group |
| `failure_breakdown` | Failure count and rate by categorical column |

This set covers the core reliability workflow from exploratory summary through
parametric modelling through design risk interpretation.

`dataset_summary` is intentionally first in the list because it is often the
right first call when a user asks a broad question like "summarise this dataset."
The model learns this ordering through both the tool descriptions and the system
prompt instruction to use computed numbers.

### Tool executor and column validation

The tool executor is a dispatch function that validates inputs and then
calls the appropriate engine function. Validation has two layers. First,
the executor checks that required columns exist in the dataset. Second,
it checks that any column the model requests is in the pre-computed safe
set for that operation — safe group columns, safe covariate columns, or
safe DOE factor columns. If either check fails, the executor returns a
structured error dict rather than raising an exception.

This validation is the primary defense against the model attempting to
use columns it should not touch — either sensitive columns that were
blocked by the privacy filter, or columns with cardinalities too high to
produce meaningful grouped analyses. Returning a structured error dict
rather than raising an exception means the tool-use loop can continue:
the model sees the error, understands why the column is blocked, and can
either ask a different question or use an allowed column instead.

---

## Data flow and session lifecycle

A session begins when the user selects a dataset and enters an API key.
The dataset is loaded from disk (for bundled samples) or from an in-memory
upload buffer (for user uploads). It is parsed into a pandas DataFrame and
subjected to size validation: no more than 5 MB, 25,000 rows, or 80 columns.
These limits prevent memory exhaustion on the shared cloud host and cap the
token cost of sending column metadata to the language model.

Once the DataFrame is available, the agent is initialized. Initialization
runs schema detection — heuristic column name matching to identify the
time, event, and group columns — and analysis metadata construction, which
classifies every column as blocked, a safe group column, a safe covariate,
a binary DOE factor, or uncategorized. The results are stored on the agent
object and injected into the system prompt. The agent object is stored in
Streamlit session state, keyed by a fingerprint of the API key and the
dataset label, so that switching datasets or API keys correctly re-initializes
the agent rather than contaminating an existing session.

Each user message triggers the tool-use loop. The loop is synchronous and
blocking during the Streamlit rendering cycle, which means the UI displays
a spinner while analysis is running. On completion, the assistant reply is
appended to the rendered chat history and to the agent's internal history.
The Streamlit session state holds the rendered message list separately from
the agent's internal history to allow the UI to reconstruct the conversation
on a page re-render without re-running the agent.

A session ends when the user clears the conversation, switches datasets or
API keys, or reaches the ten-question limit. Clearing the conversation resets
the agent's internal history but preserves the loaded dataset and schema,
so the agent remains ready for a new conversation without re-parsing the
dataset.

---

## Schema detection and column safety

Automatic schema detection uses keyword matching on column names. Time
columns are identified by keywords like "time," "cycle," "hour," "age,"
"duration," and "life." Event columns are identified by "event," "fail,"
"return," "defect," and "fault." Group columns are identified by "group,"
"stress," "condition," "profile," "region," "lot," and "mode." The first
matching column for each role is selected.

This heuristic works well for the typical reliability dataset structure but
will miss unconventionally named columns. The schema is surfaced to the user
in the sidebar so they can verify the detection before asking questions.
Future versions could allow manual override of detected columns.

The column safety classification algorithm processes every column in the
DataFrame. Columns are blocked if their name contains tokens associated with
personally identifiable information: account, address, birth, client,
customer, email, id, identifier, name, owner, person, phone, ssn, tax, or
user. The matching is case-insensitive and tokenizes on non-alphanumeric
characters, so a column named "customer_id" is caught by both "customer"
and "id."

Columns that pass the privacy filter are then classified by dtype and
cardinality. Numeric columns are candidates for safe covariate or safe
group roles. A numeric column is a safe group column if it has at most 12
distinct values — a threshold chosen to ensure that per-group Weibull fits
have enough observations per group to converge. Categorical columns are
safe group columns if they have at most 25 distinct values. Binary columns
— those containing only 0 and 1 values — are further tagged as DOE
factors.

The safe column lists are embedded in the system prompt at agent
initialization and enforced at tool execution time. This two-layer
enforcement ensures that even if the model generates a tool call referencing
a blocked column, the executor will reject it before touching the data.

---

## Privacy and security controls

The security model is designed around the assumption that the application
will be deployed publicly on Streamlit Community Cloud, where any user
with the URL can access it. The key controls are as follows.

**API key handling:** The application explicitly does not auto-use an
`ANTHROPIC_API_KEY` environment variable or deployment secret. Each user
must supply their own key in the sidebar for that session. The key is used
only to construct the Anthropic client object; it is never logged, never
written to disk, and never stored beyond the current session state. A
SHA-256 fingerprint of the key is stored in session state solely to detect
key changes that require agent re-initialization.

**Upload limits:** User-uploaded CSVs are capped at 5 MB, 25,000 rows, and
80 columns. The size check is performed against the raw byte count before
parsing, preventing memory exhaustion from maliciously crafted CSV files.
The row count is enforced by reading at most one row beyond the limit
and then checking.

**Prompt limits:** Each session is capped at 10 user questions, and each
prompt is capped at 2,000 characters. These limits cap the maximum API
spend attributable to a single session and reduce the attack surface for
prompt injection attempts.

**Column blocking:** Sensitive-looking columns are identified and blocked
from model-driven analysis before any tool call is made. This prevents
a dataset containing personal information from having that information
passed to the language model as part of a grouping or covariate analysis.
The user is informed of the count of blocked columns in the sidebar.

**Data notice:** The upload interface displays a notice that uploaded CSVs
may be processed by the language model provider during analysis, and
instructs users not to upload secrets, personal data, or regulated data.
This is a transparency control, not a technical one, and it reflects the
reality that any data sent to a cloud language model API may be subject
to the provider's data processing policies.

The application does not implement authentication or multi-tenancy. All
session state is isolated per browser session by Streamlit's session
state mechanism, but there is no user account system and no persistent
storage of any kind. Each session is ephemeral.

---

## User interface design

The interface is a sidebar-plus-main-area layout using Streamlit's wide
mode. The sidebar handles all configuration — API key entry, dataset
selection, and schema information — while the main area is dedicated to
the chat interaction.

The dataset selection offers two paths: bundled sample datasets and
user-uploaded CSVs. The sample datasets are curated to cover the major
failure regimes (wear-out, random, infant mortality, degradation threshold)
and to work well with all available tools. They serve both as a quick-start
path for users exploring the application and as a demonstration of the
analytical capabilities.

A metrics strip below the chat title shows the dataset name, row count,
failure rate, and time range at a glance. These figures orient the user
before they start asking questions and provide an immediate sanity check
on the loaded data.

Quick-start prompt buttons are shown before the first message in a session.
The six suggested questions cover the most common analysis workflow —
dataset summary, Weibull fit interpretation, B-life by group, Kaplan-Meier
breakdown, stress factor identification, and design risk summary. These
buttons are implemented as Streamlit button widgets that queue the prompt
text and trigger a rerun, which then processes the prompt through the normal
submission path. This ensures consistent behavior between quick-start and
typed prompts.

The session usage counter is displayed above the chat area, showing current
and maximum question counts. This transparency helps users plan their
session and avoids surprise when the limit is reached.

The clear conversation button in the sidebar resets the chat history and
the agent's internal history without re-initializing the agent, preserving
the loaded dataset and schema.

---

## Deployment approach

The application is designed for zero-configuration deployment on Streamlit
Community Cloud. All three Python modules live at the repository root because
that is the path Streamlit Community Cloud expects when looking for the
entry point and its local imports.

The requirements file pins major versions but allows minor version updates
within a major release — for example, NumPy 2.x and pandas 2.x — which
provides a balance between reproducibility and receiving security patches.
Exact pinning would improve reproducibility but requires more frequent
manual updates.

There are no environment variables required to deploy the application.
The bring-your-own-key design means deployment secrets are intentionally
left empty. This makes the repository safe to share publicly without
the risk of accidentally exposing API credentials.

The application can also be run locally by installing the five dependencies
and invoking Streamlit. No database, no background workers, no external
services beyond the language model API are required. The simplicity of the
runtime is intentional: it reduces operational burden and makes the system
easy to audit and understand.

---

## Trade-offs and design decisions

Several significant trade-offs were made in building this system, and each
deserves explicit discussion.

**Parametric simplicity over statistical completeness:** The Weibull fit does
not compute confidence intervals on the parameter estimates. A production
reliability analysis would include asymptotic or bootstrap confidence
intervals on beta and eta, and Greenwood standard errors on the KM curve.
The omission keeps the implementation lean and avoids the added complexity
of communicating interval estimates to the language model in a form it
can reliably interpret and present. The gap is acknowledged and could be
addressed by adding confidence interval computation to the engine functions.

**Synchronous tool execution over streaming:** Tool calls are executed
synchronously, and the complete response is returned as a single string.
Streamlit's streaming API could be used to show the response as it is
generated, improving perceived latency. The synchronous approach was
chosen for simplicity and because the dominant latency is tool execution
(statistical computation) rather than token generation for typical datasets.

**Heuristic schema detection over user configuration:** Column roles are
inferred from column names rather than requiring the user to explicitly
specify the time, event, and group columns. This reduces friction for
users with conventionally named datasets but fails silently for
unconventionally named datasets. The detection result is surfaced in the
sidebar so users can notice errors, but there is no manual override
mechanism in the current version.

**Session-scoped history over persistent history:** Conversation history is
held in Streamlit session state and is lost when the browser tab is closed.
There is no persistence layer. This eliminates the need for a database
or cloud storage service but means users cannot resume a previous session.
For the intended exploratory use case this is acceptable; a production
deployment supporting formal analysis workflows would need persistent
sessions.

**Blocking column filter over consent model:** Sensitive-looking columns are
blocked from model analysis entirely rather than asking the user for
explicit consent to include them. This is a conservative default that
may occasionally block non-sensitive columns with names that happen to
contain flagged tokens. The alternative — a consent dialog for each
flagged column — adds friction and complexity. The blocked column list
is visible in the sidebar, allowing users to understand what was excluded.

**Single-model, single-provider architecture:** The agent is hard-wired to
use a specific Claude model via the Anthropic client. Supporting multiple
models or providers would require abstracting the client behind an interface
and mapping tool definitions to each provider's format. The current design
prioritizes simplicity; model selection can be surfaced as a configuration
option without changing the tool or engine layers.

---

## Extensibility and future directions

The layered architecture makes the system straightforward to extend in
several directions without restructuring the existing code.

New statistical methods can be added to the engine as pure functions. Adding
a log-rank test for comparing KM curves between groups, for example, would
require adding one function to the engine, one tool definition in the agent,
and one dispatch case in the tool executor. The language model does not need
to be retrained or fine-tuned; the new tool description is sufficient to
make it available for tool-use decisions.

New dataset types can be supported by extending the schema detection
heuristics. Degradation data — where the observation is a continuous
measurement over time rather than a binary event — would require new
keyword patterns for the measurement column and a degradation-threshold
model in the engine. The bundled data directory already includes degradation
curve files alongside the event files, making this extension natural.

The column safety filter can be refined with domain-specific rules. In a
medical device context, additional PII tokens would be appropriate. In an
internal enterprise deployment where data is already governed by access
controls, the filter could be loosened. Because the filter is implemented
as a simple token set, these changes require only configuration edits.

Authentication and multi-tenancy could be layered on top of the existing
application by integrating Streamlit's authentication options or wrapping
the application in a reverse proxy with authentication. The session-scoped
design already isolates users from each other; adding authentication would
restrict who can create sessions rather than changing what they can do within
one.

Persistent sessions would require a key-value store or a relational database
to save and restore conversation history and session state. The agent's
history list is already a plain list of dicts, which serializes trivially
to JSON. The main engineering work would be in session identification,
expiry, and secure storage of the API key reference across sessions.

Confidence intervals on Weibull parameters could be added using the Fisher
information matrix computed from the Hessian of the log-likelihood at the
MLE solution. The BFGS optimizer returns an approximation of the inverse
Hessian, which can be used directly for asymptotic standard errors on the
log-scale parameters. Propagating these through the B-life formula using
the delta method would yield approximate confidence intervals on B10 and
B50.

The design prioritizes correctness of the statistical foundations,
transparency of the privacy controls, and simplicity of the operational
model. Each layer — pure statistics engine, structured tool-use agent,
Streamlit front end — is independently replaceable, making the system
straightforward to audit, extend, and redeploy without the cost or
operational burden of traditional reliability software.
