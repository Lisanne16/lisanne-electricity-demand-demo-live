"""
Live Demand Forecast — Explainability Demo
===========================================
One-story Streamlit app for a 5-minute MSc dissertation viva:

    Forecast -> SHAP -> LLM explanation -> Faithfulness -> Final comparison

Reuses ONLY the precomputed CSVs already sitting in:
    <BASE_DIR>/live_demo/...
No live model inference, no LLM API calls at runtime — everything is
read from the CSVs produced by the LIVE_DEMAND_FORECAST_DEMO.ipynb pipeline.
"""

import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 0. CONFIG — only thing you should need to touch
# ---------------------------------------------------------------------------
def _resolve_base_dir() -> str:
    """
    Resolution order:
      1. LIVE_DEMO_DATA_DIR env var (set this if you want to override)
      2. The Colab Google Drive path (works while running inside Colab)
      3. A local './data' folder sitting next to this script (works when
         hosted, e.g. on Streamlit Community Cloud, with the CSVs committed
         to the repo instead of read from Drive)
    """
    env_dir = os.environ.get("LIVE_DEMO_DATA_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    colab_path = "/content/drive/My Drive/dissertation_data_v2/live_demo"
    if os.path.isdir(colab_path):
        return colab_path

    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    return local_path


BASE_DIR = _resolve_base_dir()

PRED_DIR = os.path.join(BASE_DIR, "live_demo_predictions")
SHAP_DIR = os.path.join(BASE_DIR, "live_demo_shap")
LLM_CASES_DIR = os.path.join(BASE_DIR, "live_llm_cases")
LLM_RESP_DIR = os.path.join(BASE_DIR, "live_llm_responses")
EVAL_DIR = os.path.join(BASE_DIR, "live_evaluation_results")

COUNTRIES = {
    "DE": "Germany",
    "FR": "France",
    "ES": "Spain",
    "PT": "Portugal",
    "IE": "Ireland",
}

# short llm key (used in FINAL_case_level_scores.csv) -> display name + response file suffix
LLM_INFO = {
    "gpt_oss":  {"label": "GPT-OSS",  "suffix": "gpt_oss_20b"},
    "gemma":    {"label": "Gemma",    "suffix": "gemma_2_2b"},
    "qwen":     {"label": "Qwen",     "suffix": "qwen_2_5_7b"},
    "deepseek": {"label": "DeepSeek", "suffix": "deepseek_r1_qwen_7b"},
}

st.set_page_config(page_title="Live Demand Forecast — Explainability Demo", layout="wide")

# ---------------------------------------------------------------------------
# 1. DATA LOADING (cached — reads once per session)
# ---------------------------------------------------------------------------

def _tz_naive(series: pd.Series) -> pd.Series:
    """Force any timestamp column to naive UTC so joins across files are
    consistent, regardless of whether the source CSV included a UTC offset."""
    s = pd.to_datetime(series, utc=True)
    return s.dt.tz_localize(None)


def _check_base_dir():
    if not os.path.isdir(BASE_DIR):
        st.error(
            "Can't find the data folder.\n\n"
            f"Looked for: `{BASE_DIR}`\n\n"
            "**If you're in Colab:** make sure Drive is mounted:\n\n"
            "```python\nfrom google.colab import drive\ndrive.mount('/content/drive')\n```\n\n"
            "**If this is hosted (e.g. Streamlit Community Cloud):** make sure a `data/` "
            "folder sits next to `app.py` in your repo, containing the same subfolders as "
            "`live_demo/` (`live_demo_predictions/`, `live_demo_shap/`, `live_llm_cases/`, "
            "`live_llm_responses/`, `live_evaluation_results/`)."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def load_predictions(country_code: str) -> pd.DataFrame:
    path = os.path.join(PRED_DIR, f"{country_code}_t24_predictions.csv")
    df = pd.read_csv(path)
    df["forecast_origin"] = _tz_naive(df["forecast_origin"])
    df["target_timestamp"] = _tz_naive(df["target_timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_model_eval() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PRED_DIR, "t24_model_evaluation.csv"))


@st.cache_data(show_spinner=False)
def load_shap_top10() -> pd.DataFrame:
    path = os.path.join(SHAP_DIR, "live_t24_local_shap_top10.csv")
    df = pd.read_csv(path)
    df["forecast_origin"] = _tz_naive(df["forecast_origin"])
    df["target_timestamp"] = _tz_naive(df["target_timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_llm_cases() -> pd.DataFrame:
    path = os.path.join(LLM_CASES_DIR, "live_llm_cases.csv")
    df = pd.read_csv(path)
    df["forecast_origin"] = _tz_naive(df["forecast_origin"])
    df["target_timestamp"] = _tz_naive(df["target_timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_llm_response(suffix: str) -> pd.DataFrame:
    path = os.path.join(LLM_RESP_DIR, f"live_responses_{suffix}.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_case_level_scores() -> pd.DataFrame:
    path = os.path.join(EVAL_DIR, "FINAL_case_level_scores.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_final_comparison() -> pd.DataFrame:
    path = os.path.join(EVAL_DIR, "FINAL_MODEL_COMPARISON_CORRECTED.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. SIDEBAR — the only controls in the whole demo
# ---------------------------------------------------------------------------

_check_base_dir()

st.sidebar.title("Demo controls")

country_code = st.sidebar.selectbox(
    "Country",
    options=list(COUNTRIES.keys()),
    format_func=lambda c: f"{COUNTRIES[c]} ({c})",
)

llm_cases = load_llm_cases()
country_cases = (
    llm_cases[llm_cases["country_code"] == country_code]
    .sort_values("target_timestamp")
    .reset_index(drop=True)
)

case_labels = country_cases["target_timestamp"].dt.strftime("%a %d %b, %H:%M")
case_idx = st.sidebar.selectbox(
    "Forecast case (hour)",
    options=list(range(len(country_cases))),
    format_func=lambda i: case_labels.iloc[i],
    index=min(8, len(country_cases) - 1),  # default to a mid-morning hour
)
selected_case = country_cases.iloc[case_idx]
case_id = selected_case["case_id"]
forecast_origin = selected_case["forecast_origin"]
target_timestamp = selected_case["target_timestamp"]

shap_model = st.sidebar.radio(
    "SHAP reference model", options=["XGBoost", "LightGBM"], horizontal=True
)

llm_key = st.sidebar.selectbox(
    "LLM explanation",
    options=list(LLM_INFO.keys()),
    format_func=lambda k: LLM_INFO[k]["label"],
)

st.sidebar.caption(
    "All data is precomputed — nothing here calls a model or an LLM live."
)

# ---------------------------------------------------------------------------
# 3. HEADER
# ---------------------------------------------------------------------------

st.title("Live Electricity Demand Forecast — Explainability Demo")
st.caption(
    f"{COUNTRIES[country_code]} · forecasting "
    f"{target_timestamp.strftime('%A %d %B %Y, %H:%M')} "
    f"(made {forecast_origin.strftime('%d %b, %H:%M')}, 24h ahead)"
)

st.divider()

# ---------------------------------------------------------------------------
# 4. STEP 1 — LIVE FORECAST
# ---------------------------------------------------------------------------

st.header("1. Live Forecast")

pred_df = load_predictions(country_code)
row = pred_df[pred_df["forecast_origin"] == forecast_origin]

if row.empty:
    st.warning("No prediction row found for this case.")
else:
    row = row.iloc[0]
    col1, col2 = st.columns([1.3, 1])

    with col1:
        bars = go.Figure(
            data=[
                go.Bar(
                    x=["Actual", "XGBoost", "LightGBM", "Ensemble"],
                    y=[
                        row["actual_MW"],
                        row["xgb_predicted_MW"],
                        row["lgbm_predicted_MW"],
                        row["ensemble_predicted_MW"],
                    ],
                    marker_color=["#2c3e50", "#3498db", "#e67e22", "#27ae60"],
                    text=[
                        f"{v:,.0f} MW"
                        for v in [
                            row["actual_MW"],
                            row["xgb_predicted_MW"],
                            row["lgbm_predicted_MW"],
                            row["ensemble_predicted_MW"],
                        ]
                    ],
                    textposition="outside",
                )
            ]
        )
        bars.update_layout(
            title="Actual vs Predicted Demand",
            yaxis_title="MW",
            showlegend=False,
            height=380,
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(bars, use_container_width=True)

    with col2:
        st.subheader("Model accuracy")
        eval_df = load_model_eval()
        eval_country = eval_df[eval_df["country_code"] == country_code].copy()
        eval_country = eval_country[["model", "MAPE", "accuracy_pct"]].rename(
            columns={"MAPE": "MAPE %", "accuracy_pct": "Accuracy %"}
        )
        eval_country["MAPE %"] = eval_country["MAPE %"].round(2)
        eval_country["Accuracy %"] = eval_country["Accuracy %"].round(2)
        st.dataframe(eval_country, hide_index=True, use_container_width=True)
        best = eval_country.loc[eval_country["Accuracy %"].idxmax()]
        st.metric(f"Best model for {COUNTRIES[country_code]}", best["model"], f"{best['Accuracy %']}% accuracy")

st.divider()

# ---------------------------------------------------------------------------
# 5. STEP 2 — SHAP EXPLANATION
# ---------------------------------------------------------------------------

st.header("2. SHAP Explanation")
st.caption(
    "SHAP is used here as the **reference explanation** to evaluate LLM faithfulness — "
    "it is not treated as absolute ground truth about how the grid actually behaves."
)

shap_df = load_shap_top10()
shap_case = shap_df[
    (shap_df["country_code"] == country_code)
    & (shap_df["forecast_origin"] == forecast_origin)
    & (shap_df["model"] == shap_model)
].sort_values("rank")

if shap_case.empty:
    st.warning("No SHAP rows found for this case/model.")
else:
    shap_plot = shap_case.sort_values("shap_value")
    colors = ["#27ae60" if v > 0 else "#c0392b" for v in shap_plot["shap_value"]]
    fig = go.Figure(
        go.Bar(
            x=shap_plot["shap_value"],
            y=shap_plot["feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+,.0f} MW" for v in shap_plot["shap_value"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top SHAP features — {shap_model}",
        xaxis_title="SHAP value (impact on predicted MW)",
        height=420,
        margin=dict(t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 Green pushes demand up · 🔴 Red pushes demand down")

st.divider()

# ---------------------------------------------------------------------------
# 6. STEP 3 — LLM EXPLANATION
# ---------------------------------------------------------------------------

st.header("3. LLM Explanation")

resp_df = load_llm_response(LLM_INFO[llm_key]["suffix"])
resp_row = resp_df[resp_df["case_id"] == case_id]

st.subheader(f"{LLM_INFO[llm_key]['label']}'s explanation for this exact forecast")

if resp_row.empty:
    st.warning("No LLM response found for this case.")
else:
    resp_row = resp_row.iloc[0]
    response_text = resp_row.get("response", "")
    status_ok = str(resp_row.get("status", "ok")).lower() == "ok"
    has_text = isinstance(response_text, str) and response_text.strip() != ""

    if not status_ok:
        st.error(f"This model's response failed for this case: {resp_row.get('error', 'unknown error')}")
    elif not has_text:
        st.warning(
            f"{LLM_INFO[llm_key]['label']} did not produce a saved explanation for this "
            "case (empty response in the saved results) — pick a different hour or model "
            "from the sidebar."
        )
    else:
        st.info(response_text)

st.divider()

# ---------------------------------------------------------------------------
# 7. STEP 4 — FAITHFULNESS
# ---------------------------------------------------------------------------

st.header("4. Faithfulness: LLM vs SHAP")

scores_df = load_case_level_scores()
score_row = scores_df[
    (scores_df["case_id"] == case_id)
    & (scores_df["llm"] == llm_key)
    & (scores_df["shap_model"] == shap_model)
]

if score_row.empty:
    st.warning("No faithfulness score found for this case/model/shap combination.")
else:
    score_row = score_row.iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Feature Agreement", f"{score_row['feature_agreement_pct']:.0f}%")
    if bool(score_row["has_directional_claims"]):
        m2.metric("Directional Accuracy", f"{score_row['direction_agreement_pct']:.0f}%")
    else:
        m2.metric("Directional Accuracy", "N/A", "no directional claims made")
    m3.metric("Forecast error tier", str(score_row["error_tier"]).title())

    mentioned = [f for f in str(score_row["mentioned_features"]).split("|") if f and f != "nan"]
    unmentioned = [f for f in str(score_row["unmentioned_features"]).split("|") if f and f != "nan"]

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**✅ SHAP features the LLM mentioned**")
        if mentioned:
            for f in mentioned:
                st.markdown(f"- {f}")
        else:
            st.markdown("_None of the top SHAP features were mentioned._")
    with colB:
        st.markdown("**❌ SHAP features the LLM missed**")
        if unmentioned:
            for f in unmentioned:
                st.markdown(f"- {f}")
        else:
            st.markdown("_None — every top SHAP feature was covered._")

    # visual bar: agreement vs gap
    fig2 = go.Figure(
        go.Bar(
            x=["Feature Agreement", "Gap"],
            y=[score_row["feature_agreement_pct"], 100 - score_row["feature_agreement_pct"]],
            marker_color=["#27ae60", "#ecf0f1"],
        )
    )
    fig2.update_layout(
        barmode="stack",
        showlegend=False,
        height=120,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(visible=False),
        xaxis=dict(visible=False),
    )
    # simpler: single horizontal progress-style bar
    fig2 = go.Figure(
        go.Bar(
            x=[score_row["feature_agreement_pct"]],
            y=["Agreement"],
            orientation="h",
            marker_color="#27ae60",
            text=f"{score_row['feature_agreement_pct']:.0f}%",
            textposition="inside",
        )
    )
    fig2.add_vline(x=score_row["feature_agreement_pct"], line_width=0)
    fig2.update_layout(
        xaxis=dict(range=[0, 100], title="% of top SHAP features mentioned"),
        height=140,
        margin=dict(t=10, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 8. STEP 5 — FINAL COMPARISON
# ---------------------------------------------------------------------------

st.header("5. Final Model Comparison")
st.caption("Pulled directly from the saved evaluation results — nothing recalculated here.")

final_df = load_final_comparison()
final_shap = final_df[final_df["shap_model"] == shap_model].copy()
final_shap["llm_label"] = final_shap["llm"].map(lambda k: LLM_INFO.get(k, {}).get("label", k))

display_cols = [
    "llm_label",
    "mean_feature_agree",
    "mean_direction_agree",
    "n_cases",
]
table = final_shap[display_cols].rename(
    columns={
        "llm_label": "Model",
        "mean_feature_agree": "Mean Feature Agreement %",
        "mean_direction_agree": "Mean Directional Accuracy %",
        "n_cases": "Cases evaluated",
    }
).reset_index(drop=True)

best_feature_model = table.loc[table["Mean Feature Agreement %"].idxmax(), "Model"]
best_direction_model = table.loc[table["Mean Directional Accuracy %"].idxmax(), "Model"]


def highlight(row):
    styles = [""] * len(row)
    if row["Model"] == best_feature_model:
        styles[table.columns.get_loc("Mean Feature Agreement %")] = "background-color: #d4efdf; font-weight: bold"
    if row["Model"] == best_direction_model:
        styles[table.columns.get_loc("Mean Directional Accuracy %")] = "background-color: #d4efdf; font-weight: bold"
    return styles


st.dataframe(
    table.style.apply(highlight, axis=1).format(
        {"Mean Feature Agreement %": "{:.1f}", "Mean Directional Accuracy %": "{:.1f}"}
    ),
    hide_index=True,
    use_container_width=True,
)

c1, c2 = st.columns(2)
c1.success(f"🏆 Strongest on **Feature Agreement**: {best_feature_model}")
c2.success(f"🏆 Strongest on **Directional Accuracy**: {best_direction_model}")

st.caption(f"SHAP reference model shown above: {shap_model}. Switch it in the sidebar to compare.")
