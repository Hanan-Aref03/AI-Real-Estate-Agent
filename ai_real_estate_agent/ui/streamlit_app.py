"""Clean, light Streamlit UI for the AI Real Estate Agent backend."""

from __future__ import annotations

import time
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="AI Real Estate Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FEATURE_FIELDS = [
    ("lot_area", "Lot Area", "9600"),
    ("year_built", "Year Built", "2003"),
    ("year_remod_add", "Year Remodeled", "2005"),
    ("mas_vnr_area", "Masonry Veneer", "0"),
    ("bsmt_unf_sf", "Unfinished Basement", "100"),
    ("total_bsmt_sf", "Total Basement", "950"),
    ("first_flr_sf", "First Floor", "1200"),
    ("garage_area", "Garage Area", "500"),
    ("living_area", "Living Area", "1800"),
]

DEFAULT_QUERY = (
    "A single-family home built in 2003 on a 9,600 sq ft lot, with 1,800 sq ft of living area, "
    "1,200 sq ft on the first floor, a 500 sq ft garage, and a 950 sq ft total basement."
)


def initialize_state() -> None:
    """Initialize session state defaults."""

    defaults = {
        "api_url": "http://localhost:8000",
        "main_query": DEFAULT_QUERY,
        "assistant_query": DEFAULT_QUERY,
        "last_prediction": None,
        "last_missing_prediction": None,
        "last_assistant_result": None,
        "autofill_notice": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    for field_name, _, _ in FEATURE_FIELDS:
        st.session_state.setdefault(f"field_{field_name}", "")


def apply_styles() -> None:
    """Apply a light, blue professional UI style."""

    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f8fc;
            --surface: #ffffff;
            --surface-soft: #f8fbff;
            --text: #1e2a3a;
            --muted: #62748a;
            --blue: #2f6fed;
            --blue-soft: #eaf2ff;
            --border: #dce7f5;
            --success: #e9f7ef;
            --warning: #fff5e8;
            --danger: #fff0f0;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(47, 111, 237, 0.10), transparent 22%),
                linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 1.8rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: var(--text) !important;
            font-family: "Segoe UI", Arial, sans-serif !important;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #f3f8ff 100%);
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .hero p {
            margin: 0.55rem 0 0 0;
            color: var(--muted);
            font-size: 1.05rem;
        }

        .mini-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.05rem;
            margin-bottom: 1rem;
        }

        .section-title {
            font-size: 1.55rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .section-copy {
            color: var(--muted);
            margin-bottom: 0;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0;
        }

        .chip {
            border-radius: 999px;
            padding: 0.35rem 0.75rem;
            font-size: 0.86rem;
            border: 1px solid var(--border);
            color: var(--text);
            background: var(--surface-soft);
        }

        .chip-missing {
            background: var(--warning);
        }

        .chip-ready {
            background: var(--blue-soft);
        }

        .result-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1.2rem;
            margin-top: 1rem;
        }

        .result-label {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .result-price {
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 700;
            color: var(--blue);
            margin: 0.15rem 0 0.85rem 0;
        }

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid #2f6fed !important;
            background: linear-gradient(135deg, #2f6fed, #4e8cff) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 22px rgba(47, 111, 237, 0.16);
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextInput"] input {
            border-radius: 12px !important;
            border: 1px solid var(--border) !important;
            background: #ffffff !important;
            color: var(--text) !important;
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.8rem 1rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding-left: 0.2rem;
            padding-right: 0.2rem;
            color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
            color: var(--blue) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def humanize_feature_name(feature_name: str) -> str:
    return feature_name.replace("_", " ").title()


def parse_feature_value(raw_value: str) -> float | int | None:
    cleaned = raw_value.strip()
    if not cleaned:
        return None
    numeric = float(cleaned.replace(",", ""))
    return int(numeric) if numeric.is_integer() else numeric


def collect_manual_features() -> tuple[dict[str, float | int], list[str]]:
    """Collect manual feature overrides."""

    values: dict[str, float | int] = {}
    errors: list[str] = []
    for field_name, label, _ in FEATURE_FIELDS:
        raw_value = st.session_state.get(f"field_{field_name}", "")
        if not raw_value.strip():
            continue
        try:
            values[field_name] = parse_feature_value(raw_value)  # type: ignore[assignment]
        except ValueError:
            errors.append(f"{label} must be a valid number.")
    return values, errors


def render_progress(done_label: str) -> None:
    """Show a short progress bar."""

    progress = st.progress(0)
    note = st.empty()
    steps = [(25, "Reading request"), (60, "Analyzing details"), (100, done_label)]
    for value, label in steps:
        note.caption(label)
        progress.progress(value)
        time.sleep(0.1)
    note.empty()


def request_api(endpoint: str, payload: dict[str, Any], done_label: str) -> tuple[int, Any]:
    """Call backend and parse JSON safely."""

    render_progress(done_label)
    response = requests.post(endpoint, json=payload, timeout=60)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"detail": response.text}


def render_chips(items: list[str], kind: str) -> None:
    css_class = "chip chip-missing" if kind == "missing" else "chip chip-ready"
    chips = "".join(f'<span class="{css_class}">{humanize_feature_name(item)}</span>' for item in items)
    st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)


def apply_typical_values_for_missing(missing_payload: dict[str, Any]) -> None:
    """Fill missing values with training-set medians."""

    feature_statistics = missing_payload.get("stats_summary", {}).get("feature_statistics", {})
    applied: list[str] = []
    for field_name in missing_payload.get("missing_fields", []):
        median = feature_statistics.get(field_name, {}).get("median")
        if median is None:
            continue
        st.session_state[f"field_{field_name}"] = (
            str(int(median)) if float(median).is_integer() else f"{float(median):.2f}"
        )
        applied.append(humanize_feature_name(field_name))

    if applied:
        st.session_state.autofill_notice = "Filled: " + ", ".join(applied)


def render_prediction_result(data: dict[str, Any]) -> None:
    """Render pricing result."""

    stats = data.get("stats_summary", {})
    feature_importance = stats.get("feature_importance", {})
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Property Value</div>
            <div class="result-price">${data["predicted_price"]:,.0f}</div>
            <div>{data["interpretation"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Median", f'${(stats.get("median_sale_price") or 0):,.0f}')
    with metric_cols[1]:
        st.metric("Min", f'${(stats.get("min_sale_price") or 0):,.0f}')
    with metric_cols[2]:
        st.metric("Max", f'${(stats.get("max_sale_price") or 0):,.0f}')

    insights: list[str] = []
    median_price = stats.get("median_sale_price")
    if median_price:
        if data["predicted_price"] > median_price:
            insights.append("This estimate is above the market median in the training data.")
        elif data["predicted_price"] < median_price:
            insights.append("This estimate is below the market median in the training data.")
        else:
            insights.append("This estimate is close to the market median in the training data.")

    top_drivers = list(feature_importance.keys())[:3]
    if top_drivers:
        insights.append("Top value drivers: " + ", ".join(humanize_feature_name(item) for item in top_drivers))

    if data.get("user_benefit_summary"):
        insights.append(data["user_benefit_summary"])

    if insights:
        st.markdown(
            """
            <div class="mini-card">
                <div class="section-title">Market Insights</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for insight in insights:
            st.write(f"- {insight}")

    if data.get("features_used"):
        st.caption("Features used")
        render_chips(list(data["features_used"].keys()), "ready")


def render_missing_state(data: dict[str, Any]) -> None:
    """Render missing-field state."""

    st.warning(data.get("user_message") or "Some details are still missing.")
    if data.get("missing_fields"):
        render_chips(data["missing_fields"], "missing")

    feature_statistics = data.get("stats_summary", {}).get("feature_statistics", {})
    if data.get("missing_fields") and feature_statistics:
        suggestion_cols = st.columns(min(3, len(data["missing_fields"])))
        for index, field_name in enumerate(data["missing_fields"]):
            median = feature_statistics.get(field_name, {}).get("median")
            if median is None:
                continue
            with suggestion_cols[index % len(suggestion_cols)]:
                st.metric(
                    humanize_feature_name(field_name),
                    f"{median:,.0f}" if float(median).is_integer() else f"{median:,.2f}",
                )
        if st.button("Use AI Assistant Suggestions", use_container_width=True):
            apply_typical_values_for_missing(data)
            st.rerun()


def render_assistant_result(data: dict[str, Any]) -> None:
    """Render AI Assistant result."""

    if data.get("missing_fields"):
        st.caption("Missing details")
        render_chips(data["missing_fields"], "missing")
    else:
        st.caption("Ready details")
        render_chips(list(data.get("features", {}).keys()), "ready")
    st.json(data.get("features", {}), expanded=False)


initialize_state()
apply_styles()

with st.sidebar:
    st.text_input("API URL", key="api_url")
    if st.button("Load Example", use_container_width=True):
        st.session_state.main_query = DEFAULT_QUERY
        st.session_state.assistant_query = DEFAULT_QUERY
        st.rerun()

st.markdown(
    """
    <div class="hero">
        <h1>AI Real Estate Agent</h1>
        <p>Predict property prices and get real estate insights powered by AI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_assistant = st.tabs(["Price Prediction", "AI Assistant"])

with tab_predict:
    if st.session_state.autofill_notice:
        st.success(st.session_state.autofill_notice)
        st.session_state.autofill_notice = None

    st.markdown(
        """
        <div class="mini-card">
            <div class="section-title">Estimate a Property Value</div>
            <p class="section-copy">Describe the property, then add any missing details if needed.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.text_area("Property Description", key="main_query", height=150)
    st.caption("Optional details")

    cols = st.columns(3)
    for index, (field_name, label, placeholder) in enumerate(FEATURE_FIELDS):
        with cols[index % 3]:
            st.text_input(label, key=f"field_{field_name}", placeholder=placeholder)

    if st.button("Predict Price", use_container_width=True):
        manual_features, validation_errors = collect_manual_features()
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        elif not st.session_state.main_query.strip():
            st.error("Enter a property description.")
        else:
            payload = {
                "query": st.session_state.main_query,
                "user_filled_features": manual_features or None,
            }
            try:
                status_code, data = request_api(
                    f"{st.session_state.api_url.rstrip('/')}/predict",
                    payload,
                    "Ready",
                )
                if status_code == 200:
                    st.session_state.last_prediction = data
                    st.session_state.last_missing_prediction = None
                elif status_code == 400:
                    st.session_state.last_prediction = None
                    st.session_state.last_missing_prediction = data
                else:
                    st.session_state.last_prediction = None
                    st.session_state.last_missing_prediction = None
                    st.error(data.get("detail", data))
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to {st.session_state.api_url}")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.last_prediction:
        render_prediction_result(st.session_state.last_prediction)
    elif st.session_state.last_missing_prediction:
        render_missing_state(st.session_state.last_missing_prediction)

with tab_assistant:
    st.markdown(
        """
        <div class="mini-card">
            <div class="section-title">Real Estate AI Assistant</div>
            <p class="section-copy">Let the assistant read the property details and prepare the feature set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assistant_query = st.text_area(
        "Ask the AI Assistant",
        value=st.session_state.assistant_query,
        height=150,
        key="assistant_query",
    )

    if st.button("Ask AI", use_container_width=True):
        manual_features, validation_errors = collect_manual_features()
        if validation_errors:
            for error in validation_errors:
                st.error(error)
        elif not assistant_query.strip():
            st.error("Enter a property description.")
        else:
            payload = {
                "query": assistant_query,
                "user_filled_features": manual_features or None,
            }
            try:
                status_code, data = request_api(
                    f"{st.session_state.api_url.rstrip('/')}/extract",
                    payload,
                    "Ready",
                )
                if status_code == 200:
                    st.session_state.last_assistant_result = data
                else:
                    st.error(data.get("detail", data))
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to {st.session_state.api_url}")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.last_assistant_result:
        render_assistant_result(st.session_state.last_assistant_result)
