"""Streamlit UI for the AI Real Estate Agent backend."""

from __future__ import annotations

import requests
import streamlit as st

st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏠", layout="wide")

st.sidebar.title("Configuration")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000").rstrip("/")

st.title("AI Real Estate Agent")
st.markdown("Describe a property in natural language, fill any missing features, and get a price estimate.")

tab1, tab2 = st.tabs(["Price Prediction", "Feature Extraction"])

feature_fields = [
    ("lot_area", "Lot Area (sq ft)", 8000.0),
    ("year_built", "Year Built", 2000.0),
    ("year_remod_add", "Year Remodeled", 2000.0),
    ("mas_vnr_area", "Masonry Veneer Area", 0.0),
    ("bsmt_unf_sf", "Unfinished Basement (sq ft)", 0.0),
    ("total_bsmt_sf", "Total Basement (sq ft)", 1000.0),
    ("first_flr_sf", "First Floor (sq ft)", 1200.0),
    ("garage_area", "Garage Area (sq ft)", 500.0),
    ("living_area", "Living Area (sq ft)", 1800.0),
]

default_query = (
    "A single-family home built in 2003 with a lot area of 9600 sq ft, "
    "1800 sq ft of living area, 1200 sq ft on the first floor, a 500 sq ft garage, "
    "and 950 sq ft total basement."
)


def build_user_filled_features() -> dict[str, float]:
    """Collect optional feature overrides from the sidebar form."""

    values: dict[str, float] = {}
    for field_name, label, default_value in feature_fields:
        raw_value = st.session_state.get(field_name)
        if raw_value not in (None, ""):
            values[field_name] = float(raw_value)
        elif default_value is not None:
            values[field_name] = float(default_value)
    return values


with tab1:
    st.header("Property Price Prediction")
    query = st.text_area(
        "Property description",
        value=default_query,
        height=140,
        help="Describe the property in natural language. The backend will extract ML features from this text.",
    )

    st.subheader("Optional feature overrides")
    cols = st.columns(3)
    for index, (field_name, label, default_value) in enumerate(feature_fields):
        with cols[index % 3]:
            st.number_input(label, min_value=0.0, value=float(default_value), key=field_name)

    if st.button("Predict Price", use_container_width=True):
        payload = {
            "query": query,
            "user_filled_features": build_user_filled_features(),
        }
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=60)
            data = response.json()

            if response.status_code == 200:
                st.metric("Predicted Price", f"${data['predicted_price']:,.2f}")
                st.info(data["interpretation"])
                st.json(data["features_used"], expanded=False)
                if data.get("warnings"):
                    for warning in data["warnings"]:
                        st.warning(warning)
            elif response.status_code == 400:
                st.error("Prediction blocked because required features are still missing.")
                st.write("Missing fields:", ", ".join(data.get("missing_fields", [])) or "Unknown")
                if data.get("extraction"):
                    st.json(data["extraction"], expanded=False)
            else:
                st.error(data)
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to API at {api_url}")
        except Exception as exc:
            st.error(str(exc))

with tab2:
    st.header("Feature Extraction Preview")
    extract_query = st.text_area(
        "Extraction-only query",
        value=default_query,
        height=140,
        key="extract_query",
    )

    if st.button("Run Extraction", use_container_width=True):
        payload = {
            "query": extract_query,
            "user_filled_features": build_user_filled_features(),
        }
        try:
            response = requests.post(f"{api_url}/extract", json=payload, timeout=60)
            data = response.json()

            if response.status_code == 200:
                st.success("Extraction completed")
                st.json(data, expanded=True)
            else:
                st.error(data)
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to API at {api_url}")
        except Exception as exc:
            st.error(str(exc))

st.markdown("---")
st.caption("Powered by FastAPI, scikit-learn, and a two-stage LLM workflow.")
