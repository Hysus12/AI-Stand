from __future__ import annotations

import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000"


def post_json(path: str, payload: dict) -> dict:
    response = httpx.post(f"{API_BASE_URL}{path}", json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def main() -> None:
    st.set_page_config(page_title="SPBCE Demo", layout="wide")
    st.title("SPBCE")
    st.caption("Survey prior prediction, calibration notes, and synthetic respondent sampling.")

    question_text = st.text_area("Question text", "Do you support this proposal?")
    options_text = st.text_input(
        "Answer options (comma separated)", "Strongly support, Support, Oppose"
    )
    population_text = st.text_input("Target population", "Adults in Taiwan")
    region = st.text_input("Population region", "Taiwan")
    category = st.text_input("Product or domain", "public_opinion")
    price = st.number_input("Price", min_value=0.0, value=0.0)
    sample_n = st.slider("Synthetic respondents", min_value=10, max_value=1000, value=100, step=10)

    options = [option.strip() for option in options_text.split(",") if option.strip()]
    survey_payload = {
        "question_text": question_text,
        "options": options,
        "population_text": population_text,
        "population_struct": {"region": region},
        "context": {"product_category": category, "price": price},
    }

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Predict survey", type="primary"):
            result = post_json("/predict-survey", survey_payload)
            st.subheader("Predicted distribution")
            st.bar_chart(result["distribution"])
            st.write("Uncertainty:", result["uncertainty"])
            st.write("OOD flag:", result["ood_flag"])
            st.write("Support notes:", result["support_notes"])
            st.write("Calibration notes:", result["calibration_notes"])
    with col2:
        if st.button("Sample respondents"):
            result = post_json(
                "/sample-respondents", {"survey_payload": survey_payload, "n": sample_n}
            )
            st.subheader("Synthetic sample")
            st.dataframe(result["respondents"][:20], use_container_width=True)
            st.write(result["sampling_notes"])


if __name__ == "__main__":
    main()
