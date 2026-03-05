import streamlit as st
import pandas as pd

from api_client import *
from charts import *

st.set_page_config(page_title="Bias Mitigation Toolkit")

st.title("Bias Mitigation Prototype")

for key in ["smote_result", "reweighting_result", "threshold_result", "ranking"]:
    st.session_state.setdefault(key, None)

# ------------------------------------------------
# Upload
# ------------------------------------------------

st.header("1️⃣ Upload Dataset and Model")

dataset = st.file_uploader("Upload CSV dataset")
model = st.file_uploader("Upload trained model (.pkl)")

if st.button("Upload"):

    result = upload_files(dataset, model)

    if "upload_id" not in result:
        st.error(f"Upload failed: {result.get('detail', result)}")
        st.stop()

    st.session_state["upload_id"] = result["upload_id"]
    st.session_state["columns"] = result["dataset_info"]["columns"]

    st.success(f"Upload successful. ID = {result['upload_id']}")

# ------------------------------------------------
# Select attributes
# ------------------------------------------------

if "columns" in st.session_state:

    st.header("2️⃣ Select Attributes")

    target = st.selectbox("Target column", st.session_state["columns"])

    sensitive = st.selectbox("Sensitive attribute", st.session_state["columns"])

    if st.button("Run Baseline Audit"):

        baseline = run_baseline(st.session_state["upload_id"], target, sensitive)

        if "baseline_metrics" not in baseline:
            st.error(f"Baseline failed: {baseline.get('detail', baseline)}")
            st.stop()

        st.session_state["baseline"] = baseline
        st.session_state["target_column"] = target
        st.session_state["sensitive_attribute"] = sensitive

# ------------------------------------------------
# Show baseline
# ------------------------------------------------

if "baseline" in st.session_state:

    st.header("3️⃣ Baseline Metrics")

    metrics = st.session_state["baseline"]["baseline_metrics"]

    st.subheader("Performance")

    performance_table(metrics["performance"])

    st.subheader("Fairness")

    fairness_chart(metrics["fairness"]["selection_rate"])

# ------------------------------------------------
# Recommendation
# ------------------------------------------------

if "baseline" in st.session_state:

    st.header("4️⃣ Strategy Recommendation")

    recommendation = recommend_strategy(
        st.session_state["upload_id"], st.session_state["baseline"]["baseline_metrics"]
    )

    if "recommendation" in recommendation:
        st.session_state["recommendation"] = recommendation
        rec = recommendation["recommendation"]
        st.write("Recommended Strategy:", rec["recommended_strategy"])
        st.write(rec["explanation"])
    else:
        st.error(
            "Recommendation failed: " f"{recommendation.get('detail', recommendation)}"
        )

# ------------------------------------------------
# Apply mitigation
# ------------------------------------------------

if "recommendation" in st.session_state:

    st.header("5️⃣ Apply Mitigation")

    strategy = st.selectbox("Choose strategy", ["smote", "reweighting", "threshold"])

    if st.button("Run Mitigation"):

        payload = {
            "upload_id": st.session_state["upload_id"],
            "target_column": st.session_state.get("target_column"),
            "sensitive_attribute": st.session_state.get("sensitive_attribute"),
            "strategy": strategy,
            "strategy_config": {"test_size": 0.2},
        }

        if not payload["target_column"] or not payload["sensitive_attribute"]:
            st.error(
                "Please run Baseline Audit first to set target and sensitive columns."
            )
            st.stop()

        result = apply_mitigation(payload)

        if "before" not in result or "after" not in result:
            st.error(f"Mitigation failed: {result.get('detail', result)}")
            st.stop()

        st.session_state["result"] = result
        st.session_state[f"{strategy}_result"] = result

# ------------------------------------------------
# Results
# ------------------------------------------------

if "result" in st.session_state:

    st.header("6️⃣ Mitigation Results")

    before = st.session_state["result"]["before"]
    after = st.session_state["result"]["after"]

    st.subheader("Before")

    performance_table(before["performance"])
    fairness_chart(before["fairness"]["selection_rate"])

    st.subheader("After")

    performance_table(after["performance"])
    fairness_chart(after["fairness"]["selection_rate"])

    comparison = st.session_state["result"]["comparison"]

    st.header("Mitigation Assessment")

    mitigation_summary(comparison)

    st.success("Mitigated model saved!")

    st.write("Model Path:", st.session_state["result"]["artifact_model"])


st.header("7️⃣ Automatic Strategy Ranking")

if st.button("Run Automatic Strategy Ranking"):

    missing = [
        name
        for name in ["smote", "reweighting", "threshold"]
        if st.session_state.get(f"{name}_result") is None
    ]

    if missing:
        st.warning(
            "Run mitigation for all strategies before ranking. "
            f"Missing: {', '.join(missing)}"
        )
        st.stop()

    results = {
        "smote": st.session_state.get("smote_result"),
        "reweighting": st.session_state.get("reweighting_result"),
        "threshold": st.session_state.get("threshold_result"),
    }

    ranking = auto_rank_strategies(results)

    st.session_state["ranking"] = ranking

if st.session_state.get("ranking") is not None:

    st.subheader("Strategy Ranking")

    ranking_data = st.session_state["ranking"]["ranking"]

    df = pd.DataFrame(ranking_data)

    st.table(df)

    st.success(f"Best Strategy: {st.session_state['ranking']['best_strategy']}")
