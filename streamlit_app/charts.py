import pandas as pd
import streamlit as st


def fairness_chart(selection_rates):

    df = pd.DataFrame(
        {
            "group": list(selection_rates.keys()),
            "selection_rate": list(selection_rates.values()),
        }
    )

    st.bar_chart(df.set_index("group"))

def fairness_table(fairness_metrics):

    df = pd.DataFrame(
        {
            "Metric": ["DPD", "DIR", "EOD"],
            "Value": [
                fairness_metrics["dpd"],
                fairness_metrics["dir"],
                fairness_metrics["eod"],
            ],
        }
    )

    st.table(df)
def performance_table(metrics):

    df = pd.DataFrame(metrics, index=["value"]).T

    st.table(df)


def mitigation_summary(comparison):

    fairness = comparison["fairness_changes"]
    performance = comparison["performance_changes"]

    improved_metrics = [m for m, v in fairness.items() if v["improved"]]

    if improved_metrics:
        st.success(f"Fairness improved in: {', '.join(improved_metrics)}")
    else:
        st.error("Fairness did NOT improve")

    if performance["accuracy"]["change"] < 0:
        st.warning(
            f"Accuracy decreased by {abs(performance['accuracy']['change']):.3f}"
        )
    else:
        st.info("Accuracy maintained or improved")
