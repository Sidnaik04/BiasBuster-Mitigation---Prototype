def _extract_fairness_metrics(baseline_metrics: dict) -> dict:
    if not isinstance(baseline_metrics, dict):
        raise ValueError("baseline_metrics must be a dictionary")

    fairness = baseline_metrics.get("fairness")
    if isinstance(fairness, dict):
        return fairness

    wrapped = baseline_metrics.get("baseline_metrics")
    if isinstance(wrapped, dict) and isinstance(wrapped.get("fairness"), dict):
        return wrapped["fairness"]

    raise ValueError(
        "Missing fairness metrics. Expected key 'fairness' or 'baseline_metrics.fairness'."
    )


def recommend_strategy(baseline_metrics: dict):
    """
    Recommend a bias mitigation strategy based on baseline fairness metrics.

    Input:
        baseline_metrics = {
            "performance": {...},
            "fairness": {
                "dpd": float,
                "eod": float,
                "dir": float
            }
        }
    """

    fairness = _extract_fairness_metrics(baseline_metrics)

    dpd = abs(fairness.get("dpd", 0))
    eod = abs(fairness.get("eod", 0))
    dir_ratio = fairness.get("dir", 1.0)

    violations = {
        "dpd": dpd > 0.10,
        "eod": eod > 0.10,
        "dir": dir_ratio < 0.80,
    }

    explanations = []
    alternatives = []

    # Priority-based recommendation
    if violations["dir"]:
        recommended = "smote"
        explanations.append(
            "Disparate Impact Ratio is below 0.8, indicating representation imbalance "
            "across sensitive groups."
        )
        alternatives = ["reweighting", "threshold"]

    elif violations["eod"]:
        recommended = "threshold"
        explanations.append(
            "Equal Opportunity Difference is high, suggesting unequal true positive rates "
            "across sensitive groups."
        )
        alternatives = ["reweighting"]

    elif violations["dpd"]:
        recommended = "reweighting"
        explanations.append(
            "Demographic Parity Difference is high, indicating unequal selection rates "
            "between groups."
        )
        alternatives = ["threshold"]

    else:
        recommended = "none"
        explanations.append(
            "No significant fairness violations detected. Mitigation may not be necessary."
        )

    return {
        "recommended_strategy": recommended,
        "violations": violations,
        "explanation": " ".join(explanations),
        "alternatives": alternatives,
        "note": (
            "Mitigation may affect model accuracy. "
            "Human review is recommended before applying any intervention."
        ),
    }
