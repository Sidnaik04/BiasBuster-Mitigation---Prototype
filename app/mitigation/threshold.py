from fairlearn.postprocessing import ThresholdOptimizer


def apply_threshold_optimizer(
    model,
    X_train,
    y_train,
    sensitive_train,
    grid_size=200,
):
    """
    Wraps the base model using Fairlearn ThresholdOptimizer.
    """

    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        predict_method="predict_proba",
        grid_size=grid_size,
    )

    optimizer.fit(
        X_train,
        y_train,
        sensitive_features=sensitive_train,
    )

    return optimizer