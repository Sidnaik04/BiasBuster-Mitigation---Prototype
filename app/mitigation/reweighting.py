def compute_sample_weights(y, sensitive):
    import pandas as pd

    df = pd.DataFrame({
        "y": y,
        "s": sensitive
    })

    # Count combinations (label + sensitive)
    group_counts = df.groupby(["y", "s"]).size()

    total = len(df)

    # Assign weight based on combination
    weights = df.apply(
        lambda row: total / group_counts[(row["y"], row["s"])],
        axis=1
    )

    return weights