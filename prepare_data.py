from pathlib import Path

import numpy as np
import pandas as pd
import tensorly as tl


def subsample_data(df: pd.DataFrame) -> np.ndarray:
    """Sub-samples the data to make it more manageable for this assignment

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to subsample

    Returns
    -------
    np.ndarray
        Sub-sampled array ready to merged into a tensor
    """

    df = df.set_index(["timestamp", "forecast_timestamp"])
    df = df[~df.index.duplicated()]

    # Each timestamp has 24.5 hours worth of forecasts; just grab the first one
    unique_timestamps = df.index.get_level_values("timestamp").unique()
    first_forecasts = unique_timestamps + pd.Timedelta(30, "min")
    idx = zip(unique_timestamps, first_forecasts)
    df = df.loc[idx]

    # Some of the weather features are categories; we'll get rid of those
    # for the purpose of this exercise
    drop_cols = ["cloud", "lightning_prob", "precip", "cloud_ceiling", "visibility"]
    df = df.drop(columns=drop_cols)
    df = df.dropna()

    # Let's grab 2000 random samples from the data to help with SVD convergence
    rng = np.random.default_rng(17)
    idx = rng.choice(np.arange(df.shape[0]), size=2000, replace=False)
    return df.iloc[idx, :].to_numpy()


def main():
    p = Path("data")
    files = p.glob("*.bz2")
    dfs = [
        pd.read_csv(file, parse_dates=["timestamp", "forecast_timestamp"])
        for file in files
    ]

    # We'll subsample the data to include only 2000 instances for the purposes
    # of this assignment this should give a (2000 x J X K) tensor
    arrs = [subsample_data(df) for df in dfs]

    # Merge the three datasets into a tensor
    X = np.stack(arrs, axis=2)

    # Write to disk
    np.save(p / "data.npy", X)


if __name__ == "__main__":
    main()
