from pathlib import Path

import numpy as np
import pandas as pd


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
    drop_cols = ["cloud", "lightning_prob", "precip"]
    df = df.drop(columns=drop_cols)

    # We'll limit the dataset to the first 1000 samples
    return df.iloc[:1000, :].to_numpy()


def main():
    p = Path("data")
    atl = pd.read_csv(
        p / "katl_lamp.csv.bz2", parse_dates=["timestamp", "forecast_timestamp"]
    )

    clt = pd.read_csv(
        p / "kclt_lamp.csv.bz2", parse_dates=["timestamp", "forecast_timestamp"]
    )

    den = pd.read_csv(
        p / "kden_lamp.csv.bz2", parse_dates=["timestamp", "forecast_timestamp"]
    )

    # We'll subsample the data to include only 100 instances for the purposes
    # of this assignment this should give a (100 x K x 3) tensor
    atl = subsample_data(atl)
    clt = subsample_data(clt)
    den = subsample_data(den)

    # Finally merge the three datasets into a tensor
    X = np.stack((atl, clt, den), axis=2)

    # Write to disk
    np.save(p / "data.npy", X)


if __name__ == "__main__":
    main()
