"""Categorical preprocessing shared by LightGBM training and scoring."""

import pandas as pd


def fit_category_levels(fit: pd.DataFrame, categorical) -> dict:
    """Learn string category levels from the rows used to fit the model."""
    return {column: sorted(fit[column].dropna().astype(str).unique()) for column in categorical}


def lightgbm_frame(frame: pd.DataFrame, features, category_levels: dict) -> pd.DataFrame:
    result = frame[list(features)].copy()
    for column, levels in category_levels.items():
        # Unseen levels become missing and follow LightGBM's missing-value routing.
        values = result[column].astype("string")
        result[column] = pd.Categorical(values.where(values.isin(levels)), categories=levels)
    return result


def saved_category_levels(booster, categorical) -> dict:
    """Read the pandas category levels persisted in the LightGBM model file."""
    columns = [column for column in booster.feature_name() if column in categorical]
    return dict(zip(columns, booster.pandas_categorical, strict=True))
