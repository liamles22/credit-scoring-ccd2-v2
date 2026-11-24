"""Load the models saved by notebook 04 and score any sample."""
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier

from src.data import PROJECT_ROOT
from src.lightgbm_features import lightgbm_frame, saved_category_levels

MODELS = PROJECT_ROOT / "models"
MODEL_NAMES = ["Scorecard", "LightGBM", "LightGBM monotone", "CatBoost", "CatBoost monotone"]


def _file(name, ext):
    return MODELS / f"{name.lower().replace(' ', '_')}.{ext}"


class Models:
    def __init__(self, categories: dict | None = None):
        """Load saved models. `categories` is a fallback for older model files without levels."""
        with open(MODELS / "scorecard_logit.pkl", "rb") as fh:
            self.scorecard = pickle.load(fh)
        with open(MODELS / "boosting_meta.pkl", "rb") as fh:
            meta = pickle.load(fh)
        self.features, self.categorical = meta["features"], meta["categorical"]
        self.lgb = {n: lgb.Booster(model_file=str(_file(n, "txt"))) for n in ["LightGBM", "LightGBM monotone"]}
        self.category_levels = {
            name: saved_category_levels(booster, self.categorical)
            if booster.pandas_categorical is not None else categories
            for name, booster in self.lgb.items()
        }
        if any(levels is None for levels in self.category_levels.values()):
            raise ValueError("LightGBM model has no saved category levels; provide its training levels")
        self.cb = {n: CatBoostClassifier().load_model(str(_file(n, "cbm"))) for n in ["CatBoost", "CatBoost monotone"]}

    def _lgb_frame(self, df, name):
        return lightgbm_frame(df, self.features, self.category_levels[name])

    def _cb_frame(self, df):
        X = df[self.features].copy()
        for c in self.categorical:
            X[c] = X[c].astype(str).fillna("missing")
        return X

    def predict(self, df: pd.DataFrame) -> dict:
        """PD of every model for every row of `df`."""
        W = self.scorecard["binning"].transform(df)[self.scorecard["features"]]
        out = {"Scorecard": np.asarray(self.scorecard["model"].predict(sm.add_constant(W, has_constant="add")))}
        Xc = self._cb_frame(df)
        out.update({n: b.predict(self._lgb_frame(df, n)) for n, b in self.lgb.items()})
        out.update({n: m.predict_proba(Xc)[:, 1] for n, m in self.cb.items()})
        return out
