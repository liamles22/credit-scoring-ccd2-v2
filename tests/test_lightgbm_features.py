import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.lightgbm_features import fit_category_levels, lightgbm_frame, saved_category_levels


def test_saved_lightgbm_treats_unseen_category_as_missing(tmp_path):
    features = ["purpose"]
    fit = pd.DataFrame({"purpose": ["car", "home", None] * 8})
    y = [0, 1, 1] * 8
    holdout = pd.DataFrame({"purpose": ["new-purpose", None, "car"]})
    levels = fit_category_levels(fit, features)

    model = lgb.LGBMClassifier(n_estimators=5, min_child_samples=1, n_jobs=1, verbosity=-1)
    model.fit(lightgbm_frame(fit, features, levels), y)
    path = tmp_path / "model.txt"
    model.booster_.save_model(str(path))
    restored = lgb.Booster(model_file=str(path))

    saved_levels = saved_category_levels(restored, features)
    assert saved_levels == {"purpose": ["car", "home"]}
    scored = lightgbm_frame(holdout, features, saved_levels)
    assert scored["purpose"].cat.codes.tolist() == [-1, -1, 0]

    predictions = restored.predict(scored, num_threads=1)
    np.testing.assert_allclose(predictions, model.predict_proba(scored)[:, 1])
    assert predictions[0] == pytest.approx(predictions[1])
    assert predictions[0] > predictions[2]
