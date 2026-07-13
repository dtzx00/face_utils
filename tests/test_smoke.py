"""End-to-end smoke test for face_utils.

Covers the statistics module and the network-free image/analysis utilities.
The Face++ API functions require live credentials + a face image and are not
exercised here.

Run: python tests/test_smoke.py   (or: python -m pytest tests/)
"""
import warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from face_utils import utils, stats

np.random.seed(0)


def test_stats_formatting():
    assert stats.round_str(0.12345, 3) == "0.123"
    assert stats.stars(0.5, 0.001).endswith("**")
    assert stats.pval(0.0001) == "<.001"
    assert stats.convert_ci_str((0.1, 0.2), 2) == "[0.10, 0.20]"


def test_cohen_effect_size():
    # legacy cat-based path
    assert np.isfinite(stats.cohen_effect_size(0.6, 0.4, 0.04, 0.04, "men"))
    # explicit sample sizes override cat
    assert np.isfinite(stats.cohen_effect_size(0.6, 0.4, 0.04, 0.05, n1=100, n2=120))
    # corrected pooled-variance formula: equal var=1, equal n -> d = mean diff
    d = stats.cohen_effect_size(1.0, 0.0, 1.0, 1.0, n1=50, n2=50, correct=True)
    assert abs(d - 1.0) < 1e-9


def test_delong_auc():
    truth = np.array([0] * 100 + [1] * 100)
    good = np.concatenate([np.random.normal(0.3, 0.2, 100),
                           np.random.normal(0.7, 0.2, 100)])
    weak = np.concatenate([np.random.normal(0.45, 0.3, 100),
                           np.random.normal(0.55, 0.3, 100)])
    logp = stats.delong_roc_test(truth, good, weak)
    p = 10 ** float(np.array(logp).ravel()[0])
    assert 0.0 <= p <= 1.0
    auc, var = stats.delong_roc_variance(truth, good)
    assert 0.5 < auc <= 1.0 and var >= 0


def test_image_utils():
    img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    assert utils.Get_Euclidean_Distance(np.array([0, 0, 0]),
                                        np.array([3, 4, 0])) == 5.0
    flat = utils.Get_Flattened_Dict({"a": {"b": 1}, "d": 3})
    assert flat["a_b"] == 1 and flat["d"] == 3
    assert utils.to_grayscale(img).shape[:2] == (224, 224)
    assert utils.get_rotated_image(img, (80, 90), (150, 90)).shape[:2] == (224, 224)


def test_modeling():
    n = 140
    df = pd.DataFrame({
        "type": [0] * 70 + [1] * 70,
        "x1": np.concatenate([np.random.normal(0, 1, 70),
                              np.random.normal(1.3, 1, 70)]),
        "x2": np.concatenate([np.random.normal(0, 1, 70),
                              np.random.normal(-0.9, 1, 70)]),
    })
    res = utils.logistic_regression(df)
    assert res is not None
    acc = utils.Classify_LR(df, yvar="type")
    assert acc is not None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"[PASS] {fn.__name__}"); passed += 1
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)
