import numpy as np
import pytest

from lncfit.resample import METHODS, resample


def _imbalanced(n=1000, pos_rate=0.045, n_feat=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    y = (rng.random(n) < pos_rate).astype(int)
    # guarantee enough positives for SMOTE's k-NN (needs >= k_neighbors+1)
    y[:10] = 1
    return X, y


def test_none_is_passthrough():
    X, y = _imbalanced()
    X_res, y_res = resample(X, y, method="none")
    assert X_res is X and y_res is y


def test_unknown_method_raises():
    X, y = _imbalanced()
    with pytest.raises(ValueError, match="Unknown resample method"):
        resample(X, y, method="bogus")


@pytest.mark.parametrize("method", [m for m in METHODS if m != "none"])
def test_every_method_balances_and_preserves_width(method):
    X, y = _imbalanced()
    X_res, y_res = resample(X, y, method=method, seed=42)
    assert X_res.shape[1] == X.shape[1], "feature count must not change"
    assert len(X_res) == len(y_res)
    # "auto" targets a 1:1 balance; smote_tomek then removes some pairs, so allow
    # a little slack rather than demanding exactly 50%.
    assert 0.4 <= y_res.mean() <= 0.6, f"{method} left rate at {y_res.mean():.3f}"


def test_oversampling_grows_and_undersampling_shrinks():
    X, y = _imbalanced()
    over_X, _ = resample(X, y, method="random_over", seed=42)
    under_X, _ = resample(X, y, method="random_under", seed=42)
    assert len(over_X) > len(X)
    assert len(under_X) < len(X)


def test_random_under_keeps_only_real_rows():
    # Undersampling must not synthesize anything -- every kept row should be an
    # exact row of the original X (this is the property that distinguishes it
    # from SMOTE).
    X, y = _imbalanced(n=300)
    X_res, _ = resample(X, y, method="random_under", seed=42)
    original = {tuple(np.round(r, 10)) for r in X}
    assert all(tuple(np.round(r, 10)) in original for r in X_res)


def test_smote_synthesizes_new_rows():
    X, y = _imbalanced(n=300)
    X_res, y_res = resample(X, y, method="smote", seed=42)
    original = {tuple(np.round(r, 10)) for r in X}
    synthetic = [r for r in X_res if tuple(np.round(r, 10)) not in original]
    assert synthetic, "SMOTE should interpolate rows that aren't in the original X"


def test_float_ratio_gives_partial_rebalancing():
    X, y = _imbalanced()
    _, y_res = resample(X, y, method="smote", ratio=0.3, seed=42)
    rate = y_res.mean()
    # ratio=0.3 => minority ~30% of majority => ~23% of the total, i.e. well
    # above the natural ~4.5% but clearly short of a full 1:1 balance.
    assert y.mean() < rate < 0.5


def test_is_deterministic_for_a_given_seed():
    X, y = _imbalanced()
    a_X, a_y = resample(X, y, method="smote", seed=7)
    b_X, b_y = resample(X, y, method="smote", seed=7)
    assert np.array_equal(a_X, b_X) and np.array_equal(a_y, b_y)
