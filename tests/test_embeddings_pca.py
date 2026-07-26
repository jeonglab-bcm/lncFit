import numpy as np
import pytest

from lncfit.embeddings import reduce_embeddings_pca


def _fixture(n_rows=100, n_dims=50, seed=0):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_rows, n_dims)).astype(np.float32)
    index = {f"T{i}": i for i in range(n_rows)}
    return matrix, index


def test_reduces_to_n_components_and_preserves_index():
    matrix, index = _fixture()
    train = {f"T{i}" for i in range(80)}
    reduced, out_index = reduce_embeddings_pca((matrix, index), train, 10)
    assert reduced.shape == (100, 10)
    assert reduced.dtype == np.float32
    # index must be unchanged -- callers swap the tuple in and keep using target ids
    assert out_index == index


def test_fitting_on_train_only_differs_from_fitting_on_all():
    # The whole point of passing train_targets: the projection must not be
    # informed by held-out rows.
    matrix, index = _fixture()
    train = {f"T{i}" for i in range(80)}
    train_fit, _ = reduce_embeddings_pca((matrix, index), train, 10, seed=42)
    all_fit, _ = reduce_embeddings_pca((matrix, index), None, 10, seed=42)
    assert not np.allclose(train_fit, all_fit)


def test_transforms_heldout_rows_too():
    matrix, index = _fixture()
    train = {f"T{i}" for i in range(80)}
    reduced, _ = reduce_embeddings_pca((matrix, index), train, 10)
    # rows 80..99 were not fit on, but must still be projected (not zero/NaN)
    heldout = reduced[80:]
    assert np.isfinite(heldout).all()
    assert np.abs(heldout).sum() > 0


def test_n_components_clamped_to_available_dims():
    matrix, index = _fixture(n_dims=50)
    reduced, _ = reduce_embeddings_pca((matrix, index), None, 500)
    assert reduced.shape[1] == 50


def test_too_few_training_rows_raises():
    matrix, index = _fixture()
    with pytest.raises(ValueError, match="at least n_components"):
        reduce_embeddings_pca((matrix, index), {"T0", "T1"}, 10)


def test_constant_columns_do_not_produce_nans():
    # std==0 columns would divide by zero if not guarded.
    matrix, index = _fixture()
    matrix[:, 3] = 7.0
    reduced, _ = reduce_embeddings_pca((matrix, index), None, 10)
    assert np.isfinite(reduced).all()


def test_targets_missing_from_index_are_ignored_when_fitting():
    matrix, index = _fixture()
    train = {f"T{i}" for i in range(80)} | {"NOT_IN_INDEX"}
    reduced, _ = reduce_embeddings_pca((matrix, index), train, 10)
    assert reduced.shape == (100, 10)
