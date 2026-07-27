from __future__ import annotations

import numpy as np

from scripts.build_day14_loco_ensemble import blend_components


def test_blend_components_minmax_normalizes_and_weights(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    common = {
        "target": np.asarray(["T1", "T2"]),
        "cell_line": np.asarray(["HAP1", "K562"]),
        "y": np.asarray([0, 1]),
    }
    np.savez(first, **common, model_a=np.asarray([10.0, 20.0]))
    np.savez(second, **common, model_b=np.asarray([4.0, 2.0]))

    targets, cells, prediction, labels = blend_components(
        [(first, "model_a", 0.75), (second, "model_b", 0.25)]
    )

    np.testing.assert_array_equal(targets, ["T1", "T2"])
    np.testing.assert_array_equal(cells, ["HAP1", "K562"])
    np.testing.assert_array_equal(labels, [0, 1])
    np.testing.assert_allclose(prediction, [0.25, 0.75])
