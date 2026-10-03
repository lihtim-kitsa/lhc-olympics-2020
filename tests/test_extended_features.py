import h5py
import numpy as np
import pandas as pd

from scripts.build_extended_features import build_extended_features


def test_extended_schema_preserves_canonical_inputs_mass_and_labels(tmp_path):
    base = np.array(
        [[10, 2, 0.2, 0.3, 3, 3500, 1], [20, 3, 0.4, 0.5, 2, 3000, 0]], dtype=np.float32
    )
    base_path, high_path, output = (
        str(tmp_path / name) for name in ("base.h5", "high.h5", "extended.h5")
    )
    with h5py.File(base_path, "w") as target:
        target["features"] = base
        target["features"].attrs[
            "feature_names"
        ] = "mJ1,dmJ,tau21_J1,tau21_J2,dRJJ,mJJ,label"
    pd.DataFrame(
        {
            "tau2j1": [2.0, 0.0],
            "tau3j1": [1.0, 0.0],
            "tau2j2": [4.0, 2.0],
            "tau3j2": [1.0, 0.5],
        }
    ).to_hdf(high_path, key="df")
    build_extended_features(base_path, high_path, output)
    with h5py.File(output) as source:
        values = source["features"][:]
        assert source["features"].attrs["feature_schema_version"] == "2.0.0"
    np.testing.assert_array_equal(values[:, :5], base[:, :5])
    np.testing.assert_array_equal(values[:, 7:], base[:, 5:])
    np.testing.assert_array_equal(values[:, 5], [0.5, 0.0])
    np.testing.assert_array_equal(values[:, 6], [0.25, 0.25])
