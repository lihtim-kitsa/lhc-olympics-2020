import h5py
import numpy as np
import pytest

from src.data.make_dataset import create_3prong_split, create_splits


@pytest.fixture
def features(tmp_path):
    rows = np.random.default_rng(42).uniform(1, 10, (100, 7))
    rows[:, 6] = np.repeat([0, 1], 50)
    rows[0, 0] = np.nan
    rows[50, 5] = 0
    path = tmp_path / 'features.h5'
    with h5py.File(path, 'w') as output:
        output['features'] = rows
    return path, rows


def test_splits_are_disjoint_exhaustive_and_deterministic(features, tmp_path):
    path, rows = features
    for directory in (tmp_path / 'first', tmp_path / 'second'):
        create_splits(path, directory, seed=42)
    for label, prefix in enumerate(('background', 'signal2')):
        parts = [np.load(tmp_path / 'first' / f'{prefix}_{s}.npy')
                 for s in ('train', 'val', 'test')]
        assert list(map(len, parts)) == [29, 9, 11]
        combined = np.concatenate(parts)
        assert len(np.unique(combined)) == len(combined)
        expected = np.flatnonzero((rows[:, 6] == label)
                                  & np.isfinite(rows[:, :6]).all(axis=1)
                                  & (rows[:, 5] > 0))
        np.testing.assert_array_equal(np.sort(combined), expected)
        for split, part in zip(('train', 'val', 'test'), parts):
            np.testing.assert_array_equal(
                part, np.load(tmp_path / 'second' / f'{prefix}_{split}.npy'))


def test_held_out_topology_excludes_invalid_rows(features, tmp_path):
    path, _ = features
    create_3prong_split(path, tmp_path)
    indices = np.load(tmp_path / 'signal3_test.npy')
    assert len(indices) == 98
    assert not np.isin([0, 50], indices).any()
