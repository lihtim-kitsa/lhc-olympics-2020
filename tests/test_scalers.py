import h5py
import numpy as np
import pytest

from scripts.train import load_data
from src.data.make_dataset import create_splits


def test_scaler_and_labels_do_not_leak_validation_or_test(tmp_path):
    rows = np.random.default_rng(42).uniform(1, 10, (200, 7))
    rows[:, 6] = np.repeat([0, 1], 100)
    path = tmp_path / 'features.h5'
    with h5py.File(path, 'w') as output:
        output['features'] = rows
    splits = tmp_path / 'splits'
    create_splits(path, splits)
    config = {'model': {'name': 'M4_DeepSAD'},
              'data': {'f_train': 10, 'k_labels': 10},
              'training': {'seed': 42}}
    first = load_data(splits, path, config)
    train = np.load(splits / 'background_train.npy')
    np.testing.assert_allclose(first[6].mean_, rows[train, :5].mean(axis=0))
    assert not np.intersect1d(first[7], first[8]).size
    assert len(first[8]) == 10
    assert len(first[7]) == 66
    assert not first[1].any()
    assert first[3].all()
    held_out = np.concatenate([np.load(splits / f'{prefix}_{split}.npy')
                               for prefix in ('background', 'signal2')
                               for split in ('val', 'test')])
    assert not np.intersect1d(np.r_[first[7], first[8]], held_out).size
    rows[held_out, :5] += 10000
    with h5py.File(path, 'w') as output:
        output['features'] = rows
    second = load_data(splits, path, config)
    np.testing.assert_array_equal(first[6].mean_, second[6].mean_)
    np.testing.assert_array_equal(first[0], second[0])
    config['data']['k_labels'] = 1000
    with pytest.raises(ValueError, match='distinct signal train events'):
        load_data(splits, path, config)
