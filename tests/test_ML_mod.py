import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from PIL import Image
import builtins
import ML_mod

@pytest.fixture
def temp_image_dir():
    temp_dir = tempfile.mkdtemp()
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (8, 8), dtype=np.uint8))
        img.save(os.path.join(temp_dir, f"img_{i}.png"))
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def dummy_prePCA_csv(tmp_path):
    df = pd.DataFrame({'objid': ['a', 'b'], 'pixel1': [1, 2], 'pixel2': [3, 4]})
    csv_path = tmp_path / "prePCA_data_test.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def dummy_postPCA_csv(tmp_path):
    df = pd.DataFrame({'objid': ['a', 'b'], 'PC_1': [0.1, 0.2], 'PC_2': [0.3, 0.4]})
    csv_path = tmp_path / "postPCA_data_test.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def dummy_zoo_csv(tmp_path):
    df = pd.DataFrame({'objid': ['a', 'b', 'c'], 'spiral': [1, 0, 1], 'elliptical': [0, 1, 0]})
    csv_path = tmp_path / "ZooSpecPhotoDR19_torradeflot.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def dummy_merged_csv(tmp_path):
    # Numeric-only features + labels
    df = pd.DataFrame({
        'PC_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'PC_2': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'spiral': [1, 0, 1, 0, 1, 0],      # target
        'elliptical': [0, 1, 0, 1, 0, 1]   # additional label
    })
    csv_path = tmp_path / "merged_data_test.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)




def test_read_data(monkeypatch, temp_image_dir):
    monkeypatch.setattr(builtins, 'print', lambda *a, **k: None)
    monkeypatch.setattr(ML_mod, 'Image', Image)
    monkeypatch.setattr(ML_mod.os, 'walk', lambda d: [(temp_image_dir, [], os.listdir(temp_image_dir))])
    monkeypatch.setattr(ML_mod, 'pd', pd)
    monkeypatch.setattr(ML_mod, 'np', np)
    df = ML_mod.read_data()
    assert isinstance(df, pd.DataFrame)
    assert 'objid' in df.columns


def test_PCA_analysis(monkeypatch, dummy_prePCA_csv):
    real_read_csv = pd.read_csv
    monkeypatch.setattr(ML_mod.pd, 'read_csv', lambda *a, **k: real_read_csv(dummy_prePCA_csv))
    ML_mod.PCA_analysis()
    assert os.path.exists('postPCA_data.csv')


def test_concatenate_data(monkeypatch, dummy_postPCA_csv, dummy_zoo_csv):
    real_read_csv = pd.read_csv

    def read_csv_side_effect(path, *args, **kwargs):
        if "postPCA_data.csv" in path:
            return real_read_csv(dummy_postPCA_csv)
        elif "ZooSpecPhotoDR19_torradeflot.csv" in path:
            return real_read_csv(dummy_zoo_csv)
        else:
            return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(ML_mod.pd, 'read_csv', read_csv_side_effect)
    merged_df = ML_mod.concatenate_data()
    assert isinstance(merged_df, pd.DataFrame)
    assert 'objid' in merged_df.columns


def test_custom_LinearSVC(monkeypatch, dummy_merged_csv):
    builtins.print = lambda *a, **k: None  # silence print

    # Patch pd.read_csv to load our dummy CSV
    real_read_csv = pd.read_csv
    monkeypatch.setattr(ML_mod.pd, 'read_csv', lambda *a, **k: real_read_csv(dummy_merged_csv))

    # Patch sns.heatmap to avoid plotting
    monkeypatch.setattr(ML_mod.sns, 'heatmap', lambda *a, **k: None)

    # Optionally patch train_test_split to avoid stratification issues in small test
    def fake_train_test_split(X, y, **kwargs):
        return X, X, y, y
    monkeypatch.setattr(ML_mod, 'train_test_split', fake_train_test_split)

    # Run the function — should no longer raise errors
    ML_mod.custom_LinearSVC()


def test_pca_plot(monkeypatch, dummy_prePCA_csv):
    real_read_csv = pd.read_csv
    monkeypatch.setattr(ML_mod.pd, 'read_csv', lambda *a, **k: real_read_csv(dummy_prePCA_csv))
    monkeypatch.setattr(ML_mod.plt, 'show', lambda: None)
    ML_mod.pca_plot()
