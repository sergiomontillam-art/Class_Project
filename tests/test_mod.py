import pandas as pd
import numpy as np
import mod
import requests
from PIL import Image as PILImage


def make_test_csv(tmp_path, n_rows=10):
    # Create a DataFrame with all required columns and save as CSV
    data = {
        "dr7objid": np.arange(n_rows),
        "ra": np.linspace(0, 360, n_rows),
        "dec": np.linspace(-90, 90, n_rows),
        "p_el_debiased": np.random.uniform(0.8, 1.0, n_rows),
        "p_cs_debiased": np.random.uniform(0.8, 1.0, n_rows),
        "spiral": np.random.randint(0, 2, n_rows),
        "elliptical": np.random.randint(0, 2, n_rows),
        "petroR50_r": np.random.uniform(1, 10, n_rows),
        "petroR90_r": np.random.uniform(10, 20, n_rows),
    }
    for band in "ugriz":
        data[f"modelMag_{band}"] = np.random.uniform(15, 25, n_rows)
        data[f"modelMagErr_{band}"] = np.random.uniform(0.01, 0.04, n_rows)
        data[f"extinction_{band}"] = np.random.uniform(0, 1, n_rows)
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def test_filtering(tmp_path):
    csv_path, df = make_test_csv(tmp_path, n_rows=20)
    # Set some rows to fail the error/probability filter
    df.loc[0, "modelMagErr_u"] = 1.0
    df.loc[1, "p_el_debiased"] = 0.5
    df.to_csv(csv_path, index=False)
    filtered = mod.filtering(str(csv_path))
    # Should not include row 0 or 1
    assert 0 not in filtered["dr7objid"].values
    assert 1 not in filtered["dr7objid"].values
    # Should only have the main columns
    expected_cols = (
        [
            "dr7objid",
            "ra",
            "dec",
            "p_el_debiased",
            "p_cs_debiased",
            "spiral",
            "elliptical",
            "petroR50_r",
            "petroR90_r",
        ]
        + [f"modelMag_{b}" for b in "ugriz"]
        + [f"extinction_{b}" for b in "ugriz"]
    )
    assert list(filtered.columns) == expected_cols


def test_sampling():
    df = pd.DataFrame({"a": range(100)})
    sample = mod.sampling(df, 10, random_state=123)
    assert len(sample) == 10
    # Should be reproducible
    sample2 = mod.sampling(df, 10, random_state=123)
    pd.testing.assert_frame_equal(sample, sample2)
    # Index should be reset
    assert list(sample.index) == list(range(10))


def test_download_cutout_creates_folder_and_file(tmp_path, monkeypatch):
    # Patch IMG_BASE_OUT_DIR to a temp directory
    monkeypatch.setattr(mod, "IMG_BASE_OUT_DIR", tmp_path)

    # Patch requests.get to return a fake response
    class FakeResponse:
        status_code = 200
        content = b"fake image"

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: FakeResponse())
    index = 0
    ra, dec = 10.0, 20.0
    result = mod.download_cutout(index, ra, dec)
    assert "Downloaded" in result
    folder = tmp_path / "batch_00"
    file = folder / "galaxy_0.jpg"
    assert file.exists()
    # Test skipping if file exists
    result2 = mod.download_cutout(index, ra, dec)
    assert "Skipped" in result2


def test_download_cutout_handles_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMG_BASE_OUT_DIR", tmp_path)

    class FakeResponse:
        status_code = 404
        content = b""

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: FakeResponse())
    result = mod.download_cutout(1, 10.0, 20.0)
    assert "Failed" in result


def test_download_cutout_handles_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMG_BASE_OUT_DIR", tmp_path)

    def raise_exc(*a, **k):
        raise requests.RequestException("fail")

    monkeypatch.setattr(mod.requests, "get", raise_exc)
    result = mod.download_cutout(2, 10.0, 20.0)
    assert "Error" in result


def test_parallel_download_prints(monkeypatch):
    # Prepare a small DataFrame
    df = pd.DataFrame({"ra": [1.0, 2.0], "dec": [3.0, 4.0]})
    # Patch download_cutout to return a known string
    monkeypatch.setattr(mod, "download_cutout", lambda i, ra, dec: f"Downloaded {i}")
    # Patch print to capture output
    printed = []
    monkeypatch.setattr("builtins.print", lambda x: printed.append(x))
    # Patch time.sleep to avoid delay
    monkeypatch.setattr(mod.time, "sleep", lambda x: None)
    mod.parallel_download(df, max_workers=2)
    assert any("Downloaded 0" in s or "Downloaded 1" in s for s in printed)


def test_print_image_found(tmp_path, monkeypatch):
    # Patch IMG_BASE_OUT_DIR
    monkeypatch.setattr(mod, "IMG_BASE_OUT_DIR", tmp_path)
    folder = tmp_path / "batch_00"
    folder.mkdir(parents=True)
    img_path = folder / "galaxy_0.jpg"
    # Create a dummy image file
    img = PILImage.new("RGB", (10, 10))
    img.save(img_path)
    # Patch display to check it's called
    called = {}

    def fake_display(img):
        called["displayed"] = True

    monkeypatch.setattr(mod, "display", fake_display)
    mod.print_image(0)
    assert called.get("displayed", False)


def test_print_image_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "IMG_BASE_OUT_DIR", tmp_path)
    # Patch print to capture output
    printed = []
    monkeypatch.setattr("builtins.print", lambda x: printed.append(x))
    mod.print_image(999)
    assert any("not found" in s for s in printed)
