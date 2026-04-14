"""
Tests for the new BCI device adapters:
  - EmotivAdapter          (emotiv.py)
  - MuseAdapter            (muse.py)
  - NeurosityAdapter       (neurosity.py)
  - NeuropixelsAdapter     (neuropixels.py)
  - BrainProductsAdapter   (brainproducts.py)
  - GTecAdapter            (gtec.py)
  - BlackrockAdapter.from_nev_file  (stub implementation)
  - IntanAdapter.from_rhd_file      (stub implementation)
"""

from __future__ import annotations

import io
import os
import struct
import tempfile
import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pytest


# -- helpers ------------------------------------------------------------------

def rand_data(n_ch: int, n_samp: int = 1000, dtype=np.float64) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_ch, n_samp)).astype(dtype)


# =============================================================================
# EmotivAdapter
# =============================================================================

class TestEmotivAdapter:

    def test_epoc_14ch_init(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_14ch")
        assert a.sampling_rate == 128
        assert a.mapping["channels"] == 14

    def test_epoc_plus_init(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_plus")
        assert a.sampling_rate == 256

    def test_insight_init(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="insight_5ch")
        assert a.mapping["channels"] == 5

    def test_flex_32ch_init(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="flex_32ch")
        assert a.mapping["channels"] == 32
        assert a.sampling_rate == 256

    def test_invalid_device(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        with pytest.raises(ValueError):
            EmotivAdapter(device="emotiv_nonexistent")

    def test_convert_shape(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_14ch")
        data = rand_data(14, 500)
        out = a.convert(data)
        assert out.shape == (14, 500)

    def test_to_microvolts(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_14ch")
        raw = np.ones((14, 100), dtype=np.int16) * 1000
        uv = a.to_microvolts(raw)
        assert uv.dtype == np.float64
        np.testing.assert_allclose(uv[0, 0], 1000 * 0.51)

    def test_resample_to(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_plus")
        data = rand_data(14, 256)
        out = a.resample_to(data, target_rate=128)
        assert out.shape[0] == 14
        assert out.shape[1] == 128

    def test_get_channel_groups(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_14ch")
        groups = a.get_channel_groups()
        assert "frontal" in groups
        assert "occipital" in groups

    def test_get_channel_names(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_14ch")
        names = a.get_channel_names()
        assert names[0] == "AF3"
        assert len(names) == 14

    def test_from_flex_labels(self):
        from bci_compression.adapters.emotiv import EmotivAdapter
        labels = ["Fp1", "Fp2", "C3", "C4", "Pz"]
        a = EmotivAdapter.from_flex_labels(labels, sampling_rate=256)
        assert a.mapping["channels"] == 5
        assert a.get_channel_names() == labels

    def test_save_load_mapping(self, tmp_path):
        from bci_compression.adapters.emotiv import EmotivAdapter
        a = EmotivAdapter(device="epoc_14ch")
        fp = str(tmp_path / "emotiv.yaml")
        a.save_mapping(fp)
        b = EmotivAdapter.from_file(fp)
        assert b.mapping["device"] == a.mapping["device"]

    def test_convert_emotiv_to_standard(self):
        from bci_compression.adapters.emotiv import convert_emotiv_to_standard
        raw = rand_data(14, 200).astype(np.int16)
        out = convert_emotiv_to_standard(raw, device="epoc_14ch", scale_to_uv=True)
        assert out.shape[0] == 14
        assert out.dtype == np.float64


# =============================================================================
# MuseAdapter
# =============================================================================

class TestMuseAdapter:

    def test_muse_2_init(self):
        from bci_compression.adapters.muse import MuseAdapter
        a = MuseAdapter(device="muse_2")
        assert a.sampling_rate == 256
        assert a.mapping["channels"] == 4

    def test_muse_1_init(self):
        from bci_compression.adapters.muse import MuseAdapter
        a = MuseAdapter(device="muse_1")
        assert a.sampling_rate == 220

    def test_muse_s_init(self):
        from bci_compression.adapters.muse import MuseAdapter
        a = MuseAdapter(device="muse_s")
        assert a.sampling_rate == 256

    def test_invalid_device(self):
        from bci_compression.adapters.muse import MuseAdapter
        with pytest.raises(ValueError):
            MuseAdapter(device="muse_99")

    def test_convert_shape(self):
        from bci_compression.adapters.muse import MuseAdapter
        a = MuseAdapter(device="muse_2")
        data = rand_data(4, 256)
        out = a.convert(data)
        assert out.shape == (4, 256)

    def test_channel_names(self):
        from bci_compression.adapters.muse import MuseAdapter
        a = MuseAdapter(device="muse_2")
        assert a.get_channel_names() == ["TP9", "AF7", "AF8", "TP10"]

    def test_channel_groups(self):
        from bci_compression.adapters.muse import MuseAdapter
        a = MuseAdapter(device="muse_2")
        groups = a.get_channel_groups()
        assert "temporal" in groups
        assert "frontal" in groups

    def test_band_power(self):
        from bci_compression.adapters.muse import MuseAdapter
        a = MuseAdapter(device="muse_2")
        t = np.linspace(0, 2, 512)
        sig = np.sin(2 * np.pi * 10 * t)
        data = np.tile(sig, (4, 1))
        alpha_power = a.compute_band_power(data, band=(8, 13))
        assert alpha_power.shape == (4,)
        assert np.all(alpha_power > 0)

    def test_from_csv(self, tmp_path):
        from bci_compression.adapters.muse import MuseAdapter
        csv_path = tmp_path / "session.csv"
        csv_path.write_text(
            "TimeStamp,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10\n"
            "1000.0,10.5,20.1,-5.3,8.2\n"
            "1000.004,11.0,19.8,-4.9,8.5\n"
            "1000.008,10.2,20.5,-5.1,8.0\n"
        )
        adapter, ts, eeg = MuseAdapter.from_csv(str(csv_path), device="muse_2")
        assert eeg.shape == (4, 3)
        assert len(ts) == 3
        np.testing.assert_allclose(eeg[0, 0], 10.5)

    def test_convert_muse_to_standard(self):
        from bci_compression.adapters.muse import convert_muse_to_standard
        data = rand_data(4, 256)
        out = convert_muse_to_standard(data, device="muse_2")
        assert out.shape[0] == 4


# =============================================================================
# NeurosityAdapter
# =============================================================================

class TestNeurosityAdapter:

    def test_crown_init(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="crown")
        assert a.sampling_rate == 256
        assert a.mapping["channels"] == 8

    def test_shift_init(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="shift")
        assert a.mapping["channels"] == 4

    def test_invalid_device(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        with pytest.raises(ValueError):
            NeurosityAdapter(device="crown_pro")

    def test_crown_channel_names(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="crown")
        names = a.get_channel_names()
        assert names[0] == "CP3"
        assert len(names) == 8

    def test_convert(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="crown")
        data = rand_data(8, 256)
        out = a.convert(data)
        assert out.shape == data.shape

    def test_band_powers(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="crown")
        t = np.linspace(0, 2, 512)
        alpha_wave = np.sin(2 * np.pi * 10.0 * t)
        data = np.tile(alpha_wave, (8, 1))
        bp = a.band_powers(data)
        assert "alpha" in bp
        assert "beta" in bp
        assert bp["alpha"].shape == (8,)
        assert np.mean(bp["alpha"]) > np.mean(bp["beta"])

    def test_focus_index(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="crown")
        data = rand_data(8, 512)
        fi = a.focus_index(data)
        assert isinstance(fi, float)
        assert fi >= 0

    def test_calm_index(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="crown")
        data = rand_data(8, 512)
        ci = a.calm_index(data)
        assert isinstance(ci, float)
        assert ci >= 0

    def test_channel_groups(self):
        from bci_compression.adapters.neurosity import NeurosityAdapter
        a = NeurosityAdapter(device="crown")
        groups = a.get_channel_groups()
        assert "left_hemisphere" in groups
        assert "right_hemisphere" in groups


# =============================================================================
# NeuropixelsAdapter
# =============================================================================

class TestNeuropixelsAdapter:

    def test_np10_init(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        a = NeuropixelsAdapter(probe="np1.0")
        assert a.sampling_rate == 30_000
        assert a.mapping["channels"] == 384

    def test_np20_init(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        a = NeuropixelsAdapter(probe="np2.0")
        assert a.sampling_rate == 30_000

    def test_ultra_init(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        a = NeuropixelsAdapter(probe="ultra")
        assert a.mapping["channels"] == 384

    def test_invalid_probe(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        with pytest.raises(ValueError):
            NeuropixelsAdapter(probe="np99")

    def test_to_microvolts(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        a = NeuropixelsAdapter(probe="np1.0", ap_gain=500)
        raw = np.ones((384, 100), dtype=np.int16)
        uv = a.to_microvolts(raw, band="ap")
        assert uv.dtype == np.float64
        expected = a._uv_per_lsb(500)
        np.testing.assert_allclose(uv[0, 0], expected)

    def test_site_coordinates_shape(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        a = NeuropixelsAdapter(probe="np1.0")
        coords = a.site_coordinates()
        assert coords.shape == (384, 2)
        assert np.all(coords[:, 1] >= 0)

    def test_sites_in_depth_range(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        a = NeuropixelsAdapter(probe="np1.0")
        indices = a.sites_in_depth_range(0, 200)
        assert len(indices) > 0
        assert len(indices) < 384

    def test_get_channel_groups(self):
        from bci_compression.adapters.neuropixels import NeuropixelsAdapter
        a = NeuropixelsAdapter(probe="np1.0")
        groups = a.get_channel_groups()
        assert "bank_0" in groups
        assert len(groups["bank_0"]) == 96

    def test_convert_neuropixels_to_standard(self):
        from bci_compression.adapters.neuropixels import convert_neuropixels_to_standard
        raw = np.ones((384, 100), dtype=np.int16)
        out = convert_neuropixels_to_standard(raw, probe="np1.0", ap_gain=500)
        assert out.dtype == np.float64
        assert out.shape == (384, 100)

    def test_np10_site_coords_columns(self):
        from bci_compression.adapters.neuropixels import NP10_SITE_COORDS
        assert NP10_SITE_COORDS[0, 0] == pytest.approx(11.0)
        assert NP10_SITE_COORDS[1, 0] == pytest.approx(43.0)


# =============================================================================
# BrainProductsAdapter
# =============================================================================

def _write_brainvision_files(directory, n_ch=4, n_samples=100, fs=500.0):
    eeg_name = "test_rec.eeg"
    interval_us = int(1_000_000 / fs)
    ch_lines = "\n".join(f"Ch{i+1}=EEG{i+1},,0.1,uV" for i in range(n_ch))
    vhdr_content = (
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "; Generated by test suite\n\n"
        "[Common Infos]\n"
        f"DataFile={eeg_name}\n"
        "DataFormat=BINARY\n"
        "DataOrientation=MULTIPLEXED\n"
        f"NumberOfChannels={n_ch}\n"
        f"SamplingInterval={interval_us}\n\n"
        "[Binary Infos]\n"
        "BinaryFormat=INT_16\n\n"
        "[Channel Infos]\n"
        + ch_lines + "\n"
    )
    vhdr_path = directory / "test_rec.vhdr"
    vhdr_path.write_text(vhdr_content, encoding="utf-8")
    rng = np.random.default_rng(7)
    raw = rng.integers(-1000, 1000, (n_samples, n_ch), dtype=np.int16)
    (directory / eeg_name).write_bytes(raw.tobytes())
    return vhdr_path, raw


class TestBrainProductsAdapter:

    def test_preset_init(self):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        a = BrainProductsAdapter(device="brainamp_64ch")
        assert a.sampling_rate == 500.0
        assert a.mapping["channels"] == 64

    def test_actichamp_init(self):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        a = BrainProductsAdapter(device="actichamp_128ch")
        assert a.mapping["channels"] == 128
        assert a.sampling_rate == 1000.0

    def test_invalid_device(self):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        with pytest.raises(ValueError):
            BrainProductsAdapter(device="brainamp_999ch")

    def test_channel_names_32(self):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        a = BrainProductsAdapter(device="brainamp_32ch")
        names = a.get_channel_names()
        assert names[0] == "Fp1"
        assert "Cz" in names

    def test_convert_scaling(self):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        a = BrainProductsAdapter(device="brainamp_32ch")
        a._resolutions = np.full(32, 0.5)
        raw = np.ones((32, 10), dtype=np.int16) * 2
        out = a.convert(raw, apply_scaling=True)
        np.testing.assert_allclose(out, 1.0)

    def test_resample_to(self):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        a = BrainProductsAdapter(device="brainamp_32ch")
        data = rand_data(32, 500)
        out = a.resample_to(data, target_rate=250)
        assert out.shape[0] == 32
        assert abs(out.shape[1] - 250) <= 2

    def test_from_vhdr_file(self, tmp_path):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        vhdr_path, raw_int16 = _write_brainvision_files(tmp_path, n_ch=4, n_samples=100)
        adapter, data = BrainProductsAdapter.from_vhdr_file(str(vhdr_path))
        assert data.shape == (4, 100)
        assert data.dtype == np.float64
        assert adapter.sampling_rate == 500.0

    def test_from_vhdr_channel_names(self, tmp_path):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        vhdr_path, _ = _write_brainvision_files(tmp_path, n_ch=4)
        adapter, _ = BrainProductsAdapter.from_vhdr_file(str(vhdr_path))
        names = adapter.get_channel_names()
        assert names == ["EEG1", "EEG2", "EEG3", "EEG4"]

    def test_from_vhdr_scaling(self, tmp_path):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        vhdr_path, raw_int16 = _write_brainvision_files(
            tmp_path, n_ch=4, n_samples=50, fs=1000.0
        )
        adapter, data = BrainProductsAdapter.from_vhdr_file(str(vhdr_path))
        expected_ch0 = raw_int16[:, 0].astype(np.float64) * 0.1
        np.testing.assert_allclose(data[0], expected_ch0)

    def test_from_vhdr_missing_eeg(self, tmp_path):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        vhdr_path, _ = _write_brainvision_files(tmp_path, n_ch=4)
        (tmp_path / "test_rec.eeg").unlink()
        with pytest.raises(FileNotFoundError):
            BrainProductsAdapter.from_vhdr_file(str(vhdr_path))

    def test_load_brainvision_data(self, tmp_path):
        from bci_compression.adapters.brainproducts import load_brainvision_data
        vhdr_path, _ = _write_brainvision_files(tmp_path, n_ch=4, n_samples=500)
        adapter, data = load_brainvision_data(str(vhdr_path), target_rate=250.0)
        assert data.shape[0] == 4
        assert abs(data.shape[1] - 250) <= 2

    def test_save_load_mapping(self, tmp_path):
        from bci_compression.adapters.brainproducts import BrainProductsAdapter
        a = BrainProductsAdapter(device="brainamp_32ch")
        fp = str(tmp_path / "bp.json")
        a.save_mapping(fp)
        b = BrainProductsAdapter.from_file(fp)
        assert b.mapping["device"] == a.mapping["device"]


# =============================================================================
# GTecAdapter
# =============================================================================

class TestGTecAdapter:

    def test_hiamp_64ch_init(self):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="hiamp_64ch")
        assert a.mapping["channels"] == 64

    def test_usbamp_16ch_init(self):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="usbamp_16ch")
        assert a.mapping["channels"] == 16

    def test_nautilus_8ch_init(self):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="nautilus_8ch")
        assert a.mapping["channels"] == 8
        assert a.sampling_rate == 250.0

    def test_invalid_device(self):
        from bci_compression.adapters.gtec import GTecAdapter
        with pytest.raises(ValueError):
            GTecAdapter(device="gtec_xxx")

    def test_convert_dtype(self):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="hiamp_64ch")
        raw = rand_data(64, 100).astype(np.float32)
        out = a.convert(raw)
        assert out.dtype == np.float64

    def test_resample(self):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="hiamp_64ch")
        data = rand_data(64, 256)
        out = a.resample_to(data, target_rate=128)
        assert out.shape[0] == 64
        assert abs(out.shape[1] - 128) <= 2

    def test_channel_groups_populated(self):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="hiamp_64ch")
        groups = a.get_channel_groups()
        assert "frontal" in groups
        assert "central" in groups
        assert len(groups["frontal"]) > 0

    def test_compute_erd(self):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="hiamp_64ch")
        n_samp = 512
        data = rand_data(64, n_samp)
        erd = a.compute_erd(data, band=(8.0, 13.0), baseline_sec=1.0)
        assert erd.shape == (64,)
        assert np.all(np.isfinite(erd))

    def test_convert_gtec_to_standard(self):
        from bci_compression.adapters.gtec import convert_gtec_to_standard
        data = rand_data(64, 256)
        out = convert_gtec_to_standard(data, device="hiamp_64ch")
        assert out.dtype == np.float64
        assert out.shape == data.shape

    def test_save_load_mapping(self, tmp_path):
        from bci_compression.adapters.gtec import GTecAdapter
        a = GTecAdapter(device="usbamp_32ch")
        fp = str(tmp_path / "gtec.yaml")
        a.save_mapping(fp)
        b = GTecAdapter.from_file(fp)
        assert b.mapping["channels"] == 32


# =============================================================================
# Blackrock.from_nev_file  (pure-Python NEV parser)
# =============================================================================

def _write_minimal_nev(path: str, n_neuevwav: int = 96) -> None:
    ext_bytes = n_neuevwav * 32
    total_header = 336 + ext_bytes
    sample_resolution = 30_000
    timestamp_resolution = 30_000

    with open(path, "wb") as f:
        f.write(b"NEURALEV")
        f.write(struct.pack("<BB", 2, 3))
        f.write(struct.pack("<H", 0))
        f.write(struct.pack("<I", total_header))
        f.write(struct.pack("<I", 28))
        f.write(struct.pack("<I", timestamp_resolution))
        f.write(struct.pack("<I", sample_resolution))
        f.write(b"\x00" * 16)
        f.write(b"TestApp\x00" + b"\x00" * 24)
        f.write(b"Test file\x00" + b"\x00" * 246)
        f.write(struct.pack("<I", 0))
        assert f.tell() == 336, f"basic header offset wrong: {f.tell()}"
        for i in range(n_neuevwav):
            electrode_id = i + 1
            f.write(b"NEUEVWAV")
            f.write(struct.pack("<H", electrode_id))
            f.write(b"A")
            f.write(struct.pack("<B", i % 96))
            f.write(struct.pack("<H", 1000))
            f.write(struct.pack("<H", 0))
            f.write(struct.pack("<h", -5000))
            f.write(struct.pack("<h", 0))
            f.write(struct.pack("<B", 3))
            f.write(struct.pack("<B", 2))
            f.write(b"\x00" * 10)


class TestBlackrockFromNEV:

    def test_reads_valid_nev(self, tmp_path):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        nev_path = str(tmp_path / "test.nev")
        _write_minimal_nev(nev_path, n_neuevwav=96)
        adapter = BlackrockAdapter.from_nev_file(nev_path)
        assert adapter.mapping["channels"] == 96
        assert adapter.sampling_rate == 30_000

    def test_channel_names_from_neuevwav(self, tmp_path):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        nev_path = str(tmp_path / "test96.nev")
        _write_minimal_nev(nev_path, n_neuevwav=96)
        adapter = BlackrockAdapter.from_nev_file(nev_path)
        m = adapter.mapping["mapping"]
        assert m["ch_0"] == "electrode_001"
        assert m["ch_95"] == "electrode_096"

    def test_file_version_in_mapping(self, tmp_path):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        nev_path = str(tmp_path / "ver.nev")
        _write_minimal_nev(nev_path)
        adapter = BlackrockAdapter.from_nev_file(nev_path)
        assert adapter.mapping["file_version"] == "2.3"

    def test_invalid_magic_raises(self, tmp_path):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        bad_path = str(tmp_path / "bad.nev")
        Path(bad_path).write_bytes(b"BADMAGIC" + b"\x00" * 400)
        with pytest.raises(ValueError, match="Not a Blackrock NEV"):
            BlackrockAdapter.from_nev_file(bad_path)

    def test_missing_file_raises(self):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        with pytest.raises(FileNotFoundError):
            BlackrockAdapter.from_nev_file("/nonexistent/path/recording.nev")

    def test_fallback_to_96ch_when_no_neuevwav(self, tmp_path):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        nev_path = str(tmp_path / "empty_ext.nev")
        _write_minimal_nev(nev_path, n_neuevwav=0)
        adapter = BlackrockAdapter.from_nev_file(nev_path)
        assert adapter.mapping["channels"] == 96

    def test_channel_groups_present(self, tmp_path):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        nev_path = str(tmp_path / "grp.nev")
        _write_minimal_nev(nev_path, n_neuevwav=96)
        adapter = BlackrockAdapter.from_nev_file(nev_path)
        groups = adapter.mapping["channel_groups"]
        assert "all" in groups
        assert len(groups["all"]) == 96

    def test_partial_neuevwav_channels(self, tmp_path):
        from bci_compression.adapters.blackrock import BlackrockAdapter
        nev_path = str(tmp_path / "small.nev")
        _write_minimal_nev(nev_path, n_neuevwav=10)
        adapter = BlackrockAdapter.from_nev_file(nev_path)
        assert adapter.mapping["channels"] == 10


# =============================================================================
# IntanAdapter.from_rhd_file  (pure-Python RHD parser)
# =============================================================================

def _write_qstring(fid, s: str) -> None:
    if s == "":
        fid.write(struct.pack("<I", 0xFFFFFFFF))
        return
    encoded = s.encode("utf-16-le")
    fid.write(struct.pack("<I", len(encoded)))
    fid.write(encoded)


def _write_minimal_rhd(
    path: str,
    n_amplifier_channels: int = 32,
    sample_rate: float = 20_000.0,
    major: int = 2,
    minor: int = 0,
) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 0xC6912702))
        f.write(struct.pack("<hh", major, minor))
        f.write(struct.pack("<f", sample_rate))
        f.write(struct.pack("<hffffff", 1, 1.0, 300.0, 6000.0, 1.0, 300.0, 6000.0))
        f.write(struct.pack("<h", 0))
        f.write(struct.pack("<ff", 1000.0, 1000.0))
        _write_qstring(f, "Test note 1")
        _write_qstring(f, "")
        _write_qstring(f, "")
        if (major == 1 and minor >= 1) or major > 1:
            f.write(struct.pack("<h", 0))
        if (major == 1 and minor >= 3) or major > 1:
            f.write(struct.pack("<h", 0))
        if major > 1:
            _write_qstring(f, "REF")
        f.write(struct.pack("<h", 1))
        _write_qstring(f, "Port A")
        _write_qstring(f, "A")
        f.write(struct.pack("<hhh", 1, n_amplifier_channels, 0))
        for i in range(n_amplifier_channels):
            _write_qstring(f, f"A-{i:03d}")
            _write_qstring(f, f"amp_ch_{i:02d}")
            f.write(struct.pack("<hhhhhh", i, i, 0, 1, i % 32, 0))
            f.write(struct.pack("<hhhh", 0, -5000, 0, 0))
            f.write(struct.pack("<ff", 0.0, 0.0))


class TestIntanFromRHD:

    def test_reads_valid_rhd(self, tmp_path):
        from bci_compression.adapters.intan import IntanAdapter
        rhd_path = str(tmp_path / "rec.rhd")
        _write_minimal_rhd(rhd_path, n_amplifier_channels=32, sample_rate=20000.0)
        adapter = IntanAdapter.from_rhd_file(rhd_path)
        assert adapter.sampling_rate == 20_000
        assert adapter.mapping["channels"] == 32

    def test_channel_names_from_file(self, tmp_path):
        from bci_compression.adapters.intan import IntanAdapter
        rhd_path = str(tmp_path / "names.rhd")
        _write_minimal_rhd(rhd_path, n_amplifier_channels=4)
        adapter = IntanAdapter.from_rhd_file(rhd_path)
        m = adapter.mapping["mapping"]
        assert m["ch_0"] == "amp_ch_00"
        assert m["ch_3"] == "amp_ch_03"

    def test_version_in_mapping(self, tmp_path):
        from bci_compression.adapters.intan import IntanAdapter
        rhd_path = str(tmp_path / "ver.rhd")
        _write_minimal_rhd(rhd_path, major=2, minor=0)
        adapter = IntanAdapter.from_rhd_file(rhd_path)
        assert adapter.mapping["file_version"] == "2.0"

    def test_invalid_magic_raises(self, tmp_path):
        from bci_compression.adapters.intan import IntanAdapter
        bad = str(tmp_path / "bad.rhd")
        Path(bad).write_bytes(b"\xff\xff\xff\xff" + b"\x00" * 200)
        with pytest.raises(ValueError, match="Not a valid Intan RHD"):
            IntanAdapter.from_rhd_file(bad)

    def test_missing_file_raises(self):
        from bci_compression.adapters.intan import IntanAdapter
        with pytest.raises(FileNotFoundError):
            IntanAdapter.from_rhd_file("/no/such/file.rhd")

    def test_64ch_file(self, tmp_path):
        from bci_compression.adapters.intan import IntanAdapter
        rhd_path = str(tmp_path / "rec64.rhd")
        _write_minimal_rhd(rhd_path, n_amplifier_channels=64, sample_rate=20000.0)
        adapter = IntanAdapter.from_rhd_file(rhd_path)
        assert adapter.mapping["channels"] == 64

    def test_device_string_contains_version(self, tmp_path):
        from bci_compression.adapters.intan import IntanAdapter
        rhd_path = str(tmp_path / "dv.rhd")
        _write_minimal_rhd(rhd_path, major=1, minor=3)
        adapter = IntanAdapter.from_rhd_file(rhd_path)
        assert "1.3" in adapter.mapping["device"]


# =============================================================================
# __init__.py re-exports
# =============================================================================

class TestAdaptersInitReexports:

    def test_emotiv_in_init(self):
        from bci_compression.adapters import EmotivAdapter
        assert EmotivAdapter(device="epoc_14ch").mapping["channels"] == 14

    def test_muse_in_init(self):
        from bci_compression.adapters import MuseAdapter
        assert MuseAdapter(device="muse_2").mapping["channels"] == 4

    def test_neurosity_in_init(self):
        from bci_compression.adapters import NeurosityAdapter
        assert NeurosityAdapter(device="crown").mapping["channels"] == 8

    def test_neuropixels_in_init(self):
        from bci_compression.adapters import NeuropixelsAdapter
        assert NeuropixelsAdapter(probe="np1.0").mapping["channels"] == 384

    def test_brainproducts_in_init(self):
        from bci_compression.adapters import BrainProductsAdapter
        assert BrainProductsAdapter(device="brainamp_64ch").mapping["channels"] == 64

    def test_gtec_in_init(self):
        from bci_compression.adapters import GTecAdapter
        assert GTecAdapter(device="hiamp_64ch").mapping["channels"] == 64

    def test_all_converters_importable(self):
        from bci_compression.adapters import (
            convert_emotiv_to_standard,
            convert_muse_to_standard,
            convert_neurosity_to_standard,
            convert_neuropixels_to_standard,
            convert_gtec_to_standard,
        )
        assert callable(convert_emotiv_to_standard)
        assert callable(convert_muse_to_standard)
        assert callable(convert_neurosity_to_standard)
        assert callable(convert_neuropixels_to_standard)
        assert callable(convert_gtec_to_standard)
