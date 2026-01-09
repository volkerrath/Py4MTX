#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_dataproc.py
================

Pytest test-suite for :mod:`dataproc`.

These tests are self-contained: they generate small synthetic EDI-like files
(classical table style and Phoenix ``>SPECTRA`` style) in a temporary folder
and validate:

- block parsing helpers (incl. Fortran ``D`` exponents and comments),
- classical EDI load/save roundtrip,
- Phoenix SPECTRA parsing and Z/T reconstruction,
- phase tensor computation with/without uncertainty propagation,
- conversion to pandas DataFrame,
- simple EMTF-XML read/write and EDI↔EMTF conversion.

Run
---
From the folder containing ``dataproc.py`` and this test file::

    python -m pytest -q

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2025-12-27 (UTC)
"""

from __future__ import annotations

import math
import inspect
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

import importlib.util
import sys

# --- Robust import of dataproc.py without triggering a potentially broken package __init__.py ---
_HERE = Path(__file__).resolve().parent

# Prefer dataproc.py in the same directory as this test file; fall back to parent dir.
_candidates = [_HERE / "dataproc.py", _HERE.parent / "dataproc.py"]
_dataproc_path = next((c for c in _candidates if c.exists()), None)
if _dataproc_path is None:
    raise FileNotFoundError("Could not locate dataproc.py next to the tests or in the parent directory.")

_spec = importlib.util.spec_from_file_location("dataproc_under_test", str(_dataproc_path))
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not create import spec for {_dataproc_path}")

dp = importlib.util.module_from_spec(_spec)
sys.modules["dataproc_under_test"] = dp
_spec.loader.exec_module(dp)


def _rng(seed: int = 1234) -> np.random.Generator:
    """Create a deterministic NumPy RNG for tests."""
    return np.random.default_rng(seed)


def _make_invertible_real_Z(
    n: int,
    *,
    diag: float = 10.0,
    off: float = 0.25,
    seed: int = 1234,
) -> np.ndarray:
    """Create a complex Z array whose real parts are safely invertible.

    Parameters
    ----------
    n : int
        Number of frequencies/samples.
    diag : float
        Base diagonal level for the real part.
    off : float
        Off-diagonal level for the real part.
    seed : int
        RNG seed.

    Returns
    -------
    Z : ndarray, shape (n, 2, 2), complex128
        Synthetic impedance tensor with invertible real part per frequency.
    """
    rng = _rng(seed)
    Z = np.zeros((n, 2, 2), dtype=np.complex128)

    for k in range(n):
        # Real part: diagonally dominant matrix
        X = np.array([[diag, off], [off * 0.7, diag * 1.1]], dtype=float)
        X += 0.05 * rng.standard_normal((2, 2))

        # Imag part: mild variability
        Y = 0.5 * rng.standard_normal((2, 2))
        Z[k] = X + 1j * Y

    return Z


def _write_text(path: Path, text: str) -> None:
    """Write text to file with latin-1 encoding (EDI-friendly)."""
    path.write_text(text, encoding="latin-1")


def test_extract_block_values_parses_D_exponent_and_comments() -> None:
    """_extract_block_values should parse numbers, ignore // comments, and support D exponents."""
    text = """>HEAD
  DATAID="TEST"
>FREQ
  1.0D+00  2.0D+00  // comment
  3.0E+00
>ZXXR
  1.0 2.0 3.0
"""
    lines = text.splitlines()
    freq = dp._extract_block_values(lines=lines, keyword="FREQ")
    assert freq is not None
    assert np.allclose(freq, [1.0, 2.0, 3.0])


def _make_classical_edi_dict(
    *,
    n: int = 6,
    seed: int = 5,
    err_kind: str = "var",
) -> Dict:
    """Create a minimal but complete EDI dictionary for save/load roundtrip tests."""
    rng = _rng(seed)

    # Intentionally unsorted to test load_edi sorting
    freq = np.array([10.0, 1.0, 3.0, 30.0, 0.3, 0.1], dtype=float)[:n]

    Z = _make_invertible_real_Z(n, seed=seed)
    T = (0.05 * rng.standard_normal((n, 1, 2)) + 1j * 0.05 * rng.standard_normal((n, 1, 2))).astype(
        np.complex128
    )
    rot = np.linspace(0.0, 20.0, n)

    # Provide variances by default (dataproc expects var blocks)
    Z_var = (1e-2 * np.ones_like(Z.real)).astype(float)
    T_var = (1e-3 * np.ones_like(T.real)).astype(float)

    P, P_err = dp.compute_pt(Z=Z, Z_err=Z_var, err_kind="var", nsim=60, random_state=_rng(10))

    return {
        "freq": freq,
        "Z": Z,
        "T": T,
        "Z_err": Z_var if err_kind == "var" else np.sqrt(Z_var),
        "T_err": T_var if err_kind == "var" else np.sqrt(T_var),
        "P": P,
        "P_err": P_err,
        "rot": rot,
        "err_kind": err_kind,
        "station": "TEST01",
        "lat_deg": 10.5,
        "lon_deg": 20.25,
        "elev_m": 123.0,
    }


def test_save_load_roundtrip_tables(tmp_path: Path) -> None:
    """save_edi -> load_edi should preserve main arrays (up to formatting tolerance)."""
    edi_in = _make_classical_edi_dict(n=6, err_kind="var")
    out_edi = tmp_path / "test_tables.edi"

    dp.save_edi(path=out_edi, edi=edi_in, add_pt_blocks=True)
    edi_out = dp.load_edi(path=out_edi, prefer_spectra=False, err_kind="var")

    # Ensure sorting happened
    assert np.all(np.diff(edi_out["freq"]) > 0)

    # Compare after re-sorting input to match output
    order = np.argsort(np.asarray(edi_in["freq"]))
    Zin = np.asarray(edi_in["Z"])[order]
    Tin = np.asarray(edi_in["T"])[order]
    rin = np.asarray(edi_in["rot"])[order]

    assert edi_out["source_kind"] == "tables"
    assert np.allclose(edi_out["freq"], np.asarray(edi_in["freq"])[order])
    assert np.allclose(edi_out["Z"], Zin, rtol=0, atol=5e-6)
    assert edi_out["T"] is not None
    assert np.allclose(edi_out["T"], Tin, rtol=0, atol=5e-6)
    assert edi_out["rot"] is not None
    assert np.allclose(edi_out["rot"], rin, rtol=0, atol=1e-6)

    # Variances should be present (we wrote them)
    assert edi_out["Z_err"] is not None
    assert edi_out["T_err"] is not None
    assert np.all(np.isfinite(edi_out["Z_err"]))
    assert np.all(np.isfinite(edi_out["T_err"]))


def test_compute_pt_without_errors() -> None:
    """compute_pt should return P and None when Z_err is not provided."""
    Z = _make_invertible_real_Z(5, seed=7)
    P, P_err = dp.compute_pt(Z=Z, Z_err=None)
    assert P.shape == (5, 2, 2)
    assert P_err is None
    assert np.all(np.isfinite(P))


def test_compute_pt_with_var_errors() -> None:
    """compute_pt should accept variances and return variance (or std) accordingly."""
    Z = _make_invertible_real_Z(4, seed=11)
    Z_var = 1e-3 * np.ones_like(Z.real)
    P, P_err = dp.compute_pt(Z=Z, Z_err=Z_var, err_kind="var", nsim=40, random_state=_rng(42))
    assert P.shape == (4, 2, 2)
    assert P_err is not None
    assert P_err.shape == (4, 2, 2)
    assert np.all(np.isfinite(P_err))


def test_dataframe_from_edi_has_expected_columns() -> None:
    """dataframe_from_edi should include rho/phi, tipper and pt columns when available."""
    edi = _make_classical_edi_dict(n=5, err_kind="var")
    df = dp.dataframe_from_edi(edi=edi, include_rho_phi=True, include_tipper=True, include_pt=True)

    # Core
    assert "freq" in df.columns
    assert "period" in df.columns

    # Rho/phi columns
    for comp in ("xx", "xy", "yx", "yy"):
        assert f"rho_{comp}" in df.columns
        assert f"phi_{comp}" in df.columns
        assert f"rho_{comp}_err" in df.columns
        assert f"phi_{comp}_err" in df.columns

    # Tipper
    for col in ("Tx_re", "Tx_im", "Ty_re", "Ty_im"):
        assert col in df.columns

    # PT
    for col in ("ptxx", "ptxy", "ptyx", "ptyy"):
        assert col in df.columns

    # Metadata attrs
    assert df.attrs.get("station") == "TEST01"


def _encode_mat7_from_S(S: np.ndarray) -> np.ndarray:
    """Encode a complex Hermitian 7×7 spectra matrix into Phoenix's 7×7 real format."""
    if S.shape != (7, 7):
        raise ValueError("S must be 7x7.")
    mat7 = np.zeros((7, 7), dtype=float)
    for i in range(7):
        if abs(S[i, i].imag) > 1e-12:
            raise ValueError("Diagonal must be real for Phoenix encoding.")
        mat7[i, i] = float(S[i, i].real)
        for j in range(i + 1, 7):
            mat7[j, i] = float(S[i, j].real)  # lower triangle: real part
            mat7[i, j] = float(S[i, j].imag)  # upper triangle: imag part
    return mat7


def _build_minimal_spectra_edi_text(freq_hz: float, rot_deg: float, mat7: np.ndarray) -> str:
    """Create a minimal EDI text that contains one Phoenix >SPECTRA block."""
    # Put the 49 numbers in 7 lines of 7 numbers each.
    numbers = mat7.ravel()
    lines = [
        ">HEAD",
        '  DATAID="PHOENIX01"',
        "  LAT=10.0",
        "  LON=20.0",
        "  ELEV=100.0",
        "",
        f">SPECTRA  FREQ={freq_hz: .6E} ROTSPEC={rot_deg: .6g} AVGT={1.0e3: .6E} // 49",
    ]
    for i in range(0, 49, 7):
        chunk = " ".join(f"{v: .6E}" for v in numbers[i : i + 7])
        lines.append(" " + chunk)
    # Add a dummy next block start to terminate parsing cleanly
    lines.append(">END")
    return "\n".join(lines) + "\n"


def test_parse_spectra_blocks_reconstructs_expected_values(tmp_path: Path) -> None:
    """SPECTRA parsing should find blocks, reconstruct S, and load_edi should recover Z/T."""
    # Construct a Hermitian spectra matrix with only the required sub-blocks non-zero.
    S = np.zeros((7, 7), dtype=np.complex128)

    # SHH (H1,H2)
    S[0, 0] = 2.0
    S[1, 1] = 3.0
    S[0, 1] = 0.5 + 0.1j
    S[1, 0] = 0.5 - 0.1j

    # SEH rows (Ex,Ey) vs (H1,H2)
    S[3, 0] = 1.0 + 0.2j
    S[3, 1] = -0.3 + 0.1j
    S[4, 0] = 0.5 - 0.1j
    S[4, 1] = 0.8 + 0.05j

    # Enforce Hermitian counterparts
    S[0, 3] = np.conj(S[3, 0])
    S[1, 3] = np.conj(S[3, 1])
    S[0, 4] = np.conj(S[4, 0])
    S[1, 4] = np.conj(S[4, 1])

    # SBH row (Hz) vs (H1,H2)
    S[2, 0] = 0.1 + 0.0j
    S[2, 1] = -0.05 + 0.02j
    S[0, 2] = np.conj(S[2, 0])
    S[1, 2] = np.conj(S[2, 1])

    mat7 = _encode_mat7_from_S(S)
    text = _build_minimal_spectra_edi_text(freq_hz=10.0, rot_deg=15.0, mat7=mat7)
    edi_path = tmp_path / "phoenix.edi"
    _write_text(edi_path, text)

    # Parse blocks
    blocks = dp.parse_spectra_blocks(edi_text=text)
    assert len(blocks) == 1
    f, avgt, rot, mat7_out = blocks[0]
    assert math.isclose(f, 10.0, rel_tol=0, abs_tol=1e-12)
    assert math.isfinite(avgt)
    assert math.isclose(rot, 15.0, rel_tol=0, abs_tol=1e-12)
    assert mat7_out.shape == (7, 7)

    # Reconstruct S and compare to original
    S_rec = dp.reconstruct_S_phoenix(mat7=mat7_out)
    assert np.allclose(S_rec, S, atol=1e-12)

    # Expected Z/T (manual, mirroring the math but not calling ZT_from_S)
    SHH = np.array([[S[0, 0], S[0, 1]], [S[1, 0], S[1, 1]]], dtype=np.complex128)
    SEH = np.array([[S[3, 0], S[3, 1]], [S[4, 0], S[4, 1]]], dtype=np.complex128)
    SBH = np.array([[S[2, 0], S[2, 1]]], dtype=np.complex128)
    Z_exp = (SEH @ np.linalg.inv(SHH)) / 1.0e3
    T_exp = (SBH @ np.linalg.inv(SHH))

    edi = dp.load_edi(path=edi_path, prefer_spectra=True, ref="RH")
    assert edi["source_kind"] == "spectra"
    assert edi["freq"].shape == (1,)
    assert np.allclose(edi["freq"][0], 10.0)
    assert edi["rot"] is not None
    assert np.allclose(edi["rot"][0], 15.0)

    assert edi["Z"].shape == (1, 2, 2)
    assert np.allclose(edi["Z"][0], Z_exp, atol=1e-12)

    assert edi["T"] is not None
    assert edi["T"].shape == (1, 1, 2)
    assert np.allclose(edi["T"][0], T_exp, atol=1e-12)


def test_rotate_data_increases_rot_and_rotates_shapes() -> None:
    """rotate_data should rotate Z/T and update the rot array consistently."""
    edi = _make_classical_edi_dict(n=4, err_kind="var")
    rot0 = np.asarray(edi["rot"]).copy()

    edi_r = dp.rotate_data(edi_dict=edi, angle=10.0, degrees=True)
    assert edi_r["Z"].shape == (4, 2, 2)
    assert edi_r["T"] is not None and edi_r["T"].shape == (4, 1, 2)
    assert edi_r["rot"] is not None
    assert np.allclose(np.asarray(edi_r["rot"]), rot0 + 10.0)


def test_interpolate_data_changes_frequency_grid() -> None:
    """interpolate_data should produce arrays evaluated on a new frequency grid."""
    edi = _make_classical_edi_dict(n=6, err_kind="var")

    newfreqs_log10 = np.linspace(np.log10(edi["freq"].min()), np.log10(edi["freq"].max()), 9)
    edi_i = dp.interpolate_data(edi_dict=edi, method={"newfreqs": newfreqs_log10})

    assert edi_i["freq"].shape == (9,)
    assert edi_i["Z"].shape == (9, 2, 2)
    if edi_i.get("T") is not None:
        assert edi_i["T"].shape == (9, 1, 2)


def test_emtf_xml_roundtrip_and_conversion(tmp_path: Path) -> None:
    """write_emtf_xml/read_emtf_xml and edi_to_emtf/emtf_to_edi should roundtrip basic content."""
    edi = _make_classical_edi_dict(n=5, err_kind="var")

    edi_path = tmp_path / "orig.edi"
    dp.save_edi(path=edi_path, edi=edi, add_pt_blocks=False)

    emtf_path = tmp_path / "out.xml"
    dp.edi_to_emtf(edi_path=edi_path, emtf_path=emtf_path)

    data = dp.read_emtf_xml(path=emtf_path)
    assert "metadata" in data and "transfer_functions" in data
    assert "freq" in data["transfer_functions"]

    edi_back_path = tmp_path / "back.edi"
    dp.emtf_to_edi(emtf_path=emtf_path, edi_path=edi_back_path)

    edi_back = dp.load_edi(path=edi_back_path, prefer_spectra=False)
    assert np.allclose(np.sort(edi_back["freq"]), np.sort(edi["freq"]))
    # Impedance components should exist and be finite
    assert np.all(np.isfinite(edi_back["Z"].real))
    assert np.all(np.isfinite(edi_back["Z"].imag))
