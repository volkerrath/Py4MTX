#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ft_convention.py — FT sign-convention correction for MT transfer functions.
============================================================================

MT processing software applies one of two Fourier-transform conventions:

    e^{-iωt}  — standard geophysical convention  (Metronix ADU, DELTA, …)
    e^{+iωt}  — engineering / Phoenix MTU convention

Under e^{+iωt} all frequency-domain quantities become the complex conjugate
of their e^{-iωt} equivalents.  For the impedance tensor and tipper this means:

    Z_{phoenix}(ω)  = Z*_{standard}(ω)   →  Im(Z) sign-flipped
    T_{phoenix}(ω)  = T*_{standard}(ω)   →  Im(T) sign-flipped

|Z|, apparent resistivity, and error magnitudes are unaffected.
The impedance phase (arctan(Im/Re)) changes sign — it appears in the
3rd/4th quadrant instead of the 1st/2nd quadrant for passive media.

Public API
----------
correct_ft_convention(data_dict, *, from_convention, to_convention="e-iwt")
    In-place correction of a py4mt data dict.

apply_conjugation(data_dict)
    Low-level: conjugate Z and T (and their errors) in-place.

is_corrected(data_dict)
    Return True if the dict already carries the standard e^{-iωt} convention.

load_edi_corrected(path, *, manufacturer, **kwargs)
    Load an EDI file via data_proc.load_edi and apply FT correction.

save_edi_corrected(data_dict, *, path, to_convention="e+iwt", **kwargs)
    Re-conjugate to a target convention (if needed) and write an EDI file.

convert_edi(src_path, dst_path, *, from_convention, to_convention="e-iwt", **kwargs)
    One-shot EDI file conversion between FT conventions.

Author: Volker Rath (DIAS)
Modified: 2026-03-25 — initial version; Claude Sonnet 4.6 (Anthropic)
Modified: 2026-03-25 — load_edi_corrected, save_edi_corrected, convert_edi; Claude Sonnet 4.6 (Anthropic)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical string for the standard geophysical convention.
CONV_STANDARD = "e-iwt"

#: Canonical string for the Phoenix / engineering convention.
CONV_PHOENIX = "e+iwt"

#: Set of manufacturer names that use e^{+iωt} by default.
PHOENIX_MANUFACTURERS = {"phoenix"}

#: Set of manufacturer names that use e^{-iωt} by default.
STANDARD_MANUFACTURERS = {"metronix", "delta"}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def apply_conjugation(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Conjugate Z and T (and their error magnitudes) in *data_dict* **in place**.

    This is the primitive operation: it flips the sign of Im(Z) and Im(T)
    without any bookkeeping.  Errors (Z_err, T_err) are left unchanged because
    they are real-valued magnitudes (variances or standard deviations) and are
    not affected by a sign flip of the imaginary part.

    Parameters
    ----------
    data_dict : dict
        py4mt site dictionary as returned by :func:`data_proc.load_edi`.

    Returns
    -------
    dict
        The same dict (modified in place).
    """
    Z = data_dict.get("Z")
    if Z is not None:
        data_dict["Z"] = np.asarray(Z, dtype=np.complex128).conj()

    T = data_dict.get("T")
    if T is not None:
        data_dict["T"] = np.asarray(T, dtype=np.complex128).conj()

    # Phase tensor P = Re(Z)^{-1} Im(Z): Re(Z) is unchanged, Im(Z) flips.
    # P therefore flips sign → conjugate P as well (P is real-valued, so
    # "conjugation" here means negation).
    P = data_dict.get("P")
    if P is not None:
        data_dict["P"] = -np.asarray(P, dtype=float)

    return data_dict


def is_corrected(data_dict: Dict[str, Any]) -> bool:
    """Return ``True`` if *data_dict* is already in the standard e^{-iωt} convention.

    Checks the ``"ft_convention"`` key when present.  If the key is absent,
    falls back to checking ``"manufacturer"``: Phoenix → not yet corrected;
    all others → assumed standard.  If neither key is present, returns ``True``
    (i.e. no correction assumed needed).
    """
    ft = data_dict.get("ft_convention")
    if ft is not None:
        return ft in (CONV_STANDARD, "e+iwt_corrected")

    mfr = str(data_dict.get("manufacturer", "metronix")).lower()
    return mfr not in PHOENIX_MANUFACTURERS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_ft_convention(
    data_dict: Dict[str, Any],
    *,
    from_convention: Optional[str] = None,
    to_convention: str = CONV_STANDARD,
) -> Dict[str, Any]:
    """Correct the FT sign convention of a py4mt site dictionary **in place**.

    If *from_convention* is ``None`` the source convention is inferred from
    ``data_dict["ft_convention"]`` (if present) or from
    ``data_dict["manufacturer"]``.

    A correction is applied only when *from_convention* ≠ *to_convention*.
    Repeated calls are idempotent: calling twice with the same arguments
    returns the dict to its original state (conjugation is its own inverse),
    but the bookkeeping keys will reflect the final state.

    Parameters
    ----------
    data_dict : dict
        py4mt site dictionary (modified in place).
    from_convention : {"e-iwt", "e+iwt"} or None
        Source FT convention.  When ``None``, inferred automatically.
    to_convention : {"e-iwt", "e+iwt"}
        Target convention.  Default is the standard geophysical ``"e-iwt"``.

    Returns
    -------
    dict
        The same dict (modified in place).

    Raises
    ------
    ValueError
        If *from_convention* or *to_convention* are not recognised strings.

    Examples
    --------
    >>> # Load a Phoenix EDI without the manufacturer flag and correct manually
    >>> site = data_proc.load_edi("SITE.edi")          # loaded as Metronix by default
    >>> ft_convention.correct_ft_convention(site, from_convention="e+iwt")
    >>> # site["Z"] and site["T"] are now in the standard e^{-iωt} convention

    >>> # Idiomatic usage when manufacturer is known at load time
    >>> site = data_proc.load_edi("SITE.edi", manufacturer="phoenix")
    >>> # No further correction needed — load_edi already applied it.
    >>> ft_convention.is_corrected(site)
    True
    """
    _valid = {CONV_STANDARD, CONV_PHOENIX}

    # Infer from_convention if not provided
    if from_convention is None:
        ft = data_dict.get("ft_convention")
        if ft is not None:
            # Already corrected → from_convention is the standard one
            if ft in (CONV_STANDARD, "e+iwt_corrected"):
                from_convention = CONV_STANDARD
            else:
                from_convention = CONV_PHOENIX
        else:
            mfr = str(data_dict.get("manufacturer", "metronix")).lower()
            from_convention = CONV_PHOENIX if mfr in PHOENIX_MANUFACTURERS else CONV_STANDARD

    from_convention = from_convention.lower()
    to_convention = to_convention.lower()

    # Accept "e+iwt_corrected" as an alias for the standard convention
    if from_convention == "e+iwt_corrected":
        from_convention = CONV_STANDARD

    if from_convention not in _valid:
        raise ValueError(
            f"Unknown from_convention {from_convention!r}; "
            f"expected one of {sorted(_valid)}."
        )
    if to_convention not in _valid:
        raise ValueError(
            f"Unknown to_convention {to_convention!r}; "
            f"expected one of {sorted(_valid)}."
        )

    if from_convention == to_convention:
        # Nothing to do — update bookkeeping to reflect target convention
        _record_convention(data_dict, to_convention)
        return data_dict

    # Apply conjugation
    apply_conjugation(data_dict)

    # Update bookkeeping keys
    _record_convention(data_dict, to_convention)

    return data_dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_convention(data_dict: Dict[str, Any], convention: str) -> None:
    """Write ft_convention (and manufacturer if inferrable) into *data_dict*."""
    if convention == CONV_STANDARD:
        # Record whether this was a corrected Phoenix file or native standard
        mfr = str(data_dict.get("manufacturer", "metronix")).lower()
        if mfr in PHOENIX_MANUFACTURERS:
            data_dict["ft_convention"] = "e+iwt_corrected"
        else:
            data_dict["ft_convention"] = CONV_STANDARD
    else:
        data_dict["ft_convention"] = CONV_PHOENIX


# ---------------------------------------------------------------------------
# Convenience batch function
# ---------------------------------------------------------------------------

def correct_batch(
    sites: list,
    *,
    from_convention: Optional[str] = None,
    to_convention: str = CONV_STANDARD,
) -> list:
    """Apply :func:`correct_ft_convention` to a list of site dicts in place.

    Parameters
    ----------
    sites : list of dict
        py4mt site dictionaries.
    from_convention, to_convention
        Forwarded to :func:`correct_ft_convention`.

    Returns
    -------
    list
        The same list (dicts modified in place).
    """
    for site in sites:
        correct_ft_convention(
            site,
            from_convention=from_convention,
            to_convention=to_convention,
        )
    return sites

# ---------------------------------------------------------------------------
# EDI I/O with convention correction
# ---------------------------------------------------------------------------

def load_edi_corrected(
    path: str | Path,
    *,
    manufacturer: str,
    to_convention: str = CONV_STANDARD,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Load an EDI file and ensure its transfer functions are in *to_convention*.

    This is a thin wrapper around :func:`data_proc.load_edi` that passes
    *manufacturer* through so that the FT correction is applied automatically
    at load time.  The returned dict always carries ``"ft_convention"`` and
    ``"manufacturer"`` keys.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the EDI file.
    manufacturer : {"metronix", "phoenix", "delta"}
        Instrument manufacturer — determines whether Im(Z) and Im(T) need
        to be sign-flipped.
    to_convention : {"e-iwt", "e+iwt"}
        Target convention.  Default is the standard geophysical ``"e-iwt"``.
        Pass ``"e+iwt"`` if you want to keep (or force) the Phoenix convention.
    **kwargs
        Forwarded verbatim to :func:`data_proc.load_edi` (e.g. ``freq_order``,
        ``err_kind``, ``prefer_spectra``).

    Returns
    -------
    dict
        Site dictionary in *to_convention*.

    Examples
    --------
    >>> import mt_ft_convention as ftc
    >>> site = ftc.load_edi_corrected("SITE_PHX.edi", manufacturer="phoenix")
    >>> site["ft_convention"]
    'e+iwt_corrected'

    >>> # Force-keep Phoenix convention (no correction)
    >>> site = ftc.load_edi_corrected("SITE_PHX.edi", manufacturer="phoenix",
    ...                               to_convention="e+iwt")
    >>> site["ft_convention"]
    'e+iwt'
    """
    import data_proc  # imported here to avoid a hard circular dependency

    # load_edi already applies the conjugation for manufacturer="phoenix"
    data_dict = data_proc.load_edi(str(path), manufacturer=manufacturer, **kwargs)

    # If the caller wants a non-standard target, apply an additional flip
    correct_ft_convention(data_dict, to_convention=to_convention)

    return data_dict


def save_edi_corrected(
    data_dict: Dict[str, Any],
    *,
    path: str | Path,
    to_convention: str = CONV_PHOENIX,
    **kwargs: Any,
) -> None:
    """Write *data_dict* to an EDI file after converting to *to_convention*.

    EDI files produced by Phoenix and Metronix processing software are
    conventionally stored in their native FT convention (Phoenix: e⁺ⁱωᵗ,
    Metronix: e⁻ⁱωᵗ).  Use this function when you need to write an EDI
    that a specific downstream tool expects in a particular convention.

    The dict is **not modified** — a temporary copy is made for the conjugation
    before writing.

    Parameters
    ----------
    data_dict : dict
        py4mt site dictionary.  Its current convention is read from
        ``data_dict["ft_convention"]`` (or inferred from ``"manufacturer"``).
    path : str or pathlib.Path
        Output EDI path.
    to_convention : {"e-iwt", "e+iwt"}
        Target convention for the written file.  Default is ``"e+iwt"``
        (Phoenix / engineering convention), which is the most common reason
        to call this function.
    **kwargs
        Forwarded to :func:`data_proc.save_edi` (e.g. ``add_pt_blocks``,
        ``numbers_per_line``).

    Examples
    --------
    >>> # Load as standard, write back as Phoenix convention
    >>> site = data_proc.load_edi("SITE.edi", manufacturer="metronix")
    >>> ftc.save_edi_corrected(site, path="SITE_phoenix.edi", to_convention="e+iwt")

    >>> # Round-trip: Phoenix → standard → Phoenix  (data_dict unchanged)
    >>> site = data_proc.load_edi("SITE_PHX.edi", manufacturer="phoenix")
    >>> ftc.save_edi_corrected(site, path="SITE_PHX_out.edi")
    """
    import data_proc  # imported here to avoid a hard circular dependency
    import copy

    # Work on a shallow copy so the caller's dict is never mutated
    out_dict = copy.copy(data_dict)
    # Arrays are mutable, so copy the ones we might conjugate
    for key in ("Z", "T", "P"):
        if out_dict.get(key) is not None:
            out_dict[key] = np.array(out_dict[key])

    correct_ft_convention(out_dict, to_convention=to_convention)
    data_proc.save_edi(out_dict, path=path, **kwargs)


def convert_edi(
    src_path: str | Path,
    dst_path: str | Path,
    *,
    from_convention: str,
    to_convention: str = CONV_STANDARD,
    manufacturer: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Convert an EDI file from one FT convention to another.

    Reads *src_path*, flips the sign of Im(Z) and Im(T) if the conventions
    differ, and writes the result to *dst_path*.

    Parameters
    ----------
    src_path : str or pathlib.Path
        Input EDI file.
    dst_path : str or pathlib.Path
        Output EDI file (may be the same path for in-place conversion).
    from_convention : {"e-iwt", "e+iwt"}
        FT convention of the source file.
    to_convention : {"e-iwt", "e+iwt"}
        FT convention to write.  Default ``"e-iwt"`` (standard).
    manufacturer : str or None
        Passed to :func:`data_proc.load_edi` as *manufacturer*.  When
        ``None``, defaults to ``"phoenix"`` if *from_convention* is
        ``"e+iwt"``, otherwise ``"metronix"``.
    **kwargs
        Additional keyword arguments forwarded to :func:`data_proc.load_edi`.
        ``save_edi`` kwargs (``add_pt_blocks``, ``numbers_per_line``) are
        **not** forwarded — call :func:`save_edi_corrected` directly for
        full control.

    Examples
    --------
    >>> # Convert a Phoenix EDI to standard convention for an e-iwt inversion code
    >>> ftc.convert_edi("SITE_PHX.edi", "SITE_std.edi",
    ...                 from_convention="e+iwt", to_convention="e-iwt")

    >>> # Batch-convert a directory
    >>> from pathlib import Path
    >>> for edi in Path("raw/").glob("*.edi"):
    ...     ftc.convert_edi(edi, Path("corrected") / edi.name,
    ...                     from_convention="e+iwt")
    """
    import data_proc  # imported here to avoid a hard circular dependency

    if manufacturer is None:
        manufacturer = "phoenix" if from_convention == CONV_PHOENIX else "metronix"

    # Load with the declared manufacturer so load_edi applies its own correction
    # (for phoenix: Im already flipped → now in e-iwt).  Then let
    # correct_ft_convention handle whatever residual flip is needed.
    data_dict = data_proc.load_edi(str(src_path), manufacturer=manufacturer, **kwargs)
    correct_ft_convention(data_dict, to_convention=to_convention)
    data_proc.save_edi(data_dict, path=dst_path)
