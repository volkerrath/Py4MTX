"""
Petrophysical models: resistivity & permeability
=================================================
Implements the following model families:

  1. Archie (1942)                  — clean-formation resistivity
  2. Simandoux (1963)               — shaly-sand resistivity
  3. Dual-porosity / fractured      — Warren-Root parallel resistivity network
  4. RGPZ (Glover et al. 2006)      — permeability from grain size & formation factor
  5. Hashin-Shtrikman (1962)        — rigorous conductivity bounds for two-phase and
                                      N-phase composites; HS± and HS mixing formula
  6. Glover et al. (2000)           — modified Archie law for two conducting phases
  7. Glover (2010)                  — generalized Archie law for n conducting phases
  8. Sinmyo & Keppler (2017)        — NaCl-H2O fluid conductivity to 600 °C / 1 GPa
  9. Guo & Keppler (2019)           — NaCl-H2O fluid conductivity to 900 °C / 5 GPa

All inputs/outputs use SI-consistent units unless noted.
Resistivity in Ω·m; permeability in m² internally, converted to mD on output;
grain diameter input in µm for convenience.

References
----------
Archie, G.E. (1942) Trans. AIME, 146, 54-62.
Simandoux, P. (1963) Rev. Inst. Fr. Pétrole, 18 (suppl.), 193-220.
Warren, J.E. & Root, P.J. (1963) SPE-J, 3(3), 245-255.
Glover, P.W.J., Zadjali, I.I. & Frew, K.A. (2006) Geophysics, 71(4), F49-F60.
Glover, P.W.J. (2010) Solid Earth, 1, 85-91.
Walker, E. & Glover, P.W.J. (2010) Geophysics, 75(6), E229-E245.
Sen, P.N. & Goode, P.A. (1992) Geophysics, 57(1), 89-96.
Hilchie, D.W. (1982) Applied Open-Hole Log Interpretation, Golden, CO.
Hashin, Z. & Shtrikman, S. (1962) J. Appl. Phys., 33(10), 3125-3131.
Berryman, J.G. (1995) in: Rock Physics & Phase Relations (AGU Ref. Shelf 3).
Glover, P.W.J., Hole, M.J. & Pous, J. (2000) Geophys. J. Int., 142, 516-526.
Sinmyo, R. & Keppler, H. (2017) Contrib. Mineral. Petrol., 172, 4.
Guo, H. & Keppler, H. (2019) J. Geophys. Res. Solid Earth, 124, 1760-1771.

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-04-03
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Brine conductivity / resistivity (salinity + temperature → σw, Rw)
# ---------------------------------------------------------------------------

def brine_conductivity_sen_goode(
    salinity_ppm: float,
    temp_c: float,
) -> float:
    """
    NaCl brine electrical conductivity via Sen & Goode (1992).

    This is the standard petrophysical formula for formation-water
    conductivity, derived from a polynomial fit to experimental NaCl
    solution data.  It is physically more rigorous than Arps/Hilchie
    and is valid over the full range of formation-brine conditions:

        T  :   0 – 300 °C
        C  :   0 – 300 g/L  (≈ 0 – 300,000 ppm)

    Governing equation (Sen & Goode 1992, eq. 1)
    ---------------------------------------------
    σ_w(T, C) = C · [5.6 + 0.27·T - 1.5×10⁻⁴·T²]
                  - C^1.5 · [2.36 + 0.099·T] / (1 + 0.214·C^0.5)

    where
        σ_w  conductivity in S/m
        C    NaCl concentration in mol/L  (molarity)
        T    temperature in °C

    Salinity conversion
    -------------------
    The function accepts salinity in ppm (mg NaCl per kg solution) and
    converts internally:

        C [mol/L] ≈ salinity_ppm × ρ_brine / (58,443 × 10⁶)

    Brine density ρ_brine is estimated from the Batzle & Wang (1992)
    approximation for NaCl solutions:

        ρ_brine [kg/m³] = ρ_water(T) + 0.668·S + 0.44·S²
                          + 1e-6·S·(300·P - 2400·S + T·(80 + 3T - 3300·S))

    where S is salinity in g/cm³ and P is pressure in MPa.
    At typical reservoir pressures (< 50 MPa) and moderate salinities,
    the pressure correction is small; this function uses P = 0 MPa
    (surface / low-pressure approximation).  For high-pressure reservoirs
    supply the corrected density via :func:`brine_conductivity_sen_goode_mol`
    directly.

    Parameters
    ----------
    salinity_ppm : float
        NaCl equivalent salinity in mg/kg (ppm by mass).
        Range: 1 000 – 300 000 ppm.
    temp_c : float
        Temperature in °C.  Range: 0 – 300 °C.

    Returns
    -------
    float
        Brine conductivity σ_w in S/m.

    See Also
    --------
    brine_conductivity_sen_goode_mol : same model but accepts molarity directly.
    brine_resistivity_sen_goode      : returns Rw = 1/σ_w in Ω·m.
    brine_resistivity                : legacy Arps/Hilchie approximation.

    References
    ----------
    Sen, P.N. & Goode, P.A. (1992) Geophysics 57(1), 89-96.
    Batzle, M. & Wang, Z. (1992) Geophysics 57(11), 1396-1408.

    Examples
    --------
    >>> sw = brine_conductivity_sen_goode(30_000, 60)
    >>> print(f"σ_w = {sw:.4f} S/m")
    >>> print(f"Rw  = {1/sw:.4f} Ω·m")
    """
    if salinity_ppm <= 0:
        raise ValueError("salinity_ppm must be positive")
    if not 0 <= temp_c <= 300:
        raise ValueError(f"temp_c must be in [0, 300] °C for Sen & Goode, got {temp_c}")
    if salinity_ppm > 350_000:
        raise ValueError(f"salinity_ppm={salinity_ppm} exceeds NaCl solubility limit (~360 g/kg)")

    # --- Brine density (Batzle & Wang 1992, P = 0 MPa approximation) ---
    S_frac   = salinity_ppm * 1e-6              # mass fraction (kg NaCl / kg solution)
    T        = temp_c
    # Pure water density polynomial (valid 0–350 °C, Batzle & Wang eq. 27a)
    rho_w = (1.0
             + 1e-6 * (-80.0*T - 3.3*T**2 + 0.00175*T**3
                       + 489.0*300*0         # pressure term zeroed
                       - 2*T*300*0
                       + 0.016*T**2*0))
    # Simpler pure-water density fit good to ±0.5 % for 0–200 °C:
    rho_w = (999.842594
              + 6.793952e-2 * T
              - 9.095290e-3 * T**2
              + 1.001685e-4 * T**3
              - 1.120083e-6 * T**4
              + 6.536332e-9 * T**5) / 1000.0  # → g/cm³

    S_gcm3   = S_frac                           # ≈ g NaCl / cm³ solution at low salinity
    rho_b    = rho_w + 0.668*S_gcm3 + 0.44*S_gcm3**2   # g/cm³  (Batzle & Wang eq. 27b, P=0)

    # --- Molarity C [mol/L] ---
    # C = (mass fraction × density [g/mL]) / molar mass [g/mol]
    M_NaCl   = 58.443                           # g/mol
    C        = (S_frac * rho_b * 1000.0) / M_NaCl   # mol/L

    return brine_conductivity_sen_goode_mol(C, temp_c)


def brine_conductivity_sen_goode_mol(
    C: float,
    temp_c: float,
) -> float:
    """
    Sen & Goode (1992) conductivity formula in its native molarity form.

    Parameters
    ----------
    C : float
        NaCl molarity in mol/L.  Range: 0 – ~5.5 mol/L (saturation).
    temp_c : float
        Temperature in °C.  Range: 0 – 300 °C.

    Returns
    -------
    float
        Brine conductivity σ_w in S/m.
    """
    if C < 0:
        raise ValueError(f"Molarity C must be ≥ 0, got {C}")
    if C == 0:
        return 0.0
    if not 0 <= temp_c <= 300:
        raise ValueError(f"temp_c must be in [0, 300] °C, got {temp_c}")

    T  = temp_c
    # Sen & Goode (1992) eq. 1
    sigma = (C * (5.6 + 0.27*T - 1.5e-4*T**2)
             - C**1.5 * (2.36 + 0.099*T) / (1.0 + 0.214*C**0.5))
    return max(sigma, 0.0)


def brine_resistivity_sen_goode(
    salinity_ppm: float,
    temp_c: float,
) -> float:
    """
    NaCl brine resistivity Rw via Sen & Goode (1992).

    Convenience wrapper: returns 1 / σ_w from
    :func:`brine_conductivity_sen_goode`.

    Parameters
    ----------
    salinity_ppm : float
        NaCl equivalent salinity in ppm (mg/kg).
    temp_c : float
        Temperature in °C.

    Returns
    -------
    float
        Brine resistivity Rw in Ω·m.

    Examples
    --------
    >>> rw = brine_resistivity_sen_goode(30_000, 60)
    >>> print(f"Rw = {rw:.4f} Ω·m")
    """
    sigma = brine_conductivity_sen_goode(salinity_ppm, temp_c)
    if sigma == 0.0:
        return math.inf
    return 1.0 / sigma


def brine_resistivity(salinity_ppm: float, temp_c: float) -> float:
    """
    Estimate brine resistivity Rw — **Arps/Hilchie empirical correlation**.

    Kept for backward compatibility and quick estimates.  For new work
    prefer :func:`brine_resistivity_sen_goode`, which is physically
    grounded and valid to 300 °C / 300 g/L.

    Arps/Hilchie equations
    ----------------------
        Rw(25 °C) = 0.0123 + 3647.5 / salinity_ppm^0.955
        Rw(T)     = Rw(25°C) × (25 + 21.5) / (T + 21.5)

    Valid range: ~1 000 – 200 000 ppm, 15 – 150 °C.

    Parameters
    ----------
    salinity_ppm : float
        NaCl equivalent salinity in ppm (mg/kg).
    temp_c : float
        Temperature in °C.

    Returns
    -------
    float
        Brine resistivity Rw in Ω·m.
    """
    if salinity_ppm <= 0:
        raise ValueError("salinity_ppm must be positive")
    if temp_c < 0:
        raise ValueError("temp_c must be ≥ 0 °C")

    rw_25 = 0.0123 + 3647.5 / (salinity_ppm ** 0.955)
    rw_t  = rw_25 * (25.0 + 21.5) / (temp_c + 21.5)
    return rw_t

# ---------------------------------------------------------------------------
# High-P/T NaCl-H2O fluid conductivity models from the geophysical literature
# ---------------------------------------------------------------------------

@dataclass
class FluidConductivityResult:
    """
    Result container for explicit fluid conductivity / resistivity models.

    Parameters are echoed in a model-agnostic way so that downstream code can
    compare different parameterizations without special handling.
    """
    model: str
    sigma: float
    resistivity: float
    temp_c: float
    salinity_wt_pct: float
    density_g_cm3: float
    pressure_gpa: Optional[float] = None
    notes: str = ""


def _validate_log_model_inputs(
    salinity_wt_pct: float,
    temp_c: float,
    density_g_cm3: float,
) -> tuple[float, float, float]:
    """Validate common inputs used by the Sinmyo/Guo log10 conductivity models."""
    if salinity_wt_pct <= 0:
        raise ValueError(f"salinity_wt_pct must be positive, got {salinity_wt_pct}")
    if density_g_cm3 <= 0:
        raise ValueError(f"density_g_cm3 must be positive, got {density_g_cm3}")
    T_k = temp_c + 273.15
    if T_k <= 0:
        raise ValueError(f"Absolute temperature must be positive, got {T_k}")
    return T_k, salinity_wt_pct, density_g_cm3


def _lambda0_nacl_infinite_dilution(
    T_k: float,
    density_g_cm3: float,
) -> float:
    """
    Infinite-dilution molar conductivity for NaCl used by Sinmyo/Keppler and
    Guo/Keppler.

    Returns
    -------
    float
        Λ0 in S cm^2 mol^-1.
    """
    lam0 = (1573.0
            - 1212.0 * density_g_cm3
            + 537_062.0 / T_k
            - 208_122_721.0 / (T_k ** 2))
    return max(lam0, 1e-12)


def brine_conductivity_sinmyo_keppler(
    salinity_wt_pct: float,
    temp_c: float,
    density_g_cm3: float,
    pressure_gpa: Optional[float] = None,
) -> FluidConductivityResult:
    """
    NaCl-H2O fluid conductivity using the Sinmyo & Keppler (2017) regression.

    The model is

        log10(sigma) = -1.7060 - 93.78/T
                       + 0.8075 log10(c)
                       + 3.0781 log10(rho)
                       + log10(Lambda0(T, rho))

    where sigma is in S/m, T is in K, c is NaCl concentration in wt%, rho is
    the density of pure water in g/cm^3 at the target P-T conditions, and
    Lambda0 is the NaCl limiting molar conductivity in S cm^2 mol^-1.

    This implementation deliberately requires the water density as an input
    rather than embedding a separate equation of state. That keeps the model
    transparent and avoids mixing conductivity fitting with an uncertain EOS
    approximation.
    """
    T_k, c_wt, rho = _validate_log_model_inputs(salinity_wt_pct, temp_c, density_g_cm3)
    if not 100.0 <= temp_c <= 600.0:
        raise ValueError("Sinmyo & Keppler (2017) is intended for ~100–600 °C")
    if pressure_gpa is not None and not 0.0 <= pressure_gpa <= 1.0:
        raise ValueError("Sinmyo & Keppler (2017) is intended for pressures up to 1 GPa")

    lam0 = _lambda0_nacl_infinite_dilution(T_k, rho)
    log10_sigma = (-1.7060
                   - 93.78 / T_k
                   + 0.8075 * math.log10(c_wt)
                   + 3.0781 * math.log10(rho)
                   + math.log10(lam0))
    sigma = 10.0 ** log10_sigma
    return FluidConductivityResult(
        model="SinmyoKeppler2017",
        sigma=sigma,
        resistivity=(1.0 / sigma) if sigma > 0 else math.inf,
        temp_c=temp_c,
        salinity_wt_pct=salinity_wt_pct,
        density_g_cm3=density_g_cm3,
        pressure_gpa=pressure_gpa,
        notes="Requires pure-water density at target P-T as input.",
    )


def brine_conductivity_guo_keppler(
    salinity_wt_pct: float,
    temp_c: float,
    density_g_cm3: float,
    pressure_gpa: Optional[float] = None,
) -> FluidConductivityResult:
    """
    NaCl-H2O fluid conductivity using the Guo & Keppler (2019) regression.

    The model is

        log10(sigma) = -0.919 - 872.5/T
                       + 0.852 log10(c)
                       + 7.61 log10(rho)
                       + log10(Lambda0(T, rho))

    where the symbols and units are the same as for
    :func:`brine_conductivity_sinmyo_keppler`.
    """
    T_k, c_wt, rho = _validate_log_model_inputs(salinity_wt_pct, temp_c, density_g_cm3)
    if not 150.0 <= temp_c <= 900.0:
        raise ValueError("Guo & Keppler (2019) is intended for ~150–900 °C")
    if pressure_gpa is not None and not 0.0 <= pressure_gpa <= 5.0:
        raise ValueError("Guo & Keppler (2019) is intended for pressures up to 5 GPa")

    lam0 = _lambda0_nacl_infinite_dilution(T_k, rho)
    log10_sigma = (-0.919
                   - 872.5 / T_k
                   + 0.852 * math.log10(c_wt)
                   + 7.61 * math.log10(rho)
                   + math.log10(lam0))
    sigma = 10.0 ** log10_sigma
    return FluidConductivityResult(
        model="GuoKeppler2019",
        sigma=sigma,
        resistivity=(1.0 / sigma) if sigma > 0 else math.inf,
        temp_c=temp_c,
        salinity_wt_pct=salinity_wt_pct,
        density_g_cm3=density_g_cm3,
        pressure_gpa=pressure_gpa,
        notes="Requires pure-water density at target P-T as input.",
    )





# ---------------------------------------------------------------------------
# 1. Archie model (clean formation)
# ---------------------------------------------------------------------------

@dataclass
class ArchieResult:
    """Results from the Archie model."""
    Rw:  float          # brine resistivity, Ω·m
    F:   float          # formation factor  F = a / φ^m
    Ro:  float          # 100 % water-saturated resistivity, Ω·m
    Rt:  float          # true formation resistivity, Ω·m
    RI:  float          # resistivity index  RI = Rt / Ro  (= Sw^-n)
    Sw:  float          # water saturation (input echo)


def archie(
    phi:          float,
    Sw:           float,
    Rw:           float,
    m:            float = 2.0,
    n:            float = 2.0,
    a:            float = 1.0,
) -> ArchieResult:
    """
    Archie (1942) clean-formation resistivity model.

    Governing equations
    -------------------
    Formation factor :  F  = a / φ^m          (Archie's first law)
    100 % Sw resisivity: Ro = F · Rw
    True resistivity :  Rt = Ro / Sw^n        (Archie's second law)
    Resistivity index:  RI = Rt / Ro = Sw^-n

    Parameters
    ----------
    phi : float
        Total (connected) porosity, fraction  0 < φ ≤ 1.
    Sw : float
        Water saturation, fraction  0 < Sw ≤ 1.
    Rw : float
        Brine resistivity in Ω·m.  Use :func:`brine_resistivity` to
        compute from salinity + temperature.
    m : float
        Cementation exponent (default 2.0).
        Typical ranges: 1.3–1.7 (fractures/vugs), 1.8–2.2 (clastics),
        2.0–3.0 (carbonates).
    n : float
        Saturation exponent (default 2.0).  Usually 1.5–2.5.
    a : float
        Tortuosity factor (default 1.0).  Glover (2010) argues a must
        equal 1 to be physically consistent; use 1.0 unless fitting
        legacy datasets that require a ≠ 1.

    Returns
    -------
    ArchieResult
        Named tuple with Rw, F, Ro, Rt, RI, Sw.

    Raises
    ------
    ValueError
        If any parameter is outside its physical range.

    Examples
    --------
    >>> rw = brine_resistivity(30_000, 60)
    >>> r  = archie(phi=0.20, Sw=0.80, Rw=rw, m=2.0, n=2.0)
    >>> print(f"Rt = {r.Rt:.3f} Ω·m")
    """
    if not 0 < phi <= 1:
        raise ValueError(f"phi must be in (0, 1], got {phi}")
    if not 0 < Sw <= 1:
        raise ValueError(f"Sw must be in (0, 1], got {Sw}")
    if Rw <= 0:
        raise ValueError(f"Rw must be positive, got {Rw}")
    if m <= 0 or n <= 0 or a <= 0:
        raise ValueError("m, n, a must all be positive")

    F  = a / (phi ** m)
    Ro = F * Rw
    Rt = Ro / (Sw ** n)
    RI = Rt / Ro

    return ArchieResult(Rw=Rw, F=F, Ro=Ro, Rt=Rt, RI=RI, Sw=Sw)


# ---------------------------------------------------------------------------
# 2. Simandoux model (shaly sand)
# ---------------------------------------------------------------------------

@dataclass
class SimandouxResult:
    """Results from the Simandoux model."""
    Rw:   float     # brine resistivity, Ω·m
    F:    float     # clean-sand formation factor (Archie)
    Ro:   float     # 100 % Sw resistivity (shaly), Ω·m
    Rt:   float     # true resistivity, Ω·m
    RI:   float     # resistivity index Rt / Ro
    Sw:   float     # water saturation (input echo)
    Vsh:  float     # shale volume (input echo)


def simandoux(
    phi:          float,
    Sw:           float,
    Rw:           float,
    Vsh:          float,
    Rsh:          float,
    m:            float = 2.0,
    n:            float = 2.0,
    a:            float = 1.0,
) -> SimandouxResult:
    """
    Simandoux (1963) shaly-sand resistivity model.

    Adds a parallel conduction path through dispersed clay to the
    Archie clean-sand framework.

    Governing equation
    ------------------
    Conductivity form (solves for Rt):

        1/Rt = (φ^m · Sw^n) / (a · Rw)  +  Vsh · Sw / Rsh

    At Sw = 1 this reduces to:

        1/Ro = φ^m / (a · Rw)  +  Vsh / Rsh

    which equals pure Archie when Vsh → 0.

    Parameters
    ----------
    phi : float
        Total porosity, fraction  0 < φ ≤ 1.
    Sw : float
        Water saturation, fraction  0 < Sw ≤ 1.
    Rw : float
        Brine resistivity in Ω·m.
    Vsh : float
        Shale (clay) volume fraction  0 ≤ Vsh < 1.
    Rsh : float
        Shale resistivity in Ω·m (measured from pure shale baseline).
    m : float
        Cementation exponent for the clean sand (default 2.0).
    n : float
        Saturation exponent (default 2.0).
    a : float
        Tortuosity factor (default 1.0).

    Returns
    -------
    SimandouxResult

    Notes
    -----
    The model is most reliable for Vsh < 0.5.  For highly shaly
    formations consider Indonesia, Waxman-Smits, or Dual-Water models
    which account for bound-water conductivity more rigorously.

    Examples
    --------
    >>> rw = brine_resistivity(30_000, 60)
    >>> r  = simandoux(phi=0.20, Sw=0.70, Rw=rw, Vsh=0.15, Rsh=3.0)
    >>> print(f"Rt = {r.Rt:.3f} Ω·m")
    """
    if not 0 < phi <= 1:
        raise ValueError(f"phi must be in (0, 1], got {phi}")
    if not 0 < Sw <= 1:
        raise ValueError(f"Sw must be in (0, 1], got {Sw}")
    if not 0 <= Vsh < 1:
        raise ValueError(f"Vsh must be in [0, 1), got {Vsh}")
    if Rw <= 0 or Rsh <= 0:
        raise ValueError("Rw and Rsh must be positive")

    F       = a / (phi ** m)
    C_sand  = (phi ** m * Sw ** n) / (a * Rw)
    C_shale = Vsh * Sw / Rsh
    Ct      = C_sand + C_shale
    Rt      = 1.0 / Ct

    # Ro = Rt at Sw = 1
    Ct_ro = phi ** m / (a * Rw) + Vsh / Rsh
    Ro    = 1.0 / Ct_ro
    RI    = Rt / Ro

    return SimandouxResult(Rw=Rw, F=F, Ro=Ro, Rt=Rt, RI=RI, Sw=Sw, Vsh=Vsh)


# ---------------------------------------------------------------------------
# 3. Dual-porosity / fractured model
# ---------------------------------------------------------------------------

@dataclass
class DualPorosityResult:
    """Results from the dual-porosity (matrix + fracture) model."""
    Rw:        float   # brine resistivity, Ω·m
    Rt_matrix: float   # matrix resistivity (Archie), Ω·m
    Rt_frac:   float   # fracture network resistivity (Archie), Ω·m
    Rt:        float   # combined true resistivity (parallel), Ω·m
    Ro_matrix: float   # matrix Ro (Sw=1), Ω·m
    Ro_frac:   float   # fracture Ro (Sw=1), Ω·m
    Ro:        float   # combined Ro (parallel), Ω·m
    RI:        float   # resistivity index Rt / Ro
    F_matrix:  float   # matrix formation factor
    F_frac:    float   # fracture formation factor
    Sw_matrix: float   # matrix water saturation (input echo)
    Sw_frac:   float   # fracture water saturation (input echo)
    phi_matrix: float  # matrix porosity (input echo)
    phi_frac:  float   # fracture porosity (input echo)


def dual_porosity(
    phi_matrix:  float,
    Sw_matrix:   float,
    Rw:          float,
    phi_frac:    float,
    Sw_frac:     float   = 1.0,
    m_matrix:    float   = 2.0,
    n_matrix:    float   = 2.0,
    m_frac:      float   = 1.3,
    n_frac:      float   = 1.5,
    a_matrix:    float   = 1.0,
    a_frac:      float   = 1.0,
) -> DualPorosityResult:
    """
    Dual-porosity (Warren-Root) resistivity model for fractured formations.

    The matrix and fracture networks are treated as two independent
    Archie conductors arranged in **electrical parallel**:

        1/Rt = 1/Rt_matrix + 1/Rt_frac

    Each end-member follows its own Archie law with independent
    cementation and saturation exponents.

    Parameters
    ----------
    phi_matrix : float
        Matrix (intergranular) porosity, fraction.  Typically 0.05–0.35.
    Sw_matrix : float
        Water saturation in the matrix, fraction.
    Rw : float
        Brine resistivity in Ω·m (same fluid assumed in both systems).
    phi_frac : float
        Fracture porosity, fraction.  Typically 0.001–0.02.
    Sw_frac : float
        Water saturation in the fracture network (default 1.0 — fractures
        are commonly fully brine-saturated even when matrix has hydrocarbons).
    m_matrix : float
        Matrix cementation exponent (default 2.0).
    n_matrix : float
        Matrix saturation exponent (default 2.0).
    m_frac : float
        Fracture cementation exponent (default 1.3).
        Open fractures range 1.0–1.5; partially healed fractures 1.5–2.0.
    n_frac : float
        Fracture saturation exponent (default 1.5).
    a_matrix : float
        Tortuosity factor for the matrix (default 1.0).
    a_frac : float
        Tortuosity factor for the fracture network (default 1.0).

    Returns
    -------
    DualPorosityResult

    Notes
    -----
    Fractures dramatically lower the bulk resistivity when they are
    brine-saturated (Sw_frac = 1), because even tiny fracture porosity
    (~0.5 %) creates a very low-resistivity parallel path (F_frac is
    small for m_frac ≈ 1.3).

    For a triple-porosity system (matrix + fractures + vugs) call this
    function twice: first combine matrix + fractures, then combine the
    result with a vug Archie term.

    Examples
    --------
    >>> rw = brine_resistivity(30_000, 60)
    >>> r  = dual_porosity(phi_matrix=0.20, Sw_matrix=0.70, Rw=rw,
    ...                    phi_frac=0.005, Sw_frac=1.0)
    >>> print(f"Rt = {r.Rt:.3f} Ω·m  (matrix alone: {r.Rt_matrix:.2f} Ω·m)")
    """
    if not 0 < phi_matrix <= 1:
        raise ValueError(f"phi_matrix must be in (0,1], got {phi_matrix}")
    if not 0 < phi_frac <= 1:
        raise ValueError(f"phi_frac must be in (0,1], got {phi_frac}")
    if not 0 < Sw_matrix <= 1:
        raise ValueError(f"Sw_matrix must be in (0,1], got {Sw_matrix}")
    if not 0 < Sw_frac <= 1:
        raise ValueError(f"Sw_frac must be in (0,1], got {Sw_frac}")
    if Rw <= 0:
        raise ValueError("Rw must be positive")

    # --- Matrix (Archie)
    F_m      = a_matrix / (phi_matrix ** m_matrix)
    Ro_m     = F_m * Rw
    Rt_m     = Ro_m / (Sw_matrix ** n_matrix)

    # --- Fracture network (Archie with lower m)
    F_f      = a_frac / (phi_frac ** m_frac)
    Ro_f     = F_f * Rw
    Rt_f     = Ro_f / (Sw_frac ** n_frac)

    # --- Parallel combination
    Rt       = 1.0 / (1.0 / Rt_m + 1.0 / Rt_f)
    Ro       = 1.0 / (1.0 / Ro_m + 1.0 / Ro_f)
    RI       = Rt / Ro

    return DualPorosityResult(
        Rw        = Rw,
        Rt_matrix = Rt_m,
        Rt_frac   = Rt_f,
        Rt        = Rt,
        Ro_matrix = Ro_m,
        Ro_frac   = Ro_f,
        Ro        = Ro,
        RI        = RI,
        F_matrix  = F_m,
        F_frac    = F_f,
        Sw_matrix = Sw_matrix,
        Sw_frac   = Sw_frac,
        phi_matrix = phi_matrix,
        phi_frac  = phi_frac,
    )


# ---------------------------------------------------------------------------
# Convenience: invert for Sw from measured Rt
# ---------------------------------------------------------------------------

def solve_Sw_archie(
    Rt:   float,
    phi:  float,
    Rw:   float,
    m:    float = 2.0,
    n:    float = 2.0,
    a:    float = 1.0,
) -> float:
    """
    Invert Archie's law to recover Sw from a measured Rt.

    Sw = (a · Rw / (φ^m · Rt))^(1/n)

    Parameters
    ----------
    Rt : float
        Measured true resistivity in Ω·m.
    phi, Rw, m, n, a : float
        Same as in :func:`archie`.

    Returns
    -------
    float
        Water saturation Sw, clipped to [0, 1].
    """
    F   = a / (phi ** m)
    Ro  = F * Rw
    Sw  = (Ro / Rt) ** (1.0 / n)
    return max(0.0, min(1.0, Sw))


def solve_Sw_simandoux(
    Rt:   float,
    phi:  float,
    Rw:   float,
    Vsh:  float,
    Rsh:  float,
    m:    float = 2.0,
    n:    float = 2.0,
    a:    float = 1.0,
    tol:  float = 1e-8,
    maxiter: int = 200,
) -> float:
    """
    Invert the Simandoux equation for Sw using Newton-Raphson iteration.

    Solves:

        f(Sw) = φ^m · Sw^n / (a · Rw)  +  Vsh · Sw / Rsh  -  1/Rt = 0

    Parameters
    ----------
    Rt : float
        Measured true resistivity in Ω·m.
    phi, Rw, Vsh, Rsh, m, n, a : float
        Same as in :func:`simandoux`.
    tol : float
        Convergence tolerance on |f(Sw)| (default 1e-8).
    maxiter : int
        Maximum Newton iterations (default 200).

    Returns
    -------
    float
        Water saturation Sw in [0, 1].

    Raises
    ------
    RuntimeError
        If Newton-Raphson does not converge within `maxiter` iterations.
    """
    A  = phi ** m / (a * Rw)   # coefficient for Sw^n term
    B  = Vsh / Rsh             # coefficient for Sw term
    C  = 1.0 / Rt

    Sw = 0.5  # initial guess
    for _ in range(maxiter):
        f   = A * Sw ** n + B * Sw - C
        df  = A * n * Sw ** (n - 1) + B
        dSw = -f / df
        Sw  = Sw + dSw
        Sw  = max(1e-6, min(1.0, Sw))
        if abs(f) < tol:
            return Sw

    raise RuntimeError(
        f"solve_Sw_simandoux did not converge after {maxiter} iterations "
        f"(last Sw={Sw:.4f}, residual={abs(f):.2e})"
    )


# ---------------------------------------------------------------------------
# 4. RGPZ permeability model (Glover et al. 2006)
# ---------------------------------------------------------------------------

#: 1 Darcy in m²  (exact SI conversion)
_DARCY_m2 = 9.869_233e-13

#: Packing constant for quasi-spherical grains (= 8/3)
PACKING_SPHERICAL: float = 8.0 / 3.0


@dataclass
class RGPZResult:
    """Results from the RGPZ permeability model."""
    k_m2:        float   # permeability in m²
    k_mD:        float   # permeability in milliDarcy
    F:           float   # formation factor  φ^-m
    S:           float   # connectivity  φ^m  (= 1/F)
    tortuosity:  float   # electrical tortuosity  T = F · φ
    d_geom_um:   float   # grain diameter used (µm, input echo)
    phi:         float   # porosity (input echo)
    m:           float   # cementation exponent (input echo)
    a_pack:      float   # packing constant (input echo)


def rgpz(
    phi:       float,
    d_geom_um: float,
    m:         float = 2.0,
    a_pack:    float = PACKING_SPHERICAL,
) -> RGPZResult:
    """
    RGPZ permeability model (Revil, Glover, Pezard & Zamora 1999/2006).

    Derives permeability from porosity, cementation exponent, and grain
    size — quantities accessible from resistivity logs plus a grain-size
    measurement (sieve, laser diffraction, or image analysis).

    Governing equation
    ------------------

        k = d̄² · φ^(3m) / (4 · a · m²)          [m²]

    where

        φ^(3m) = φ^m · φ^m · φ^m = S³   (connectivity cubed)

    Equivalently, using the formation factor F = φ^(-m):

        k = d̄² / (4 · a · m² · F³)

    Key derived quantities
    ----------------------
    Formation factor  F = φ^(-m)   (Archie's first law, a = 1)
    Connectivity      S = φ^m  = 1/F
    Tortuosity        T = F · φ

    Parameters
    ----------
    phi : float
        Connected porosity, fraction  0 < φ ≤ 1.
    d_geom_um : float
        **Geometric mean** grain diameter in µm.  Glover et al. (2006)
        demonstrated that the geometric mean gives significantly better
        predictions than the arithmetic mean.  For unimodal, roughly
        log-normal grain size distributions convert from sieve data via:
            d_geom = exp(mean(ln(d_i) · f_i))
        where d_i are sieve midpoints and f_i the mass fractions.
    m : float
        Cementation exponent (default 2.0).  The same m used in Archie's
        formation-factor law F = φ^(-m).  Can be derived from resistivity
        and porosity logs, or from MICP.
    a_pack : float
        Packing constant (default 8/3 ≈ 2.667 for quasi-spherical grains,
        following Glover et al. 2006).  For angular grains or other
        geometries this may be fitted to core data.

    Returns
    -------
    RGPZResult
        Dataclass containing k in m² and mD, plus F, S, tortuosity, and
        echoed inputs.

    Raises
    ------
    ValueError
        If any parameter is outside its physical range.

    Notes
    -----
    The model inherently accounts for dead-end and unconnected porosity
    through the φ^(3m) term, which is why it outperforms Kozeny-Carman
    (which uses φ³/(1-φ)²) for tight and heterogeneous rocks.

    The Kozeny-Carman equation for comparison:

        k_KC = d̄² · φ³ / (72 · τ · (1 - φ)²)

    where τ is tortuosity (≈ 2.5 for random sphere packs).

    Unit conversion: 1 mD = 9.869 × 10⁻¹⁶ m²

    Examples
    --------
    >>> r = rgpz(phi=0.20, d_geom_um=150.0, m=2.0)
    >>> print(f"k = {r.k_mD:.2f} mD")

    >>> # Derive m from a resistivity + porosity measurement:
    >>> import math
    >>> F_measured = 28.5          # from Rt / Rw at Sw = 1
    >>> phi_core   = 0.18
    >>> m_fit      = -math.log(F_measured) / math.log(phi_core)
    >>> r2 = rgpz(phi=phi_core, d_geom_um=120.0, m=m_fit)
    >>> print(f"m = {m_fit:.3f},  k = {r2.k_mD:.2f} mD")
    """
    if not 0 < phi <= 1:
        raise ValueError(f"phi must be in (0, 1], got {phi}")
    if d_geom_um <= 0:
        raise ValueError(f"d_geom_um must be positive, got {d_geom_um}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if a_pack <= 0:
        raise ValueError(f"a_pack must be positive, got {a_pack}")

    d_m   = d_geom_um * 1e-6          # µm → m
    k_m2  = (d_m ** 2) * (phi ** (3 * m)) / (4.0 * a_pack * m ** 2)
    k_mD  = k_m2 / _DARCY_m2 * 1000.0

    F          = phi ** (-m)
    S          = phi ** m              # connectivity = 1/F
    tortuosity = F * phi               # T = F·φ

    return RGPZResult(
        k_m2       = k_m2,
        k_mD       = k_mD,
        F          = F,
        S          = S,
        tortuosity = tortuosity,
        d_geom_um  = d_geom_um,
        phi        = phi,
        m          = m,
        a_pack     = a_pack,
    )


def rgpz_from_formation_factor(
    F:         float,
    phi:       float,
    d_geom_um: float,
    a_pack:    float = PACKING_SPHERICAL,
) -> RGPZResult:
    """
    RGPZ permeability using a *measured* formation factor F directly.

    This avoids fitting m explicitly: it derives m = -ln(F)/ln(φ) and
    then calls :func:`rgpz`.  Useful when F is read directly from a
    resistivity log (F = Rt/Rw at Sw = 1) and core porosity is known.

    Parameters
    ----------
    F : float
        Measured formation factor F = Ro / Rw  (dimensionless, > 1).
    phi : float
        Porosity, fraction.
    d_geom_um : float
        Geometric mean grain diameter in µm.
    a_pack : float
        Packing constant (default 8/3).

    Returns
    -------
    RGPZResult
        Same as :func:`rgpz`.

    Examples
    --------
    >>> r = rgpz_from_formation_factor(F=28.5, phi=0.18, d_geom_um=120.0)
    >>> print(f"m = {r.m:.3f},  k = {r.k_mD:.2f} mD")
    """
    if F <= 1:
        raise ValueError(f"Formation factor F must be > 1, got {F}")
    if not 0 < phi < 1:
        raise ValueError(f"phi must be in (0, 1), got {phi}")

    m_derived = -math.log(F) / math.log(phi)
    return rgpz(phi=phi, d_geom_um=d_geom_um, m=m_derived, a_pack=a_pack)


def kozeny_carman(
    phi:       float,
    d_geom_um: float,
    tau:       float = 2.5,
) -> float:
    """
    Kozeny-Carman permeability for reference / comparison with RGPZ.

        k = d̄² · φ³ / (72 · τ · (1 - φ)²)     [mD]

    Parameters
    ----------
    phi : float
        Porosity, fraction.
    d_geom_um : float
        Grain diameter in µm.
    tau : float
        Tortuosity (default 2.5, typical for random sphere packs).

    Returns
    -------
    float
        Permeability in milliDarcy.
    """
    d_m  = d_geom_um * 1e-6
    k_m2 = (d_m ** 2) * phi ** 3 / (72.0 * tau * (1.0 - phi) ** 2)
    return k_m2 / _DARCY_m2 * 1000.0




# ---------------------------------------------------------------------------
# Glover conductivity mixing laws
# ---------------------------------------------------------------------------

@dataclass
class GloverTwoPhaseResult:
    """Results for the modified Archie law of Glover et al. (2000)."""
    sigma_bulk: float
    resistivity_bulk: float
    sigma_1: float
    sigma_2: float
    phi_1: float
    phi_2: float
    p: float
    m: float


def glover_modified_archie_two_phase(
    sigma_1: float,
    sigma_2: float,
    phi_2: float,
    m: float = 2.0,
    p: Optional[float] = None,
) -> GloverTwoPhaseResult:
    """
    Modified Archie law for two conducting phases (Glover et al., 2000).

    The model is

        sigma_bulk = sigma_1 * phi_1**p + sigma_2 * phi_2**m

    with

        phi_1 = 1 - phi_2
        p = ln(1 - phi_2**m) / ln(1 - phi_2)

    when `p` is not supplied explicitly.
    """
    if sigma_1 < 0 or sigma_2 < 0:
        raise ValueError("sigma_1 and sigma_2 must be non-negative")
    if not 0 < phi_2 < 1:
        raise ValueError(f"phi_2 must be in (0, 1), got {phi_2}")
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")

    phi_1 = 1.0 - phi_2
    if p is None:
        p = math.log(1.0 - phi_2 ** m) / math.log(1.0 - phi_2)
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")

    sigma_bulk = sigma_1 * (phi_1 ** p) + sigma_2 * (phi_2 ** m)
    return GloverTwoPhaseResult(
        sigma_bulk=sigma_bulk,
        resistivity_bulk=(1.0 / sigma_bulk) if sigma_bulk > 0 else math.inf,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        phi_1=phi_1,
        phi_2=phi_2,
        p=p,
        m=m,
    )


@dataclass
class GloverNPhaseResult:
    """Results for the generalized Archie law of Glover (2010)."""
    sigma_bulk: float
    resistivity_bulk: float
    sigmas: list[float]
    fractions: list[float]
    exponents: list[float]


def glover_generalized_archie_n_phase(
    sigmas: list[float],
    fractions: list[float],
    exponents: list[float],
) -> GloverNPhaseResult:
    """
    Generalized Archie law for n conducting phases (Glover, 2010).

        sigma_bulk = sum_i sigma_i * phi_i**m_i

    This is the explicit no-Stieltjes-integral form discussed by Glover
    (2010), i.e. the same approximation level adopted in the two-phase paper.
    """
    if not (len(sigmas) == len(fractions) == len(exponents)):
        raise ValueError("sigmas, fractions, and exponents must have the same length")
    if len(sigmas) == 0:
        raise ValueError("At least one phase must be provided")
    if any(s < 0 for s in sigmas):
        raise ValueError("All conductivities must be non-negative")
    if any(f < 0 for f in fractions):
        raise ValueError("All fractions must be non-negative")
    if any(m <= 0 for m in exponents):
        raise ValueError("All exponents must be positive")

    fsum = sum(fractions)
    if not math.isclose(fsum, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"fractions must sum to 1.0, got {fsum}")

    sigma_bulk = sum(s * (f ** m) for s, f, m in zip(sigmas, fractions, exponents))
    return GloverNPhaseResult(
        sigma_bulk=sigma_bulk,
        resistivity_bulk=(1.0 / sigma_bulk) if sigma_bulk > 0 else math.inf,
        sigmas=list(sigmas),
        fractions=list(fractions),
        exponents=list(exponents),
    )


# ---------------------------------------------------------------------------
# 5. Hashin-Shtrikman bounds on effective conductivity
# ---------------------------------------------------------------------------

@dataclass
class HSBoundsResult:
    """
    Hashin-Shtrikman upper and lower conductivity bounds for a two-phase
    composite, plus the HS mixing-formula estimate.

    Naming convention (conductivity space → resistivity space):
        sigma_upper / R_lower  — HS upper bound on σ  = lower bound on R
        sigma_lower / R_upper  — HS lower bound on σ  = upper bound on R
    """
    sigma_upper:  float   # HS upper bound on σ_eff  [S/m]
    sigma_lower:  float   # HS lower bound on σ_eff  [S/m]
    sigma_hs_mix: float   # HS mixing formula (geometric mean)  [S/m]
    R_upper:      float   # highest resistivity bound (= 1/sigma_lower)  [Ω·m]
    R_lower:      float   # lowest  resistivity bound (= 1/sigma_upper)  [Ω·m]
    R_hs_mix:     float   # HS mixing resistivity  [Ω·m]
    R_parallel:   float   # Voigt / parallel bound  [Ω·m]
    R_series:     float   # Reuss / series bound  [Ω·m]
    sigma_1:      float   # phase-1 conductivity (input echo)  [S/m]
    sigma_2:      float   # phase-2 conductivity (input echo)  [S/m]
    f1:           float   # volume fraction of phase 1 (input echo)
    f2:           float   # volume fraction of phase 2 (input echo)


def hashin_shtrikman_two_phase(
    sigma_1: float,
    sigma_2: float,
    f1:      float,
) -> HSBoundsResult:
    """
    Hashin-Shtrikman (1962) conductivity bounds for a two-phase composite.

    Returns the HS upper bound, HS lower bound, and a geometric mixing
    estimate, together with the classical Voigt (parallel) and Reuss
    (series) arithmetic bounds for comparison.

    Background
    ----------
    Hashin & Shtrikman (1962) derived the tightest possible bounds on the
    effective conductivity σ_eff of a statistically isotropic two-phase
    composite, given only the phase conductivities and volume fractions —
    without any assumption about microgeometry.

    For a composite with phase conductivities σ₁ ≤ σ₂ and volume
    fractions f₁ + f₂ = 1:

    HS lower bound (phase 2 = matrix, phase 1 = inclusions):

        σ_HS⁻ = σ₁ + f₂ / [1/(σ₂ - σ₁) + f₁/(3σ₁)]

    HS upper bound (phase 1 = matrix, phase 2 = inclusions):

        σ_HS⁺ = σ₂ + f₁ / [1/(σ₁ - σ₂) + f₂/(3σ₂)]

    The HS bounds are always tighter than the Voigt-Reuss (parallel-series)
    bounds:

        σ_Reuss  ≤  σ_HS⁻  ≤  σ_eff  ≤  σ_HS⁺  ≤  σ_Voigt

    Petrophysical context
    ---------------------
    For a rock with two phases (e.g. pore fluid and mineral matrix):

        Phase 1 = pore fluid  (higher σ, e.g. brine:  ~5 S/m)
        Phase 2 = mineral     (lower σ,  e.g. quartz: ~1e-9 S/m)
        f₁      = porosity φ

    The HS bounds then bracket the effective formation conductivity for
    any pore geometry.  Archie's law should fall inside the HS bounds; if
    it does not, the cementation exponent m is likely inconsistent with
    an isotropic composite assumption.

    HS mixing formula
    -----------------
    Glover, Hole & Pous (2000) proposed a self-consistent HS mixing
    formula that interpolates between bounds using the volume fractions
    directly:

        σ_mix = σ₁^f₁ · σ₂^f₂      (geometric mean)

    This is equivalent to the HS mixing model for the special case of
    equal volume fractions and gives a single-valued estimate inside
    the HS envelope.

    Parameters
    ----------
    sigma_1 : float
        Conductivity of phase 1 in S/m  (e.g. pore fluid).
    sigma_2 : float
        Conductivity of phase 2 in S/m  (e.g. mineral matrix).
    f1 : float
        Volume fraction of phase 1 (0 < f1 < 1).
        f2 is computed as 1 - f1.

    Returns
    -------
    HSBoundsResult
        Upper bound, lower bound, mixing estimate, Voigt and Reuss bounds —
        all in both S/m (conductivity) and Ω·m (resistivity).

    Raises
    ------
    ValueError
        If conductivities are non-positive or volume fractions are out of range.

    Notes
    -----
    The formula requires σ₁ ≠ σ₂.  If the two phases have the same
    conductivity, σ_eff equals that conductivity regardless of geometry
    and the function returns that value for all bounds.

    For N > 2 phases use :func:`hashin_shtrikman_n_phase`.

    Examples
    --------
    >>> # Brine-saturated sandstone: φ = 0.20, σ_fluid = 5 S/m, σ_grain ≈ 0
    >>> r = hashin_shtrikman_two_phase(sigma_1=5.0, sigma_2=1e-9, f1=0.20)
    >>> print(f"HS+  = {r.R_upper:.4f} Ω·m")
    >>> print(f"HS-  = {r.R_lower:.6f} Ω·m")

    >>> # Cross-check: Archie Rt should lie between HS- and HS+
    >>> rw = 1/5.0                    # 0.20 Ω·m
    >>> ar = archie(phi=0.20, Sw=1.0, Rw=rw, m=2.0, n=2.0)
    >>> assert r.R_lower <= ar.Rt <= r.R_upper, "Archie outside HS bounds!"
    """
    if sigma_1 <= 0 or sigma_2 <= 0:
        raise ValueError("Both phase conductivities must be positive (S/m)")
    if not 0.0 < f1 < 1.0:
        raise ValueError(f"f1 must be strictly in (0, 1), got {f1}")

    f2 = 1.0 - f1

    # --- Voigt (parallel) and Reuss (series) bounds ---
    sigma_voigt = f1 * sigma_1 + f2 * sigma_2
    sigma_reuss = 1.0 / (f1 / sigma_1 + f2 / sigma_2)

    if math.isclose(sigma_1, sigma_2, rel_tol=1e-10):
        # Degenerate: both phases identical
        s = sigma_1
        return HSBoundsResult(
            sigma_upper=s, sigma_lower=s, sigma_hs_mix=s,
            R_upper=1/s, R_lower=1/s, R_hs_mix=1/s,
            R_parallel=1/s, R_series=1/s,
            sigma_1=sigma_1, sigma_2=sigma_2, f1=f1, f2=f2,
        )

    # Ensure σ_lo ≤ σ_hi for consistent bound labelling
    s_lo, s_hi = min(sigma_1, sigma_2), max(sigma_1, sigma_2)
    f_lo = f2 if sigma_1 > sigma_2 else f1   # volume fraction of the LOW-σ phase
    f_hi = 1.0 - f_lo

    # --- HS lower bound  (low-σ phase is the matrix) ---
    # σ_HS⁻ = σ_lo + f_hi / [1/(σ_hi - σ_lo) + f_lo/(3·σ_lo)]
    hs_lower = s_lo + f_hi / (1.0 / (s_hi - s_lo) + f_lo / (3.0 * s_lo))

    # --- HS upper bound  (high-σ phase is the matrix) ---
    # σ_HS⁺ = σ_hi + f_lo / [1/(σ_lo - σ_hi) + f_hi/(3·σ_hi)]
    hs_upper = s_hi + f_lo / (1.0 / (s_lo - s_hi) + f_hi / (3.0 * s_hi))

    # --- HS geometric mixing (Glover et al. 2000) ---
    hs_mix = (sigma_1 ** f1) * (sigma_2 ** f2)

    def safe_R(s: float) -> float:
        return 1.0 / s if s > 0 else math.inf

    # HS+ is the upper bound on conductivity → lower bound on resistivity
    # HS- is the lower bound on conductivity → upper bound on resistivity
    return HSBoundsResult(
        sigma_upper  = hs_upper,
        sigma_lower  = hs_lower,
        sigma_hs_mix = hs_mix,
        R_upper      = safe_R(hs_lower),   # highest R ← lowest σ
        R_lower      = safe_R(hs_upper),   # lowest  R ← highest σ
        R_hs_mix     = safe_R(hs_mix),
        R_parallel   = safe_R(sigma_voigt),
        R_series     = safe_R(sigma_reuss),
        sigma_1      = sigma_1,
        sigma_2      = sigma_2,
        f1           = f1,
        f2           = f2,
    )


@dataclass
class HSNPhaseResult:
    """
    Results from the N-phase Hashin-Shtrikman generalisation.

    Naming convention (conductivity space → resistivity space):
        sigma_upper / R_lower  — HS upper bound on σ  = lower bound on R
        sigma_lower / R_upper  — HS lower bound on σ  = upper bound on R
    """
    sigma_upper:  float          # HS upper bound on σ_eff  [S/m]
    sigma_lower:  float          # HS lower bound on σ_eff  [S/m]
    R_upper:      float          # highest R bound (= 1/sigma_lower)  [Ω·m]
    R_lower:      float          # lowest  R bound (= 1/sigma_upper)  [Ω·m]
    R_parallel:   float          # Voigt bound     [Ω·m]
    R_series:     float          # Reuss bound     [Ω·m]
    sigmas:       list[float]    # input conductivities (echo)
    fractions:    list[float]    # input volume fractions (echo)


def hashin_shtrikman_n_phase(
    sigmas:    list[float],
    fractions: list[float],
) -> HSNPhaseResult:
    """
    Generalised Hashin-Shtrikman bounds for an N-phase isotropic composite.

    Uses the Berryman (1995) / Milton (1981) N-phase HS formulation:

        σ_HS± = reference phase conductivity + correction sum

    The HS bounds are computed by choosing the reference phase σ_ref
    as the minimum conductivity phase (lower bound) or the maximum
    conductivity phase (upper bound):

        σ_HS = σ_ref  +  [Σᵢ fᵢ / (1/(σᵢ - σ_ref) + 1/(d·σ_ref))]⁻¹

    where d = 3 for three-dimensional composites (isotropic), and the
    sum excludes the reference phase itself (its contribution vanishes).

    Parameters
    ----------
    sigmas : list[float]
        Conductivity of each phase in S/m.  Length N ≥ 2.
    fractions : list[float]
        Volume fraction of each phase.  Must sum to 1.  Length N.

    Returns
    -------
    HSNPhaseResult

    Raises
    ------
    ValueError
        If lengths differ, fractions do not sum to 1, or any value is
        out of physical range.

    Examples
    --------
    >>> # Three-phase: brine pores + quartz + clay
    >>> r = hashin_shtrikman_n_phase(
    ...     sigmas    = [5.0,   1e-9, 0.05],
    ...     fractions = [0.20,  0.70, 0.10],
    ... )
    >>> print(f"HS+ = {r.R_upper:.4f} Ω·m,  HS- = {r.R_lower:.6f} Ω·m")
    """
    if len(sigmas) != len(fractions):
        raise ValueError("sigmas and fractions must have the same length")
    if len(sigmas) < 2:
        raise ValueError("Need at least 2 phases")
    if any(s <= 0 for s in sigmas):
        raise ValueError("All conductivities must be positive")
    if any(f < 0 for f in fractions):
        raise ValueError("All volume fractions must be ≥ 0")

    total_f = sum(fractions)
    if not math.isclose(total_f, 1.0, rel_tol=1e-6):
        raise ValueError(f"Volume fractions must sum to 1, got {total_f:.6f}")

    d = 3  # spatial dimension (isotropic 3-D)

    def _hs_bound(sigma_ref: float) -> float:
        """HS bound with a given reference-phase conductivity."""
        # Σ fᵢ / (1/(σᵢ - σ_ref) + 1/(d·σ_ref))
        # = Σ fᵢ · (σᵢ - σ_ref) · d·σ_ref / (d·σ_ref + σᵢ - σ_ref)
        numerator = 0.0
        for s_i, f_i in zip(sigmas, fractions):
            if math.isclose(s_i, sigma_ref, rel_tol=1e-12):
                continue   # reference phase: term is 0/0 → 0
            numerator += f_i / (1.0 / (s_i - sigma_ref) + 1.0 / (d * sigma_ref))
        return sigma_ref + numerator

    sigma_lower = _hs_bound(min(sigmas))
    sigma_upper = _hs_bound(max(sigmas))

    # For very high conductivity contrasts (many decades) the polynomial
    # N-phase HS formula can return a slightly negative sigma_upper due to
    # floating-point cancellation.  Clamp to zero (physically: the true
    # upper bound approaches the Voigt value, which is always valid).
    sigma_lower = max(sigma_lower, 0.0)
    sigma_upper = max(sigma_upper, 0.0)

    # Voigt and Reuss
    sigma_voigt = sum(f * s for f, s in zip(fractions, sigmas))
    sigma_reuss = 1.0 / sum(f / s for f, s in zip(fractions, sigmas))

    def safe_R(s: float) -> float:
        return 1.0 / s if s > 0 else math.inf

    return HSNPhaseResult(
        sigma_upper = sigma_upper,
        sigma_lower = sigma_lower,
        R_upper     = safe_R(sigma_lower),   # highest R ← lowest σ
        R_lower     = safe_R(sigma_upper),   # lowest  R ← highest σ
        R_parallel  = safe_R(sigma_voigt),
        R_series    = safe_R(sigma_reuss),
        sigmas      = list(sigmas),
        fractions   = list(fractions),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Petrophysical models — quick demo")
    print("=" * 60)

    # Common inputs
    salinity = 30_000   # ppm NaCl
    temp     = 60.0     # °C
    phi      = 0.20
    Sw       = 0.70
    m, n     = 2.0, 2.0

    # --- Brine conductivity comparison ---
    print("\n--- Brine conductivity / resistivity ---")
    sw_sg  = brine_conductivity_sen_goode(salinity, temp)
    rw_sg  = brine_resistivity_sen_goode(salinity, temp)
    rw_arps = brine_resistivity(salinity, temp)
    print(f"  Salinity = {salinity:,} ppm NaCl,  T = {temp} °C")
    print(f"  Sen & Goode (1992):  σ_w = {sw_sg:.4f} S/m,  Rw = {rw_sg:.4f} Ω·m")
    print(f"  Arps/Hilchie:                            Rw = {rw_arps:.4f} Ω·m")
    print(f"  Difference: {abs(rw_sg - rw_arps)/rw_sg*100:.1f} %")

    rho_demo = 0.90
    sk = brine_conductivity_sinmyo_keppler(5.0, 400.0, rho_demo, pressure_gpa=0.8)
    gk = brine_conductivity_guo_keppler(5.0, 700.0, 1.15, pressure_gpa=3.0)

    print("\n  -- σ_w vs T  (30,000 ppm NaCl) --")
    print(f"  {'T (°C)':>8}  {'σ_w S&G (S/m)':>14}  {'Rw S&G (Ω·m)':>14}  {'Rw Arps (Ω·m)':>14}")
    for t in [25, 60, 100, 150, 200]:
        sg  = brine_conductivity_sen_goode(salinity, t)
        rsg = 1.0 / sg
        ra  = brine_resistivity(salinity, t)
        print(f"  {t:>8}  {sg:>14.4f}  {rsg:>14.4f}  {ra:>14.4f}")

    print("\n  -- σ_w vs salinity  (T = 60 °C) --")
    print(f"  {'ppm':>10}  {'σ_w (S/m)':>12}  {'Rw (Ω·m)':>12}")
    for sal in [1_000, 5_000, 15_000, 30_000, 80_000, 150_000, 250_000]:
        sg = brine_conductivity_sen_goode(sal, 60)
        print(f"  {sal:>10,}  {sg:>12.4f}  {1/sg:>12.5f}")

    # Use Sen & Goode Rw for all petrophysical models below
    Rw = rw_sg
    print(f"\n  Using Sen & Goode Rw = {Rw:.4f} Ω·m for models below.")

    # 1. Archie
    print("\n--- Archie (clean formation) ---")
    ar = archie(phi=phi, Sw=Sw, Rw=Rw, m=m, n=n)
    print(f"  F  = {ar.F:.2f}")
    print(f"  Ro = {ar.Ro:.3f} Ω·m")
    print(f"  Rt = {ar.Rt:.3f} Ω·m")
    print(f"  RI = {ar.RI:.3f}")

    # Round-trip inversion
    Sw_inv = solve_Sw_archie(Rt=ar.Rt, phi=phi, Rw=Rw, m=m, n=n)
    print(f"  Sw (inverted from Rt) = {Sw_inv:.4f}  [input was {Sw}]")

    # 2. Simandoux
    print("\n--- Simandoux (shaly sand) ---")
    Vsh, Rsh = 0.15, 3.0
    sr = simandoux(phi=phi, Sw=Sw, Rw=Rw, Vsh=Vsh, Rsh=Rsh, m=m, n=n)
    print(f"  Vsh = {Vsh},  Rsh = {Rsh} Ω·m")
    print(f"  F   = {sr.F:.2f}  (clean-sand factor)")
    print(f"  Ro  = {sr.Ro:.3f} Ω·m")
    print(f"  Rt  = {sr.Rt:.3f} Ω·m")
    print(f"  RI  = {sr.RI:.3f}")

    Sw_sim = solve_Sw_simandoux(
        Rt=sr.Rt, phi=phi, Rw=Rw, Vsh=Vsh, Rsh=Rsh, m=m, n=n
    )
    print(f"  Sw (Newton-Raphson inversion) = {Sw_sim:.4f}  [input was {Sw}]")

    # 3. Dual-porosity
    print("\n--- Dual-porosity / fractured ---")
    dr = dual_porosity(
        phi_matrix=phi, Sw_matrix=Sw, Rw=Rw,
        phi_frac=0.005, Sw_frac=1.0,
        m_matrix=m, n_matrix=n, m_frac=1.3, n_frac=1.5,
    )
    print(f"  Matrix:   F={dr.F_matrix:.1f}  Ro={dr.Ro_matrix:.3f}  Rt={dr.Rt_matrix:.3f} Ω·m")
    print(f"  Fracture: F={dr.F_frac:.1f}  Ro={dr.Ro_frac:.3f}  Rt={dr.Rt_frac:.3f} Ω·m")
    print(f"  Combined (parallel):  Rt={dr.Rt:.3f} Ω·m")
    print(f"  RI = {dr.RI:.3f}")
    print(f"  Fracture short-circuit: Rt reduced by "
          f"{100*(dr.Rt_matrix - dr.Rt)/dr.Rt_matrix:.1f} % vs matrix alone")

    # 4. RGPZ
    print("\n--- RGPZ permeability (Glover et al. 2006) ---")
    d_geom = 150.0   # µm geometric mean grain diameter

    rr = rgpz(phi=phi, d_geom_um=d_geom, m=m)
    print(f"  d_geom = {d_geom} µm,  φ = {phi},  m = {m}")
    print(f"  F          = {rr.F:.2f}   (= φ^-m)")
    print(f"  Connectivity S = {rr.S:.5f}  (= φ^m)")
    print(f"  Tortuosity T   = {rr.tortuosity:.3f}  (= F·φ)")
    print(f"  k (RGPZ)   = {rr.k_mD:.4f} mD")

    kc = kozeny_carman(phi=phi, d_geom_um=d_geom)
    print(f"  k (K-C)    = {kc:.4f} mD  [for comparison]")
    print(f"  Ratio RGPZ/KC = {rr.k_mD/kc:.3f}")

    # From a measured formation factor
    print("\n  -- From measured F directly --")
    F_meas = archie(phi=phi, Sw=1.0, Rw=Rw, m=m).F   # simulated log reading
    rr2 = rgpz_from_formation_factor(F=F_meas, phi=phi, d_geom_um=d_geom)
    print(f"  F (measured) = {F_meas:.2f}  →  m derived = {rr2.m:.4f}")
    print(f"  k (RGPZ)     = {rr2.k_mD:.4f} mD")

    # Sensitivity: effect of m on k
    print("\n  -- k sensitivity to m (d=150 µm, φ=0.20) --")
    print(f"  {'m':>5}  {'F':>8}  {'k (mD)':>12}")
    for m_i in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
        ri = rgpz(phi=phi, d_geom_um=d_geom, m=m_i)
        print(f"  {m_i:>5.1f}  {ri.F:>8.2f}  {ri.k_mD:>12.5f}")

    # 5. Hashin-Shtrikman bounds
    print("\n--- Hashin-Shtrikman bounds ---")

    # Two-phase: brine-filled pores (σ_fluid) + quartz matrix (σ_grain ≈ 0)
    sigma_fluid = 1.0 / Rw          # S/m from Sen & Goode Rw
    sigma_grain = 1e-9              # quartz ≈ insulating
    print(f"\n  Two-phase (brine pores + quartz matrix), φ = {phi}")
    print(f"  σ_fluid = {sigma_fluid:.4f} S/m,  σ_quartz ≈ {sigma_grain:.0e} S/m")
    hs2 = hashin_shtrikman_two_phase(sigma_1=sigma_fluid, sigma_2=sigma_grain, f1=phi)
    print(f"  Reuss  (series,  highest R) : {hs2.R_series:.4f} Ω·m")
    print(f"  HS−    (upper R bound)      : {hs2.R_upper:.4f} Ω·m")
    print(f"  HS mix (geometric mean)     : {hs2.R_hs_mix:.4f} Ω·m")
    print(f"  HS+    (lower R bound)      : {hs2.R_lower:.4f} Ω·m")
    print(f"  Voigt  (parallel, lowest R) : {hs2.R_parallel:.4f} Ω·m")

    # Cross-check: Archie Ro should lie between HS+ and HS- (in R space)
    Ro_archie = archie(phi=phi, Sw=1.0, Rw=Rw, m=m).Ro
    inside = hs2.R_lower <= Ro_archie <= hs2.R_upper
    print(f"  Archie Ro = {Ro_archie:.4f} Ω·m  →  within HS bounds: {inside}")

    # Three-phase: brine + quartz + clay
    print(f"\n  Three-phase (brine + quartz + clay), φ=0.20, Vclay=0.10")
    sigma_clay = 0.05   # S/m — clay is moderately conductive
    hs3 = hashin_shtrikman_n_phase(
        sigmas    = [sigma_fluid, sigma_grain, sigma_clay],
        fractions = [0.20,        0.70,        0.10],
    )
    print(f"  σ_clay = {sigma_clay} S/m")
    print(f"  Reuss  (series)      : {hs3.R_series:.4f} Ω·m")
    print(f"  HS−    (upper R)     : {hs3.R_upper:.4f} Ω·m")
    print(f"  HS+    (lower R)     : {hs3.R_lower:.4f} Ω·m")
    print(f"  Voigt  (parallel)    : {hs3.R_parallel:.4f} Ω·m")

    # Sweep: show how bounds narrow as σ_grain increases (e.g. conductive minerals)
    print(f"\n  -- HS bounds vs mineral conductivity (φ=0.20, σ_fluid={sigma_fluid:.2f} S/m) --")
    print(f"  {'σ_grain (S/m)':>16}  {'HS+ / R_lower (Ω·m)':>22}  {'HS- / R_upper (Ω·m)':>22}")
    for sg in [1e-9, 1e-6, 1e-3, 0.01, 0.1, 1.0]:
        h = hashin_shtrikman_two_phase(sigma_1=sigma_fluid, sigma_2=sg, f1=phi)
        print(f"  {sg:>16.0e}  {h.R_lower:>22.4f}  {h.R_upper:>22.4f}")

    print("\n" + "=" * 60)
