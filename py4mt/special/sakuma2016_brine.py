"""sakuma2016_brine.py

Python implementation of the H2O-NaCl brine electrical conductivity
parameterizations used in the supplied MATLAB files.

This module provides two functions:
- sigbrine_high: high-temperature / pressure-dependent parameterization
- sigbrine_low: low-temperature parameterization

The implementation follows the coefficients and equations in the uploaded
MATLAB source files derived from Sakuma and Ichiki (2016).

Author: Volker Rath (DIAS)
Created with the help of ChatGPT (GPT-5 Thinking) on 2026-04-11
"""

from __future__ import annotations

import numpy as np


ArrayLike = float | int | np.ndarray


def _as_float_array(x: ArrayLike) -> np.ndarray:
    """Convert input to a NumPy float array.

    Parameters
    ----------
    x : float, int, or numpy.ndarray
        Scalar or array-like input value.

    Returns
    -------
    numpy.ndarray
        Input converted to a NumPy array with dtype float.
    """
    return np.asarray(x, dtype=float)



def sigbrine_high(T_c: ArrayLike, C_wt: ArrayLike, P_mpa: ArrayLike) -> np.ndarray:
    """Compute H2O-NaCl brine conductivity for the high-temperature model.

    This function is a direct Python translation of the supplied MATLAB
    function ``sigbrine_high.m``. It uses the pressure-dependent polynomial
    parameterization with coefficients attributed there to Sakuma and Ichiki
    (2016).

    Parameters
    ----------
    T_c : float, int, or numpy.ndarray
        Temperature in degrees Celsius.
    C_wt : float, int, or numpy.ndarray
        NaCl concentration in weight percent.
    P_mpa : float, int, or numpy.ndarray
        Pressure in MPa.

    Returns
    -------
    numpy.ndarray
        Electrical conductivity in S/m.

    Notes
    -----
    The MATLAB source converts temperature from Celsius to Kelvin-like units
    via ``T = 273.16 + T``; the same convention is preserved here.
    """
    T_c = _as_float_array(T_c)
    C_wt = _as_float_array(C_wt)
    P_mpa = _as_float_array(P_mpa)

    # Coefficients from the uploaded MATLAB file sigbrine_high.m
    g111 = -2.76823e-12
    g112 = 2.86668e-11
    g113 = -1.01120e-11

    g121 = 6.32515e-9
    g122 = -6.35950e-8
    g123 = 2.14326e-8

    g131 = -2.92588e-6
    g132 = 2.69121e-5
    g133 = -9.20740e-6

    g211 = 6.52051e-9
    g212 = -7.43514e-8
    g213 = 2.23618e-8

    g221 = -1.47966e-5
    g222 = 1.67038e-4
    g223 = -4.54299e-5

    g231 = 6.88977e-3
    g232 = -7.25629e-3
    g233 = 1.89836e-2

    g311 = -2.60077e-6
    g312 = 3.64027e-5
    g313 = -7.50611e-6

    g321 = 6.12874e-3
    g322 = -9.01143e-2
    g323 = 1.51621e-2

    g331 = -3.17282e0
    g332 = 5.2186e1
    g333 = -6.22277e0

    C2 = C_wt * C_wt

    b11 = g111 * C2 + g112 * C_wt + g113
    b12 = g121 * C2 + g122 * C_wt + g123
    b13 = g131 * C2 + g132 * C_wt + g133
    b21 = g211 * C2 + g212 * C_wt + g213
    b22 = g221 * C2 + g222 * C_wt + g223
    b23 = g231 * C2 + g232 * C_wt + g233
    b31 = g311 * C2 + g312 * C_wt + g313
    b32 = g321 * C2 + g322 * C_wt + g323
    b33 = g331 * C2 + g332 * C_wt + g333

    T = 273.16 + T_c
    T2 = T * T

    a1 = b11 * T2 + b12 * T + b13
    a2 = b21 * T2 + b22 * T + b23
    a3 = b31 * T2 + b32 * T + b33

    P2 = P_mpa * P_mpa

    sigma = a1 * P2 + a2 * P_mpa + a3
    return sigma



def sigbrine_low(T_c: ArrayLike, C_wt: ArrayLike) -> np.ndarray:
    """Compute H2O-NaCl brine conductivity for the low-temperature model.

    This function is a direct Python translation of the supplied MATLAB
    function ``sigbrine_low.m``. It uses the low-temperature polynomial
    parameterization with coefficients attributed there to Sakuma and Ichiki
    (2016).

    Parameters
    ----------
    T_c : float, int, or numpy.ndarray
        Temperature in degrees Celsius.
    C_wt : float, int, or numpy.ndarray
        NaCl concentration in weight percent.

    Returns
    -------
    numpy.ndarray
        Electrical conductivity in S/m.

    Notes
    -----
    The MATLAB source converts temperature from Celsius to Kelvin-like units
    via ``T = 273.16 + T``; the same convention is preserved here.
    """
    T_c = _as_float_array(T_c)
    C_wt = _as_float_array(C_wt)

    # Coefficients from the uploaded MATLAB file sigbrine_low.m
    f11 = -1.61994e-12
    f12 = 4.32808e-11
    f13 = 1.15235e-11
    f14 = 2.52257e-10
    f21 = 1.88235e-9
    f22 = -5.82409e-8
    f23 = -3.37538e-7
    f24 = -4.53779e-7
    f31 = -5.65158e-7
    f32 = 2.70538e-5
    f33 = 2.40270e-4
    f34 = 2.97574e-5
    f41 = 4.64690e-5
    f42 = -6.70560e-3
    f43 = -2.69091e-2
    f44 = -8.37212e-2
    f51 = 2.58834e-3
    f52 = 6.92510e-1
    f53 = -3.22923e0
    f54 = 8.48091e0

    C2 = C_wt * C_wt
    C3 = C2 * C_wt

    d1 = f11 * C3 + f12 * C2 + f13 * C_wt + f14
    d2 = f21 * C3 + f22 * C2 + f23 * C_wt + f24
    d3 = f31 * C3 + f32 * C2 + f33 * C_wt + f34
    d4 = f41 * C3 + f42 * C2 + f43 * C_wt + f44
    d5 = f51 * C3 + f52 * C2 + f53 * C_wt + f54

    T = 273.16 + T_c
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    sigma = d1 * T4 + d2 * T3 + d3 * T2 + d4 * T + d5
    return sigma



def conductivity_to_resistivity(sigma: ArrayLike) -> np.ndarray:
    """Convert conductivity to resistivity.

    Parameters
    ----------
    sigma : float, int, or numpy.ndarray
        Electrical conductivity in S/m.

    Returns
    -------
    numpy.ndarray
        Electrical resistivity in ohm m.
    """
    sigma = _as_float_array(sigma)
    return 1.0 / sigma


if __name__ == "__main__":
    T = 400.0
    C = 3.5
    P = 1000.0
    sigma = sigbrine_high(T, C, P)
    rho = conductivity_to_resistivity(sigma)
    print(
        f"HIGH T = {T:g} C   P = {P:g} MPa   {C:g} wt%     "
        f"sigma = {np.asarray(sigma).item():g} S/m ({np.asarray(rho).item():g} Ohm.m)"
    )

    T = 250.0
    C = 3.5
    P = 100.0
    sigma = sigbrine_low(T, C)
    rho = conductivity_to_resistivity(sigma)
    print(
        f"LOW  T = {T:g} C   P = {P:g} MPa   {C:g} wt%     "
        f"sigma = {np.asarray(sigma).item():g} S/m ({np.asarray(rho).item():g} Ohm.m)"
    )
