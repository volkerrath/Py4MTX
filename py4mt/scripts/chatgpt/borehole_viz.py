
"""
borehole_viz.py

Plotting helpers for vertical borehole resistivity profiles.

Functions
---------
- plot_vertical_profile(z, v, label=None, ax=None, invert_z=True, logx=False)
- plot_vertical_profiles(z, profiles, labels=None, logx=True, z_positive_down=True, ax=None)

Notes
-----
- Uses pure Matplotlib (no seaborn, no custom styles).
- One chart per call; no subplots created here.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_vertical_profile(
    z: np.ndarray,
    v: np.ndarray,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    invert_z: bool = True,
    logx: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a single vertical profile v(z).

    Parameters
    ----------
    z : (P,) array
        Vertical coordinates (depth or elevation).
    v : (P,) array
        Profile values at z.
    label : str or None
        Legend label.
    ax : matplotlib.axes.Axes or None
        Optional axis to draw on.
    invert_z : bool, default True
        If True, invert y-axis (common for depth plots).
    logx : bool, default False
        If True, use log-scale for the x-axis.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))
    else:
        fig = ax.figure

    ax.plot(v, z, label=label)
    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Z")
    if logx:
        ax.set_xscale("log")
    if invert_z:
        ax.invert_yaxis()
    if label:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_vertical_profiles(
    z: np.ndarray,
    profiles: List[np.ndarray],
    labels: Optional[List[str]] = None,
    logx: bool = True,
    z_positive_down: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot multiple vertical profiles on a single Matplotlib axis.

    Parameters
    ----------
    z : (P,) array
        Vertical coordinates (depth or elevation).
    profiles : list of (P,) arrays
        Each entry is a profile sampled at the same z.
    labels : list[str] or None
        Legend labels for each profile.
    logx : bool, default True
        If True, use a logarithmic x-scale for resistivity.
    z_positive_down : bool, default True
        If True, configure axis so that z increases downward.
    ax : matplotlib.axes.Axes or None
        Optional axis to draw on.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 6.5))
    else:
        fig = ax.figure

    for i, prof in enumerate(profiles):
        lab = labels[i] if (labels and i < len(labels)) else None
        ax.plot(prof, z, label=lab)

    ax.set_xlabel("Resistivity")
    ax.set_ylabel("Z")
    if logx:
        ax.set_xscale("log")

    # Configure vertical direction
    if z_positive_down:
        # If z is descending (e.g., 0, -10, -20), invert to make positive downward display
        if z.size >= 2 and z[1] < z[0]:
            ax.invert_yaxis()
    else:
        # If z is ascending and user wants positive upward, invert
        if z.size >= 2 and z[1] > z[0]:
            ax.invert_yaxis()

    if labels:
        ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax
