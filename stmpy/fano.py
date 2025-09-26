"""
fano.py — Utilities for Fano resonance fitting

Author: Extracted by ChatGPT for Zeyu (2025-07-01)

This module defines reusable Fano lineshape functions, including fitting routines
and optional plotting. Designed for STM/STS spectroscopy with asymmetry and background.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def fano(x: np.ndarray, q: float, ek: float, t: float) -> np.ndarray:
    """
    Basic Fano resonance lineshape (unit-normalized).

    Parameters:
        x : np.ndarray - Energy axis
        q : float      - Asymmetry parameter
        ek : float     - Resonance energy
        t : float      - Width (inverse lifetime)

    Returns:
        np.ndarray - Fano lineshape
    """
    eps = (x - ek) / t
    return (q + eps)**2 / (1 + eps**2)


def fano_fit(x: np.ndarray, q: float, ek: float, t: float, k: float, a: float, b: float, c: float) -> np.ndarray:
    """
    Fano lineshape with polynomial background.

    Parameters:
        x  : np.ndarray - Energy axis
        q  : float      - Asymmetry
        ek : float      - Resonance energy
        t  : float      - Width
        k  : float      - Amplitude scaling
        a, b, c : float - Quadratic background coefficients

    Returns:
        np.ndarray - Combined Fano + background model
    """
    return k * fano(x, q, ek, t) + a * x**2 + b * x + c


def fit_fano(x: np.ndarray, y: np.ndarray, p0=None, maxfev=10000):
    """
    Fit data to the Fano + background model.

    Parameters:
        x : np.ndarray - x data (e.g., energy)
        y : np.ndarray - y data (e.g., dI/dV)
        p0: list       - Initial guess [q, ek, t, k, a, b, c]

    Returns:
        popt, pcov : fitted parameters and covariance matrix
    """
    if p0 is None:
        # rough guess: q=1, ek=center, t=range/10, k=1, background=flat
        ek0 = x[np.argmax(y)]
        p0 = [1.0, ek0, (x.max() - x.min()) / 10, 1.0, 0.0, 0.0, np.min(y)]

    popt, pcov = curve_fit(fano_fit, x, y, p0=p0, maxfev=maxfev)
    return popt, pcov


def plot_fano_fit(x: np.ndarray, y: np.ndarray, popt, ax=None, label_data='Data', label_fit='Fit'):
    """
    Plot the raw data and fitted Fano model.

    Parameters:
        x, y     : np.ndarray - data
        popt     : list       - fitted parameters
        ax       : plt.Axes or None - where to plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(x, y, 'ro', markerfacecolor='none', label=label_data)
    ax.plot(x, fano_fit(x, *popt), 'k-', label=label_fit)
    ax.set_xlabel('Bias (V)')
    ax.set_ylabel('dI/dV')
    ax.legend()
    ax.set_title("Fano Fit")
    return ax
