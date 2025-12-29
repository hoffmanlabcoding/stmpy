
import numpy as np
from typing import Tuple, List, Sequence, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def cutdata(x: np.ndarray, y: np.ndarray, xmin: float, xmax: float):
    """
    Return x, y restricted to xmin <= x <= xmax.
    Preserves original order.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]

def lorentz(x: np.ndarray, ctr: float, amp: float, wid: float):
    x = np.asarray(x)
    return amp / (1.0 + 4.0 * (x - ctr)**2 / wid**2)

def moving_ave(x: np.ndarray, y: np.ndarray, window: int):
    """
    Centered moving average with window size `window`.
    Returns (x_avg, y_avg) using 'valid' convolution so both arrays are shorter.
    x is averaged over each window to keep alignment.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if window < 1:
        raise ValueError("window must be >= 1")
    if window == 1:
        return x.copy(), y.copy()
    # Use simple uniform kernel
    kernel = np.ones(window, dtype=float) / window
    y_ma = np.convolve(y, kernel, mode='valid')
    # x positions centered over the window
    x_ma = np.convolve(x, kernel, mode='valid')
    return x_ma, y_ma

def smooth(y: np.ndarray, window: int = 5):
    """
    1D moving average smoother for convenience. Returns array of same length by padding at edges.
    """
    y = np.asarray(y, dtype=float)
    if window < 1:
        return y.copy()
    if window == 1:
        return y.copy()
    pad = window // 2
    # pad edges by reflection
    ypad = np.pad(y, (pad, pad), mode='edge')
    kern = np.ones(window, dtype=float) / window
    out = np.convolve(ypad, kern, mode='valid')
    return out[:len(y)]

def _poly_background(x: np.ndarray, y: np.ndarray, order: int):
    """
    Fit a polynomial background of given order to (x,y).
    """
    coeff = np.polyfit(x, y, order)
    bg = np.polyval(coeff, x)
    return bg, coeff

def remove_bg(x: np.ndarray,
              y: np.ndarray,
              order: int = 3,
              shift: bool = False,
              plot_bg: bool = False,
              savename: Optional[str] = None,
              labels: Optional[str] = None):
    """
    Fit and remove a smooth polynomial background.
    - If shift=True: return (x, y / bg)  (baseline normalization)
    - If shift=False: return (x, y - bg) (baseline subtraction)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bg, coeff = _poly_background(x, y, order)
    if shift:
        # Avoid divide-by-zero
        eps = np.finfo(float).eps
        y_out = y / (bg + eps)
    else:
        y_out = y - bg
    if plot_bg:
        plt.figure()
        plt.plot(x*1e3, y, label='Original Data' if labels is None else labels + ' Data', color='k', linewidth=0.5)
        plt.plot(x*1e3, bg, label='Fitted Background' if labels is None else labels + ' Background', color='gray', linewidth=1)
        plt.xlabel('Bias voltage (mV)')
        plt.ylabel('dI/dV (arb. units)')
        plt.xlim(100,280)
        plt.legend()
        if savename is not None:
            plt.savefig(savename)
        plt.show()

    return x.copy(), y_out, coeff

def multi_lorentz(x: np.ndarray, params: Sequence[float]):
    """
    Sum of multiple Lorentz peaks plus constant offset.
    params = [ctr1, amp1, wid1, ctr2, amp2, wid2, ..., c0]
    """
    x = np.asarray(x, dtype=float)
    params = list(params)
    if len(params) < 4 or (len(params)-1) % 3 != 0:
        raise ValueError("Parameter vector must be triplets per peak plus one constant offset at the end.")
    c0 = params[-1]
    model = np.zeros_like(x, dtype=float) + c0
    for i in range(0, len(params)-1, 3):
        ctr, amp, wid = params[i], params[i+1], params[i+2]
        model += lorentz(x, ctr, amp, wid)
    return model


from scipy.signal import find_peaks

def detect_peaks(x, y, prominence=None, distance=None, max_peaks=None):
    """
    Detect positive peaks in y. Returns peak indices (sorted by x).
    Tune prominence/distance (most important).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    idx, props = find_peaks(y, prominence=prominence, distance=distance)
    if idx.size == 0:
        return idx, props

    # Optionally keep strongest peaks only (by prominence)
    if max_peaks is not None and idx.size > max_peaks:
        order = np.argsort(props["prominences"])[::-1][:max_peaks]
        idx = idx[order]
        for k in props:
            props[k] = props[k][order]

    # sort by x (energy)
    idx = idx[np.argsort(x[idx])]
    return idx, props


def make_p0_from_peaks(x, y, peak_idx, wid0=None, c0=None):
    """
    Build p0 = [ctr1, amp1, wid1, ..., c0] for your multi_lorentz.
    wid0 is FWHM initial guess (same units as x).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    dx = np.median(np.diff(x))
    if wid0 is None:
        wid0 = max(3*dx, 2.0)  # FWHM guess

    if c0 is None:
        c0 = np.median(y)

    p0 = []
    for i in peak_idx:
        ctr = x[i]
        # convert a rough "height" guess into "amp" (area): amp ≈ height * (pi*wid)/2
        height0 = max(y[i] - c0, 0.0)
        amp0 = height0 * (np.pi * wid0) / 2.0
        p0 += [ctr, amp0, wid0]

    p0 += [c0]
    return p0

def fitwithCF(x: np.ndarray,
              y: np.ndarray,
              p0: Sequence[float],
              p_fixed: Optional[Sequence[bool]] = None,
              limit: Optional[Sequence[float]] = None,
              plot: bool = True,
              savename: Optional[str] = None,
              ax=None,
              xlim_plot: Optional[Sequence[float]] = None,
              fulloutput: bool = False,
              maxfev: int = 200000):
    """
    Fit a sum of Lorentzians plus constant offset to (x,y), optionally fixing parameters.

    p0: full parameter list [ctr1, amp1, wid1, ctr2, amp2, wid2, ..., c0]
    p_fixed: same length as p0. True => keep that parameter fixed at p0 value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Drop NaNs
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if limit is not None:
        xmin, xmax = float(limit[0]), float(limit[1])
        mask = (x >= xmin) & (x <= xmax)
        x_fitdata, y_fitdata = x[mask], y[mask]
    else:
        x_fitdata, y_fitdata = x, y
        xmin, xmax = float(np.min(x)), float(np.max(x))

    p0 = np.asarray(p0, dtype=float)
    n_params = len(p0)
    if (n_params - 1) % 3 != 0:
        raise ValueError("p0 must be [ctr, amp, wid]*N + [c0].")
    n_peaks = (n_params - 1) // 3

    if p_fixed is None:
        p_fixed = np.zeros(n_params, dtype=bool)
    p_fixed = np.asarray(p_fixed, dtype=bool)
    if p_fixed.shape != (n_params,):
        raise ValueError("p_fixed must have the same length as p0")

    free_idx = np.where(~p_fixed)[0]
    p0_free = p0[free_idx]

    # ---- Full bounds (then reduced to free params) ----
    lb_full = np.full(n_params, -np.inf, dtype=float)
    ub_full = np.full(n_params,  np.inf, dtype=float)

    # widths (FWHM) must be > 0
    for i in range(2, n_params - 1, 3):
        lb_full[i] = 0

    # optional: if you want amplitudes >= 0 uncomment:
    for i in range(1, n_params - 1, 3):
        lb_full[i] = 0.0
        # ub_full[i] = np.max(y_fitdata) * 10.0  # reasonable upper bound

    lb_free = lb_full[free_idx]
    ub_free = ub_full[free_idx]

    # Model that only exposes free parameters to curve_fit
    def model_free(xx, *p_free):
        p_full = p0.copy()
        p_full[free_idx] = np.asarray(p_free, dtype=float)
        return multi_lorentz(xx, p_full)

    # ---- Fit ----
    if curve_fit is None:
        popt_full = p0.copy()
        pcov = np.full((n_params, n_params), np.nan)
    else:
        try:
            popt_free, pcov_free = curve_fit(
                model_free, x_fitdata, y_fitdata,
                p0=p0_free, bounds=(lb_free, ub_free),
                maxfev=maxfev
            )
            popt_full = p0.copy()
            popt_full[free_idx] = popt_free
        except Exception:
            print("Warning: fit failed; returning initial parameters.")
            popt_full = p0.copy()
            pcov_free = None

    # ---- Uncertainties (NaN for fixed params) ----
    perr_full = np.full(n_params, np.nan, dtype=float)
    if curve_fit is not None and pcov_free is not None and np.all(np.isfinite(pcov_free)):
        perr_free = np.sqrt(np.diag(pcov_free))
        perr_full[free_idx] = perr_free

    # ---- Parse peaks ----
    peaks = []
    for k in range(n_peaks):
        ctr, amp, wid = popt_full[3*k], popt_full[3*k+1], popt_full[3*k+2]
        ctr_err, amp_err, wid_err = perr_full[3*k], perr_full[3*k+1], perr_full[3*k+2]
        peaks.append(dict(ctr=ctr, amp=amp, fwhm=wid, ctr_err=ctr_err, amp_err=amp_err, fwhm_err=wid_err))
    if plot:
        for k, peak in enumerate(peaks):
            # Print peak parameters with error
            print(f"Peak {k+1}: ctr = {peak['ctr']:.3f} ± {perr_full[3*k]:.3f}, "
                f"amp = {peak['amp']:.3f} ± {perr_full[3*k+1]:.3f}, "
                f"fwhm = {peak['fwhm']:.3f} ± {perr_full[3*k+2]:.3f}")

    # ---- Plot ----
    x_dense = np.linspace(xmin, xmax, 1000)
    y_dense = multi_lorentz(x_dense, popt_full)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    
    ax.plot(x, y, 'b-', label='Data', linewidth=0.5)
    ax.plot(x_dense, y_dense, 'r-', label='Fit', linewidth=1)
    ax.set_xlabel('Bias voltage (mV)')
    ax.set_ylabel('dI/dV (arb. units)')
    # Tickes inside all borders
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False)
    if xlim_plot is not None:
        ax.set_xlim(xlim_plot)
    if savename is not None:
        # Create folder if not exists
        import os
        folder = os.path.dirname(savename)
        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(savename)
    if not plot:
        plt.close()
    if fulloutput:
        return popt_full, perr_full, peaks, x_dense, y_dense
    return popt_full, perr_full

def multi_lorentz_plus_poly(x: np.ndarray,
                            params: Sequence[float],
                            poly_order: int,
                            xmid: float,
                            xscale: float):
    """
    params = [ctr1, amp1, wid1, ctr2, amp2, wid2, ..., (poly coeffs), ]
    where poly coeffs are [b0, b1, ..., b_poly_order] in normalized coordinate
      t = (x - xmid) / xscale
      poly(t) = b0 + b1*t + ... + bM*t^M
    """
    x = np.asarray(x, dtype=float)
    params = list(params)

    n_poly = poly_order + 1
    if len(params) < 3 + n_poly or (len(params) - n_poly) % 3 != 0:
        raise ValueError("params must be [ctr,amp,wid]*N + [b0..bM].")

    # split params
    poly = params[-n_poly:]
    peak_params = params[:-n_poly]

    # peaks
    model = np.zeros_like(x, dtype=float)
    for i in range(0, len(peak_params), 3):
        ctr, amp, wid = peak_params[i], peak_params[i+1], peak_params[i+2]
        model += lorentz(x, ctr, amp, wid)

    # polynomial background in normalized coord
    t = (x - xmid) / xscale
    bg = np.zeros_like(x, dtype=float)
    for k, bk in enumerate(poly):
        bg += bk * (t ** k)

    return model + bg


def fitwithCF_plus_poly(x: np.ndarray,
              y: np.ndarray,
              p0: Sequence[float],
              p_fixed: Optional[Sequence[bool]] = None,
              limit: Optional[Sequence[float]] = None,
              poly_order: Optional[int] = None,   # <--- NEW: None means old behavior
              plot: bool = True,
              savename: Optional[str] = None,
              ax=None,
              fulloutput: bool = False,
              maxfev: int = 200000):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if limit is not None:
        xmin, xmax = float(limit[0]), float(limit[1])
        mask = (x >= xmin) & (x <= xmax)
        x_fitdata, y_fitdata = x[mask], y[mask]
    else:
        x_fitdata, y_fitdata = x, y
        xmin, xmax = float(np.min(x)), float(np.max(x))

    p0 = np.asarray(p0, dtype=float)
    n_params = len(p0)

    if p_fixed is None:
        p_fixed = np.zeros(n_params, dtype=bool)
    p_fixed = np.asarray(p_fixed, dtype=bool)
    if p_fixed.shape != (n_params,):
        raise ValueError("p_fixed must have the same length as p0")

    free_idx = np.where(~p_fixed)[0]
    p0_free = p0[free_idx]

    # ---- bounds on full vector ----
    lb_full = np.full(n_params, -np.inf, dtype=float)
    ub_full = np.full(n_params,  np.inf, dtype=float)

    if poly_order is None:
        # Old: params are [ctr,amp,wid]*N + [c0]
        if (n_params - 1) % 3 != 0:
            raise ValueError("p0 must be [ctr, amp, wid]*N + [c0].")
        # widths > 0
        for i in range(2, n_params - 1, 3):
            lb_full[i] = 0.0
        # amps >= 0
        for i in range(1, n_params - 1, 3):
            lb_full[i] = 0.0

        def model_full(xx, p_full):
            return multi_lorentz(xx, p_full)

        # parse counts
        n_peaks = (n_params - 1) // 3

    else:
        # New: params are [ctr,amp,wid]*N + [b0..bM]
        n_poly = poly_order + 1
        if len(p0) < 3 + n_poly or (len(p0) - n_poly) % 3 != 0:
            raise ValueError("With poly_order, p0 must be [ctr,amp,wid]*N + [b0..bM].")

        n_peaks = (n_params - n_poly) // 3

        # bounds only apply to peak part; poly part stays unbounded
        peak_end = n_params - n_poly
        for i in range(2, peak_end, 3):
            lb_full[i] = 0.0        # wid > 0
        for i in range(1, peak_end, 3):
            lb_full[i] = 0.0        # amp >= 0

        # normalization for polynomial stability
        xmid = 0.5 * (xmin + xmax)
        xscale = max(xmax - xmin, 1e-12)

        def model_full(xx, p_full):
            return multi_lorentz_plus_poly(xx, p_full, poly_order, xmid, xscale)

    lb_free = lb_full[free_idx]
    ub_free = ub_full[free_idx]

    def model_free(xx, *p_free):
        p_full = p0.copy()
        p_full[free_idx] = np.asarray(p_free, dtype=float)
        return model_full(xx, p_full)

    try:
        popt_free, pcov_free = curve_fit(
            model_free, x_fitdata, y_fitdata,
            p0=p0_free, bounds=(lb_free, ub_free),
            maxfev=maxfev
        )
        popt_full = p0.copy()
        popt_full[free_idx] = popt_free
    except Exception:
        print("Warning: fit failed; returning initial parameters.")
        popt_full = p0.copy()
        pcov_free = None

    perr_full = np.full(n_params, np.nan, dtype=float)
    if pcov_free is not None and np.all(np.isfinite(pcov_free)):
        perr_free = np.sqrt(np.diag(pcov_free))
        perr_full[free_idx] = perr_free

    # ---- plotting ----
    x_dense = np.linspace(xmin, xmax, 1000)
    y_dense = model_full(x_dense, popt_full)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(x, y, 'b-', label='Data', linewidth=0.5)
    ax.plot(x_dense, y_dense, 'r-', label='Fit', linewidth=1)
    ax.set_xlabel('Bias voltage (mV)')
    ax.set_ylabel('dI/dV (arb. units)')
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(frameon=False)

    if savename is not None:
        import os
        folder = os.path.dirname(savename)
        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(savename)

    if not plot:
        plt.close()

    if fulloutput:
        return popt_full, perr_full, x_dense, y_dense
    return popt_full, perr_full