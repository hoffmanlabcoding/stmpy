import stmpy
import stmpy.driftcorr as dfc
import matplotlib.patches as patches
import pylab
import pandas
import rhk_sm4.rhk_sm4 as sm4

import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.integrate import cumtrapz, trapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz, trapezoid as trapz

from importlib import reload

from pathlib import Path
import os
import stmpy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit, OptimizeWarning
from tqdm import tqdm  # For a progress bar
import warnings

def plot_FFT_data(data, 
                  k_crop_n = 0, 
                  k_box_center=None,  # e.g. (x, y) to center a box
                  k_box_size=None,    # e.g. (nx, ny) to define box
                  cmap=stmpy.cm.gray_r,  # colormap
                  clim=None,                 # e.g. (vmin, vmax) to force limits
                  sigma=5,                 
                  center='mean',             # 'mean', 0, or a numeric value
                  prc=None,                  # e.g. (1, 99) to use percentiles instead of std
                  ax=None,
                  add_colorbar=False,
                  make_circle=None,
                  savename=None,
                  scalebar=None,
                  
                  ):

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("`data` must be a 2D array.")

    # Crop
    if k_crop_n < 0:
        raise ValueError("k_crop_n must be >= 0")
    if k_crop_n > 0:
        h, w = arr.shape
        if 2*k_crop_n >= min(h, w):
            raise ValueError(f"k_crop_n too large for image {h}x{w}. Need 2*k_crop_n < min(h,w).")
        arr = arr[k_crop_n:-k_crop_n, k_crop_n:-k_crop_n]

    if k_box_center is not None and k_box_size is not None:
        h, w = arr.shape
        cx, cy = k_box_center
        sx, sy = k_box_size
        if sx < 0 or sy < 0:
            raise ValueError("k_box_size must be non-negative.")
        if cx - sx//2 < 0 or cx + (sx+1)//2 > w or cy - sy//2 < 0 or cy + (sy+1)//2 > h:
            raise ValueError("Box exceeds image boundaries.")
        arr = arr[cy - sy//2 : cy + (sy+1)//2, cx - sx//2 : cx + (sx+1)//2]

    # Determine color limits
    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("All values are NaN/inf.")

    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    def _auto_clim(arr, sigma=2.0, center='mean', prc=None):
        x = arr[np.isfinite(arr)].astype(float)
        if prc is not None:
            return tuple(np.nanpercentile(x, prc))
        c = x.mean() if center == 'mean' else (0.0 if center == 0 else float(center))
        mad = np.nanmedian(np.abs(x - np.nanmedian(x)))
        s = 1.4826 * mad if mad > 0 else np.nanstd(x)
        lo, hi = c - sigma*s, c + sigma*s
        return max(lo, np.nanmin(x)), min(hi, np.nanmax(x))

    # inside plot_FFT_data(...)
    if clim is None:
        if sigma is not None:
            clim = _auto_clim(arr, sigma=sigma, center=center, prc=prc)


    ax.imshow(arr, origin='lower', cmap=cmap, clim=clim, interpolation='none')
    if add_colorbar:
        
        stmpy.image.add_colorbar(ax=ax, loc=0, label='FFT Amplitude', fs=8)
    if scalebar is not None:
        stmpy.image.add_scale_bar(scalebar[0], scalebar[1], scalebar[2], ax=ax, fs=12, unit='Å^{-1}', color='black', barheight=1e-3)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    if make_circle is not None:
        cx, cy = arr.shape[1]//2, arr.shape[0]//2
        print(cx, cy, make_circle)
        circ = patches.Circle((cx, cy), make_circle, fill=False, edgecolor='cyan', lw=1.5)
        ax.add_patch(circ)
    
        
    if savename is not None:
        fig.savefig(savename)
    return fig, ax

def _robust_line_subtract(Z, order=1, n_sigma=2.5, n_iter=3, min_frac=0.3):
    '''Line-by-line polynomial background subtraction that ignores bright/dark
    outliers (adsorbates, defects, bad pixels) when estimating each line's
    background.

    Plain lineSubtract fits every pixel in a row, so a bright feature drags the
    polynomial upward and the rest of that row gets over-subtracted, leaving a
    dark streak/halo around the feature. Here each row is fit iteratively:
    after each fit, pixels whose residual exceeds n_sigma * (robust MAD-based
    sigma) are masked out and the fit is repeated, so only the true background
    sets the baseline. The full row (features included) is then subtracted by
    that background fit.

    Inputs:
        Z        - 2D numpy array (rows = fast scan lines).
        order    - Polynomial order fit to each line (default 1).
        n_sigma  - Residual threshold in robust sigma to mask a pixel (default 2.5).
        n_iter   - Max refit iterations per line (default 3).
        min_frac - Min fraction of a line that must stay unmasked to trust the
                   refit; below this the previous fit is kept (default 0.3).

    Returns:
        2D array with the per-line background removed.
    '''
    Z = np.asarray(Z, dtype=float)
    out = np.empty_like(Z)
    ncol = Z.shape[-1]
    x = np.linspace(0.0, 1.0, ncol)
    min_pts = max(order + 1, int(min_frac * ncol))
    for i, line in enumerate(Z):
        good = np.isfinite(line)
        if good.sum() < min_pts:
            out[i] = line - np.nanmean(line)
            continue
        coeffs = np.polyfit(x[good], line[good], order)
        for _ in range(n_iter):
            resid = line - np.polyval(coeffs, x)
            med = np.median(resid[good])
            mad = np.median(np.abs(resid[good] - med))
            sigma = 1.4826 * mad
            if sigma == 0:
                break
            new_good = np.isfinite(line) & (np.abs(resid - med) < n_sigma * sigma)
            if new_good.sum() < min_pts or np.array_equal(new_good, good):
                break
            good = new_good
            coeffs = np.polyfit(x[good], line[good], order)
        out[i] = line - np.polyval(coeffs, x)
    return out


def _destripe_rows(Z, n_iter=1):
    '''Remove residual horizontal stripes (row-to-row offset noise) by
    "median of differences" row alignment.

    Line-by-line leveling reduces but doesn't fully remove the small,
    scan-to-scan vertical offset each fast-scan row carries; the leftover shows
    up as faint horizontal stripes. Walking down the rows, this subtracts from
    each row the running sum of the median difference to the row above it, so
    consecutive rows are brought onto a common level. Because the offset is
    estimated from the *median* over the whole row, localized features
    (adatoms), which are a minority of the pixels, don't bias it -- real
    structure is preserved while the common per-row offset is removed.

    Inputs:
        Z       - 2D numpy array (rows = fast scan lines; stripes run horizontally).
        n_iter  - Number of alignment passes (default 1; 2 can help stubborn stripes).

    Returns:
        2D array with per-row offsets aligned and the global median re-zeroed.
    '''
    out = np.asarray(Z, dtype=float).copy()
    nrow = out.shape[0]
    for _ in range(max(1, int(n_iter))):
        offsets = np.zeros(nrow)
        for i in range(1, nrow):
            diff = out[i] - out[i - 1]
            diff = diff[np.isfinite(diff)]
            offsets[i] = offsets[i - 1] + (np.median(diff) if diff.size else 0.0)
        out = out - offsets[:, None]
    out = out - np.nanmedian(out)
    return out


def _background_clim(img, pct=(1, 99), exclude_adatoms=True, adatom_sigma=3.0):
    '''Color limits (vmin, vmax) computed from the background only.

    A surface covered in adatoms otherwise stretches the color scale to the
    adatoms' height and washes out the subtle background corrugation. When
    exclude_adatoms is True, pixels brighter than
    median + adatom_sigma * (MAD-based sigma) are flagged as adatoms and dropped
    before the percentile range is taken, so the color scale spans the
    background and the adatoms simply saturate at the top of the colormap.

    The median/MAD are robust to a large adatom coverage (they hold as long as
    adatoms occupy < ~50% of the frame), so detection is automatic and needs no
    hand-tuned threshold.

    Inputs:
        img             - 2D numpy array (already leveled, e.g. Z_ls or Z_ps).
        pct             - (low, high) percentiles used for the color limits.
        exclude_adatoms - If True, drop bright-outlier (adatom) pixels first.
        adatom_sigma    - Robust-sigma cutoff above the median for flagging adatoms.

    Returns:
        [vmin, vmax] numpy array for imshow(clim=...), or None if img is empty.
    '''
    flat = np.asarray(img, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    if exclude_adatoms:
        med = np.median(flat)
        mad = np.median(np.abs(flat - med))
        sigma = 1.4826 * mad
        if sigma > 0:
            bg = flat[flat <= med + adatom_sigma * sigma]
            if bg.size >= 0.2 * flat.size:   # keep only if plenty of background remains
                flat = bg
    return np.nanpercentile(flat, list(pct))


def _process_pipeline(Z, window_type='hanning', ls_order=2, ps_order=2, n_sigma_correction=5,
                      ls_robust=True, ls_n_sigma=2.5, ls_n_iter=3, destripe=True, destripe_iter=1):
    
    Z_gc = stmpy.tools.nsigma_global(Z, n=n_sigma_correction, M=5, repeat=100)
    Z_lc = stmpy.tools.nsigma_local(Z_gc, n=n_sigma_correction, N=4, M=5, repeat=50)

    if ls_robust:
        Z_ls = _robust_line_subtract(Z_lc, order=ls_order, n_sigma=ls_n_sigma, n_iter=ls_n_iter)
    else:
        Z_ls = stmpy.tools.lineSubtract(Z_lc, ls_order)
    if destripe:
        Z_ls = _destripe_rows(Z_ls, n_iter=destripe_iter)
    Z_ps = stmpy.tools.plane_subtract(Z_lc, ps_order, include_cross_terms=True, preserve_units=True)
    FZ_ls = stmpy.tools.fft(Z_lc, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    FZ_ps = stmpy.tools.fft(Z_ps, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    return Z_gc, Z_lc, Z_ls, FZ_ls, Z_ps, FZ_ps

def _process_pipeline_LIY(data,smooth_window=10, window_type='hanning', repeat=2, n_sigma_correction=3):
    data.LIY_smoothed = smooth_LIY(data.LIY, window=smooth_window, axis=0, mode='reflect')
    data.LIY_gc = stmpy.tools.nsigma_global(data.LIY_smoothed, n=n_sigma_correction, M=3, repeat=repeat)
    data.LIY_lc = stmpy.tools.nsigma_local(data.LIY_gc, n=n_sigma_correction, N=4, M=3, repeat=repeat)
    data.I_smoothed = smooth_LIY(data.I, window=smooth_window, axis=0, mode='reflect')
    data.Feenstra = data.LIY / data.I * data.en[:, None, None]
    data.Feenstra_smoothed = data.LIY_smoothed / data.I_smoothed * data.en[:, None, None]

    data.FLIY = stmpy.tools.fft(data.LIY, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    data.FLIY_gc = stmpy.tools.fft(data.LIY_gc, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    data.FLIY_lc = stmpy.tools.fft(data.LIY_lc, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    data.FLIY_smoothed = stmpy.tools.fft(data.LIY_smoothed, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    data.Fdidv = stmpy.tools.fft(data.didv, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    data.FI_smoothed = stmpy.tools.fft(data.I_smoothed, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    
def normalize_LIY_by_area(en, LIY, E1, E2, eps=1e-24, window_type='hanning',return_area=True):
    en = np.asarray(en)
    LIY = np.asarray(LIY)
    sort_idx = np.argsort(en)
    en = en[sort_idx]
    LIY = LIY[sort_idx, :, :]

    # Ensure E1 < E2
    E_low, E_high = sorted([E1, E2])

    # Mask energy window
    mask = (en >= E_low) & (en <= E_high)
    if not np.any(mask):
        raise ValueError("No energy points found between E1 and E2")

    # Extract window
    en_win = en[mask]
    LIY_win = LIY[mask, :, :]

    # Integrate using trapezoidal rule
    area_map = np.trapz(LIY_win, en_win, axis=0)

    # Protect against zero / bad setpoints
    area_safe = np.where(np.abs(area_map) < eps, np.nan, area_map)
    # Find the average area over valid pixels
    avg_area = np.nanmean(area_safe)

    # Normalize
    LIY_norm = LIY / area_safe[None, :, :] * avg_area
    FLIY_norm = stmpy.tools.fft(LIY_norm, zeroDC=True, window=window_type, units='amplitude', output='absolute')
    if return_area:
        return LIY_norm, FLIY_norm, area_map
    else:
        return LIY_norm, FLIY_norm
    
def add_corrections_and_plot(data, dos_map: bool = False, 
                             r_crop_n=0,
                             r_box_center=None,
                             r_box_size=None,
                             idx= None, 
                             ls_order=2,
                             ps_order=2,
                             n_sigma_correction=3,
                             ls_robust=True,
                             ls_n_sigma=2.5,
                             ls_n_iter=3,
                             destripe=True,
                             destripe_iter=1,
                             add_colorbar=True,
                             colorbar_range=None,
                             colorbar_range_FFT=None,
                             colorbar_range_ps=None,
                             exclude_adatoms=True,
                             adatom_sigma=3.0,
                             k_kept_n_ratio = 6.3,
                             add_label=True,
                             sym=None,
                             savepath=None, 
                             savepath2=None,
                             smooth_window=10,
                             savename=None, make_plots=True, show=True, 
                             return_figs=False, silent=True, RHK_format=True,
                             lateral_calibration_factor = 1,
                             z_calibration_factor = 1):

    scan_size = data.scan_info['scan_size'] * lateral_calibration_factor  # in nm
    n_pixels = data.scan_info['n_pixels']    
    set_current = data.scan_info['set_current']  # in pA
    set_voltage = data.scan_info['set_voltage']  # in V

    delta_q = 2 * np.pi / scan_size * 0.1 # in armstrong^-1
    FFT_scan_size = delta_q * n_pixels # in armstrong^-1

    figs = {}
    data.Z = data.Z * z_calibration_factor  # Apply z calibration factor
    
    # --- Topography corrections ---
    data.Z_gc, data.Z_lc, data.Z_ls, data.FZ_ls, data.Z_ps, data.FZ_ps = _process_pipeline(data.Z, ls_order=ls_order, ps_order=ps_order, n_sigma_correction=n_sigma_correction, ls_robust=ls_robust, ls_n_sigma=ls_n_sigma, ls_n_iter=ls_n_iter, destripe=destripe, destripe_iter=destripe_iter)
    data.FZ_ls = stmpy.tools.fft(data.Z_ls, zeroDC=True, window='hanning', units='amplitude', output='absolute')
    
    k_kept_n = k_kept_n_ratio * scan_size
    k_crop_n = int(0.5 * (n_pixels - k_kept_n))  # crop 10 nm in FFT
    # print(k_kept_n,k_crop_n)
    k_crop_n = 0 if k_crop_n < 0 else k_crop_n
    
    # Default topography color scale computed from the BACKGROUND only: bright
    # adatoms are detected (median + adatom_sigma * robust sigma) and excluded
    # before taking the 2nd-98th percentile, so they don't wash out the
    # underlying corrugation -- the adatoms simply saturate at the top of the
    # scale. Set exclude_adatoms=False to include them (full-range scaling).
    # Shared by the FWD and BWD panels for direct comparison.
    if colorbar_range is None:
        colorbar_range = _background_clim(data.Z_ls, exclude_adatoms=exclude_adatoms,
                                          adatom_sigma=adatom_sigma)
    if colorbar_range_ps is None:
        colorbar_range_ps = _background_clim(data.Z_ps, exclude_adatoms=exclude_adatoms,
                                             adatom_sigma=adatom_sigma)

    if hasattr(data, "Z_BWD"):
        data.Z_BWD = data.Z_BWD * z_calibration_factor 
        data.Z_BWD_gc, data.Z_BWD_lc, data.Z_BWD_ls, data.FZ_BWD_ls, data.Z_BWD_ps, data.FZ_BWD_ps = _process_pipeline(data.Z_BWD, ls_order=ls_order, ps_order=ps_order, n_sigma_correction=n_sigma_correction, ls_robust=ls_robust, ls_n_sigma=ls_n_sigma, ls_n_iter=ls_n_iter, destripe=destripe, destripe_iter=destripe_iter)
        data.FZ_BWD_ls = stmpy.tools.fft(data.Z_BWD_ls, zeroDC=True, window='hanning', units='amplitude', output='absolute')
        if make_plots:
            if sym is not None:
                fig_topo, ax_topo = plt.subplots(2, 8, figsize=(26, 10), constrained_layout=True)
                data.FZ_ls_s = dfc.sym(data.FZ_ls, sym)
                data.FZ_BWD_ls_s = dfc.sym(data.FZ_BWD_ls, sym)
            else:
                fig_topo, ax_topo = plt.subplots(2, 7, figsize=(23, 10))
            ax_topo_ls_FWD = ax_topo[0,4]
            ax_topo_ps_FWD = ax_topo[0,3]
            ax_topo_ls_BWD = ax_topo[1,4]
            ax_topo_ps_BWD = ax_topo[1,3]

            ax_topo[0,0].imshow(data.Z,     cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[0,0].set_title('Raw Z')
            ax_topo[0,1].imshow(data.Z_gc,  cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[0,1].set_title('Global Corrected Z')
            ax_topo[0,2].imshow(data.Z_lc,  cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[0,2].set_title('Local Corrected Z')
            ax_topo_ls_FWD.imshow(data.Z_ls,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range, interpolation='none'); ax_topo_ls_FWD.set_title(f'Line Subtracted Z (order={ls_order})')
            ax_topo_ps_FWD.imshow(data.Z_ps,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range_ps, interpolation='none'); ax_topo_ps_FWD.set_title(f'Plane Subtracted Z (order={ps_order})')
        
            ax_topo[1,0].imshow(data.Z_BWD,     cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[1,0].set_title('Raw Z BWD')
            ax_topo[1,1].imshow(data.Z_BWD_gc,  cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[1,1].set_title('Global Corrected Z BWD')
            ax_topo[1,2].imshow(data.Z_BWD_lc,  cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[1,2].set_title('Local Corrected Z BWD')
            ax_topo_ls_BWD.imshow(data.Z_BWD_ls,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range, interpolation='none'); ax_topo_ls_BWD.set_title(f'Line Subtracted Z BWD (order={ls_order})')
            ax_topo_ps_BWD.imshow(data.Z_BWD_ps,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range_ps, interpolation='none'); ax_topo_ps_BWD.set_title(f'Plane Subtracted Z BWD (order={ps_order})')

            for i in range(2):
                for j in range(7):
                    ax_topo[i,j].set_xlabel('')
                    ax_topo[i,j].set_ylabel('')
                    # aspect ratio
                    ax_topo[i,j].set_aspect('equal')    
                        
                    if j < 5:
                        if r_crop_n > 0:
                            # make a white dashed square to indicate cropped region
                            h, w = data.Z.shape
                            rect = patches.Rectangle((r_crop_n, r_crop_n), w - 2*r_crop_n, h - 2*r_crop_n, linewidth=1, edgecolor='w', facecolor='none')
                        elif r_box_center is not None and r_box_size is not None:
                            cx, cy = r_box_center
                            sx, sy = np.array(r_box_size) / 2
                            rect = patches.Rectangle((cx - sx, cy - sy), 2*sx, 2*sy, linewidth=1, edgecolor='w', facecolor='none')
                            ax_topo[i,j].add_patch(rect)
                        ax_topo[i,j].set_axis_off()
                        stmpy.image.add_scale_bar(5, scan_size, n_pixels, fs=12, ax=ax_topo[i,j])
                        if add_colorbar:
                            stmpy.image.add_colorbar(ax=ax_topo[i,j], label='Topography (m)', fs=8)
                        if add_label:
                            stmpy.image.add_label(f'{set_voltage:.2f}V {np.abs(set_current):.0f}pA', ax=ax_topo[i,j], fs=8)
   
            plot_FFT_data(data.FZ_ls, k_crop_n=k_crop_n, ax=ax_topo[0,5], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                          scalebar = [1, FFT_scan_size, n_pixels])
            plot_FFT_data(data.FZ_ls, k_crop_n=0, ax=ax_topo[0,6], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                          scalebar = [1, FFT_scan_size, n_pixels])
            plot_FFT_data(data.FZ_BWD_ls, k_crop_n=k_crop_n, ax=ax_topo[1,5], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                          scalebar = [1, FFT_scan_size, n_pixels])
            plot_FFT_data(data.FZ_BWD_ls, k_crop_n=0, ax=ax_topo[1,6], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                          scalebar = [1, FFT_scan_size, n_pixels])
            if sym is not None:
                plot_FFT_data(data.FZ_ls_s, k_crop_n=k_crop_n, ax=ax_topo[0,7], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                              scalebar = [1, FFT_scan_size, n_pixels])
                plot_FFT_data(data.FZ_BWD_ls_s, k_crop_n=k_crop_n, ax=ax_topo[1,7], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                              scalebar = [1, FFT_scan_size, n_pixels])
            
    else:
        if make_plots:
            fig_topo, ax_topo = plt.subplots(1, 7, figsize=(23, 5))
            ax_topo[0].imshow(data.Z,     cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[0].set_title('Raw Z')
            ax_topo[1].imshow(data.Z_gc,  cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[1].set_title('Global Corrected Z')
            ax_topo[2].imshow(data.Z_lc,  cmap=stmpy.cm.Blues_r, origin='lower', interpolation='none'); ax_topo[2].set_title('Local Corrected Z')
            ax_topo[3].imshow(data.Z_ls,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range, interpolation='none'); ax_topo[3].set_title('Line Subtracted Z')
            ax_topo[4].imshow(data.Z_ps,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range_ps, interpolation='none'); ax_topo[4].set_title('Plane Subtracted Z')
            for i, a in enumerate(ax_topo): 
                a.set_xlabel('')
                a.set_ylabel('')
                a.set_aspect('equal')
                if i < 5:
                    a.set_axis_off()
                    stmpy.image.add_scale_bar(5, scan_size, n_pixels, fs=12,  ax=a)
                    if add_colorbar:
                            stmpy.image.add_colorbar(ax=ax_topo[i], label='Topography (m)', fs=8)
                    if add_label:
                            stmpy.image.add_label(f'{set_voltage:.2f}V {np.abs(set_current):.0f}pA', ax=ax_topo[i], fs=8)

            plot_FFT_data(data.FZ_ls, k_crop_n=k_crop_n, ax=ax_topo[5], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                          scalebar = [1, FFT_scan_size, n_pixels])
            plot_FFT_data(data.FZ_ls, k_crop_n=0, ax=ax_topo[6], add_colorbar=add_colorbar, clim=colorbar_range_FFT,
                          scalebar = [1, FFT_scan_size, n_pixels])

    if make_plots:
        # fig_topo.tight_layout()
        #add title
        if not dos_map:
            if not RHK_format:
                scan_offset = data.header['scan_offset']
                scan_angle = data.header['scan_angle']
                fig_topo.suptitle(data.info_str + ' (' + f'{scan_offset[0]*1e9:.2f}, {scan_offset[1]*1e9:.2f})nm, {scan_angle} deg', fontsize=16)
            else:
                fig_topo.suptitle(data.info_str, fontsize=16)
        figs['topo'] = (fig_topo, ax_topo)

    # --- DOS map (dI/dV) corrections & plots ---
    if dos_map:
        if not hasattr(data, 'LIY'):
            raise AttributeError("dos_map=True but `data.LIY` not found.")
        # Corrections on the full energy stack 
        _process_pipeline_LIY(data, smooth_window=smooth_window, window_type='hanning', n_sigma_correction=n_sigma_correction)
    

        # Mean over energy
       
        mean_gc  = np.mean(data.LIY_gc, axis=0)
        mean_lc  = np.mean(data.LIY_lc, axis=0)

        if make_plots:
            fig_dm, ax_dm = plt.subplots(1, 3, figsize=(15, 5))
            ax_dm[0].imshow(data.didv, origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
            ax_dm[0].set_title('Raw dI/dV at mean V')
            ax_dm[1].imshow(mean_gc, origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
            ax_dm[1].set_title('Global Corrected dI/dV at mean V')
            ax_dm[2].imshow(mean_lc, origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
            ax_dm[2].set_title('Local Corrected dI/dV at mean V')
            for a in ax_dm: 
                a.set_axis_off()
                stmpy.image.add_scale_bar(5, scan_size, n_pixels, fs=12, ax=a)
            fig_dm.tight_layout()
            figs['dos_mean'] = (fig_dm, ax_dm)

            if idx is not None:
                # Single energy slice
                nE = data.LIY.shape[0]
                idx = int(np.clip(idx, 0, nE - 1))
                # Energies if present; otherwise index-based label
                if hasattr(data, 'en') and getattr(data, 'en') is not None and len(data.en) == nE:
                    en_label = f"{data.en[idx]:.2f} V"
                else:
                    en_label = f"index {idx}"

                fig_ds, ax_ds = plt.subplots(1, 3, figsize=(15, 5))
                ax_ds[0].imshow(data.LIY[idx], origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
                ax_ds[0].set_title(f'Raw dI/dV at {en_label}')
                ax_ds[1].imshow(data.LIY_gc[idx], origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
                ax_ds[1].set_title(f'Global Corrected dI/dV at {en_label}')
                ax_ds[2].imshow(data.LIY_lc[idx], origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
                ax_ds[2].set_title(f'Local Corrected dI/dV at {en_label}')
                for a in ax_ds: 
                    a.set_axis_off()
                    stmpy.image.add_scale_bar(5, scan_size, n_pixels, fs=12,  ax=a)
                fig_ds.tight_layout()
                figs['dos_idx'] = (fig_ds, ax_ds)

    if savename is not None:
        savename = data.info_str + "_" + savename.replace(".sm4", "") +  ".pdf"
        fig_topo.savefig(savepath+'/'+savename)
        if savepath2 is not None:
            fig_topo.savefig(savepath2+'/'+savename)
        if not silent:
            print(f"Saved topo figure to {savepath+'/'+savename}")

    if show:
        plt.show()
    else:
        plt.close('all')

    if return_figs:
        return figs
    


def cumulative_integral_zero_at_origin(x, y):
    # Sort x and reorder y
    idx = np.argsort(x)
    x_sorted = np.array(x)[idx]
    y_sorted = np.array(y)[idx]

    # Compute cumulative integral (same length as x, with initial=0)
    I = cumtrapz(y_sorted, x_sorted, initial=0)

    # Find index of closest point to x=0
    zero_idx = np.argmin(np.abs(x_sorted - 0))

    # Shift so that I(x=0) = 0
    I = I - I[zero_idx]
    return x_sorted, y_sorted, I



def compute_shape_params(
    data,
    *,
    shape1=(0.0, 0.5, 1.3),   # (e0, e1, e2):  ∫(e0→e1) / ∫(e1→e2)
    shape2=(0.7, 1.0, 1.3),   # (e3, e4, e5):  ∫(e3→e4) / ∫(e4→e5)
    method="interp",          # "interp" (linear) or "nearest" (snap to grid)
    grid=False,
    clamp_negative=True,      # clamp negative shape_para1 to 0 (keeps your original behavior)
    store=True,                # write iv_math/shape_para* back to `data`
    im_show=True             # show images and scatter plot
):
    """
    Compute shape parameters from a dI/dV hypercube.

    Definitions
    -----------
    shape1 = (e0, e1, e2)  ⇒  shape_para1 = ∫(e0→e1) / ∫(e1→e2)
    shape2 = (e3, e4, e5)  ⇒  shape_para2 = ∫(e3→e4) / ∫(e4→e5)

    Inputs (required on `data`)
    ---------------------------
    data.en  : (nE,) energy axis (must be monotonic)
    data.LIY : (nE, nx, ny) dI/dV spectra grid

    Outputs (stored on `data` if store=True)
    ----------------------------------------
    data.iv_math     : (nE, nx, ny) cumulative integral C(E) = ∫ dI/dV dE with C(en[0])=0
    data.shape_para1 : (nx, ny)
    data.shape_para2 : (nx, ny)

    Returns
    -------
    iv_math, shape_para1, shape_para2
    """
    en = np.asarray(data.en)
    if grid:
        LIY = np.asarray(data.LIY)
    else:
        LIY = np.asarray(data.LIY_lc)
    
    if LIY.shape[0] != en.size:
        raise ValueError("data.LIY first dimension must equal len(data.en).")

    # Ensure strictly increasing energy axis (sort if needed)
    if np.any(np.diff(en) <= 0):
        order = np.argsort(en)
        en = en[order]
        LIY = LIY[order]

    # Cumulative integral along energy (same length via initial=0)
    iv_math = cumtrapz(LIY, en, axis=0, initial=0)  # (nE, nx, ny)

    # --- helpers ---
    def _validate_triple(tri, name):
        e0, e1, e2 = tri
        if not (e0 < e1 < e2):
            raise ValueError(f"{name} must be strictly increasing (got {tri}).")
        if e0 < en[0] or e2 > en[-1]:
            raise ValueError(
                f"{name} values must lie within energy range "
                f"[{en[0]:.6g}, {en[-1]:.6g}] (got {tri})."
            )
        return e0, e1, e2

    e0, e1, e2 = _validate_triple(shape1, "shape1")
    e3, e4, e5 = _validate_triple(shape2, "shape2")

    # Interpolate cumulative integral C(E) at arbitrary energies (vectorized)
    nE, nx, ny = iv_math.shape
    iv2d = iv_math.reshape(nE, nx * ny)  # (nE, nPix)

    def _cum_at(E):
        if method == "nearest":
            k = int(np.argmin(np.abs(en - E)))
            return iv2d[k]
        # piecewise-linear interpolation
        k = np.searchsorted(en, E, side="right")
        if k == 0 or k == nE:
            raise ValueError(f"Energy {E} outside grid [{en[0]}, {en[-1]}].")
        x0, x1 = en[k - 1], en[k]
        y0, y1 = iv2d[k - 1], iv2d[k]
        t = (E - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # Interval integrals via C(b) - C(a)
    C0, C1, C2 = _cum_at(e0), _cum_at(e1), _cum_at(e2)
    C3, C4, C5 = _cum_at(e3), _cum_at(e4), _cum_at(e5)

    int1 = C1 - C0
    int2 = C2 - C1
    int3 = C4 - C3
    int4 = C5 - C4

    sp1 = np.divide(int1, int2, out=np.full_like(int1, np.nan), where=(int2 != 0))
    if clamp_negative:
        sp1 = np.where(sp1 < 0, 0, sp1)
    sp2 = np.divide(int3, int4, out=np.full_like(int3, np.nan), where=(int4 != 0))

    sp1 = sp1.reshape(nx, ny)
    sp2 = sp2.reshape(nx, ny)

    if store:
        data.iv_math = iv_math
        data.shape_para1 = sp1
        data.shape_para2 = sp2
        data.shape_para1a = int1.reshape(nx, ny)
        data.shape_para1b = int2.reshape(nx, ny)
        data.shape_para2a = int3.reshape(nx, ny)
        data.shape_para2b = int4.reshape(nx, ny)

    if im_show:
        plot_rk_space(data, ens=[e0, e1, e2, e3, e4, e5])
        

def smooth_LIY(LIY, window=10, axis=0, mode='reflect'):
    """
    Smooths the LIY data by a moving mean along the specified axis.

    Parameters
    ----------
    LIY : np.ndarray
        3D array (E, I, J)
    window : int
        Size of the moving average window (must be >= 1)
    axis : int
        Axis along which to smooth (default 0: energy axis)
    mode : str
        How to handle edges. Options: 'reflect', 'nearest', 'constant', 'wrap'.
        Passed to np.pad.

    Returns
    -------
    np.ndarray
        Smoothed array with same shape as LIY.
    """
    if window < 2:
        return LIY.copy()

    LIY = np.asarray(LIY)
    pad = window // 2
    LIY_padded = np.pad(LIY, 
                        [(pad, pad) if a == axis else (0, 0) for a in range(LIY.ndim)],
                        mode=mode)
    cumsum = np.cumsum(LIY_padded, axis=axis)
    # difference between cumulative sums gives moving average
    slices1 = [slice(None)] * LIY.ndim
    slices2 = [slice(None)] * LIY.ndim
    slices1[axis] = slice(window, None)
    slices2[axis] = slice(None, -window)
    smoothed = (cumsum[tuple(slices1)] - cumsum[tuple(slices2)]) / window
    return smoothed

def plot_all_didv(data, ax=None, alpha=0.3,grid='False'):
    """
    Plot (1) all dI/dV spectra, (2) all I spectra (if present),
    and (3) all integrated I (= iv_math). No computation here.

    Requires:
      data.en:     (nE,)
      data.LIY:    (nE, nx, ny)
      data.iv_math (nE, nx, ny)   # from compute_shape_params(...)
      Optional: data.I (nE, nx, ny)

    Returns:
      fig, ax  # ax is length-3 array of axes
    """
    if not hasattr(data, "iv_math"):
        raise ValueError("data.iv_math not found. Run compute_shape_params(...) first.")

    en = np.asarray(data.en)
    LIY = np.asarray(data.LIY)          # (nE, nx, ny)
   
    iv_math = np.asarray(data.iv_math)  # (nE, nx, ny)
    has_I = hasattr(data, "I")

    # Set up axes
    if ax is None:
        if grid is True:

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        else:
            LIY_lc = np.asarray(data.LIY_lc)          # (nE, nx, ny)
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    else:
        fig = ax[0].get_figure()

    nx, ny = LIY.shape[1], LIY.shape[2]

    # Plot all pixels
    for i in range(nx):
        for j in range(ny):
            ax[0].plot(en, LIY[:, i, j], color='gray', alpha=alpha)
            if has_I:
                ax[1].plot(en, data.I[:, i, j], color='gray', alpha=alpha)
            ax[2].plot(en, iv_math[:, i, j], color='gray', alpha=alpha)
            if grid is False:
                ax[3].plot(en, LIY_lc[:, i, j], color='gray', alpha=alpha)

    ax[0].set_title('dI/dV at all pixels'); ax[0].set_xlabel('Energy'); ax[0].set_ylabel('dI/dV')
    if has_I:
        ax[1].set_title('I at all pixels'); ax[1].set_xlabel('Energy'); ax[1].set_ylabel('I')
    else:
        ax[1].set_title('I channel not present'); ax[1].set_xlabel('Energy'); ax[1].set_ylabel('I')
    ax[2].set_title('Integrated I (cumtrapz)'); ax[2].set_xlabel('Energy'); ax[2].set_ylabel('∫ dI/dV dE')
    if grid is False:
        ax[3].set_title('Global and then Local Corrected dI/dV at all pixels'); ax[3].set_xlabel('Energy'); ax[3].set_ylabel('dI/dV')

    return fig, ax



def sts_model(V, amp_L, edge_L, amp_R, edge_R, broad, bump_amp, bump_pos, bump_width, offset):
    """
    Defines the fitting model for a single STS spectrum.

    V: Voltage/Energy array
    amp_L, amp_R: Amplitude of Valence/Conduction bands
    edge_L, edge_R: Position of band edges (e.g. -0.5 and +0.5)
    broad: Broadening factor (smoothness of the turn-on)
    bump_*: Parameters for the mid-gap bump (Gaussian)
    offset: vertical offset (noise floor)
    """
    
    # Left side (Valence Band) - turns on as V goes negative
    left_side = amp_L / (1 + np.exp((V - edge_L) / broad))
    
    # Right side (Conduction Band) - turns on as V goes positive
    right_side = amp_R / (1 + np.exp(-(V - edge_R) / broad))
    
    # The Bump (Gaussian)
    bump = bump_amp * np.exp(-(V - bump_pos)**2 / (2 * bump_width**2))
    
    # Total signal
    return left_side + right_side + bump + offset



def fit_sts_map(x_data, y_data_cube, model, p0, bounds=None):
    """
    Fits an STS model to every pixel in a 3D data cube (E, Y, X).

    Args:
        x_data (1D array): The energy/voltage array.
        y_data_cube (3D array): The LIY data, with shape (N_E, N_Y, N_X).
        model (function): The fitting function to use (e.g., sts_model).
        p0 (list): Initial guess for parameters.
        bounds (tuple): (min_bounds, max_bounds) for parameters.

    Returns:
        param_map (3D array): A map of fitted parameters.
                              Shape is (N_Y, N_X, N_Params).
        error_map (2D array): A map of the fit error (sum of cov diagonal).
                              Shape is (N_Y, N_X).
        liy_fitted_cube (3D array): A cube of the fitted LIY spectra.
                                    Shape is (N_E, N_Y, N_X).
    """
    
    # Get dimensions
    if y_data_cube.ndim != 3:
        raise ValueError(f"Input data cube must be 3D (E, Y, X), but got {y_data_cube.ndim} dimensions")
    
    N_E, N_Y, N_X = y_data_cube.shape
    N_params = len(p0)
    
    # Create empty arrays to store the results
    # We use np.nan to mark pixels where the fit fails
    param_map = np.full((N_Y, N_X, N_params), np.nan)
    error_map = np.full((N_Y, N_X), np.nan)
    liy_fitted_cube = np.full((N_E, N_Y, N_X), np.nan)
    
    # Loop over every pixel (Y, X)
    # We use tqdm to create a nice progress bar
    for i in tqdm(range(N_Y), desc="Fitting STS Map Rows"):
        for j in range(N_X):
            
            # Get the 1D spectrum for this pixel
            y_pixel = y_data_cube[:, i, j]
            
            # Skip if data is all NaN or all zero
            if np.all(np.isnan(y_pixel)) or not np.any(y_pixel):
                continue

            try:
                # Ignore OptimizeWarnings (e.g. covariance not estimated)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)

                    popt, pcov = curve_fit(
                        model,
                        x_data,
                        y_pixel,
                        p0=p0,
                        bounds=bounds if bounds is not None else (-np.inf, np.inf),
                        maxfev=5000,  # a bit more robust
                    )

            except (RuntimeError, ValueError):
                # Fit failed for this pixel – leave NaNs and move on
                continue

            # Store parameters
            param_map[i, j, :] = popt

            # Simple error metric: sqrt(sum of variances)
            if pcov is not None and np.all(np.isfinite(pcov)):
                error_map[i, j] = np.sqrt(np.sum(np.diag(pcov)))

            # Store fitted spectrum
            liy_fitted_cube[:, i, j] = model(x_data, *popt)

    return param_map, error_map, liy_fitted_cube

def analyze_dos_data(data_object, crop_rect=None, scale_factor=1e12, p0=None, bounds=None, cmap='viridis', plot_results=True, check_pixel=(5,5)):
    """
    Runs the full STS fitting analysis on a given data object.

    Args:
        data_object: An object (like dos_E) that has .en (1D) and .LIY (3D) attributes.
                     The results will be attached to this object.
        crop_rect (tuple): A tuple of slices for (Y, X) dimensions, e.g., (slice(0, 10), slice(0, 10)).
                           If None, the full map is used.
        scale_factor (float): Factor to multiply LIY data by (e.g., 1e12).
        p0 (list): Initial parameter guess. Uses a default if None.
        bounds (tuple): Parameter bounds. Uses a default if None.
        plot_results (bool): If True, generates and shows the result plots.
        check_pixel (tuple): (Y, X) coordinates for the single pixel plot.
    """
    
    # --- 1. Set up data and parameters ---
    x_data = data_object.en
    
    if crop_rect is None:
        # Use full map
        crop_rect = (slice(None), slice(None))
        
    y_data_cube = data_object.LIY[:, crop_rect[0], crop_rect[1]] * scale_factor

    N_E, N_Y, N_X = y_data_cube.shape

    print(f"--- Analyzing Data Object ---")
    print(f"x_data shape: {x_data.shape}")
    print(f"Cropped y_data_cube shape: {y_data_cube.shape} (Y={N_Y}, X={N_X})")

    # Set default p0 and bounds if not provided
    if p0 is None:
        p0 = [3, -0.8, 3, 0.5, 0.1, 0, 0, 0.2, 0]
    if bounds is None:
        bounds = ([0, -2, 0, 0, 0.01, 0, -0.1, 0, -1], 
                  [100, 0, 100, 2, 1.0, 10, 0.5, 1, 1])

    # --- 2. Run the fit ---
    print("Starting fit... (This may take a while)")
    param_map, error_map, liy_fitted_cube = fit_sts_map(x_data, y_data_cube, sts_model, p0, bounds)
    print("Fit complete.")
    
    # --- 3. Attach results to the data object ---
    data_object.param_map = param_map
    data_object.error_map = error_map
    data_object.liy_fitted = liy_fitted_cube
    data_object.liy_residual = y_data_cube - liy_fitted_cube # Calculate residuals
    
    # Attach individual parameter maps
    data_object.amp_L = data_object.param_map[:, :, 0]
    data_object.edge_L = data_object.param_map[:, :, 1]
    data_object.amp_R = data_object.param_map[:, :, 2]
    data_object.edge_R = data_object.param_map[:, :, 3]
    data_object.broadening = data_object.param_map[:, :, 4]
    data_object.bump_amp = data_object.param_map[:, :, 5]
    data_object.bump_pos = data_object.param_map[:, :, 6]
    data_object.bump_width = data_object.param_map[:, :, 7]
    data_object.offset = data_object.param_map[:, :, 8]
    data_object.gap_map = data_object.edge_R - data_object.edge_L
    
    print("Results attached to data object.")

    # --- 4. Visualize the results (if requested) ---
    if plot_results:
        print("Generating plots...")
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        
        # Plot Gap Map
        im0 = axes[0].imshow(data_object.gap_map, origin='lower', aspect='equal', cmap=cmap, 
                             extent=[0, N_X, 0, N_Y], interpolation='none')
        axes[0].set_title("Fitted Gap Size (edge_R - edge_L)")
        fig.colorbar(im0, ax=axes[0], label="Gap (V)")

        # Plot Bump Amplitude Map
        im1 = axes[1].imshow(data_object.bump_amp, origin='lower', aspect='equal', cmap=cmap, 
                             extent=[0, N_X, 0, N_Y], interpolation='none')
        axes[1].set_title("Fitted Bump Amplitude")
        fig.colorbar(im1, ax=axes[1], label="Amplitude (a.u.)")

        # Plot Error Map
        im2 = axes[2].imshow(data_object.error_map, origin='lower', aspect='equal', cmap=cmap, 
                             extent=[0, N_X, 0, N_Y], interpolation='none')
        axes[2].set_title("Fit Error (sqrt(sum(diag(pcov))))")
        fig.colorbar(im2, ax=axes[2], label="Error (a.u.)")
        
        im3 = axes[3].imshow(data_object.edge_L, origin='lower', aspect='equal', cmap=cmap,
                             extent=[0, N_X, 0, N_Y], interpolation='none')
        axes[3].set_title("Fitted Valence Band Edge (edge_L)")
        fig.colorbar(im3, ax=axes[3], label="Energy (V)")

        im4 = axes[4].imshow(data_object.edge_R, origin='lower', aspect='equal', cmap=cmap,
                             extent=[0, N_X, 0, N_Y], interpolation='none')
        axes[4].set_title("Fitted Conduction Band Edge (edge_R)")
        fig.colorbar(im4, ax=axes[4], label="Energy (V)")

        plt.tight_layout()
        plt.show()

        # 5. VISUALIZE A SINGLE PIXEL FIT (to check)
        px_i, px_j = check_pixel
        
        # Check if pixel is within the fitted map
        if px_i < N_Y and px_j < N_X:
            y_raw_pixel = y_data_cube[:, px_i, px_j]
            y_fit_pixel = data_object.liy_fitted[:, px_i, px_j]
            fitted_params = data_object.param_map[px_i, px_j, :]
            
            # 2 axes plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
            ax1.plot(x_data, y_raw_pixel, 'k-', alpha=0.3, label=f'Raw Data (pixel {px_i}, {px_j})')
            ax1.plot(x_data, y_fit_pixel, 'r--', lw=2, label='Fit')

            ax2.plot(x_data, y_raw_pixel - y_fit_pixel, 'b-', alpha=0.7, label='Residual')
            ax2.axhline(0, color='gray', linestyle='--', lw=1)
            ax2.set_ylabel("Residual (LIY units)")
            ax2.set_xlabel("Energy (V)")
            # Check if fit was successful before indexing params
            if not np.all(np.isnan(fitted_params)):
                ax1.set_title(f"Single Pixel Fit Check (Gap: {fitted_params[3]-fitted_params[1]:.2f} V)")
            else:
                ax1.set_title(f"Single Pixel Fit Check (pixel {px_i}, {px_j}) - Fit Failed")

            ax1.legend()
            ax1.set_xlabel("Energy (V)")
            ax1.set_ylabel("LIY (scaled)")
            plt.show()
        else:
            print(f"Check pixel ({px_i}, {px_j}) is outside the bounds of the fitted map ({N_Y}, {N_X}). Skipping pixel plot.")
            
    print("--- Analysis complete ---")


def animate_ldos_with_topo(data, plot_range=None, interval=80, repeat=True,
                           use_global_ylim=False, cmap=stmpy.cm.Blues_r,
                           start_ij=(0,0), click_to_jump=True):
    """
    Show topo on the left and LDOS(E) on the right; animate across all spatial pixels.
    A red marker on the topo shows the currently plotted (i, j).

    Parameters
    ----------
    data
        en : (E,) array
        LIY_3d : (E, I, J) array
        topo_2d : (I, J) array
    plot_range
    interval : ms between frames
    repeat : loop animation
    use_global_ylim : fix y-limits from global min/max for steadier view
    cmap : colormap for topo
    start_ij : starting (i, j) index
    click_to_jump : click on topo to jump to that pixel’s spectrum
    """
    if plot_range is None:
        LIY_3d = np.asarray(data.LIY)
        LIY_smoothed = np.asarray(data.LIY_smoothed)
        LIY_fitted = np.asarray(data.liy_fitted)*1e-12
        topo_2d = np.asarray(data.Z_ls)
        edge_L = np.asarray(data.edge_L)
        edge_R = np.asarray(data.edge_R)
        bump_amp = np.asarray(data.bump_amp)
        bump_pos = np.asarray(data.bump_pos)
    else:
        LIY_3d = np.asarray(data.LIY)[:, plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
        LIY_smoothed = np.asarray(data.LIY_smoothed)[:, plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
        LIY_fitted = np.asarray(data.liy_fitted)[:, plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]*1e-12
        topo_2d = np.asarray(data.Z_ls)[plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
        edge_L = np.asarray(data.edge_L)[plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
        edge_R = np.asarray(data.edge_R)[plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
        bump_amp = np.asarray(data.bump_amp)[plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
        bump_pos = np.asarray(data.bump_pos)[plot_range[0]:plot_range[1], plot_range[2]:plot_range[3]]
    en = np.asarray(data.en)
   

    assert LIY_3d.ndim == 3 and topo_2d.ndim == 2, "Shapes: LIY (E,I,J), topo (I,J)"
    E, I, J = LIY_3d.shape
    assert en.shape[0] == E and topo_2d.shape == (I, J), "Shape mismatch"

    ij_list = [(i, j) for i in range(I) for j in range(J)]
    start_i, start_j = np.clip(start_ij[0], 0, I-1), np.clip(start_ij[1], 0, J-1)
    start_frame = ij_list.index((start_i, start_j))

    # Figure layout

    fig = plt.figure(figsize=(25, 5))

    # 2 rows × 6 columns
    gs = gridspec.GridSpec(
        nrows=2, ncols=6,
        height_ratios=[1, 1],
        width_ratios=[1, 1, 1, 1, 1, 1]
    )

    # First five axes span BOTH rows → tall panels
    ax_topo     = fig.add_subplot(gs[:, 0])
    ax_edge_L   = fig.add_subplot(gs[:, 1])
    ax_edge_R   = fig.add_subplot(gs[:, 2])
    ax_bump_amp = fig.add_subplot(gs[:, 3])
    ax_bump_pos = fig.add_subplot(gs[:, 4])

    # Last column: one axis per row → same height as maps
    ax_spec     = fig.add_subplot(gs[0, 5])
    ax_residual = fig.add_subplot(gs[1, 5])



    # --- Topography
    im = ax_topo.imshow(topo_2d, origin='lower', cmap=cmap, aspect='equal', interpolation='none')
    cb = fig.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.04)
    ax_topo.set_title("Topography")

    # --- Edge L
    im2 = ax_edge_L.imshow(edge_L, origin='lower', cmap=cmap, aspect='equal', interpolation='none')
    cb2 = fig.colorbar(im2, ax=ax_edge_L, fraction=0.046, pad=0.04)
    ax_edge_L.set_title("Valence Band Edge (edge_L)")

    # --- Edge R
    im3 = ax_edge_R.imshow(edge_R, origin='lower', cmap=cmap, aspect='equal', interpolation='none')
    cb3 = fig.colorbar(im3, ax=ax_edge_R, fraction=0.046, pad=0.04)
    ax_edge_R.set_title("Conduction Band Edge (edge_R)")
    # --- Bump Amplitude
    im4 = ax_bump_amp.imshow(bump_amp, origin='lower', cmap=cmap, aspect='equal', interpolation='none')
    cb4 = fig.colorbar(im4, ax=ax_bump_amp, fraction=0.046, pad=0.04)
    ax_bump_amp.set_title("Bump Amplitude")
    # --- Bump Position
    im5 = ax_bump_pos.imshow(bump_pos, origin='lower', cmap=cmap, aspect='equal', interpolation='none')
    cb5 = fig.colorbar(im5, ax=ax_bump_pos, fraction=0.046, pad=0.04)
    ax_bump_pos.set_title("Bump Position")

    # Marker at (start_i, start_j). Note imshow uses x=j, y=i.
    marker_topo = ax_topo.scatter([start_j], [start_i], s=80, facecolors='none',
                             edgecolors='r', linewidths=1.8)
    marker_edge_L = ax_edge_L.scatter([start_j], [start_i], s=80, facecolors='none',
                             edgecolors='r', linewidths=1.8)
    marker_edge_R = ax_edge_R.scatter([start_j], [start_i], s=80, facecolors='none',
                             edgecolors='r', linewidths=1.8)            
    marker_bump_amp = ax_bump_amp.scatter([start_j], [start_i], s=80, facecolors='none',
                             edgecolors='r', linewidths=1.8)            
    marker_bump_pos = ax_bump_pos.scatter([start_j], [start_i], s=80, facecolors='none',
                             edgecolors='r', linewidths=1.8)

    # --- Spectrum
    line, = ax_spec.plot(en, LIY_3d[:, start_i, start_j], color='gray', lw=1.8, label='Raw')
    line_smoothed, = ax_spec.plot(en, LIY_smoothed[:, start_i, start_j]+1e-12, color='blue', lw=1.8, label='Smoothed')
    line_fitted, = ax_spec.plot(en, LIY_fitted[:, start_i, start_j]+2e-12, color='red', lw=1.8, label='Fitted')
    ax_spec.set_ylabel("dI/dV (a.u.)")
    ax_spec.legend()
    title_spec = ax_spec.set_title(f"LDOS at (i,j)=({start_i},{start_j})")
    txt = ax_spec.text(0.98, 0.92, f"({start_i},{start_j})", transform=ax_spec.transAxes,
                       ha='right', va='top', fontsize=10, alpha=0.8)

    if use_global_ylim:
        ymin = np.nanmin(LIY_3d)
        ymax = np.nanmax(LIY_3d)
        if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
            ymin, ymax = -1, 1
        ax_spec.set_ylim(ymin, ymax)

    # --- Residuals
    residuals = LIY_3d - LIY_fitted
    line_residual, = ax_residual.plot(en, residuals[:, start_i, start_j], color='gray', lw=1.8)
    ax_residual.set_xlabel("Bias (V)")
    ax_residual.set_ylabel("Residuals (a.u.)")
    ax_residual.axhline(0, color='black', lw=0.8, ls='--')
    ax_residual.set_ylim(-np.nanmax(np.abs(residuals)), np.nanmax(np.abs(residuals)))

    plt.tight_layout()
    plt.show()
    # --- Update function
    def update(frame):
        i, j = ij_list[frame]
        # update spectrum
        line.set_ydata(LIY_3d[:, i, j])
        line_fitted.set_ydata(LIY_fitted[:, i, j])
        line_smoothed.set_ydata(LIY_smoothed[:, i, j])
        line_residual.set_ydata(LIY_3d[:, i, j] - LIY_fitted[:, i, j])
        
        title_spec.set_text(f"LDOS at (i,j)=({i},{j})")
        txt.set_text(f"({i},{j})")
        if not use_global_ylim:
            ax_spec.relim()
            ax_spec.autoscale_view()

        # move marker
        marker_topo.set_offsets([[j, i]])
        marker_edge_L.set_offsets([[j, i]])
        marker_edge_R.set_offsets([[j, i]])
        marker_bump_amp.set_offsets([[j, i]])
        marker_bump_pos.set_offsets([[j, i]])
        return line, marker_topo, marker_edge_L, marker_edge_R, marker_bump_amp, marker_bump_pos, title_spec, txt

    anim = FuncAnimation(fig, update, frames=len(ij_list),
                         interval=interval, blit=False, repeat=repeat)

    # --- Optional: click on the topo to jump to a pixel
    if click_to_jump:
        def onclick(event):
            if event.inaxes is not ax_topo or event.xdata is None or event.ydata is None:
                return
            j = int(round(event.xdata))
            i = int(round(event.ydata))
            if 0 <= i < I and 0 <= j < J:
                frame = ij_list.index((i, j))
                update(frame)
                fig.canvas.draw_idle()
        fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
    return anim

# ---------- Use it ----------
# Assuming you already smoothed:
# dos_E.LIY_smoothed = smooth_LIY(dos_E.LIY, window=10, axis=0, mode='reflect')

# Run the combined animation/plot:
# anim = animate_ldos_with_topo(dos_E.en, dos_E.LIY_smoothed, dos_E.Z_ls,
#                               interval=80, repeat=True, use_global_ylim=True,
#                               start_ij=(24,40), click_to_jump=True)

# Optional saving (uncomment one):
# anim.save("ldos_with_topo.mp4", writer="ffmpeg", dpi=150, bitrate=1800)
# from matplotlib.animation import PillowWriter
# anim.save("ldos_with_topo.gif", writer=PillowWriter(fps=12))

def plot_rk_space(data, ens=None):
    if ens is None:
        n_ax = 3
    else:
        n_ax = 3 + len(ens)
    # print(n_ax)
    fig, ax = plt.subplots(2, n_ax, figsize=(n_ax*4, 4*2))
    
    extent = (0, data.scan_info['scan_size'], 0, data.scan_info['scan_size'])
     # in nm
    n_pixels = data.scan_info['n_pixels']    # number of pixels along one axis

    ax[0,0].imshow(data.Z_ls, origin='lower',  cmap=stmpy.cm.Blues_r, interpolation='none')
    ax[0,0].set_title('Z line subtracted')

    data.FZ_ls = stmpy.tools.fft(data.Z_ls, zeroDC=True, units='amplitude', output='absolute')
    plot_FFT_data(data.FZ_ls, k_crop_n=0, ax=ax[1,0])

    ax[0,1].imshow(data.shape_para1, origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
    ax[0,1].set_title('Shape Parameter 1')

    data.Fshape_para1 = stmpy.tools.fft(data.shape_para1, zeroDC=True, units='amplitude', output='absolute')
    plot_FFT_data(data.Fshape_para1, k_crop_n=0, ax=ax[1,1])

    ax[0,2].imshow(data.shape_para2, origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
    ax[0,2].set_title('Shape Parameter 2')

    data.Fshape_para2 = stmpy.tools.fft(data.shape_para2, zeroDC=True, units='amplitude', output='absolute')
    plot_FFT_data(data.Fshape_para2, k_crop_n=0, ax=ax[1,2])

    if ens is not None:
        for en_id, en in enumerate(ens):
            # print(en_id)
            id = np.argmin(np.abs(data.en - en))
            ax[0,3+en_id].imshow(data.LIY[id,:,:], origin='lower', cmap=stmpy.cm.Blues_r, interpolation='none')
            ax[0,3+en_id].set_title(f'dI/dV at {en:.2f} V')
            FLIY_idx = stmpy.tools.fft(data.LIY[id,:,:], zeroDC=True, units='amplitude', output='absolute')
            plot_FFT_data(FLIY_idx, k_crop_n=0, ax=ax[1,3+en_id])

    for ax in ax[0,:]:
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        # print(data.scan_info['scan_size'], data.scan_info['n_pixels'])
        stmpy.image.add_scale_bar(5, data.scan_info['scan_size'], data.scan_info['n_pixels'], fs=12, pad=0.1, ax=ax)


def plot_histogram(data, title='Histogram', xlabel='I (pA)', xlim=None):
    plt.figure(figsize=(5, 3))
    plt.hist(data, bins=50, alpha=0.85, edgecolor='none')
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    if xlim is not None:
        plt.xlim(xlim)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def cross_correlation_2d_plot(
    A, B,
    normalize=True,
    subtract_mean=True,
    titleA="A",
    titleB="B",
    plot=False,
    order=1,                  # interpolation order
    preserve_range=True,
    anti_aliasing=True,
    xlim=None,
    ylim=None,
    scalebar=None,
    savename=None
):
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D arrays")

    # -------------------------------------------------
    # Always resample LOW-res image → HIGH-res image
    # -------------------------------------------------
    if A.shape != B.shape:
        try:
            from skimage.transform import resize
        except Exception as e:
            raise ImportError(
                "Resampling requires scikit-image. "
                "Install with: pip install scikit-image"
            ) from e

        if A.size >= B.size:
            # A is high-res → resample B
            target_shape = A.shape
            B = resize(
                B, target_shape,
                order=order,
                mode="reflect",
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing,
            )
            high_res_label = "A"
        else:
            # B is high-res → resample A
            target_shape = B.shape
            A = resize(
                A, target_shape,
                order=order,
                mode="reflect",
                preserve_range=preserve_range,
                anti_aliasing=anti_aliasing,
            )
            high_res_label = "B"
    else:
        high_res_label = "None"  # same shape

    ny, nx = A.shape  # now guaranteed equal

    # --------------------
    # Mean subtraction
    # --------------------
    if subtract_mean:
        A0 = A - np.nanmean(A)
        B0 = B - np.nanmean(B)
    else:
        A0, B0 = A.copy(), B.copy()

    # FFT-safe NaNs
    A0 = np.where(np.isfinite(A0), A0, 0.0)
    B0 = np.where(np.isfinite(B0), B0, 0.0)

    # Zero padding to 2N × 2N
    A_pad = np.zeros((2 * ny, 2 * nx))
    B_pad = np.zeros((2 * ny, 2 * nx))
    A_pad[:ny, :nx] = A0
    B_pad[:ny, :nx] = B0

    # FFT cross-correlation
    FA = np.fft.fft2(A_pad)
    FB = np.fft.fft2(B_pad)
    C = np.fft.ifft2(FA * np.conj(FB)).real
    C = np.fft.fftshift(C)

    if normalize:
        denom = np.sqrt(np.sum(A0 * A0) * np.sum(B0 * B0))
        if denom > 0:
            C /= denom

    # Peak detection (max |corr|)
    iy, ix = np.unravel_index(np.nanargmax(np.abs(C)), C.shape)
    cy, cx = C.shape[0] // 2, C.shape[1] // 2
    dy, dx = iy - cy, ix - cx
    peak_value = C[iy, ix]
    central_value = C[cy, cx]

    # --------------------
    # Plot (optional)
    # --------------------
    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)

        axs[0].imshow(A, origin="lower", cmap=stmpy.cm.Blues_r, interpolation='none')
        axs[0].set_title(f"{titleA} ({'high-res' if high_res_label=='A' else 'resampled'})")

        # plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        stmpy.image.add_colorbar(ax=axs[0], loc=0, label='Topography (m)', fs=8,pad=0.1)

        axs[1].imshow(B, origin="lower", cmap=stmpy.cm.Blues_r, interpolation='none')
        axs[1].set_title(f"{titleB} ({'high-res' if high_res_label=='B' else 'resampled'})")

        stmpy.image.add_colorbar(ax=axs[1], loc=0, label='Topography (m)', fs=8,pad=0.1)

        # plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        vmax = np.nanmax(np.abs(C))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        extent = [-cx, cx, -cy, cy]

        axs[2].imshow(
            C, origin="lower", extent=extent,
            cmap=stmpy.cm.jason_r, norm=norm, interpolation='none'
        )
        if scalebar is not None:
            for ax_ in axs[:3]:
                stmpy.image.add_scale_bar(scalebar[0], scalebar[1], scalebar[2], fs=8, pad=0.1, ax=ax_, color='black')
        axs[2].plot(dx, dy, "ko", ms=3)
        axs[2].axhline(0, ls="--", c="gray")
        axs[2].axvline(0, ls="--", c="gray")
        axs[2].set_title(f"peak={peak_value:.3f} at {np.array(iy)}, {np.array(ix)}, center={central_value:.3f}\n")
        # axs[2].set_xlabel("dx (pixels)")
        axs[2].set_ylabel("dy (pixels)")
        if xlim is not None:
            axs[2].set_xlim(xlim)
        if ylim is not None:
            axs[2].set_ylim(ylim)
        # no xlabels and ticks
        axs[2].set_xticks([])
        stmpy.image.add_colorbar(ax=axs[2], loc=0, label='Correlation', fs=8,pad=0.1)
        # plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, )
        
        x = np.asarray(A).ravel()
        y = np.asarray(B).ravel()

        mask = np.isfinite(x) & np.isfinite(y)
        xm, ym = x[mask], y[mask]

        if xm.size < 2:
            r = np.nan
        else:
            # Compute Pearson r without SciPy
            xzc = xm - xm.mean()
            yzc = ym - ym.mean()
            denom = np.sqrt((xzc**2).sum() * (yzc**2).sum())
            r = float((xzc * yzc).sum() / denom) if denom > 0 else np.nan

        axs[3].scatter(xm, ym, s=10, alpha=0.7)
        axs[3].set_xlabel(titleA)
        axs[3].set_ylabel(titleB)
        # Optional best-fit line
        if xm.size >= 2 and np.isfinite(r):
            try:
                slope, intercept = np.polyfit(xm, ym, 1)
                xs = np.linspace(xm.min(), xm.max(), 100)
                axs[3].plot(xs, slope * xs + intercept, linewidth=2)
            except Exception:
                pass
        axs[3].set_title(f"Pearson r = {r:.3f}" if np.isfinite(r) else "Pearson r is undefined")
        
        if savename is not None:
            fig.savefig(savename)

    return C, (dy, dx), peak_value, central_value

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_topo_and_energy_energy_correlations_sharedx_square(
    topo2d,
    liy3d,
    en,
    *,
    mask=None,
    idxs=None,
    energies=None,
    subtract_mean=True,
    common_mask_topo=True,
    common_mask_energy=True,
    cmap_corr=stmpy.cm.jason_r,
    title_top="Topo–DOS correlation vs energy",
    title_bottom="Zero-shift correlation between DOS maps",
    figsize=(8, 16),
):
    topo = np.asarray(topo2d, float)
    liy = np.asarray(liy3d, float)
    en = np.asarray(en, float)
    order = np.argsort(en)
    en = en[order]
    liy = liy[order, :, :]

    nE, ny, nx = liy.shape
    if topo.shape != (ny, nx):
        raise ValueError("topo2d must match LIY spatial shape")

    if mask is None:
        mask = np.ones((ny, nx), dtype=bool)
    else:
        mask = np.asarray(mask, bool)

    # ---------- Topo–DOS correlation vs energy ----------
    topo_valid = np.isfinite(topo) & mask
    t = topo.copy()
    if subtract_mean and np.any(topo_valid):
        t -= np.nanmean(t[topo_valid])
    t = np.where(topo_valid, t, np.nan)
    t_flat = t.ravel()

    rE = np.full(nE, np.nan, float)
    for i in range(nE):
        y = liy[i].astype(float)
        y_valid = np.isfinite(y) & mask
        good = (topo_valid & y_valid) if common_mask_topo else y_valid
        if good.sum() < 3:
            continue

        yy = y.copy()
        if subtract_mean:
            yy -= np.nanmean(yy[good])

        g = good.ravel()
        xg = t_flat[g]
        yg = yy.ravel()[g]

        ok = np.isfinite(xg) & np.isfinite(yg)
        xg, yg = xg[ok], yg[ok]
        if xg.size < 3:
            continue

        xg -= xg.mean()
        yg -= yg.mean()
        denom = np.sqrt((xg @ xg) * (yg @ yg))
        rE[i] = (xg @ yg) / denom if denom > 0 else np.nan

    # ---------- Energy–Energy correlation matrix ----------
    if idxs is None:
        if energies is None:
            idxs = np.arange(nE)
        else:
            energies = np.asarray(energies, float)
            idxs = np.array([int(np.argmin(np.abs(en - e))) for e in energies], dtype=int)
    else:
        idxs = np.asarray(idxs, int)

    idxs = np.unique(idxs)
    ens = en[idxs]
    cube = liy[idxs].astype(float)

    finite_cube = np.isfinite(cube)
    if common_mask_energy:
        m = np.all(finite_cube, axis=0) & mask
        cube = np.where(m[None, :, :], cube, np.nan)
    else:
        cube = np.where(mask[None, :, :], cube, np.nan)

    X = cube.reshape(len(idxs), -1)
    if subtract_mean:
        X = X - np.nanmean(X, axis=1)[:, None]
    X = np.where(np.isfinite(X), X, 0.0)

    G = X @ X.T
    norms = np.sqrt(np.sum(X * X, axis=1))
    denom = np.outer(norms, norms)
    with np.errstate(invalid="ignore", divide="ignore"):
        Cmat = np.where(denom > 0, G / denom, np.nan)
    Cmat = np.clip(Cmat, -1.0, 1.0)

    # ---------- Plot (share x; bottom square; inset colorbar) ----------
    fig, (ax0, ax1) = plt.subplots(
        2, 1,
        figsize=figsize,
        # sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 3]}
    )

    # top
    ax0.scatter(en, rE, s=1)
    ax0.axhline(0, ls="--", lw=1, color="gray")
    ax0.label_outer()

    # bottom
    vmax = np.nanmax(np.abs(Cmat))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    # set the diagonal to zero for visualization
    # np.fill_diagonal(Cmat, 0.0)

    im = ax1.imshow(
        Cmat,
        origin="lower",
        cmap=cmap_corr,
        norm=norm,
        extent=[ens[0], ens[-1], ens[0], ens[-1]],
        aspect="equal", interpolation='none'
    )
    ax1.set_box_aspect(1)     # square panel
    ax1.set_xlim(ens[0], ens[-1])  # enforce shared x-range to match matrix
    ax1.set_xlabel("Bias (V)")
    ax1.set_ylabel("Bias (V)")

    # inset colorbar snug to the right of the matrix
    cax = inset_axes(
        ax1,
        width="6%", height="100%",
        loc="lower left",
        bbox_to_anchor=(1.05, 0.0, 1, 1),
        bbox_transform=ax1.transAxes,
        borderpad=0,
    )

    cb = fig.colorbar(im, cax=cax)
    # stmpy.image.add_colorbar(ax=ax1, loc=1, label='FFT Amplitude', fs=8)
    # add title
    ax0.set_title(title_top)
    plt.show()
    return rE, Cmat, ens, idxs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_LIY_groups_at_energy_with_binmap(
    en,
    LIY_grouped,
    bin_maps,
    E_ref,
    *,
    energy_is_index=False,
    cmap="viridis",
    linewidth=2.0,
    alpha=0.9,
    vmin=None,
    vmax=None,
):
    """
    Left: grouped spectra at chosen reference energy.
    Right: bin assignment map (which pixels belong to which percentile bin).
    """
    en = np.asarray(en)
    LIY_grouped = np.asarray(LIY_grouped)
    bin_maps = np.asarray(bin_maps)

    nE_ref, nE, n_groups = LIY_grouped.shape
    if bin_maps.shape[0] != nE_ref:
        raise ValueError("bin_maps first dimension must match LIY_grouped first dimension")

    if not energy_is_index:
        k_ref = int(np.argmin(np.abs(en - E_ref)))
    else:
        k_ref = int(E_ref)

    if k_ref < 0 or k_ref >= nE_ref:
        raise IndexError("Reference energy index out of range")

    fig, (ax0, ax2, ax1, axL) = plt.subplots(
        1, 4, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1.0, 1.0, 1]}
    )

    # --- colors for groups (same for lines and map) ---
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0.15, 0.95, n_groups))
    disc_cmap = ListedColormap(colors)

    # --- spectra panel ---
    specs = LIY_grouped[k_ref]  # (nE, n_groups)
    for g in range(n_groups):
        spec = LIY_grouped[k_ref, :, g]
        if np.all(np.isnan(spec)):
            continue
        ax0.plot(
            en,
            spec,
            color=colors[g],
            lw=linewidth,
            alpha=alpha,
            label=f"Bin {g+1}",
        )

        ax2.plot(en, spec - LIY_grouped[k_ref, :, 0], color=colors[g], lw=linewidth, alpha=alpha)

    ax0.set_xlabel("Bias (V)")
    ax0.set_ylabel("dI/dV (a.u.)")
    ax0.set_title(f"Reference bias = {en[k_ref]:.3g} V")
    # ax0.legend(fontsize=10, frameon=False)
    ax2.set_xlabel("Bias (V)")
    ax2.set_ylabel("dI/dV (a.u.) - Bin 1")
    ax2.set_title("Same spectra, shifted to Bin 1 baseline")
    # ax2.legend(fontsize=10, frameon=False)

    # --- bin map panel ---
    bm = bin_maps[k_ref]  # (Ny, Nx) with 0..n_groups-1 or -1
    bm_masked = np.ma.masked_where(bm < 0, bm)

    im = ax1.imshow(
        bm_masked,
        cmap=disc_cmap,
        interpolation="none",
        vmin=0 if vmin is None else vmin,
        vmax=(n_groups - 1) if vmax is None else vmax,
        origin="lower",
    )
    ax1.set_title("Bin map at reference bias")
    ax1.set_xticks([])
    ax1.set_yticks([])

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(n_groups))
    cbar.set_ticklabels([str(i + 1) for i in range(n_groups)])
    cbar.set_label("Bin # (low → high percentile)")

    fig.tight_layout()
    
    return fig, (ax0, ax2, ax1, axL), bm_masked, specs

def group_LIY_by_intensity_at_each_energy(
    en, LIY, LIY_original, n_groups=10, use_nanmean=True, return_bin_maps=True, return_edges=True
):
    """
    Group pixels by LIY intensity at each reference energy, then average full spectra per group.

    Returns:
      en_sorted: (nE,)
      LIY_grouped: (nE_ref, nE, n_groups)
      bin_maps (optional): (nE_ref, Ny, Nx) with bin index per pixel, or -1 if invalid.
      edges_all (optional): (nE_ref, n_groups+1) edges used at each reference energy (NaN if invalid)
    """
    en = np.asarray(en)
    LIY = np.asarray(LIY)

    if LIY.ndim != 3:
        raise ValueError("LIY must have shape (n_energy, Ny, Nx)")

    nE, Ny, Nx = LIY.shape
    if en.shape[0] != nE:
        raise ValueError("en length must match LIY.shape[0]")

    # ---- sort energies ----
    order = np.argsort(en)
    en_sorted = en[order]
    LIY_sorted = LIY[order, :, :]
    LIY_original_sorted = LIY_original[order, :, :]
    # ---- flatten pixels ----
    Npix = Ny * Nx
    LIY_flat = LIY_sorted.reshape(nE, Npix)  # (nE, Npix)
    LIY_original_flat = LIY_original_sorted.reshape(nE, Npix)  # (nE, Npix)

    mean_fn = np.nanmean if use_nanmean else np.mean
    LIY_grouped = np.full((nE, nE, n_groups), np.nan, dtype=float)
    LIY_original_grouped = np.full((nE, nE, n_groups), np.nan, dtype=float)
    # bin map output
    bin_maps = None
    if return_bin_maps:
        bin_maps = np.full((nE, Ny, Nx), -1, dtype=int)

    # edges output
    edges_all = None
    if return_edges:
        edges_all = np.full((nE, n_groups + 1), np.nan, dtype=float)

    edges_pct = np.linspace(0, 100, n_groups + 1)

    for k_ref in range(nE):
        vals = LIY_flat[k_ref, :]  # (Npix,)

        good = np.isfinite(vals)
        if not np.any(good):
            continue

        vals_good = vals[good]
        edges = np.percentile(vals_good, edges_pct)

        if return_edges:
            edges_all[k_ref] = edges

        # We'll fill a 1D bin index array for this reference energy
        if return_bin_maps:
            bin_idx_1d = np.full(Npix, -1, dtype=int)

        idx_good = np.where(good)[0]

        for g in range(n_groups):
            lo, hi = edges[g], edges[g + 1]

            if g < n_groups - 1:
                in_bin_good = (vals_good >= lo) & (vals_good < hi)
            else:
                in_bin_good = (vals_good >= lo) & (vals_good <= hi)

            if not np.any(in_bin_good):
                continue

            pix_in_bin = idx_good[in_bin_good]  # indices in [0..Npix-1]

            # assign bin id
            if return_bin_maps:
                bin_idx_1d[pix_in_bin] = g

            # average full spectrum over these pixels
            LIY_grouped[k_ref, :, g] = mean_fn(LIY_flat[:, pix_in_bin], axis=1)
            LIY_original_grouped[k_ref, :, g] = mean_fn(LIY_original_flat[:, pix_in_bin], axis=1)

        if return_bin_maps:
            bin_maps[k_ref] = bin_idx_1d.reshape(Ny, Nx)

    out = [en_sorted, LIY_grouped, LIY_original_grouped]
    if return_bin_maps:
        out.append(bin_maps)
    if return_edges:
        out.append(edges_all)

    return tuple(out)


def plot_LIY_groups_at_energy_with_binmap_and_LIYmap(
    en,
    LIY_sorted,          # <-- pass the *sorted* LIY cube (same ordering as en)
    LIY_grouped,
    bin_maps,
    edges_all,           # <-- (nE_ref, n_groups+1)
    E_ref,
    *,
    energy_is_index=False,
    cmap="viridis",
    linewidth=2.0,
    alpha=0.9,
    v_offset=0,
):
    """
    Panels:
      (1) grouped spectra at chosen reference energy
      (2) same spectra shifted to Bin 1 baseline
      (3) bin assignment map
      (4) original LIY map (unbinned) colored so bin centers match bin colors
    """
    en = np.asarray(en)
    LIY_sorted = np.asarray(LIY_sorted)
    LIY_grouped = np.asarray(LIY_grouped)
    bin_maps = np.asarray(bin_maps)
    edges_all = np.asarray(edges_all)

    nE_ref, nE, n_groups = LIY_grouped.shape

    if not energy_is_index:
        k_ref = int(np.argmin(np.abs(en - E_ref)))
    else:
        k_ref = int(E_ref)

    if k_ref < 0 or k_ref >= nE_ref:
        raise IndexError("Reference energy index out of range")

    # --- colors for groups (same for lines and both maps) ---
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0.15, 0.95, n_groups))
    disc_cmap = ListedColormap(colors)

    fig, (ax0, ax2, ax1, axL) = plt.subplots(
        1, 4, figsize=(18, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1]}
    )
    
    specs = LIY_grouped[k_ref]  # (nE, n_groups)
    # --- spectra panel ---
    for g in range(n_groups):
        spec = LIY_grouped[k_ref, :, g]
        if np.all(np.isnan(spec)):
            continue
        ax0.plot(en, spec+v_offset, color=colors[g], lw=linewidth, alpha=alpha, label=f"Bin {g+1}")
        ax2.plot(en, spec - LIY_grouped[k_ref, :, 0], color=colors[g], lw=linewidth, alpha=alpha)

    ax0.set_xlabel("Bias (V)")
    ax0.set_ylabel("dI/dV (a.u.)")
    ax0.set_title(f"Reference bias = {en[k_ref]:.3g} V")

    ax2.set_xlabel("Bias (V)")
    ax2.set_ylabel("dI/dV (a.u.) - Bin 1")
    ax2.set_title("Shifted to Bin 1 baseline")

    # --- bin map panel ---
    bm = bin_maps[k_ref]  # (Ny, Nx) with 0..n_groups-1 or -1
    bm_masked = np.ma.masked_where(bm < 0, bm)
    im_bm = ax1.imshow(
        bm_masked,
        cmap=disc_cmap,
        interpolation="none",
        vmin=0,
        vmax=n_groups - 1,
        origin="lower",
    )
    ax1.set_title("Bin map (percentile groups)")
    ax1.set_xticks([])
    ax1.set_yticks([])

    cbar_bm = fig.colorbar(im_bm, ax=ax1, fraction=0.046, pad=0.04)
    cbar_bm.set_ticks(np.arange(n_groups))
    cbar_bm.set_ticklabels([str(i + 1) for i in range(n_groups)])
    cbar_bm.set_label("Bin # (low → high)")

    from matplotlib.colors import LinearSegmentedColormap, Normalize

    # --- original LIY map panel, continuous colormap pinned at bin centers ---
    LIY_map = LIY_sorted[k_ref, :, :]
    LIY_map_masked = np.ma.masked_invalid(LIY_map)

    edges = edges_all[k_ref]                 # (n_groups+1,)
    centers = 0.5 * (edges[:-1] + edges[1:]) # (n_groups,)

    vmin, vmax = edges[0], edges[-1]
    print(vmin, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # same group colors you already use for lines/binmap:
    # colors shape (n_groups, 4)

    # Build anchor points in [0,1] with exact colors at bin centers
    anchors = [(norm(vmin), colors[0])]
    anchors += [(norm(c), colors[g]) for g, c in enumerate(centers)]
    anchors += [(norm(vmax), colors[-1])]

    cont_cmap = LinearSegmentedColormap.from_list("LIY_pinned", anchors, N=256)

    im_L = axL.imshow(
        LIY_map_masked,
        cmap=cont_cmap,
        norm=norm,
        interpolation="none",
        origin="lower",
    )

    axL.set_title(f"Original LIY map (continuous, pinned at bin centers) [{vmin*1e11:0.3g}, {vmax*1e11:0.3g}] 10^-11")
    axL.set_xticks([])
    axL.set_yticks([])

    cbar_L = fig.colorbar(im_L, ax=axL, fraction=0.046, pad=0.04)
    cbar_L.set_ticks(centers)  # optional: label the bin centers
    cbar_L.set_ticklabels([f"{c:.2g}" for c in centers])
    cbar_L.set_label("LIY (bin centers pinned to bin colors)")

    fig.tight_layout()
    return fig, (ax0, ax2, ax1, axL), bm_masked, specs


def plot_LIY_groups_at_energy(
    en,
    LIY_grouped,
    E_ref,
    *,
    energy_is_index=False,
    cmap="viridis",
    linewidth=2.0,
    alpha=0.9,
    ax=None,
):
    """
    Plot grouped LIY spectra for a chosen reference energy.

    Args
    ----
    en : (nE,) array
        Energy axis (sorted, same as returned by group_LIY_by_intensity_at_each_energy)
    LIY_grouped : (nE_ref, nE, n_groups) array
        Output of group_LIY_by_intensity_at_each_energy
    E_ref : float or int
        Reference energy value (float) or index (int)
    energy_is_index : bool
        If True, interpret E_ref as an index
    cmap : str or matplotlib colormap
        Colormap for group gradient (low → high percentile)
    linewidth : float
        Line width
    alpha : float
        Line transparency
    ax : matplotlib axis or None
        If None, create a new figure

    Returns
    -------
    ax : matplotlib axis
    """
    en = np.asarray(en)
    LIY_grouped = np.asarray(LIY_grouped)

    nE_ref, nE, n_groups = LIY_grouped.shape

    if not energy_is_index:
        # find nearest energy index
        k_ref = int(np.argmin(np.abs(en - E_ref)))
    else:
        k_ref = int(E_ref)

    if k_ref < 0 or k_ref >= nE_ref:
        raise IndexError("Reference energy index out of range")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(0.15, 0.95, n_groups))

    for g in range(n_groups):
        spec = LIY_grouped[k_ref, :, g]
        if np.all(np.isnan(spec)):
            continue

        ax.plot(
            en,
            spec,
            color=colors[g],
            lw=linewidth,
            alpha=alpha,
            label=f"Bin {g+1}",
        )

    ax.set_xlabel("Bias (V)")
    ax.set_ylabel("dI/dV (a.u.)")
    ax.set_title(f"Reference bias = {en[k_ref]:.3g} V", fontsize=12)
    ax.legend(fontsize=12, frameon=False)

    return ax


# ------------------------
# basic lorentzian
# ------------------------
def lorentz(x, A, x0, gamma):
    return A * (gamma**2) / ((x - x0)**2 + gamma**2)

def lorentz_0(x, A, gamma):
    return A * (gamma**2) / (x**2 + gamma**2)

def lorentz_plus_linear_bg(x, A, x0, gamma, m, b):
    return lorentz(x, A, x0, gamma) + m * x + b

# ------------------------
# model
# ------------------------
def model_center_plus_two_pairs(
    x,
    x0,
    A0, gamma0,
    A1, delta1, gamma1,
    A2, delta2, gamma2,
    offset
):
    y = lorentz(x, A0, x0, gamma0)

    # pair 1
    y += lorentz(x, A1, x0 + delta1, gamma1)
    y += lorentz(x, A1, x0 - delta1, gamma1)

    # pair 2
    y += lorentz(x, A2, x0 + delta2, gamma2)
    y += lorentz(x, A2, x0 - delta2, gamma2)

    return y + offset

def model_0center_plus_two_pairs(
    x,
    A0, gamma0,
    A1, delta1, gamma1,
    A2, delta2, gamma2,
    offset
):
    y = lorentz(x, A0, 0, gamma0)

    # pair 1
    y += lorentz(x, A1, 0 + delta1, gamma1)
    y += lorentz(x, A1, 0 - delta1, gamma1)

    # pair 2
    y += lorentz(x, A2, 0 + delta2, gamma2)
    y += lorentz(x, A2, 0 - delta2, gamma2)

    return y + offset


# ------------------------
# fitting + plotting
# ------------------------
def fit_center_and_two_pairs(
    x,
    y,
    p0,
    bounds=None,
    plot=False,
    maxfev=20000,
    yscale_log=False,
    return_lines=False,
):
    """
    Fit a central Lorentzian + two symmetric Lorentzian pairs.

    Parameter order:
      [ x0,
        A0, gamma0,
        A1, delta1, gamma1,
        A2, delta2, gamma2,
        offset ]
    """

    param_names = [
        "x0",
        "A0", "gamma0",
        "A1", "delta1", "gamma1",
        "A2", "delta2", "gamma2",
        "offset"
    ]

    # fit
    if bounds is None:
        popt, pcov = curve_fit(
            model_center_plus_two_pairs,
            x, y, p0=p0, maxfev=maxfev
        )
    else:
        popt, pcov = curve_fit(
            model_center_plus_two_pairs,
            x, y, p0=p0, bounds=bounds, maxfev=maxfev
        )

    # plotting
    if plot:
        x0, A0, g0, A1, d1, g1, A2, d2, g2, off = popt
        xs = np.linspace(x.min(), x.max(), 1000)
        y_fits = model_center_plus_two_pairs(xs, *popt)
        

        y_centers = lorentz(xs, A0, x0, g0) + off
        y_center = lorentz(x, A0, x0, g0) + off
        y_pair1s = (
            lorentz(xs, A1, x0 + d1, g1)
            + lorentz(xs, A1, x0 - d1, g1)
        )
        y_pair2s = (
            lorentz(xs, A2, x0 + d2, g2)
            + lorentz(xs, A2, x0 - d2, g2)
        )
        y_BPonly = y - lorentz(x, A0, x0, g0) - off - (
            lorentz(x, A1, x0 + d1, g1)
            + lorentz(x, A1, x0 - d1, g1)
        ) - (
            lorentz(x, A2, x0 + d2, g2)
            + lorentz(x, A2, x0 - d2, g2)
        )
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, "k.", ms=2, label="data")
        plt.plot(xs, y_fits, "r-", lw=1, label="total fit")
        plt.plot(xs, y_centers, "--", lw=1, label="central Lorentzian")
        plt.plot(xs, y_pair1s, "--", lw=1, label="pair 1")
        plt.plot(xs, y_pair2s, "--", lw=1, label="pair 2")

        plt.xlabel("x")
        plt.ylabel("intensity")
        plt.legend()
        plt.tight_layout()

        if yscale_log:
            plt.yscale("log")
        delta1 = popt[4]
        delta2 = popt[7]
        # plot x = delta1 and x = delta2 lines
        plt.axvline(delta1, ls="--", color="blue", label="delta1")
        plt.axvline(-delta1, ls="--", color="blue")
        plt.axvline(delta2, ls="--", color="orange", label="delta2")
        plt.axvline(-delta2, ls="--", color="orange")
        delta1_err = np.sqrt(pcov[4, 4])
        delta2_err = np.sqrt(pcov[7, 7])
        # ratio
        ratio = delta2/delta1
        ratio_err = ratio * np.sqrt(
            (delta1_err / delta1)**2 + (delta2_err / delta2)**2
        )
        plt.title(f"charge ordering period/lattice constant = {ratio:.2f} ± {ratio_err:.2f}")
        plt.show()
        # Fitted lines
        if return_lines:
            return popt, pcov, param_names, x, y_BPonly, xs, y_fits, y_centers, y_pair1s, y_pair2s
    return popt, pcov, param_names, 


def fit_0center_and_two_pairs(
    x,
    y,
    p0,
    bounds=None,
    plot=False,
    maxfev=20000,
    yscale_log=False,
    return_lines=False,
):
    param_names = [
        "x0",
        "A0", "gamma0",
        "A1", "delta1", "gamma1",
        "A2", "delta2", "gamma2",
        "offset"
    ]

    # fit
    if bounds is None:
        popt, pcov = curve_fit(
            model_0center_plus_two_pairs,
            x, y, p0=p0, maxfev=maxfev
        )
    else:
        popt, pcov = curve_fit(
            model_0center_plus_two_pairs,
            x, y, p0=p0, bounds=bounds, maxfev=maxfev
        )

    # plotting
    if plot:
        A0, g0, A1, d1, g1, A2, d2, g2, off = popt
        xs = np.linspace(x.min(), x.max(), 1000)
        y_fits = model_0center_plus_two_pairs(xs, *popt)
        

        y_centers = lorentz(xs, A0, 0, g0) + off
        y_center = lorentz(x, A0, 0, g0) + off
        y_pair1s = (
            lorentz(xs, A1, 0 + d1, g1)
            + lorentz(xs, A1, 0 - d1, g1)
        )
        y_pair2s = (
            lorentz(xs, A2, 0 + d2, g2)
            + lorentz(xs, A2, 0 - d2, g2)
        )
        y_BPonly = y - lorentz(x, A0, 0, g0) - off - (
            lorentz(x, A1, 0 + d1, g1)
            + lorentz(x, A1, 0 - d1, g1)
        ) - (
            lorentz(x, A2, 0 + d2, g2)
            + lorentz(x, A2, 0 - d2, g2)
        )
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, "k.", ms=2, label="data")
        plt.plot(xs, y_fits, "r-", lw=1, label="total fit")
        plt.plot(xs, y_centers, "--", lw=1, label="central Lorentzian")
        plt.plot(xs, y_pair1s, "--", lw=1, label="pair 1")
        plt.plot(xs, y_pair2s, "--", lw=1, label="pair 2")

        plt.xlabel("x")
        plt.ylabel("intensity")
        plt.legend()
        plt.tight_layout()

        if yscale_log:
            plt.yscale("log")
        delta1 = popt[3]
        delta2 = popt[6]
        # plot x = delta1 and x = delta2 lines
        plt.axvline(delta1, ls="--", color="blue", label="delta1")
        plt.axvline(-delta1, ls="--", color="blue")
        plt.axvline(delta2, ls="--", color="orange", label="delta2")
        plt.axvline(-delta2, ls="--", color="orange")
        delta1_err = np.sqrt(pcov[3, 3])
        delta2_err = np.sqrt(pcov[6, 6])
        # ratio
        ratio = delta2/delta1
        ratio_err = ratio * np.sqrt(
            (delta1_err / delta1)**2 + (delta2_err / delta2)**2
        )
        plt.title(f"charge ordering period/lattice constant = {ratio:.2f} ± {ratio_err:.2f}")
        plt.show()
        # Fitted lines
        if return_lines:
            return popt, pcov, param_names, x, y_BPonly, xs, y_fits, y_centers, y_pair1s, y_pair2s
    return popt, pcov, param_names, 

def model_center_plus_one_pair(x,
                              x0,            # center position
                              A0, gamma0,    # central Lorentzian
                              A1, delta1, gamma1,  # symmetric pair at x0±delta1
                              offset):       # constant background
    y = lorentz(x, A0, x0, gamma0)
    # symmetric pair
    y += lorentz(x, A1, x0 + delta1, gamma1)
    y += lorentz(x, A1, x0 - delta1, gamma1)
    return y + offset

def model_0center_plus_one_pair(x,
                              A0, gamma0,    # central Lorentzian
                              A1, delta1, gamma1,  # symmetric pair at x0±delta1
                              offset):       # constant background
    y = lorentz(x, A0, 0, gamma0)
    # symmetric pair
    y += lorentz(x, A1, delta1, gamma1)
    y += lorentz(x, A1, -delta1, gamma1)
    return y + offset

# --- Fit function with optional plotting ---
def fit_center_and_one_pair(x, y, p0, bounds=None, plot=False, show_individual_peaks=False, maxfev=20000, ax=None, yscale_log=False, 
                            mask_values=None,
                            return_lines=False):
    """
    Fit central Lorentzian + one symmetric Lorentzian pair + constant offset.

    Parameter order for p0 / popt:
      [ x0,
        A0, gamma0,
        A1, delta1, gamma1,
        offset ]

    Returns:
      popt, pcov, param_names, model_callable
    """
    param_names = [
        "x0",
        "A0", "gamma0",
        "A1", "delta1", "gamma1",
        "offset"
    ]

    # perform fit
    if bounds is None:
        popt, pcov = curve_fit(model_center_plus_one_pair, x, y, p0=p0, maxfev=maxfev)
    else:
        popt, pcov = curve_fit(model_center_plus_one_pair, x, y, p0=p0, bounds=bounds, maxfev=maxfev)

    # plotting
    if plot:
        x0, A0, g0, A1, d1, g1, off = popt
        xs = np.linspace(x.min(), x.max(), 1000)
        y_fit = model_center_plus_one_pair(xs, *popt)
        y_center = lorentz(xs, A0, x0, g0) + off
        y_pair = lorentz(xs, A1, x0 + d1, g1) + lorentz(xs, A1, x0 - d1, g1)
        y_BPonly = y - lorentz(x, A0, x0, g0) - off - (
            lorentz(x, A1, x0 + d1, g1)
            + lorentz(x, A1, x0 - d1, g1)
        )
        xplots = (xs-x0) / d1
        xplot = (x-x0) / d1
        if mask_values is None:
            mask_values = (x.min()/d1, x.max()/d1)
        mask = (xplot >= mask_values[0]) & (xplot <= mask_values[1])
        masks = (xplots >= mask_values[0]) & (xplots <= mask_values[1])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(
            xplot[mask], y[mask],
            s=4,              # size (roughly similar to ms=4)
            marker='o',
            facecolors='none', # hollow
            edgecolors='k',    # black outline
            linewidths=0.8,
            label='data'
        )
        ax.plot(xplots[masks], y_fit[masks], 'r-', lw=0.5, label='total fit')
        ax.fill_between(xplots[masks], 0, y_center[masks],  color='gray', alpha=0.25, label='central Lorentzian')
        ax.fill_between(xplots[masks], 0, y_pair[masks],  color='green', alpha=0.25, label='pair: +peak')

        ax.set_xlabel('x')
        ax.set_ylabel('intensity')
        # ax.legend()
        ax.set_xlim(mask_values[0], mask_values[1])
        if yscale_log:
            ax.set_yscale('log')
        if return_lines:
            return popt, pcov, param_names, model_center_plus_one_pair, xplot, y_BPonly, y, xplots, y_fit, y_center, y_pair, ax
    return popt, pcov, param_names, model_center_plus_one_pair, ax


def fit_0center_and_one_pair(x, y, p0, bounds=None, plot=False, show_individual_peaks=False, maxfev=20000, ax=None, yscale_log=False, 
                            mask_values=None,
                            return_lines=False):

    param_names = [
        "A0", "gamma0",
        "A1", "delta1", "gamma1",
        "offset"
    ]

    # perform fit
    if bounds is None:
        popt, pcov = curve_fit(model_0center_plus_one_pair, x, y, p0=p0, maxfev=maxfev)
    else:
        popt, pcov = curve_fit(model_0center_plus_one_pair, x, y, p0=p0, bounds=bounds, maxfev=maxfev)

    # plotting
    if plot:
        A0, g0, A1, d1, g1, off = popt
        xs = np.linspace(x.min(), x.max(), 1000)
        y_fit = model_0center_plus_one_pair(xs, *popt)
        y_center = lorentz(xs, A0, 0, g0) + off
        y_pair = lorentz(xs, A1, 0 + d1, g1) + lorentz(xs, A1, 0 - d1, g1)
        y_BPonly = y - lorentz(x, A0, 0, g0) - off - (
            lorentz(x, A1, 0 + d1, g1)
            + lorentz(x, A1, 0 - d1, g1)
        )
        xplots = xs
        xplot = x
        if mask_values is None:
            mask_values = (x.min(), x.max())
        mask = (xplot >= mask_values[0]) & (xplot <= mask_values[1])
        masks = (xplots >= mask_values[0]) & (xplots <= mask_values[1])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(
            xplot[mask], y[mask],
            s=4,              # size (roughly similar to ms=4)
            marker='o',
            facecolors='none', # hollow
            edgecolors='k',    # black outline
            linewidths=0.8,
            label='data'
        )
        ax.plot(xplots[masks], y_fit[masks], 'r-', lw=0.5, label='total fit')
        ax.fill_between(xplots[masks], 0, y_center[masks],  color='gray', alpha=0.25, label='central Lorentzian')
        ax.fill_between(xplots[masks], 0, y_pair[masks],  color='green', alpha=0.25, label='pair: +peak')

        ax.set_xlabel('x')
        ax.set_ylabel('intensity')
        # ax.legend()
        ax.set_xlim(mask_values[0], mask_values[1])
        if yscale_log:
            ax.set_yscale('log')
        if return_lines:
            return popt, pcov, param_names, model_0center_plus_one_pair, xplot, y_BPonly, y, xplots, y_fit, y_center, y_pair, ax
    return popt, pcov, param_names, model_0center_plus_one_pair, ax

def crop_square_center_max(C, center_on="max", prefer_odd=True, return_slices=False):
    """
    Crop a square subarray from 2D array C so that the chosen target point
    (by default the global maximum) is centered as well as possible.

    Parameters
    ----------
    C : (H, W) array_like
        Input 2D data.
    center_on : {"max"} or (row, col)
        - "max": center on global maximum (argmax)
        - (r, c): explicitly center on this index
    prefer_odd : bool
        If True, make the cropped square size odd so the center is a single pixel.
    return_slices : bool
        If True, also return (row_slice, col_slice)

    Returns
    -------
    Cc : (L, L) ndarray
        Cropped square.
    (optional) row_slice, col_slice : slice
        The slices applied to the original array.
    """
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError("C must be a 2D array")

    H, W = C.shape

    if center_on == "max":
        r0, c0 = np.unravel_index(np.nanargmax(np.abs(C)), C.shape)
    else:
        r0, c0 = center_on
        if not (0 <= r0 < H and 0 <= c0 < W):
            raise ValueError("center_on index out of bounds")

    # Max half-size allowed while keeping (r0,c0) inside the crop
    top = r0
    bottom = H - 1 - r0
    left = c0
    right = W - 1 - c0
    half = min(top, bottom, left, right)  # symmetric half-span

    # Choose square side length
    L = 2 * half + 1
    if not prefer_odd:
        # allow even length by trimming 1 if needed
        pass

    # If we want odd and L is odd already; good.
    # If prefer_odd=False, you could also choose L=2*half or 2*half+1; we keep odd by default.

    r_start = r0 - half
    r_end = r0 + half + 1
    c_start = c0 - half
    c_end = c0 + half + 1

    rs = slice(r_start, r_end)
    cs = slice(c_start, c_end)
    Cc = C[rs, cs]

    if return_slices:
        return Cc, rs, cs
    return Cc

import numpy as np

def crop_center(arr, frac=0.75):
    """
    Crop the central `frac` of an array in the last two dimensions.
    Works for (Ny, Nx) or (..., Ny, Nx).
    """
    if not (0 < frac <= 1):
        raise ValueError("frac must be in (0, 1].")

    Ny, Nx = arr.shape[-2], arr.shape[-1]
    cy, cx = Ny // 2, Nx // 2

    hy = int(round((frac * Ny) / 2))
    hx = int(round((frac * Nx) / 2))

    y0, y1 = cy - hy, cy + hy
    x0, x1 = cx - hx, cx + hx

    # clamp just in case rounding hits edges
    y0, y1 = max(0, y0), min(Ny, y1)
    x0, x1 = max(0, x0), min(Nx, x1)

    if arr.ndim == 2:
        return arr[y0:y1, x0:x1]
    else:
        return arr[..., y0:y1, x0:x1]
