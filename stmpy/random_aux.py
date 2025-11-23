import stmpy
import stmpy.driftcorr as dfc
import matplotlib.patches as patches
import pylab
import pandas
import rhk_sm4.rhk_sm4 as sm4

import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import cumtrapz, trapz

from importlib import reload

from pathlib import Path
import os
import stmpy
import matplotlib.pyplot as plt


def plot_FFT_data(data, 
                  k_crop_n = 0, 
                  k_box_center=None,  # e.g. (x, y) to center a box
                  k_box_size=None,    # e.g. (nx, ny) to define box
                  cmap=stmpy.cm.gray_r,  # colormap
                  clim=None,                 # e.g. (vmin, vmax) to force limits
                  sigma=3,                 # default: +/- 2 std
                  center='mean',             # 'mean', 0, or a numeric value
                  prc=None,                  # e.g. (1, 99) to use percentiles instead of std
                  ax=None,
                  add_colorbar=False):
    
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


    ax.imshow(arr, origin='lower', cmap=cmap, clim=clim)
    if add_colorbar:
        stmpy.image.add_colorbar(ax=ax, loc=0, label='FFT Amplitude', fs=8)
    # ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig, ax

def _process_pipeline(Z):
    
    Z_gc = stmpy.tools.nsigma_global(Z, n=4, M=5, repeat=100)
    Z_lc = stmpy.tools.nsigma_local(Z_gc, n=5, N=4, M=5, repeat=50)

    Z_ls = stmpy.tools.lineSubtract(Z_lc, 2)
    Z_ps = stmpy.tools.plane_subtract(Z_lc, 2, include_cross_terms=True, preserve_units=True)
    FZ_ls = stmpy.tools.fft(Z_lc, zeroDC=True, units='amplitude', output='absolute')
    FZ_ps = stmpy.tools.fft(Z_ps, zeroDC=True, units='amplitude', output='absolute')
    return Z_gc, Z_lc, Z_ls, FZ_ls, Z_ps, FZ_ps


def add_corrections_and_plot(data, dos_map: bool = False, 
                             r_crop_n=0,
                             r_box_center=None,
                             r_box_size=None,
                             idx= None, 
                             add_colorbar=False, 
                             colorbar_range=None,
                             colorbar_range_ps=None,
                             add_label=True,
                             savepath=None, savename=None, show=True, return_figs=False, silent=True):
    """
    Parameters
    ----------
    data : object with at least `Z` (2D array). Optionally:
           - `LIY` (3D array: [E, Y, X]) and `en` (1D array of energies, len E)
    dos_map : if True, also process/plot dI/dV (LIY) maps
    idx : energy index for the per-slice DOS plots

    Side effects
    ------------
    Adds to `data`:
      - Z_gc, Z_lc, Z_ls
      - (if dos_map) LIY_gc, LIY_lc

    Returns
    -------
    figs : dict
        {
          'topo': (fig_topo, ax_topo_array),
          'dos_mean': (fig_dos_mean, ax_dos_mean_array),        # if dos_map
          'dos_idx': (fig_dos_idx, ax_dos_idx_array)            # if dos_map
        }
    """

    scan_size = data.scan_info['scan_size']  # in nm
    n_pixels = data.scan_info['n_pixels']    
    set_current = data.scan_info['set_current']  # in pA
    set_voltage = data.scan_info['set_voltage']  # in V

    figs = {}

    # --- Topography corrections ---
    data.Z_gc, data.Z_lc, data.Z_ls, data.FZ_ls, data.Z_ps, data.FZ_ps = _process_pipeline(data.Z)
    data.FZ_ls = stmpy.tools.fft(data.Z_ls, zeroDC=True, window='hanning', units='amplitude', output='absolute')
    k_kept_n = 256/60 * scan_size;
    k_crop_n = int(0.5 * (n_pixels - k_kept_n))  # crop 10 nm in FFT
    # print(k_kept_n,k_crop_n)
    k_crop_n = 0 if k_crop_n < 0 else k_crop_n
    
    if hasattr(data, "Z_BWD"):
        fig_topo, ax_topo = plt.subplots(2, 7, figsize=(23, 10))
        ax_topo[0,0].imshow(data.Z,     cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[0,0].set_title('Raw Z')
        ax_topo[0,1].imshow(data.Z_gc,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[0,1].set_title('Global Corrected Z')
        ax_topo[0,2].imshow(data.Z_lc,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[0,2].set_title('Local Corrected Z')
        ax_topo[0,3].imshow(data.Z_ls,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[0,3].set_title('Line Subtracted Z')
        ax_topo[0,4].imshow(data.Z_ps,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[0,4].set_title('Plane Subtracted Z')
        
        data.Z_BWD_gc, data.Z_BWD_lc, data.Z_BWD_ls, data.FZ_BWD_ls, data.Z_BWD_ps, data.FZ_BWD_ps = _process_pipeline(data.Z_BWD)
        data.FZ_BWD_ls = stmpy.tools.fft(data.Z_BWD_ls, zeroDC=True, window='hanning', units='amplitude', output='absolute')

        ax_topo[1,0].imshow(data.Z_BWD,     cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[1,0].set_title('Raw Z BWD')
        ax_topo[1,1].imshow(data.Z_BWD_gc,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[1,1].set_title('Global Corrected Z BWD')
        ax_topo[1,2].imshow(data.Z_BWD_lc,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[1,2].set_title('Local Corrected Z BWD')
        ax_topo[1,3].imshow(data.Z_BWD_ls,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range); ax_topo[1,3].set_title('Line Subtracted Z BWD')
        ax_topo[1,4].imshow(data.Z_BWD_ps,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range_ps); ax_topo[1,4].set_title('Plane Subtracted Z BWD')

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

        plot_FFT_data(data.FZ_ls, k_crop_n=k_crop_n, ax=ax_topo[0,5], add_colorbar=add_colorbar)
        plot_FFT_data(data.FZ_ls, k_crop_n=0, ax=ax_topo[0,6], add_colorbar=add_colorbar)
        plot_FFT_data(data.FZ_BWD_ls, k_crop_n=k_crop_n, ax=ax_topo[1,5], add_colorbar=add_colorbar)
        plot_FFT_data(data.FZ_BWD_ls, k_crop_n=0, ax=ax_topo[1,6], add_colorbar=add_colorbar)
    else:
        fig_topo, ax_topo = plt.subplots(1, 7, figsize=(23, 5))

        ax_topo[0].imshow(data.Z,     cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[0].set_title('Raw Z')
        ax_topo[1].imshow(data.Z_gc,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[1].set_title('Global Corrected Z')
        ax_topo[2].imshow(data.Z_lc,  cmap=stmpy.cm.Blues_r, origin='lower'); ax_topo[2].set_title('Local Corrected Z')
        ax_topo[3].imshow(data.Z_ls,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range); ax_topo[3].set_title('Line Subtracted Z')
        ax_topo[4].imshow(data.Z_ps,  cmap=stmpy.cm.Blues_r, origin='lower', clim=colorbar_range_ps); ax_topo[4].set_title('Plane Subtracted Z')

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


        plot_FFT_data(data.FZ_ls, k_crop_n=k_crop_n, ax=ax_topo[5])
        plot_FFT_data(data.FZ_ls, k_crop_n=0, ax=ax_topo[6])


    fig_topo.tight_layout()
    #add title
    if not dos_map:
        scan_offset = data.header['scan_offset']
        scan_angle = data.header['scan_angle']
        fig_topo.suptitle(data.info_str + ' (' + f'{scan_offset[0]*1e9:.2f}, {scan_offset[1]*1e9:.2f})nm, {scan_angle} deg', fontsize=16)


    figs['topo'] = (fig_topo, ax_topo)

    # --- DOS map (dI/dV) corrections & plots ---
    if dos_map:
        if not hasattr(data, 'LIY'):
            raise AttributeError("dos_map=True but `data.LIY` not found.")
        # Corrections on the full energy stack
        data.LIY_gc = stmpy.tools.nsigma_global(data.LIY, n=3, M=3, repeat=2)
        data.LIY_lc = stmpy.tools.nsigma_local(data.LIY_gc, n=3, N=4, M=3, repeat=2)

        # Mean over energy
        mean_raw = np.mean(data.LIY, axis=0)
        mean_gc  = np.mean(data.LIY_gc, axis=0)
        mean_lc  = np.mean(data.LIY_lc, axis=0)

        fig_dm, ax_dm = plt.subplots(1, 3, figsize=(15, 5))
        ax_dm[0].imshow(mean_raw, origin='lower',                        
                        cmap=stmpy.cm.Blues)
        ax_dm[0].set_title('Raw dI/dV at mean V')
        ax_dm[1].imshow(mean_gc, origin='lower', 
                        
                        cmap=stmpy.cm.Blues)
        ax_dm[1].set_title('Global Corrected dI/dV at mean V')
        ax_dm[2].imshow(mean_lc, origin='lower', 
                        cmap=stmpy.cm.Blues)
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
          ax_ds[0].imshow(data.LIY[idx], origin='lower', cmap=stmpy.cm.Blues)
          ax_ds[0].set_title(f'Raw dI/dV at {en_label}')
          ax_ds[1].imshow(data.LIY_gc[idx], origin='lower', cmap=stmpy.cm.Blues)
          ax_ds[1].set_title(f'Global Corrected dI/dV at {en_label}')
          ax_ds[2].imshow(data.LIY_lc[idx], origin='lower', cmap=stmpy.cm.Blues)
          ax_ds[2].set_title(f'Local Corrected dI/dV at {en_label}')
          for a in ax_ds: 
             a.set_axis_off()
             stmpy.image.add_scale_bar(5, scan_size, n_pixels, fs=12,  ax=a)
          fig_ds.tight_layout()
          figs['dos_idx'] = (fig_ds, ax_ds)

    if savename is not None:
        savename = data.info_str + "_" + savename.replace(".sm4", "") +  ".pdf"
        fig_topo.savefig(savepath+'/'+savename)
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


def plot_correlation(data1, data2, xlabel=None, ylabel=None):
    """
    Plot data1 vs data2 and compute Pearson correlation.
    Accepts 1D or ND arrays (they're flattened). NaN/inf are ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    r   : float (Pearson correlation; NaN if undefined)
    """
    x = np.asarray(data1).ravel()
    y = np.asarray(data2).ravel()

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

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(data1, origin='lower', cmap=stmpy.cm.Blues)
    ax[0].set_title(xlabel if xlabel is not None else 'Data 1')
    ax[1].imshow(data2, origin='lower', cmap=stmpy.cm.Blues)
    ax[1].set_title(ylabel if ylabel is not None else 'Data 2')
    ax[2].scatter(xm, ym, s=10, alpha=0.7)
    if xlabel is not None:
        ax[2].set_xlabel(xlabel)

    if ylabel is not None:
        ax[2].set_ylabel(ylabel)
    # Optional best-fit line
    if xm.size >= 2 and np.isfinite(r):
        try:
            slope, intercept = np.polyfit(xm, ym, 1)
            xs = np.linspace(xm.min(), xm.max(), 100)
            ax.plot(xs, slope * xs + intercept, linewidth=2)
        except Exception:
            pass

    print(f"Pearson r = {r:.3f}" if np.isfinite(r) else "Pearson r is undefined")
    
    return fig, r


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

    if im_show:
        plot_rk_space(data, ens=[e0, e1, e2, e3, e4, e5])
        
        # fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        # ax[0].imshow(sp1, cmap=stmpy.cm.Blues)
        # ax[0].set_title('Shape Para 1')
        # ax[1].imshow(sp2, cmap=stmpy.cm.Blues)
        # ax[1].set_title('Shape Para 2')
        # plt.tight_layout()
        # plt.show()


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

    ax[0,0].imshow(data.Z_ls, origin='lower',  cmap=stmpy.cm.Blues)
    ax[0,0].set_title('Z line subtracted')

    data.FZ_ls = stmpy.tools.fft(data.Z_ls, zeroDC=True, units='amplitude', output='absolute')
    plot_FFT_data(data.FZ_ls, k_crop_n=0, ax=ax[1,0])

    ax[0,1].imshow(data.shape_para1, origin='lower', cmap=stmpy.cm.Blues)
    ax[0,1].set_title('Shape Parameter 1')

    data.Fshape_para1 = stmpy.tools.fft(data.shape_para1, zeroDC=True, units='amplitude', output='absolute')
    plot_FFT_data(data.Fshape_para1, k_crop_n=0, ax=ax[1,1])

    ax[0,2].imshow(data.shape_para2, origin='lower', cmap=stmpy.cm.Blues)
    ax[0,2].set_title('Shape Parameter 2')

    data.Fshape_para2 = stmpy.tools.fft(data.shape_para2, zeroDC=True, units='amplitude', output='absolute')
    plot_FFT_data(data.Fshape_para2, k_crop_n=0, ax=ax[1,2])

    if ens is not None:
        for en_id, en in enumerate(ens):
            # print(en_id)
            id = np.argmin(np.abs(data.en - en))
            ax[0,3+en_id].imshow(data.LIY[id,:,:], origin='lower', cmap=stmpy.cm.Blues)
            ax[0,3+en_id].set_title(f'dI/dV at {en:.2f} V')
            FLIY_idx = stmpy.tools.fft(data.LIY[id,:,:], zeroDC=True, units='amplitude', output='absolute')
            plot_FFT_data(FLIY_idx, k_crop_n=0, ax=ax[1,3+en_id])

    for ax in ax[0,:]:
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        print(data.scan_info['scan_size'], data.scan_info['n_pixels'])
        stmpy.image.add_scale_bar(5, data.scan_info['scan_size'], data.scan_info['n_pixels'], fs=12, pad=0.1, ax=ax)


def plot_histogram(data):
    plt.figure(figsize=(5, 3))
    plt.hist(data, bins=50, alpha=0.85, edgecolor='none')
    plt.ylabel('Count')
    plt.title('Histogram of IV at en = 1.3')
    plt.tight_layout()
    plt.show()

