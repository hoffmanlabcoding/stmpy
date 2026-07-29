import matplotlib
matplotlib.use("Agg")  # non-GUI backend to avoid macOS NSWindow issues

import base64
import tempfile
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go

import matplotlib.colors as mcolors

import stmpy
from stmpy.random_aux import *


# ---------------------------------------------------------------------
# Helper: convert Matplotlib colormap (e.g. stmpy.cm.Blues_r) to Plotly
# ---------------------------------------------------------------------
def mpl_to_plotly(cmap, n=256):
    """Convert a Matplotlib colormap to a Plotly colorscale."""
    colors = [mcolors.rgb2hex(cmap(i / (n - 1))) for i in range(n)]
    return [[i / (n - 1), colors[i]] for i in range(n)]


def thin_colorbar(title):
    """Make thin colorbars to save horizontal space."""
    return dict(
        title=title,
        thickness=10,
        len=0.9,
        outlinewidth=0.5
    )


def maybe_colorbar(title, show=True):
    return thin_colorbar(title) if show else None


def fft_log_image(F2):
    """Robust FFT visualization: log10(|F| + eps)."""
    mag = np.abs(F2)
    return np.log10(mag + 1e-12)


def _range_bounds(arr):
    """Return (min, max, step, [min, max]) for a color-range slider from data."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0, 0.01, [0.0, 1.0]
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo:
        hi = lo + 1.0
    step = (hi - lo) / 200.0
    return lo, hi, step, [lo, hi]


def _crange_vals(crange):
    """Turn a RangeSlider [lo, hi] value into (zmin, zmax); (None, None) = autorange."""
    if crange and len(crange) == 2 and crange[0] is not None and crange[1] is not None:
        lo, hi = float(crange[0]), float(crange[1])
        if hi > lo:
            return lo, hi
    return None, None


def realspace_panel(title, graph_id, slider_id, wrapper_style, graph_h=280, wrapper_id=None):
    """A real-space image panel with a vertical color-range slider on its right."""
    wrapper_kwargs = {"style": wrapper_style}
    if wrapper_id is not None:
        wrapper_kwargs["id"] = wrapper_id
    return html.Div(
        **wrapper_kwargs,
        children=[
            html.Div(title, style={"fontSize": "12px", "marginBottom": "2px"}),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "alignItems": "flex-start"},
                children=[
                    html.Div(
                        style={"flex": "1", "minWidth": "0"},
                        children=[
                            dcc.Graph(id=graph_id, style={"height": f"{graph_h}px"},
                                      config={"displayModeBar": True}),
                        ],
                    ),
                    html.Div(
                        style={"width": "50px", "paddingLeft": "2px"},
                        children=[
                            html.Div("color", style={"fontSize": "9px", "color": "#888",
                                                      "textAlign": "center"}),
                            dcc.RangeSlider(
                                id=slider_id,
                                min=0, max=1, step=0.01, value=[0, 1],
                                vertical=True, verticalHeight=graph_h - 55,
                                marks=None,
                                tooltip={"placement": "left", "always_visible": False},
                                updatemode="mouseup",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------
# Global in-memory storage for large arrays
# ---------------------------------------------------------------------
GLOBAL_DATA = {
    "en": None,           # (E,)
    "LIY": None,          # (E, I, J)
    "LIY_smooth": None,   # (E, I, J)
    "FLIY_smooth": None,  # (E, I, J)
    "FZ_ls": None,        # (I, J)
    "topo": None,         # (I, J)
    "topo_high": None,    # (I_hr, J_hr)

    # normalized products
    "LIY_norm": None,     # (E, I, J)
    "FLIY_norm": None,    # (E, I, J)
    "area_map": None,     # (I, J)
}


# ---------------------------------------------------------------------
# Helper functions to load / parse sm4
# ---------------------------------------------------------------------
def load_sm4_from_bytes(file_bytes, filename):
    """
    Take raw bytes of a .sm4 LDOS-map file, write to temp, and use stmpy.load
    to get the data object. Assumes there is LIY etc.
    """
    suffix = filename.split(".")[-1] if "." in filename else "sm4"
    with tempfile.NamedTemporaryFile(suffix="." + suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        data = stmpy.load(tmp.name)
        add_corrections_and_plot(data, dos_map=True, make_plots=False, idx=164)

    en = np.asarray(data.en)                         # (E,)
    LIY = np.asarray(data.LIY)                       # (E, I, J)
    LIY_smoothed = np.asarray(data.LIY_smoothed)     # (E, I, J)
    FLIY_smoothed = np.asarray(data.FLIY_smoothed)   # (E, I, J)
    topo = np.asarray(data.Z_ls)                     # (I, J)
    FZ_ls = np.asarray(data.FZ_ls)                   # (I, J)

    assert LIY.ndim == 3, "LIY must be (E, I, J)"
    assert LIY_smoothed.shape == LIY.shape, "LIY_smoothed must match LIY shape"
    assert FLIY_smoothed.shape == LIY.shape, "FLIY_smoothed must match LIY shape"
    assert topo.ndim == 2, "Topography must be (I, J)"
    assert LIY.shape[1:] == topo.shape, f"LIY {LIY.shape}, topo {topo.shape}"
    assert FZ_ls.shape == topo.shape, f"FZ_ls {FZ_ls.shape}, topo {topo.shape}"

    return en, LIY, LIY_smoothed, FLIY_smoothed, topo, FZ_ls


def load_topo_only_from_bytes(file_bytes, filename):
    """
    Load a high-resolution topo-only .sm4 file and return Z_ls.
    Does NOT assume LIY exists.
    """
    suffix = filename.split(".")[-1] if "." in filename else "sm4"
    with tempfile.NamedTemporaryFile(suffix="." + suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        data = stmpy.load(tmp.name)
        add_corrections_and_plot(data, dos_map=False, make_plots=False, idx=0)

    topo_high = np.asarray(data.Z_ls)
    assert topo_high.ndim == 2, "High-res topography must be (I, J)"
    return topo_high


def parse_contents(contents, filename):
    """Decode uploaded LDOS-map file contents and return numpy arrays."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return load_sm4_from_bytes(decoded, filename)


def parse_topo_contents(contents, filename):
    """Decode uploaded topo-only file contents and return topo_high (Z_ls)."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    topo_high = load_topo_only_from_bytes(decoded, filename)
    return topo_high


# ---------------------------------------------------------------------
# Dash app layout
# ---------------------------------------------------------------------
app = dash.Dash(__name__)
app.title = "RHK LDOS Map Viewer"

# ---- move Plotly modebars above each graph ----
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        .js-plotly-plot .plotly .modebar {
            top: -2000px !important;
        }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    style={"fontFamily": "Arial", "margin": "10px"},
    children=[
        html.Div(
            style={"display": "flex", "alignItems": "center", "marginBottom": "6px", "gap": "10px"},
            children=[
                html.H3("RHK LDOS Map Viewer", style={"margin": "0 10px 0 0"}),

                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["LDOS file: Drag & Drop or ", html.A("Select .sm4")]),
                    style={
                        "lineHeight": "28px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "padding": "2px 8px",
                        "fontSize": "12px",
                    },
                    multiple=False,
                ),

                dcc.Upload(
                    id="upload-topo-hires",
                    children=html.Div(["High-res topo: Drag & Drop or ", html.A("Select .sm4")]),
                    style={
                        "lineHeight": "28px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "padding": "2px 8px",
                        "fontSize": "12px",
                    },
                    multiple=False,
                ),

                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "6px", "marginLeft": "10px"},
                    children=[
                        html.Div("Normalize area:", style={"fontSize": "12px", "color": "#333"}),
                        html.Span("E1", style={"fontSize": "12px"}),
                        dcc.Input(
                            id="norm-e1",
                            type="number",
                            step=0.001,
                            debounce=True,
                            style={"width": "90px", "fontSize": "12px"},
                            placeholder="-0.05",
                        ),
                        html.Span("E2", style={"fontSize": "12px"}),
                        dcc.Input(
                            id="norm-e2",
                            type="number",
                            step=0.001,
                            debounce=True,
                            style={"width": "90px", "fontSize": "12px"},
                            placeholder="0.05",
                        ),
                        html.Div(id="norm-status", style={"fontSize": "12px", "color": "#666", "marginLeft": "8px"}),
                    ],
                ),

                dcc.Checklist(
                    id="liy-smooth-toggle",
                    options=[{"label": " Show LIY-S", "value": "show"}],
                    value=["show"],
                    style={"fontSize": "12px", "marginLeft": "10px", "whiteSpace": "nowrap"},
                    inputStyle={"marginRight": "3px"},
                ),

                html.Div(
                    id="file-info",
                    style={"marginLeft": "10px", "fontStyle": "italic", "color": "#555", "fontSize": "12px"},
                ),
            ],
        ),

        dcc.Store(id="data-store"),
        dcc.Store(id="hires-store"),
        dcc.Store(id="fov-store"),
        dcc.Store(id="selected-point-store"),
        dcc.Store(id="norm-store"),

        # ---------------- Row 1: 5 panels ----------------
        html.Div(
            style={"display": "flex", "flexDirection": "row", "height": "300px"},
            children=[
                realspace_panel("Topo (LDOS)", "topo-graph", "topo-crange",
                                {"flex": "1", "marginRight": "4px"}),
                html.Div(style={"flex": "1", "margin": "0 4px"}, children=[
                    html.Div("FFT (FZ_ls)", style={"fontSize": "12px", "marginBottom": "2px"}),
                    dcc.Graph(id="fzfft-graph", style={"height": "280px"}, config={"displayModeBar": True}),
                ]),
                realspace_panel("LIY (smoothed)", "liy-smooth-graph", "liy-smooth-crange",
                                {"flex": "1", "margin": "0 4px"}, wrapper_id="liy-smooth-panel"),
                html.Div(style={"flex": "1", "margin": "0 4px"}, children=[
                    html.Div("FFT (FLIY_smoothed)", style={"fontSize": "12px", "marginBottom": "2px"}),
                    dcc.Graph(id="fft-graph", style={"height": "280px"}, config={"displayModeBar": True}),
                ]),
                realspace_panel("High-res topo (Z_ls)", "topo-hires-graph", "topo-hires-crange",
                                {"flex": "1", "marginLeft": "4px"}),
            ],
        ),

        # Energy slider
        html.Div(
            style={"display": "flex", "alignItems": "center", "margin": "6px 4px 4px 4px"},
            children=[
                html.Div("Energy slice:", style={"fontSize": "12px", "marginRight": "8px"}),
                html.Div(
                    style={"flex": "1"},
                    children=[
                        dcc.Slider(
                            id="energy-slider",
                            min=0, max=0, step=1, value=0,
                            marks={},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Div(id="energy-label", style={"marginLeft": "10px", "fontSize": "12px", "whiteSpace": "nowrap"}),
            ],
        ),

        # ---------------- Row 2: 3 panels ----------------
        html.Div(
            style={"display": "flex", "flexDirection": "row", "height": "320px", "marginTop": "6px"},
            children=[
                realspace_panel("LIY_norm", "liy-norm-graph", "liy-norm-crange",
                                {"flex": "1", "marginRight": "6px"}, graph_h=300),
                html.Div(style={"flex": "1", "margin": "0 6px"}, children=[
                    html.Div("FFT (FLIY_norm)", style={"fontSize": "12px", "marginBottom": "2px"}),
                    dcc.Graph(id="fft-norm-graph", style={"height": "300px"}, config={"displayModeBar": True}),
                ]),
                html.Div(style={"flex": "1", "marginLeft": "6px"}, children=[
                    html.Div("LDOS at selected point", style={"fontSize": "12px", "marginBottom": "2px"}),
                    dcc.Graph(id="ldos-graph", style={"height": "300px"}, config={"displayModeBar": True}),
                ]),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------
# Callback 1: handle LDOS upload, store big arrays on server
# ALSO sets default E1/E2 in the inputs
# ---------------------------------------------------------------------
@app.callback(
    Output("data-store", "data"),
    Output("file-info", "children"),
    Output("energy-slider", "min"),
    Output("energy-slider", "max"),
    Output("energy-slider", "value"),
    Output("norm-e1", "value"),
    Output("norm-e2", "value"),
    Output("topo-crange", "min"),
    Output("topo-crange", "max"),
    Output("topo-crange", "step"),
    Output("topo-crange", "value"),
    Output("liy-smooth-crange", "min"),
    Output("liy-smooth-crange", "max"),
    Output("liy-smooth-crange", "step"),
    Output("liy-smooth-crange", "value"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    en, LIY, LIY_smooth, FLIY_smooth, topo, FZ_ls = parse_contents(contents, filename)
    E, I, J = LIY.shape
    order = np.argsort(en)

    en_sorted = en[order]
    GLOBAL_DATA["en"] = en_sorted
    GLOBAL_DATA["LIY"] = LIY[order, :, :]
    GLOBAL_DATA["LIY_smooth"] = LIY_smooth[order, :, :]
    GLOBAL_DATA["FLIY_smooth"] = FLIY_smooth[order, :, :]
    GLOBAL_DATA["topo"] = topo
    GLOBAL_DATA["FZ_ls"] = FZ_ls

    # clear old normalization products
    GLOBAL_DATA["LIY_norm"] = None
    GLOBAL_DATA["FLIY_norm"] = None
    GLOBAL_DATA["area_map"] = None

    data_store = {"I": int(I), "J": int(J), "E": int(E)}
    info = f"Loaded LDOS: {filename} | (E, I, J) = {LIY.shape}"

    emin = 0
    emax = E - 1
    e0 = E // 2

    # default normalization window around 0V, clamped to range
    e_min = float(en_sorted[0])
    e_max = float(en_sorted[-1])
    width = 0.05
    e1_default = max(e_min, min(0.0 - width, e_max))
    e2_default = max(e_min, min(0.0 + width, e_max))
    if e2_default <= e1_default:
        e1_default = e_min + 0.25 * (e_max - e_min)
        e2_default = e_min + 0.75 * (e_max - e_min)

    # color-range slider bounds for topo (low-res) and LIY smoothed (full stack)
    t_lo, t_hi, t_step, t_val = _range_bounds(GLOBAL_DATA["topo"])
    l_lo, l_hi, l_step, l_val = _range_bounds(GLOBAL_DATA["LIY_smooth"])

    return (data_store, info, emin, emax, e0, e1_default, e2_default,
            t_lo, t_hi, t_step, t_val,
            l_lo, l_hi, l_step, l_val)


# ---------------------------------------------------------------------
# Callback 1b: handle high-res topo upload
# ---------------------------------------------------------------------
@app.callback(
    Output("hires-store", "data"),
    Output("topo-hires-crange", "min"),
    Output("topo-hires-crange", "max"),
    Output("topo-hires-crange", "step"),
    Output("topo-hires-crange", "value"),
    Input("upload-topo-hires", "contents"),
    State("upload-topo-hires", "filename"),
    prevent_initial_call=True,
)
def handle_hires_upload(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    topo_high = parse_topo_contents(contents, filename)
    I_hr, J_hr = topo_high.shape

    GLOBAL_DATA["topo_high"] = topo_high
    h_lo, h_hi, h_step, h_val = _range_bounds(topo_high)
    return {"I": int(I_hr), "J": int(J_hr)}, h_lo, h_hi, h_step, h_val


# ---------------------------------------------------------------------
# Toggle: show / hide the LIY (smoothed) panel
# ---------------------------------------------------------------------
@app.callback(
    Output("liy-smooth-panel", "style"),
    Input("liy-smooth-toggle", "value"),
)
def toggle_liy_smooth_panel(value):
    base = {"flex": "1", "margin": "0 4px"}
    if not value or "show" not in value:
        base["display"] = "none"
    return base


# ---------------------------------------------------------------------
# Callback: energy label (slider position -> actual energy)
# ---------------------------------------------------------------------
@app.callback(
    Output("energy-label", "children"),
    Input("energy-slider", "value"),
    Input("data-store", "data"),
)
def update_energy_label(e_idx, data_store):
    if GLOBAL_DATA["en"] is None or data_store is None or e_idx is None:
        return ""
    en = GLOBAL_DATA["en"]
    E = data_store["E"]
    e_idx = max(0, min(E - 1, int(e_idx)))
    return f"idx {e_idx} → {en[e_idx]:.5f} V"


# ---------------------------------------------------------------------
# Compute LIY_norm / FLIY_norm whenever E1/E2 change
# ---------------------------------------------------------------------
@app.callback(
    Output("norm-store", "data"),
    Output("norm-status", "children"),
    Output("liy-norm-crange", "min"),
    Output("liy-norm-crange", "max"),
    Output("liy-norm-crange", "step"),
    Output("liy-norm-crange", "value"),
    Input("data-store", "data"),
    Input("norm-e1", "value"),
    Input("norm-e2", "value"),
    prevent_initial_call=True,
)
def recompute_norm(data_store, E1, E2):
    if data_store is None or GLOBAL_DATA["en"] is None or GLOBAL_DATA["LIY_smooth"] is None:
        raise dash.exceptions.PreventUpdate

    if E1 is None or E2 is None:
        return (dash.no_update, "set E1/E2",
                dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    en = GLOBAL_DATA["en"]
    LIY_sm = GLOBAL_DATA["LIY_smooth"]

    E1 = float(E1)
    E2 = float(E2)
    if E2 < E1:
        E1, E2 = E2, E1

    # clamp to range
    E1c = max(float(en[0]), min(float(en[-1]), E1))
    E2c = max(float(en[0]), min(float(en[-1]), E2))
    if E2c <= E1c:
        E2c = min(float(en[-1]), E1c + 1e-6)

    LIY_norm, FLIY_norm, area_map = normalize_LIY_by_area(en, LIY_sm, E1c, E2c)

    GLOBAL_DATA["LIY_norm"] = np.asarray(LIY_norm)
    GLOBAL_DATA["FLIY_norm"] = np.asarray(FLIY_norm)
    GLOBAL_DATA["area_map"] = np.asarray(area_map)

    n_lo, n_hi, n_step, n_val = _range_bounds(GLOBAL_DATA["LIY_norm"])

    return ({"E1": E1c, "E2": E2c}, f"ok: [{E1c:.4f}, {E2c:.4f}] V",
            n_lo, n_hi, n_step, n_val)


# ---------------------------------------------------------------------
# Shared FOV store (fractional coordinates in [0,1])
# sync: topo, LIY_smooth, LIY_norm, topo-hires
# ---------------------------------------------------------------------
@app.callback(
    Output("fov-store", "data"),
    Input("topo-graph", "relayoutData"),
    Input("liy-smooth-graph", "relayoutData"),
    Input("liy-norm-graph", "relayoutData"),
    Input("topo-hires-graph", "relayoutData"),
    Input("data-store", "data"),
    Input("hires-store", "data"),
    State("fov-store", "data"),
    prevent_initial_call=True,
)
def update_fov(topo_relayout, smooth_relayout, norm_relayout, hires_relayout,
               data_store, hires_store, fov_state):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    full_fov = {"fi0": 0.0, "fi1": 1.0, "fj0": 0.0, "fj1": 1.0}

    if trigger_id in ["data-store", "hires-store"]:
        return full_fov

    if fov_state is None:
        fov_state = full_fov

    if trigger_id == "topo-graph":
        rel = topo_relayout
        if data_store is None:
            raise dash.exceptions.PreventUpdate
        I = data_store["I"]; J = data_store["J"]
    elif trigger_id == "liy-smooth-graph":
        rel = smooth_relayout
        if data_store is None:
            raise dash.exceptions.PreventUpdate
        I = data_store["I"]; J = data_store["J"]
    elif trigger_id == "liy-norm-graph":
        rel = norm_relayout
        if data_store is None:
            raise dash.exceptions.PreventUpdate
        I = data_store["I"]; J = data_store["J"]
    elif trigger_id == "topo-hires-graph":
        rel = hires_relayout
        if hires_store is None:
            raise dash.exceptions.PreventUpdate
        I = hires_store["I"]; J = hires_store["J"]
    else:
        raise dash.exceptions.PreventUpdate

    if rel is None:
        return fov_state

    if "xaxis.autorange" in rel and rel["xaxis.autorange"]:
        return full_fov

    x0 = rel.get("xaxis.range[0]", 0)
    x1 = rel.get("xaxis.range[1]", J - 1)
    y0 = rel.get("yaxis.range[0]", 0)
    y1 = rel.get("yaxis.range[1]", I - 1)

    fj0 = x0 / (J - 1) if J > 1 else 0.0
    fj1 = x1 / (J - 1) if J > 1 else 1.0
    fi0 = y0 / (I - 1) if I > 1 else 0.0
    fi1 = y1 / (I - 1) if I > 1 else 1.0

    fj0 = max(0.0, min(1.0, fj0))
    fj1 = max(0.0, min(1.0, fj1))
    fi0 = max(0.0, min(1.0, fi0))
    fi1 = max(0.0, min(1.0, fi1))

    eps = 1e-6
    if fj1 <= fj0 + eps:
        fj1 = min(1.0, fj0 + eps)
    if fi1 <= fi0 + eps:
        fi1 = min(1.0, fi0 + eps)

    return {"fi0": fi0, "fi1": fi1, "fj0": fj0, "fj1": fj1}


# ---------------------------------------------------------------------
# Helper: compute integer crop indices from fractional FOV
# ---------------------------------------------------------------------
def crop_indices_from_fov(fov, I, J):
    if fov is None:
        fi0, fi1, fj0, fj1 = 0.0, 1.0, 0.0, 1.0
    else:
        fi0 = fov.get("fi0", 0.0)
        fi1 = fov.get("fi1", 1.0)
        fj0 = fov.get("fj0", 0.0)
        fj1 = fov.get("fj1", 1.0)

    i0 = int(np.floor(fi0 * (I - 1))) if I > 1 else 0
    i1 = int(np.ceil(fi1 * (I - 1)))  if I > 1 else 0
    j0 = int(np.floor(fj0 * (J - 1))) if J > 1 else 0
    j1 = int(np.ceil(fj1 * (J - 1)))  if J > 1 else 0

    i0 = max(0, min(I - 1, i0))
    i1 = max(0, min(I - 1, i1))
    j0 = max(0, min(J - 1, j0))
    j1 = max(0, min(J - 1, j1))

    if j1 <= j0:
        j1 = min(J - 1, j0 + 1)
    if i1 <= i0:
        i1 = min(I - 1, i0 + 1)

    return i0, i1, j0, j1


# ---------------------------------------------------------------------
# Shared selected point (fi,fj) from clicks on real-space panels
# sync: topo, LIY_smooth, LIY_norm, topo-hires
# ---------------------------------------------------------------------
@app.callback(
    Output("selected-point-store", "data"),
    Input("topo-graph", "clickData"),
    Input("liy-smooth-graph", "clickData"),
    Input("liy-norm-graph", "clickData"),
    Input("topo-hires-graph", "clickData"),
    State("data-store", "data"),
    State("hires-store", "data"),
    prevent_initial_call=True,
)
def update_selected_point(topo_click, smooth_click, norm_click, hires_click, data_store, hires_store):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    def extract_point(cd):
        if cd is None or "points" not in cd or len(cd["points"]) == 0:
            return None, None
        pt = cd["points"][0]
        return pt.get("x", None), pt.get("y", None)

    if trigger_id == "topo-graph":
        x, y = extract_point(topo_click)
        if x is None or y is None or data_store is None:
            raise dash.exceptions.PreventUpdate
        I = data_store["I"]; J = data_store["J"]
    elif trigger_id == "liy-smooth-graph":
        x, y = extract_point(smooth_click)
        if x is None or y is None or data_store is None:
            raise dash.exceptions.PreventUpdate
        I = data_store["I"]; J = data_store["J"]
    elif trigger_id == "liy-norm-graph":
        x, y = extract_point(norm_click)
        if x is None or y is None or data_store is None:
            raise dash.exceptions.PreventUpdate
        I = data_store["I"]; J = data_store["J"]
    elif trigger_id == "topo-hires-graph":
        x, y = extract_point(hires_click)
        if x is None or y is None or hires_store is None:
            raise dash.exceptions.PreventUpdate
        I = hires_store["I"]; J = hires_store["J"]
    else:
        raise dash.exceptions.PreventUpdate

    j_idx = max(0, min(J - 1, int(round(x))))
    i_idx = max(0, min(I - 1, int(round(y))))

    fi = i_idx / (I - 1) if I > 1 else 0.0
    fj = j_idx / (J - 1) if J > 1 else 0.0

    return {"fi": fi, "fj": fj}


# ---------------------------------------------------------------------
# Topo plot (low-res) - Row 1 (NO colorbar)
# ---------------------------------------------------------------------
@app.callback(
    Output("topo-graph", "figure"),
    Input("data-store", "data"),
    Input("fov-store", "data"),
    Input("selected-point-store", "data"),
    Input("topo-crange", "value"),
    prevent_initial_call=True,
)
def update_topo(data_store, fov, selected_point, crange):
    if GLOBAL_DATA["topo"] is None or data_store is None:
        return go.Figure()

    topo = GLOBAL_DATA["topo"]
    I, J = topo.shape
    colorscale = mpl_to_plotly(stmpy.cm.Blues_r)

    i0, i1, j0, j1 = crop_indices_from_fov(fov, I, J)
    topo_crop = topo[i0:i1 + 1, j0:j1 + 1]
    Ii, Jj = topo_crop.shape

    x = np.arange(j0, j0 + Jj)
    y = np.arange(i0, i0 + Ii)

    show_cb = False
    zmin, zmax = _crange_vals(crange)

    fig = go.Figure(
        data=[go.Heatmap(
            z=topo_crop, x=x, y=y,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            colorbar=maybe_colorbar("Z_ls", show_cb),
        )]
    )

    if selected_point is not None:
        fi = selected_point.get("fi", 0.5)
        fj = selected_point.get("fj", 0.5)
        i_sel = int(round(fi * (I - 1))) if I > 1 else 0
        j_sel = int(round(fj * (J - 1))) if J > 1 else 0
        if i0 <= i_sel <= i1 and j0 <= j_sel <= j1:
            fig.add_trace(go.Scatter(
                x=[j_sel], y=[i_sel],
                mode="markers",
                marker=dict(symbol="circle-open", size=5, line=dict(width=2, color="black")),
                showlegend=False,
            ))

    fig.update_layout(
        xaxis=dict(title="j", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="i", domain=[0, 0.95]),
        margin=dict(l=35, r=5, t=10, b=28),
    )
    return fig


# ---------------------------------------------------------------------
# LIY (smoothed) - Row 1 (NO colorbar)
# ---------------------------------------------------------------------
@app.callback(
    Output("liy-smooth-graph", "figure"),
    Input("data-store", "data"),
    Input("fov-store", "data"),
    Input("energy-slider", "value"),
    Input("selected-point-store", "data"),
    Input("liy-smooth-crange", "value"),
    prevent_initial_call=True,
)
def update_liy_smooth(data_store, fov, e_idx, selected_point, crange):
    if GLOBAL_DATA["LIY_smooth"] is None or data_store is None:
        return go.Figure()

    LIY_smooth = GLOBAL_DATA["LIY_smooth"]
    en = GLOBAL_DATA["en"]
    E, I, J = LIY_smooth.shape

    e_idx = E // 2 if e_idx is None else int(e_idx)
    e_idx = max(0, min(E - 1, e_idx))

    i0, i1, j0, j1 = crop_indices_from_fov(fov, I, J)
    liy_slice = LIY_smooth[e_idx, i0:i1 + 1, j0:j1 + 1]
    Ii, Jj = liy_slice.shape

    x = np.arange(j0, j0 + Jj)
    y = np.arange(i0, i0 + Ii)

    colorscale = mpl_to_plotly(stmpy.cm.Blues_r)
    show_cb = False
    zmin, zmax = _crange_vals(crange)

    fig = go.Figure(
        data=[go.Heatmap(
            z=liy_slice, x=x, y=y,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            colorbar=maybe_colorbar("LIY_s", show_cb),
        )]
    )

    if selected_point is not None:
        fi = selected_point.get("fi", 0.5)
        fj = selected_point.get("fj", 0.5)
        i_sel = int(round(fi * (I - 1))) if I > 1 else 0
        j_sel = int(round(fj * (J - 1))) if J > 1 else 0
        if i0 <= i_sel <= i1 and j0 <= j_sel <= j1:
            fig.add_trace(go.Scatter(
                x=[j_sel], y=[i_sel],
                mode="markers",
                marker=dict(symbol="circle-open", size=5, line=dict(width=2, color="black")),
                showlegend=False,
            ))

    fig.update_layout(
        xaxis=dict(title="j", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="i", domain=[0, 0.95]),
        margin=dict(l=35, r=5, t=10, b=28),
    )

    fig.add_annotation(
        x=0.95, y=0.92, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=f"E={en[e_idx]:.2f} V",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=3,
        font=dict(size=12, color="black"),
    )
    return fig


# ---------------------------------------------------------------------
# FFT(Z_ls): data.FZ_ls - Row 1 (NO colorbar)
# ---------------------------------------------------------------------
@app.callback(
    Output("fzfft-graph", "figure"),
    Input("data-store", "data"),
    prevent_initial_call=True,
)
def update_fzfft(data_store):
    if GLOBAL_DATA["FZ_ls"] is None or data_store is None:
        return go.Figure()

    F2 = GLOBAL_DATA["FZ_ls"]
    I, J = F2.shape

    z = F2
    z_flat = F2[np.isfinite(F2)]
    mu = np.mean(z_flat)
    sigma = np.std(z_flat)

    zmin = np.max([np.min(z_flat), 0, mu - 2 * sigma])
    zmax = np.min([np.max(z_flat), mu + 2 * sigma])

    kx = np.arange(J) - (J // 2)
    ky = np.arange(I) - (I // 2)

    colorscale_fft = mpl_to_plotly(stmpy.cm.gray_r)
    show_cb = False

    fig = go.Figure(
        data=[go.Heatmap(
            z=z, x=kx, y=ky,
            colorscale=colorscale_fft,
            zmin=zmin, zmax=zmax,
            colorbar=maybe_colorbar("a.u.", show_cb),
        )]
    )

    fig.update_layout(
        xaxis=dict(title="kx (index)", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="ky (index)", domain=[0, 0.95]),
        margin=dict(l=35, r=5, t=10, b=28),
    )
    return fig


# ---------------------------------------------------------------------
# FFT(LIY_smoothed): data.FLIY_smoothed - Row 1 (NO colorbar)
# ---------------------------------------------------------------------
@app.callback(
    Output("fft-graph", "figure"),
    Input("data-store", "data"),
    Input("energy-slider", "value"),
    prevent_initial_call=True,
)
def update_fft(data_store, e_idx):
    if GLOBAL_DATA["FLIY_smooth"] is None or data_store is None:
        return go.Figure()

    F = GLOBAL_DATA["FLIY_smooth"]
    en = GLOBAL_DATA["en"]
    E, I, J = F.shape

    e_idx = E // 2 if e_idx is None else int(e_idx)
    e_idx = max(0, min(E - 1, e_idx))

    F2 = F[e_idx, :, :]
    z = F2

    z_flat = F2[np.isfinite(F2)]
    mu = np.mean(z_flat)
    sigma = np.std(z_flat)

    zmin = np.max([np.min(z_flat), 0, mu - 2 * sigma])
    zmax = np.min([np.max(z_flat), mu + 2 * sigma])

    kx = np.arange(J) - (J // 2)
    ky = np.arange(I) - (I // 2)

    colorscale_fft = mpl_to_plotly(stmpy.cm.gray_r)
    show_cb = False

    fig = go.Figure(
        data=[go.Heatmap(
            z=z, x=kx, y=ky,
            colorscale=colorscale_fft,
            zmin=zmin, zmax=zmax,
            colorbar=maybe_colorbar("a.u.", show_cb),
        )]
    )

    fig.update_layout(
        xaxis=dict(title="kx (index)", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="ky (index)", domain=[0, 0.95]),
        margin=dict(l=35, r=5, t=10, b=28),
    )

    fig.add_annotation(
        x=0.95, y=0.92, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=f"E={en[e_idx]:.2f} V",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=3,
        font=dict(size=12, color="black"),
    )

    return fig


# ---------------------------------------------------------------------
# High-res topo - Row 1 (NO colorbar)
# ---------------------------------------------------------------------
@app.callback(
    Output("topo-hires-graph", "figure"),
    Input("hires-store", "data"),
    Input("fov-store", "data"),
    Input("selected-point-store", "data"),
    Input("topo-hires-crange", "value"),
    prevent_initial_call=True,
)
def update_topo_hires(hires_store, fov, selected_point, crange):
    if GLOBAL_DATA["topo_high"] is None or hires_store is None:
        return go.Figure()

    topo_high = GLOBAL_DATA["topo_high"]
    I_hr, J_hr = topo_high.shape

    i0, i1, j0, j1 = crop_indices_from_fov(fov, I_hr, J_hr)
    topo_crop = topo_high[i0:i1 + 1, j0:j1 + 1]
    Ii, Jj = topo_crop.shape

    x = np.arange(j0, j0 + Jj)
    y = np.arange(i0, i0 + Ii)

    colorscale = mpl_to_plotly(stmpy.cm.Blues_r)
    show_cb = False
    zmin, zmax = _crange_vals(crange)

    fig = go.Figure(
        data=[go.Heatmap(
            z=topo_crop, x=x, y=y,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            colorbar=maybe_colorbar("Z_ls", show_cb),
        )]
    )

    if selected_point is not None:
        fi = selected_point.get("fi", 0.5)
        fj = selected_point.get("fj", 0.5)
        i_sel = int(round(fi * (I_hr - 1))) if I_hr > 1 else 0
        j_sel = int(round(fj * (J_hr - 1))) if J_hr > 1 else 0
        if i0 <= i_sel <= i1 and j0 <= j_sel <= j1:
            fig.add_trace(go.Scatter(
                x=[j_sel], y=[i_sel],
                mode="markers",
                marker=dict(symbol="circle-open", size=5, line=dict(width=2, color="black")),
                showlegend=False,
            ))

    fig.update_layout(
        xaxis=dict(title="j (hi-res)", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="i (hi-res)", domain=[0, 0.95]),
        margin=dict(l=35, r=5, t=10, b=28),
    )
    return fig


# ---------------------------------------------------------------------
# LIY_norm - Row 2 (WITH colorbar)
# ---------------------------------------------------------------------
@app.callback(
    Output("liy-norm-graph", "figure"),
    Input("data-store", "data"),
    Input("norm-store", "data"),
    Input("fov-store", "data"),
    Input("energy-slider", "value"),
    Input("selected-point-store", "data"),
    Input("liy-norm-crange", "value"),
    prevent_initial_call=True,
)
def update_liy_norm(data_store, norm_store, fov, e_idx, selected_point, crange):
    if GLOBAL_DATA["LIY_norm"] is None or data_store is None or norm_store is None:
        return go.Figure()

    LIY_norm = GLOBAL_DATA["LIY_norm"]
    en = GLOBAL_DATA["en"]
    E, I, J = LIY_norm.shape

    e_idx = E // 2 if e_idx is None else int(e_idx)
    e_idx = max(0, min(E - 1, e_idx))

    i0, i1, j0, j1 = crop_indices_from_fov(fov, I, J)
    liy_slice = LIY_norm[e_idx, i0:i1 + 1, j0:j1 + 1]
    Ii, Jj = liy_slice.shape

    x = np.arange(j0, j0 + Jj)
    y = np.arange(i0, i0 + Ii)

    colorscale = mpl_to_plotly(stmpy.cm.Blues_r)
    show_cb = True
    zmin, zmax = _crange_vals(crange)

    fig = go.Figure(
        data=[go.Heatmap(
            z=liy_slice, x=x, y=y,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            colorbar=maybe_colorbar("LIY_n", show_cb),
        )]
    )

    if selected_point is not None:
        fi = selected_point.get("fi", 0.5)
        fj = selected_point.get("fj", 0.5)
        i_sel = int(round(fi * (I - 1))) if I > 1 else 0
        j_sel = int(round(fj * (J - 1))) if J > 1 else 0
        if i0 <= i_sel <= i1 and j0 <= j_sel <= j1:
            fig.add_trace(go.Scatter(
                x=[j_sel], y=[i_sel],
                mode="markers",
                marker=dict(symbol="circle-open", size=5, line=dict(width=2, color="black")),
                showlegend=False,
            ))

    fig.update_layout(
        xaxis=dict(title="j", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="i", domain=[0, 0.9]),
        margin=dict(l=45, r=10, t=10, b=30),
    )

    fig.add_annotation(
        x=0.95, y=0.9, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=f"E={en[e_idx]:.2f} V",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=3,
        font=dict(size=12, color="black"),
    )
    return fig


# ---------------------------------------------------------------------
# FFT(FLIY_norm) - Row 2 (WITH colorbar)
# ---------------------------------------------------------------------
@app.callback(
    Output("fft-norm-graph", "figure"),
    Input("data-store", "data"),
    Input("norm-store", "data"),
    Input("energy-slider", "value"),
    prevent_initial_call=True,
)
def update_fft_norm(data_store, norm_store, e_idx):
    if GLOBAL_DATA["FLIY_norm"] is None or data_store is None or norm_store is None:
        return go.Figure()

    F = GLOBAL_DATA["FLIY_norm"]
    en = GLOBAL_DATA["en"]
    E, I, J = F.shape

    e_idx = E // 2 if e_idx is None else int(e_idx)
    e_idx = max(0, min(E - 1, e_idx))

    F2 = F[e_idx, :, :]
    z = F2

    z_flat = F2[np.isfinite(F2)]
    mu = np.mean(z_flat)
    sigma = np.std(z_flat)

    zmin = np.max([np.min(z_flat), 0, mu - 2 * sigma])
    zmax = np.min([np.max(z_flat), mu + 2 * sigma])

    kx = np.arange(J) - (J // 2)
    ky = np.arange(I) - (I // 2)

    colorscale_fft = mpl_to_plotly(stmpy.cm.gray_r)
    show_cb = True

    fig = go.Figure(
        data=[go.Heatmap(
            z=z, x=kx, y=ky,
            colorscale=colorscale_fft,
            zmin=zmin, zmax=zmax,
            colorbar=maybe_colorbar("a.u.", show_cb),
        )]
    )

    fig.update_layout(
        xaxis=dict(title="kx (index)", constrain="domain", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="ky (index)", domain=[0, 0.9]),
        margin=dict(l=45, r=10, t=10, b=30),
    )

    fig.add_annotation(
        x=0.95, y=0.9, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=f"E={en[e_idx]:.2f} V",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.75)",
        borderpad=3,
        font=dict(size=12, color="black"),
    )
    return fig


# ---------------------------------------------------------------------
# LDOS spectrum – uses selected point + shows energy slider line
# ---------------------------------------------------------------------
@app.callback(
    Output("ldos-graph", "figure"),
    Input("selected-point-store", "data"),
    Input("data-store", "data"),
    Input("energy-slider", "value"),
)
def update_ldos(selected_point, data_store, e_idx):
    if GLOBAL_DATA["en"] is None or GLOBAL_DATA["LIY"] is None:
        return go.Figure(
            layout=go.Layout(
                xaxis={"title": "Energy (V)"},
                yaxis={"title": "dI/dV (a.u.)"},
                margin=dict(l=50, r=10, t=10, b=40),
            )
        )

    en = GLOBAL_DATA["en"]
    LIY = GLOBAL_DATA["LIY"]
    LIY_smooth = GLOBAL_DATA["LIY_smooth"]
    E, I, J = LIY.shape

    if selected_point is None:
        i = I // 2
        j = J // 2
    else:
        fi = selected_point.get("fi", 0.5)
        fj = selected_point.get("fj", 0.5)
        i = int(round(fi * (I - 1))) if I > 1 else 0
        j = int(round(fj * (J - 1))) if J > 1 else 0
        i = max(0, min(I - 1, i))
        j = max(0, min(J - 1, j))

    spectrum_raw = LIY[:, i, j]
    spectrum_smooth = LIY_smooth[:, i, j]

    fig = go.Figure(
        data=[
            go.Scatter(x=en, y=spectrum_raw, mode="lines", line=dict(width=1.2), name="LIY"),
            go.Scatter(x=en, y=spectrum_smooth, mode="lines", line=dict(width=1.5), name="LIY_s"),
        ]
    )

    if e_idx is not None and E > 0:
        e_idx = max(0, min(E - 1, int(e_idx)))
        e_val = en[e_idx]
        fig.add_shape(
            type="line",
            x0=e_val, x1=e_val,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="black", width=1, dash="dash"),
        )

    fig.update_layout(
        xaxis=dict(title="Bias (V)"),
        yaxis=dict(title="dI/dV (a.u.)"),
        margin=dict(l=55, r=10, t=10, b=40),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        title=dict(text=f"LDOS at (i, j) = ({i}, {j})", x=0.5, y=0.95, font=dict(size=10)),
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)