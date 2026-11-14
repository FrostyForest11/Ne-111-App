"""
NE111 Term Project - Plot Digitizer (Streamlit)

Workflow
--------
1. Upload an image of a graph (png, jpg, gif, tiff, bmp; colour or greyscale).
2. Click 3 calibration points: a reference/origin point, a point on the x-axis,
   and a point on the y-axis, then enter their true axis values in the sidebar.
   The 3-point affine calibration automatically handles scans that are rotated
   (or slightly skewed). Linear and log axes (including semi-log) are supported.
3. Click on the graph to digitize data points. Each dataset has its own colour.
4. Edit points in the live table (bonus), zoom/pan for precision (bonus),
   and download everything as a CSV.
"""

import csv
import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Plot Digitizer", layout="wide")

PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9A6324", "#800000", "#008080",
]
DISPLAY_W = 900
CAL_LABELS = ["Origin / reference point", "Point on the X-axis", "Point on the Y-axis"]
CAL_TAGS = ["O", "X", "Y"]
CAL_COLOR = "#ff00ff"

ss = st.session_state


# ----------------------------------------------------------------------------
# State helpers
# ----------------------------------------------------------------------------
def bump():
    """Changing this counter re-creates the image/table widgets (clears stale clicks)."""
    ss.nclicks += 1


def new_dataset(name=None):
    i = ss.ds_counter
    ss.ds_counter += 1
    ss.datasets.append(
        {
            "id": i,
            "name": name or f"Dataset {i + 1}",
            "color": PALETTE[i % len(PALETTE)],
            "points": [],  # stored in ORIGINAL image pixel coordinates
        }
    )


def init_state():
    ss.setdefault("datasets", [])
    ss.setdefault("ds_counter", 0)
    ss.setdefault("cal", [])
    ss.setdefault("nclicks", 0)
    ss.setdefault("file_id", None)
    ss.setdefault("image", None)
    ss.setdefault("active_idx", 0)
    ss.setdefault("new_ds_name", "")
    if not ss.datasets:
        new_dataset()


def reset_all():
    ss.datasets = []
    ss.ds_counter = 0
    ss.cal = []
    ss.active_idx = 0
    new_dataset()
    bump()


def cb_add_dataset():
    name = ss.get("new_ds_name", "").strip() or None
    new_dataset(name)
    ss.active_idx = len(ss.datasets) - 1
    ss.new_ds_name = ""
    bump()


def cb_delete_dataset():
    idx = ss.active_idx
    ss.datasets.pop(idx)
    if not ss.datasets:
        new_dataset()
    ss.active_idx = max(0, min(idx - 1, len(ss.datasets) - 1))
    bump()


def cb_undo():
    pts = ss.datasets[ss.active_idx]["points"]
    if pts:
        pts.pop()
    bump()


def cb_clear_points():
    ss.datasets[ss.active_idx]["points"] = []
    bump()


def cb_reset_cal():
    ss.cal = []
    bump()


def cb_undo_cal():
    if ss.cal:
        ss.cal.pop()
    bump()


# ----------------------------------------------------------------------------
# Image handling
# ----------------------------------------------------------------------------
def load_image(file) -> Image.Image:
    img = Image.open(file)
    try:
        img.seek(0)  # first frame of GIF / multi-page TIFF
    except Exception:
        pass
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # 16-bit / float greyscale -> 8-bit
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        arr = np.array(img).astype(float)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8))

    # Flatten transparency onto white
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, "white")
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB")  # greyscale and colour both end up as RGB


# ----------------------------------------------------------------------------
# Calibration (3-point affine => handles rotation and skew)
# ----------------------------------------------------------------------------
def fwd(v, is_log):
    return np.log10(v) if is_log else v


def inv(u, is_log):
    return 10.0 ** u if is_log else u


def validate_calibration(x0, y0, x1, y2, xlog, ylog):
    if xlog and (x0 <= 0 or x1 <= 0):
        return "Log x-axis: the origin x and the X-axis point must both be > 0."
    if ylog and (y0 <= 0 or y2 <= 0):
        return "Log y-axis: the origin y and the Y-axis point must both be > 0."
    if x1 == x0:
        return "The X-axis point must have a different x value than the origin."
    if y2 == y0:
        return "The Y-axis point must have a different y value than the origin."
    return None


class Calibration:
    """
    Pixel p = P0 + a*(P1-P0) + b*(P2-P0), where
      a = (u - u0)/(u1 - u0),  b = (v - v0)/(v2 - v0)
    and u, v are x, y (or log10 of them on log axes).
    P0: origin pixel, P1: x-axis pixel, P2: y-axis pixel.
    Because the basis vectors are arbitrary, rotated scans work automatically.
    """

    def __init__(self, cal_px, x0, y0, x1, y2, xlog, ylog):
        self.P0 = np.array(cal_px[0], dtype=float)
        P1 = np.array(cal_px[1], dtype=float)
        P2 = np.array(cal_px[2], dtype=float)
        self.A = np.column_stack([P1 - self.P0, P2 - self.P0])
        self.ok = abs(np.linalg.det(self.A)) > 1e-9
        self.Ainv = np.linalg.inv(self.A) if self.ok else None
        self.xlog, self.ylog = xlog, ylog
        self.u0, self.u1 = fwd(x0, xlog), fwd(x1, xlog)
        self.v0, self.v2 = fwd(y0, ylog), fwd(y2, ylog)

    def to_data(self, px, py):
        a, b = self.Ainv @ (np.array([px, py], dtype=float) - self.P0)
        u = self.u0 + a * (self.u1 - self.u0)
        v = self.v0 + b * (self.v2 - self.v0)
        return float(inv(u, self.xlog)), float(inv(v, self.ylog))

    def to_pixel(self, x, y):
        a = (fwd(x, self.xlog) - self.u0) / (self.u1 - self.u0)
        b = (fwd(y, self.ylog) - self.v0) / (self.v2 - self.v0)
        p = self.P0 + self.A @ np.array([a, b])
        return float(p[0]), float(p[1])


# ----------------------------------------------------------------------------
# Start-up
# ----------------------------------------------------------------------------
init_state()

st.title("📈 Plot Digitizer")
st.caption("Extract x-y data from an image of a graph.")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Image")
    up = st.file_uploader(
        "Upload a graph image",
        type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp"],
    )
    if up is not None:
        fid = f"{up.name}-{up.size}"
        if fid != ss.file_id:
            try:
                ss.image = load_image(up)
                ss.file_id = fid
                reset_all()
            except Exception as e:
                st.error(f"Could not open image: {e}")
                ss.image = None

    st.header("2. Axes")
    xlabel = st.text_input("X-axis label", value="x")
    ylabel = st.text_input("Y-axis label", value="y")
    c1, c2 = st.columns(2)
    xscale = c1.radio("X scale", ["Linear", "Log"], horizontal=True)
    yscale = c2.radio("Y scale", ["Linear", "Log"], horizontal=True)
    xlog, ylog = xscale == "Log", yscale == "Log"

    st.subheader("Calibration values")
    st.caption(
        "Click the 3 points on the image, then type the true axis values at "
        "those points. The origin can be any known point (e.g. a tick "
        "intersection) – required for log axes."
    )
    c1, c2 = st.columns(2)
    x0 = c1.number_input("Origin x", value=1.0 if xlog else 0.0, format="%g", key=f"x0_{xlog}")
    y0 = c2.number_input("Origin y", value=1.0 if ylog else 0.0, format="%g", key=f"y0_{ylog}")
    x1 = c1.number_input("X-axis point: x", value=10.0 if xlog else 1.0, format="%g", key=f"x1_{xlog}")
    y2 = c2.number_input("Y-axis point: y", value=10.0 if ylog else 1.0, format="%g", key=f"y2_{ylog}")

    b1, b2 = st.columns(2)
    b1.button("Undo cal. point", on_click=cb_undo_cal, disabled=not ss.cal)
    b2.button("Redo calibration", on_click=cb_reset_cal, disabled=not ss.cal)

    st.header("3. Datasets")
    ss.active_idx = min(ss.active_idx, len(ss.datasets) - 1)
    st.selectbox(
        "Active dataset (receives clicks)",
        options=list(range(len(ss.datasets))),
        format_func=lambda i: ss.datasets[i]["name"],
        key="active_idx",
    )
    ds = ss.datasets[ss.active_idx]
    ds["name"] = st.text_input("Dataset name", value=ds["name"], key=f"name_{ds['id']}")
    st.markdown(
        f"<span style='color:{ds['color']};font-size:1.4em'>●</span> "
        f"{len(ds['points'])} points",
        unsafe_allow_html=True,
    )
    b1, b2 = st.columns(2)
    b1.button("↩ Undo point", on_click=cb_undo, disabled=not ds["points"])
    b2.button("Clear points", on_click=cb_clear_points, disabled=not ds["points"])
    st.button("🗑 Delete dataset", on_click=cb_delete_dataset)

    st.text_input("New dataset name (optional)", key="new_ds_name")
    st.button("➕ Add dataset", on_click=cb_add_dataset)

    st.header("4. Zoom")
    zoom = st.slider("Zoom", 1.0, 10.0, 1.0, 0.5)
    pan_x = pan_y = 50
    if zoom > 1:
        pan_x = st.slider("Pan horizontally (%)", 0, 100, 50)
        pan_y = st.slider("Pan vertically (%)", 0, 100, 50)

# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
img = ss.image
if img is None:
    st.info("👈 Upload an image in the sidebar to get started.")
    st.stop()

W, H = img.size

# Calibration object (if ready)
cal_error = validate_calibration(x0, y0, x1, y2, xlog, ylog)
calib = None
if len(ss.cal) == 3 and cal_error is None:
    c = Calibration(ss.cal, x0, y0, x1, y2, xlog, ylog)
    if c.ok:
        calib = c
    else:
        cal_error = "The three calibration points are collinear – please redo calibration."

# Status message
if len(ss.cal) < 3:
    st.info(
        f"**Calibration step {len(ss.cal) + 1} of 3:** click the "
        f"**{CAL_LABELS[len(ss.cal)]}** on the image "
        f"(for rotated scans, click along the actual tilted axis)."
    )
elif cal_error:
    st.error(cal_error)
else:
    st.success(
        f"Calibrated ✔  Click on the image to add points to "
        f"**{ds['name']}**."
    )

# ---- Build the (zoomed) view --------------------------------------------------
crop_w, crop_h = W / zoom, H / zoom
left = int(round((W - crop_w) * pan_x / 100))
top = int(round((H - crop_h) * pan_y / 100))
right = min(W, left + int(round(crop_w)))
bottom = min(H, top + int(round(crop_h)))
cw, ch = right - left, bottom - top
scale = DISPLAY_W / cw

view = img.crop((left, top, right, bottom)).resize(
    (DISPLAY_W, max(1, int(round(ch * scale)))), Image.BICUBIC
)
draw = ImageDraw.Draw(view)


def to_disp(px, py):
    return (px - left) * scale, (py - top) * scale


# Calibration markers
for (px, py), tag in zip(ss.cal, CAL_TAGS):
    dx, dy = to_disp(px, py)
    draw.line([(dx - 10, dy), (dx + 10, dy)], fill=CAL_COLOR, width=2)
    draw.line([(dx, dy - 10), (dx, dy + 10)], fill=CAL_COLOR, width=2)
    draw.text((dx + 8, dy + 6), tag, fill=CAL_COLOR)

# Data point markers (same colour per dataset)
for d in ss.datasets:
    for px, py in d["points"]:
        dx, dy = to_disp(px, py)
        if -10 <= dx <= view.width + 10 and -10 <= dy <= view.height + 10:
            r = 5
            draw.ellipse(
                [dx - r, dy - r, dx + r, dy + r],
                fill=d["color"], outline="black", width=1,
            )

res = streamlit_image_coordinates(
    view, key=f"img_{ss.nclicks}_{zoom}_{pan_x}_{pan_y}"
)

# ---- Handle a click -----------------------------------------------------------
if res is not None:
    ox = left + res["x"] * cw / res["width"]
    oy = top + res["y"] * ch / res["height"]
    if len(ss.cal) < 3:
        ss.cal.append((ox, oy))
        bump()
        st.rerun()
    elif calib is not None:
        ds["points"].append((ox, oy))
        bump()
        st.rerun()
    else:
        st.warning("Fix the calibration values in the sidebar before adding points.")

# ---- Live editable table (bonus) ----------------------------------------------
st.subheader(f"Data table – {ds['name']}")
if calib is None:
    st.caption("Complete calibration to see data values.")
else:
    st.caption("Edit values or add/delete rows – the marks on the graph update automatically.")
    pts = [calib.to_data(px, py) for px, py in ds["points"]]
    df = pd.DataFrame(np.array(pts, dtype=float).reshape(-1, 2), columns=["x", "y"])
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        key=f"ed_{ds['id']}_{ss.nclicks}",
        column_config={
            "x": st.column_config.NumberColumn(xlabel or "x", format="%.6g"),
            "y": st.column_config.NumberColumn(ylabel or "y", format="%.6g"),
        },
    )
    clean = edited.dropna()
    changed = clean.shape != df.shape or (
        len(df) > 0 and not np.allclose(clean.values, df.values, rtol=1e-9, atol=0)
    )
    if changed:
        new_px = []
        for x, y in clean.values:
            if (xlog and x <= 0) or (ylog and y <= 0):
                continue  # invalid on a log axis
            new_px.append(calib.to_pixel(x, y))
        ds["points"] = new_px
        bump()
        st.rerun()

# ---- CSV download -------------------------------------------------------------
def build_csv():
    cols = []
    for d in ss.datasets:
        if calib is not None:
            vals = [calib.to_data(px, py) for px, py in d["points"]]
        else:
            vals = []
        cols.append((d["name"], vals))
    n = max((len(v) for _, v in cols), default=0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([c for name, _ in cols for c in (name, "")])
    w.writerow([c for _ in cols for c in (xlabel, ylabel)])
    for i in range(n):
        row = []
        for _, vals in cols:
            if i < len(vals):
                row += [f"{vals[i][0]:.8g}", f"{vals[i][1]:.8g}"]
            else:
                row += ["", ""]
        w.writerow(row)
    return buf.getvalue()


st.divider()
total = sum(len(d["points"]) for d in ss.datasets)
st.download_button(
    "⬇ Download CSV",
    data=build_csv(),
    file_name="digitized_data.csv",
    mime="text/csv",
    disabled=(calib is None or total == 0),
)
