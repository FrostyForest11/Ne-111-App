"""
NE111 Term Project - Plot Digitizer (Streamlit)

Workflow
--------
1. Upload an image of a graph (png, jpg, gif, tiff, bmp; colour or greyscale).
2. Select 3 calibration points on the image (Point 1 = origin / reference point,
   Point 2 = a point on the X-axis, Point 3 = a point on the Y-axis) and type their
   true values in the sidebar. The 3-point affine calibration handles rotated scans.
   Linear and log axes (including semi-log) are supported.
3. Select points on the graph to digitize data. Each dataset has its own colour AND
   its own marker shape (so colour is never the only way to tell datasets apart).
4. Edit points in the live table, zoom/pan for precision, and download a CSV.

Accessibility (WCAG 2.2 AA, where within the app's control)
-----------------------------------------------------------
* Colour is never the only cue: each dataset has a distinct marker shape + a text legend.
* Colour palette chosen for >= 3:1 contrast against white; markers have white + black
  halos so they stay visible on any image background.
* Every control has a visible text label; no emoji-only or icon-only buttons.
* Mouse-free alternatives to clicking on the image (typed pixel / data values).
* Text equivalents for the image: calibration summary table, data table, legend, counts.
* Status and error messages are written as text with instructions on how to fix them.
"""

import csv
import html
import io
import math

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Plot Digitizer", layout="wide")

# Colour-blind-friendly palette, each >= 3:1 contrast on white.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#9A6700",  # dark orange
    "#6A3D9A",  # purple
    "#A50026",  # dark red
    "#006D77",  # teal
]
SHAPES = ["circle", "square", "triangle", "diamond", "inverted triangle", "hexagon"]
SYMBOLS = ["●", "■", "▲", "◆", "▼", "⬢"]

DISPLAY_W = 900
CAL_LABELS = ["Origin (reference point)", "X-axis point", "Y-axis point"]
CAL_SHORT = ["1 Origin", "2 X-axis", "3 Y-axis"]

try:
    FONT = ImageFont.load_default(size=16)
except TypeError:  # very old Pillow
    FONT = ImageFont.load_default()

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
            "shape": i % len(SHAPES),
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
# Marker drawing (shape + colour, with halo for contrast on any background)
# ----------------------------------------------------------------------------
def shape_points(kind, cx, cy, r):
    if kind == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if kind == "triangle":
        return [(cx, cy - 1.2 * r), (cx + 1.1 * r, cy + 0.9 * r), (cx - 1.1 * r, cy + 0.9 * r)]
    if kind == "inverted triangle":
        return [(cx, cy + 1.2 * r), (cx + 1.1 * r, cy - 0.9 * r), (cx - 1.1 * r, cy - 0.9 * r)]
    if kind == "diamond":
        return [(cx, cy - 1.3 * r), (cx + 1.3 * r, cy), (cx, cy + 1.3 * r), (cx - 1.3 * r, cy)]
    if kind == "hexagon":
        return [
            (cx + r * math.cos(math.radians(60 * k)), cy + r * math.sin(math.radians(60 * k)))
            for k in range(6)
        ]
    return None  # circle


def draw_marker(draw, kind, cx, cy, r, color):
    for size, fill in ((r + 3, "white"), (r + 1.5, "black"), (r, color)):
        pts = shape_points(kind, cx, cy, size)
        if pts is None:
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=fill)
        else:
            draw.polygon(pts, fill=fill)


def draw_cal_marker(draw, cx, cy, label):
    """Crosshair with white halo + black core, plus a text label in a white box."""
    for width, fill in ((6, "white"), (2, "black")):
        draw.line([(cx - 12, cy), (cx + 12, cy)], fill=fill, width=width)
        draw.line([(cx, cy - 12), (cx, cy + 12)], fill=fill, width=width)
    bbox = draw.textbbox((cx + 10, cy + 8), label, font=FONT)
    draw.rectangle(
        [bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2], fill="white", outline="black"
    )
    draw.text((cx + 10, cy + 8), label, fill="black", font=FONT)


# ----------------------------------------------------------------------------
# Calibration (3-point affine => handles rotation and skew)
# ----------------------------------------------------------------------------
def fwd(v, is_log):
    return np.log10(v) if is_log else v


def inv(u, is_log):
    return 10.0 ** u if is_log else u


def validate_calibration(x0, y0, x1, y2, xlog, ylog):
    if xlog and (x0 <= 0 or x1 <= 0):
        return (
            "Log x-axis: Point 1's x value and Point 2's x value must both be greater "
            "than 0. Change them in the sidebar under 'Calibration values'."
        )
    if ylog and (y0 <= 0 or y2 <= 0):
        return (
            "Log y-axis: Point 1's y value and Point 3's y value must both be greater "
            "than 0. Change them in the sidebar under 'Calibration values'."
        )
    if x1 == x0:
        return "Point 2's x value must be different from Point 1's x value."
    if y2 == y0:
        return "Point 3's y value must be different from Point 1's y value."
    return None


class Calibration:
    """
    Pixel p = P0 + a*(P1-P0) + b*(P2-P0), where
      a = (u - u0)/(u1 - u0),  b = (v - v0)/(v2 - v0)
    and u, v are x, y (or log10 of them on log axes).
    P0: Point 1 pixel, P1: Point 2 pixel, P2: Point 3 pixel.
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

st.title("Plot Digitizer")
st.write(
    "Extract x-y data from an image of a graph. "
    "New here? Open the **Help and FAQ** tab for step-by-step instructions."
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Image")
    up = st.file_uploader(
        "Upload a graph image (PNG, JPG, GIF, TIFF or BMP)",
        type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp"],
    )
    if up is not None:
        fid = f"{up.name}-{up.size}"
        if fid != ss.file_id:
            try:
                ss.image = load_image(up)
                ss.file_id = fid
                ss.file_name = up.name
                reset_all()
            except Exception as e:
                st.error(f"Could not open that image ({e}). Try saving it as PNG or JPG.")
                ss.image = None

    st.header("2. Axes")
    xlabel = st.text_input("X-axis name (used as the CSV label)", value="x")
    ylabel = st.text_input("Y-axis name (used as the CSV label)", value="y")
    xscale = st.radio("X-axis scale", ["Linear", "Log"], horizontal=True)
    yscale = st.radio("Y-axis scale", ["Linear", "Log"], horizontal=True)
    xlog, ylog = xscale == "Log", yscale == "Log"

    st.header("3. Calibration values")
    st.write(
        "First select 3 points on the image (see the instructions above the image). "
        "Then type below what each point is worth on the graph's axes. "
        "Tip: use gridline crossings or labelled tick marks that are far apart."
    )

    st.subheader("Point 1: Origin (reference point)")
    st.caption("Any point whose x and y values you know. It does not have to be (0, 0).")
    x0 = st.number_input(
        "Point 1 (origin): x value", value=1.0 if xlog else 0.0, format="%g", key=f"x0_{xlog}"
    )
    y0 = st.number_input(
        "Point 1 (origin): y value", value=1.0 if ylog else 0.0, format="%g", key=f"y0_{ylog}"
    )

    st.subheader("Point 2: X-axis point")
    st.caption(
        "A point on the same horizontal line as Point 1 (so it has the same y value), "
        "away from Point 1 along the x-axis."
    )
    x1 = st.number_input(
        "Point 2 (X-axis point): x value",
        value=10.0 if xlog else 1.0,
        format="%g",
        key=f"x1_{xlog}",
    )

    st.subheader("Point 3: Y-axis point")
    st.caption(
        "A point on the same vertical line as Point 1 (so it has the same x value), "
        "away from Point 1 along the y-axis."
    )
    y2 = st.number_input(
        "Point 3 (Y-axis point): y value",
        value=10.0 if ylog else 1.0,
        format="%g",
        key=f"y2_{ylog}",
    )

    st.button("Undo last calibration point", on_click=cb_undo_cal, disabled=not ss.cal)
    st.button("Restart calibration", on_click=cb_reset_cal, disabled=not ss.cal)

    st.header("4. Datasets")
    ss.active_idx = min(ss.active_idx, len(ss.datasets) - 1)
    st.selectbox(
        "Active dataset (new points are added to this one)",
        options=list(range(len(ss.datasets))),
        format_func=lambda i: f"{SYMBOLS[ss.datasets[i]['shape']]} {ss.datasets[i]['name']}",
        key="active_idx",
    )
    ds = ss.datasets[ss.active_idx]
    ds["name"] = st.text_input("Name of active dataset", value=ds["name"], key=f"name_{ds['id']}")
    st.write(
        f"Marker: {SHAPES[ds['shape']]} {SYMBOLS[ds['shape']]}. "
        f"Points so far: {len(ds['points'])}."
    )
    st.button("Undo last point", on_click=cb_undo, disabled=not ds["points"])
    st.button("Remove all points from this dataset", on_click=cb_clear_points, disabled=not ds["points"])
    st.button("Delete this dataset", on_click=cb_delete_dataset)

    st.text_input("Name for a new dataset (optional)", key="new_ds_name")
    st.button("Add new dataset", on_click=cb_add_dataset)

    st.header("5. Zoom")
    zoom = st.slider("Zoom level (1 = whole image)", 1.0, 10.0, 1.0, 0.5)
    pan_x = pan_y = 50
    if zoom > 1:
        pan_x = st.slider("Move view left / right (%)", 0, 100, 50)
        pan_y = st.slider("Move view up / down (%)", 0, 100, 50)


# ----------------------------------------------------------------------------
# Digitizer tab
# ----------------------------------------------------------------------------
def legend_markdown(calib):
    lines = []
    for i, d in enumerate(ss.datasets):
        chip = (
            f"<span style='background:#ffffff;color:{d['color']};border:1px solid #767676;"
            f"border-radius:4px;padding:0 6px;font-size:1.1em'>{SYMBOLS[d['shape']]}</span>"
        )
        active = " (active: receives new points)" if i == ss.active_idx else ""
        n = len(d["points"])
        lines.append(
            f"- {chip} **{html.escape(d['name'])}** – {SHAPES[d['shape']]} marker, "
            f"{n} point{'s' if n != 1 else ''}{active}"
        )
    return "\n".join(lines)


def build_csv(calib):
    cols = []
    for d in ss.datasets:
        vals = [calib.to_data(px, py) for px, py in d["points"]] if calib is not None else []
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


def render_digitizer():
    img = ss.image
    if img is None:
        st.info("Upload an image using the sidebar (section 1) to get started.")
        return

    W, H = img.size
    xl, yl = (xlabel or "x"), (ylabel or "y")

    # --- Calibration object (if ready) ---
    cal_error = validate_calibration(x0, y0, x1, y2, xlog, ylog)
    calib = None
    if len(ss.cal) == 3 and cal_error is None:
        c = Calibration(ss.cal, x0, y0, x1, y2, xlog, ylog)
        if c.ok:
            calib = c
        else:
            cal_error = (
                "The three calibration points lie on one straight line. "
                "Choose 'Restart calibration' in the sidebar and select three points "
                "that form a corner shape (an L)."
            )

    # --- Status / instructions (plain text, not colour-only) ---
    st.subheader("Instructions")
    if len(ss.cal) < 3:
        n = len(ss.cal) + 1
        st.info(
            f"Calibration step {n} of 3: select **Point {n} – {CAL_LABELS[n - 1]}** on the "
            f"image below. For a rotated scan, follow the tilted axis or gridline. "
            f"(No mouse? Use 'Add points without a mouse' further down this tab.)"
        )
    elif cal_error:
        st.error(f"Calibration problem: {cal_error}")
    else:
        st.success(
            f"Calibration complete. Select points on the image to add them to the active "
            f"dataset: {ds['name']}."
        )

    # --- Calibration summary (text equivalent of the markers) ---
    pixel_txt = [
        f"column {ss.cal[i][0]:.0f}, row {ss.cal[i][1]:.0f}" if i < len(ss.cal) else "not selected yet"
        for i in range(3)
    ]
    cal_df = pd.DataFrame(
        {
            "Point": ["Point 1", "Point 2", "Point 3"],
            "Role": CAL_LABELS,
            "Selected at (image pixels)": pixel_txt,
            f"Value you entered ({xl}, {yl})": [
                f"{xl} = {x0:g}, {yl} = {y0:g}",
                f"{xl} = {x1:g}, {yl} = {y0:g} (same {yl} as Point 1)",
                f"{xl} = {x0:g} (same as Point 1), {yl} = {y2:g}",
            ],
        }
    )
    st.table(cal_df.set_index("Point"))

    # --- Build the (zoomed) view ---
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

    for (px, py), label in zip(ss.cal, CAL_SHORT):
        dx, dy = to_disp(px, py)
        draw_cal_marker(draw, dx, dy, label)

    for d in ss.datasets:
        for px, py in d["points"]:
            dx, dy = to_disp(px, py)
            if -15 <= dx <= view.width + 15 and -15 <= dy <= view.height + 15:
                draw_marker(draw, SHAPES[d["shape"]], dx, dy, 6, d["color"])

    st.subheader("Graph image")
    st.caption(
        f"File: {ss.get('file_name', 'uploaded image')} – {W} × {H} pixels. "
        f"Zoom {zoom:g}×. Showing columns {left}–{right}, rows {top}–{bottom}."
    )
    res = streamlit_image_coordinates(view, key=f"img_{ss.nclicks}_{zoom}_{pan_x}_{pan_y}")

    # --- Handle a selection on the image ---
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
            st.warning("Fix the calibration problem described above before adding points.")

    # --- Legend (text equivalent of marker colours/shapes) ---
    st.subheader("Legend")
    st.markdown(legend_markdown(calib), unsafe_allow_html=True)

    # --- Live editable table ---
    st.subheader(f"Data table for {ds['name']}")
    if calib is None:
        st.write("Complete calibration to see data values here.")
    else:
        st.caption(
            "You can edit any value, add a row, or delete a row. "
            "The markers on the graph update automatically."
        )
        pts = [calib.to_data(px, py) for px, py in ds["points"]]
        df = pd.DataFrame(np.array(pts, dtype=float).reshape(-1, 2), columns=["x", "y"])
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            key=f"ed_{ds['id']}_{ss.nclicks}",
            column_config={
                "x": st.column_config.NumberColumn(xl, format="%.6g"),
                "y": st.column_config.NumberColumn(yl, format="%.6g"),
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
                    continue  # not valid on a log axis
                new_px.append(calib.to_pixel(x, y))
            ds["points"] = new_px
            bump()
            st.rerun()

    # --- Mouse-free alternatives ---
    with st.expander("Add points without a mouse (keyboard-friendly alternative)"):
        st.write(
            "Selecting a spot on the image needs a pointing device. These fields do the same "
            "jobs using typed numbers only."
        )
        st.markdown("#### A. Set a calibration point by pixel position")
        st.caption(
            "Pixel positions are measured in the original image: column from the left edge, "
            "row from the top edge. Set the points in order: 1, then 2, then 3."
        )
        kp = st.selectbox(
            "Calibration point to set",
            [1, 2, 3],
            format_func=lambda n: f"Point {n}: {CAL_LABELS[n - 1]}",
            key="kb_cal_pt",
        )
        kx = st.number_input(
            f"Pixel column (0 to {W - 1})", min_value=0, max_value=W - 1, value=0, step=1, key="kb_px"
        )
        ky = st.number_input(
            f"Pixel row (0 to {H - 1})", min_value=0, max_value=H - 1, value=0, step=1, key="kb_py"
        )
        if st.button("Set this calibration point", key="kb_set_cal"):
            i = kp - 1
            if i < len(ss.cal):
                ss.cal[i] = (float(kx), float(ky))
                bump()
                st.rerun()
            elif i == len(ss.cal):
                ss.cal.append((float(kx), float(ky)))
                bump()
                st.rerun()
            else:
                st.error(f"Set Point {len(ss.cal) + 1} first, then continue in order.")

        st.markdown("#### B. Add a data point by its values")
        nx = st.number_input(f"{xl} value of the new point", value=0.0, format="%g", key="kb_nx")
        ny = st.number_input(f"{yl} value of the new point", value=0.0, format="%g", key="kb_ny")
        if st.button("Add this point to the active dataset", key="kb_add"):
            if calib is None:
                st.error("Complete calibration first (see Instructions at the top of this tab).")
            elif (xlog and nx <= 0) or (ylog and ny <= 0):
                st.error("Values must be greater than 0 on a log axis.")
            else:
                ds["points"].append(calib.to_pixel(nx, ny))
                bump()
                st.rerun()

    # --- Download ---
    st.subheader("Download")
    total = sum(len(d["points"]) for d in ss.datasets)
    if calib is None or total == 0:
        st.write("The download becomes available once calibration is complete and at least one point exists.")
    st.download_button(
        "Download CSV file",
        data=build_csv(calib),
        file_name="digitized_data.csv",
        mime="text/csv",
        disabled=(calib is None or total == 0),
    )


# ----------------------------------------------------------------------------
# Help tab
# ----------------------------------------------------------------------------
def render_help():
    st.header("Help and FAQ")

    st.subheader("Quick start")
    st.markdown(
        """
1. **Upload** your graph image (sidebar, section 1).
2. **Name your axes** and choose **Linear** or **Log** for each (sidebar, section 2).
3. **Calibrate:** select 3 points on the image, then type their true values (sidebar, section 3).
4. **Digitize:** select points along your curve. Use **Add new dataset** for more lines.
5. **Check and fix** points in the data table, then **Download CSV file**.
"""
    )

    st.subheader("Step-by-step guide")

    st.markdown("#### Step 1 – Upload")
    st.markdown(
        "Use the upload field in the sidebar. PNG, JPG, GIF, TIFF and BMP files work, in "
        "colour or greyscale. For multi-page TIFF or animated GIF files only the first page "
        "or frame is used."
    )

    st.markdown("#### Step 2 – Name the axes and choose the scales")
    st.markdown(
        "Type the names you want in the CSV (for example *Year* and *Sales*). Choose **Log** for "
        "an axis only if its tick labels go up by powers (1, 10, 100…) with equal spacing. "
        "Choose Log for just one axis for a semi-log plot."
    )

    st.markdown("#### Step 3 – Calibrate (this sets the scale)")
    st.markdown(
        """
The app needs three reference points to learn how image positions map to graph values:

| Point | Where to select it | What to type in the sidebar |
|---|---|---|
| **Point 1 – Origin** | Any spot where you know both the x and y values, ideally a corner of the plot area | Its x value and its y value |
| **Point 2 – X-axis point** | On the same horizontal line as Point 1, far to one side | Its x value (its y value is the same as Point 1's) |
| **Point 3 – Y-axis point** | On the same vertical line as Point 1, far above or below | Its y value (its x value is the same as Point 1's) |

Points are selected in order. The instructions at the top of the Digitizer tab always say which
one is next. The table under the instructions confirms what you selected and what values the
app is using. If a point lands in the wrong spot, use **Undo last calibration point** or
**Restart calibration** in the sidebar.

**Tips for an accurate scale**
- Choose points that are far apart; small selection errors then matter less.
- Use the **Zoom level** and **Move view** controls to place each point precisely.
- Use gridline crossings or tick marks, not the text of the labels.
"""
    )

    st.markdown("#### Step 4 – Digitize data points")
    st.markdown(
        "Select the dataset you want in **Active dataset**, then select points on the curve in "
        "the image. Each dataset has its own colour **and** marker shape, listed in the Legend. "
        "Use **Add new dataset** for another line and **Delete this dataset** to remove one."
    )

    st.markdown("#### Step 5 – Review and edit")
    st.markdown(
        "The data table shows the values of the active dataset. Edit a number, add a row or delete "
        "a row, and the marker on the image moves to match. **Undo last point** removes the most "
        "recent point."
    )

    st.markdown("#### Step 6 – Download")
    st.markdown(
        "**Download CSV file** saves every dataset. Row 1 has the dataset names, row 2 has your "
        "x and y axis names, and the following rows have the values."
    )

    st.subheader("Worked example: a line chart with years and 30–70")
    st.markdown(
        """
Suppose the x-axis shows years 2011 to 2018 (one gridline per year) and the y-axis shows 30 to 70
in steps of 10.

- Axis names: *Year* and *Value*. Both scales: **Linear**.
- **Point 1:** where the 2011 gridline crosses the 30 gridline. Type x = 2011, y = 30.
- **Point 2:** where the 2018 gridline crosses the 30 gridline. Type x = 2018.
- **Point 3:** where the 2011 gridline crosses the 70 gridline. Type y = 70.

Now one gridline step equals one year across and 10 units up, automatically.
"""
    )

    st.subheader("Rotated scans")
    st.markdown(
        "No separate rotation step is needed. Select the three calibration points along the "
        "tilted axes or gridlines, exactly as they appear. The app uses the direction of those "
        "lines to correct for the tilt. Very large tilts or mirrored images are better fixed in "
        "an image editor first."
    )

    st.subheader("Log axes")
    st.markdown(
        "On a log axis all calibration values on that axis must be greater than 0, and so must any "
        "value you type into the table. Use gridline crossings such as 1, 10, 100 as your "
        "calibration values. The app places values by their position between the gridlines "
        "on a logarithmic scale."
    )

    st.subheader("Frequently asked questions")

    with st.expander("The Download button is greyed out. Why?"):
        st.write(
            "Downloads need two things: a finished calibration (three points selected and valid "
            "values entered) and at least one data point. A message above the button says "
            "which is missing."
        )
    with st.expander("My values look wrong. What should I check?"):
        st.write(
            "1. Check the table under Instructions: are the typed values the ones the graph shows "
            "at those points? 2. Is each axis set to the right scale (Linear or Log)? "
            "3. Are Points 2 and 3 really on the same horizontal and vertical lines as Point 1? "
            "4. Restart calibration and use zoom to place the points more precisely."
        )
    with st.expander("Does the origin have to be (0, 0)?"):
        st.write(
            "No. Point 1 can be any spot where you know both values, such as the crossing of two "
            "gridlines. This is essential for log axes, which have no zero."
        )
    with st.expander("Can Point 2 or Point 3 be any point on the graph?"):
        st.write(
            "They must lie on the same horizontal line (Point 2) and the same vertical line "
            "(Point 3) as Point 1, because the app assumes Point 2 shares Point 1's y value and "
            "Point 3 shares Point 1's x value. On a rotated image, follow the tilted lines."
        )
    with st.expander("I clicked in the wrong place. How do I fix it?"):
        st.write(
            "During calibration, use **Undo last calibration point**. For data, use **Undo last "
            "point**, or fix or delete the row in the data table."
        )
    with st.expander("What happens if I change the calibration values after adding points?"):
        st.write(
            "Points are remembered by their position on the image, so the data values update "
            "automatically to match the new calibration."
        )
    with st.expander("Can I digitize more than one line from the same image?"):
        st.write(
            "Yes. Use **Add new dataset**, name it, and add points. Choose which dataset receives "
            "points with **Active dataset**. Each dataset has its own colour and marker shape."
        )
    with st.expander("What does the CSV look like?"):
        st.code(
            "Dataset 1,,Dataset 2,\nYear,Value,Year,Value\n2011,30,2012,41\n2012,40,2013,44",
            language="text",
        )
        st.write(
            "Datasets with fewer points leave blank cells at the bottom of their columns."
        )
    with st.expander("Is my work saved?"):
        st.write(
            "Only for your current browser session. Refreshing the page or closing it clears "
            "everything, so download your CSV before you leave."
        )
    with st.expander("Is my image uploaded to anyone?"):
        st.write(
            "The image is processed by this app while you use it and is not kept after your "
            "session ends."
        )
    with st.expander("Which file types and sizes work?"):
        st.write(
            "PNG, JPG, GIF, TIFF and BMP. Very large images may load slowly; if so, reduce the "
            "image size in an image editor first."
        )

    st.subheader("Accessibility")
    st.markdown(
        """
This app is designed to meet **WCAG 2.2 level AA** where it is within the app's control:

- **Not by colour alone:** each dataset has a different marker shape and a text legend, as well as a colour.
- **Contrast:** dataset colours were chosen for at least 3:1 contrast against white, and markers have
  white and black outlines so they stay visible on any image.
- **Labels and instructions:** every control has a visible text label, and error messages explain how
  to fix the problem.
- **Keyboard and no-mouse use:** the sidebar controls, tables and buttons work with the keyboard. The
  expander "Add points without a mouse" lets you set calibration points and add data points by typing numbers.
- **Text equivalents:** the calibration summary table, the legend and the data table give the same information
  as the markers on the image.

**Known limitations:** selecting a spot directly on the image needs a pointing device, so use the
keyboard alternative if you cannot. The app is built on Streamlit, whose built-in widgets set some accessibility
behaviour. If you find a barrier, please tell the webapp admin so it can be fixed.
"""
    )


# ----------------------------------------------------------------------------
# Page layout: single page with two tabs
# ----------------------------------------------------------------------------
tab_app, tab_help = st.tabs(["Digitizer", "Help and FAQ"])
with tab_app:
    render_digitizer()
with tab_help:
    render_help()
