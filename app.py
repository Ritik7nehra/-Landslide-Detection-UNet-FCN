"""
Landslide Detection Dashboard
Author: Ritik Nehra

Streamlit dashboard for landslide segmentation using RGB + NDVI imagery.
The included inference function is a realistic placeholder that can be replaced
with trained U-Net / FCN model inference using PyTorch or TensorFlow/Keras.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tifffile
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Landslide Detection Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class PredictionMetrics:
    total_pixels: int
    landslide_pixels: int
    landslide_percentage: float
    pixel_resolution_m: float
    total_area_km2: float
    landslide_area_km2: float
    severity: str


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize any numeric image array to uint8 range [0, 255]."""
    arr = np.asarray(arr).astype(np.float32)
    finite_mask = np.isfinite(arr)

    if not finite_mask.any():
        raise ValueError("Uploaded image contains no valid numeric values.")

    arr = np.where(finite_mask, arr, np.nan)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)

    if np.isclose(max_val, min_val):
        return np.zeros(arr.shape, dtype=np.uint8)

    norm = (arr - min_val) / (max_val - min_val)
    return np.clip(norm * 255, 0, 255).astype(np.uint8)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Read PNG/JPG/TIFF uploads into a NumPy array."""
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    file_name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    if file_name.endswith((".tif", ".tiff")):
        image = tifffile.imread(io.BytesIO(raw_bytes))
        return np.asarray(image)

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        return np.asarray(image)
    except Exception as exc:
        raise ValueError(f"Unable to read image file: {exc}") from exc


def prepare_rgb(rgb_arr: np.ndarray) -> np.ndarray:
    """Convert uploaded image to a clean 3-channel RGB visualization."""
    arr = np.asarray(rgb_arr)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] >= 3:
            arr = arr[..., :3]
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        else:
            raise ValueError("RGB image must contain at least 3 channels or be grayscale.")
    else:
        raise ValueError("Unsupported RGB image dimensions.")

    return normalize_to_uint8(arr)


def extract_rgb_ndvi_from_4channel(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract RGB and NDVI from a preprocessed 4-channel image."""
    arr = np.asarray(arr)

    if arr.ndim != 3 or arr.shape[-1] < 4:
        raise ValueError(
            "Preprocessed input must be a 4-channel image with channels [R, G, B, NDVI]."
        )

    rgb = prepare_rgb(arr[..., :3])
    ndvi = arr[..., 3].astype(np.float32)

    # If NDVI appears to be stored as 0-255, scale it to -1 to 1.
    if np.nanmax(ndvi) > 1.5 or np.nanmin(ndvi) < -1.5:
        ndvi = normalize_to_uint8(ndvi).astype(np.float32) / 127.5 - 1.0

    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return rgb, ndvi


def calculate_ndvi(rgb: np.ndarray, nir_arr: np.ndarray) -> np.ndarray:
    """Calculate NDVI using (NIR - Red) / (NIR + Red)."""
    red = rgb[..., 0].astype(np.float32) / 255.0

    nir = np.asarray(nir_arr)
    if nir.ndim == 3:
        # If a multi-channel NIR image is uploaded, use the first channel.
        nir = nir[..., 0]
    elif nir.ndim != 2:
        raise ValueError("NIR image must be grayscale or a readable image channel.")

    nir = normalize_to_uint8(nir).astype(np.float32) / 255.0

    if nir.shape != red.shape:
        nir = cv2.resize(nir, (red.shape[1], red.shape[0]), interpolation=cv2.INTER_LINEAR)

    denominator = nir + red
    ndvi = np.divide(nir - red, denominator + 1e-6)
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(ndvi, -1.0, 1.0)


def resize_for_model(rgb: np.ndarray, ndvi: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Create 4-channel model input: RGB + NDVI."""
    rgb_resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    ndvi_resized = cv2.resize(ndvi, (target_size, target_size), interpolation=cv2.INTER_AREA)
    rgb_float = rgb_resized.astype(np.float32) / 255.0
    ndvi_float = ndvi_resized.astype(np.float32)[..., None]
    return np.concatenate([rgb_float, ndvi_float], axis=-1)


# -----------------------------------------------------------------------------
# Model placeholder and integration point
# -----------------------------------------------------------------------------
def predict_landslide(
    rgb: np.ndarray,
    ndvi: np.ndarray,
    model_type: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder landslide prediction.

    Replace this function with your real model inference.

    Expected real model steps:
    1. Build 4-channel tensor: [R, G, B, NDVI]
    2. Resize/normalize exactly like training.
    3. Load U-Net or FCN weights.
    4. Run model.forward() or model.predict().
    5. Resize probability map back to original image size.
    6. Apply threshold to produce binary mask.

    Returns:
        probability_map: float array [H, W] in range [0, 1]
        binary_mask: uint8 array [H, W], values 0 or 255
    """
    h, w = rgb.shape[:2]

    rgb_float = rgb.astype(np.float32) / 255.0
    red = rgb_float[..., 0]
    green = rgb_float[..., 1]
    blue = rgb_float[..., 2]

    # Heuristic idea: landslide scars often appear as exposed/bare land with
    # lower vegetation, brighter soil-like surfaces, and textural edges.
    brightness = (red + green + blue) / 3.0
    soil_index = np.clip((red + green) / 2.0 - blue * 0.35, 0, 1)
    low_vegetation = np.clip((0.35 - ndvi) / 0.7, 0, 1)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 140).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (7, 7), 0)

    # Slightly different behavior for U-Net and FCN to simulate model selection.
    if model_type == "U-Net":
        probability = 0.42 * low_vegetation + 0.35 * soil_index + 0.15 * brightness + 0.08 * edges
        probability = cv2.GaussianBlur(probability, (5, 5), 0)
        kernel = np.ones((3, 3), np.uint8)
    else:  # FCN
        probability = 0.38 * low_vegetation + 0.32 * soil_index + 0.20 * brightness + 0.10 * edges
        probability = cv2.GaussianBlur(probability, (9, 9), 0)
        kernel = np.ones((5, 5), np.uint8)

    probability = normalize_probability(probability)
    binary = (probability >= threshold).astype(np.uint8) * 255

    # Remove tiny noise and smooth regions.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return probability.astype(np.float32), binary.astype(np.uint8)


def normalize_probability(arr: np.ndarray) -> np.ndarray:
    """Normalize model-like score map to [0, 1]."""
    arr = arr.astype(np.float32)
    min_val = np.min(arr)
    max_val = np.max(arr)
    if np.isclose(max_val, min_val):
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val)


# -----------------------------------------------------------------------------
# Visualization and export functions
# -----------------------------------------------------------------------------
def create_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay bright red landslide mask on RGB image."""
    overlay = rgb.copy().astype(np.float32)
    red_layer = np.zeros_like(overlay)
    red_layer[..., 0] = 255

    mask_bool = mask > 0
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * red_layer[mask_bool]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_mask_png(mask: np.ndarray) -> bytes:
    """Convert binary mask to downloadable PNG bytes."""
    image = Image.fromarray(mask.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_overlay_png(overlay: np.ndarray) -> bytes:
    """Convert RGB overlay to downloadable PNG bytes."""
    image = Image.fromarray(overlay.astype(np.uint8), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_geotiff_bytes(mask: np.ndarray) -> bytes:
    """Create a simple TIFF mask download. For true GeoTIFF, add CRS/transform using rasterio."""
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, mask.astype(np.uint8))
    return buffer.getvalue()


def calculate_metrics(mask: np.ndarray, pixel_resolution_m: float) -> PredictionMetrics:
    """Calculate area and severity metrics from binary mask."""
    total_pixels = int(mask.size)
    landslide_pixels = int(np.count_nonzero(mask > 0))
    landslide_percentage = (landslide_pixels / total_pixels) * 100 if total_pixels else 0.0

    pixel_area_m2 = pixel_resolution_m ** 2
    total_area_km2 = (total_pixels * pixel_area_m2) / 1_000_000
    landslide_area_km2 = (landslide_pixels * pixel_area_m2) / 1_000_000

    if landslide_percentage < 5:
        severity = "Low Risk"
    elif landslide_percentage < 15:
        severity = "Medium Risk"
    else:
        severity = "High Risk"

    return PredictionMetrics(
        total_pixels=total_pixels,
        landslide_pixels=landslide_pixels,
        landslide_percentage=landslide_percentage,
        pixel_resolution_m=pixel_resolution_m,
        total_area_km2=total_area_km2,
        landslide_area_km2=landslide_area_km2,
        severity=severity,
    )


def create_csv_report(metrics: PredictionMetrics, model_type: str, threshold: float) -> bytes:
    """Create CSV report bytes."""
    df = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_type,
                "threshold": threshold,
                "pixel_resolution_m": metrics.pixel_resolution_m,
                "total_pixels": metrics.total_pixels,
                "landslide_pixels": metrics.landslide_pixels,
                "total_area_km2": round(metrics.total_area_km2, 6),
                "landslide_area_km2": round(metrics.landslide_area_km2, 6),
                "landslide_percentage": round(metrics.landslide_percentage, 3),
                "severity": metrics.severity,
            }
        ]
    )
    return df.to_csv(index=False).encode("utf-8")


def create_pdf_report(metrics: PredictionMetrics, model_type: str, threshold: float) -> bytes:
    """Create a simple PDF report summary."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 60, "Landslide Detection Report")

    pdf.setFont("Helvetica", 11)
    y = height - 100
    lines = [
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Selected Model: {model_type}",
        f"Confidence Threshold: {threshold:.2f}",
        f"Pixel Resolution: {metrics.pixel_resolution_m:.2f} meters/pixel",
        f"Total Pixels Analyzed: {metrics.total_pixels:,}",
        f"Detected Landslide Pixels: {metrics.landslide_pixels:,}",
        f"Total Area Analyzed: {metrics.total_area_km2:.6f} km²",
        f"Detected Landslide Area: {metrics.landslide_area_km2:.6f} km²",
        f"Landslide Coverage: {metrics.landslide_percentage:.2f}%",
        f"Severity Status: {metrics.severity}",
    ]

    for line in lines:
        pdf.drawString(50, y, line)
        y -= 24

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        50,
        60,
        "Note: Current dashboard uses placeholder image-processing inference. Replace predict_landslide() with trained model inference.",
    )
    pdf.save()
    return buffer.getvalue()


def create_download_zip(mask: np.ndarray, overlay: np.ndarray, metrics: PredictionMetrics, model_type: str, threshold: float) -> bytes:
    """Bundle all outputs into one ZIP."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("landslide_mask.png", create_mask_png(mask))
        zf.writestr("landslide_overlay.png", create_overlay_png(overlay))
        zf.writestr("landslide_mask.tiff", create_geotiff_bytes(mask))
        zf.writestr("report.csv", create_csv_report(metrics, model_type, threshold))
        zf.writestr("report.pdf", create_pdf_report(metrics, model_type, threshold))
    return buffer.getvalue()


def plot_area_chart(metrics: PredictionMetrics) -> go.Figure:
    """Create a compact Plotly chart for area distribution."""
    safe_area = max(metrics.total_area_km2 - metrics.landslide_area_km2, 0)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Non-landslide area", "Detected landslide area"],
                values=[safe_area, metrics.landslide_area_km2],
                hole=0.55,
            )
        ]
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Area Distribution",
    )
    return fig


# -----------------------------------------------------------------------------
# Sidebar UI
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Model Configuration")
model_type = st.sidebar.selectbox("Choose segmentation model", ["U-Net", "FCN"])
threshold = st.sidebar.slider("Classification confidence threshold", 0.0, 1.0, 0.50, 0.01)
overlay_alpha = st.sidebar.slider("Mask overlay transparency", 0.0, 1.0, 0.45, 0.05)
pixel_resolution_m = st.sidebar.number_input(
    "Pixel resolution (meters/pixel)",
    min_value=0.1,
    max_value=1000.0,
    value=10.0,
    step=0.5,
    help="Used only to estimate area. Example: Sentinel-2 RGB/NIR is often 10 m per pixel.",
)

st.sidebar.divider()
st.sidebar.info(
    "This prototype uses placeholder inference. Replace predict_landslide() with your trained U-Net/FCN weights."
)


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.title("🌍 Landslide Detection Dashboard")
st.caption(
    "Deep learning-ready dashboard for U-Net / FCN segmentation using RGB + NDVI satellite imagery."
)

with st.expander("How to use this dashboard", expanded=False):
    st.markdown(
        """
        1. Select **U-Net** or **FCN** from the sidebar.  
        2. Upload either a **4-channel RGB+NDVI image** or separate **RGB and NIR** images.  
        3. Adjust the confidence threshold and overlay transparency.  
        4. View the predicted landslide mask, metrics, and download reports.  
        """
    )

input_mode = st.radio(
    "Choose input format",
    ["Upload preprocessed 4-channel image: RGB + NDVI", "Upload separate RGB and NIR images"],
    horizontal=False,
)

rgb: Optional[np.ndarray] = None
ndvi: Optional[np.ndarray] = None

try:
    if input_mode.startswith("Upload preprocessed"):
        uploaded_4ch = st.file_uploader(
            "Upload 4-channel image [R, G, B, NDVI] as TIFF/PNG if supported",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
        )
        if uploaded_4ch is not None:
            arr = read_uploaded_image(uploaded_4ch)
            rgb, ndvi = extract_rgb_ndvi_from_4channel(arr)

    else:
        col_upload_1, col_upload_2 = st.columns(2)
        with col_upload_1:
            rgb_file = st.file_uploader("Upload RGB image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        with col_upload_2:
            nir_file = st.file_uploader("Upload NIR image", type=["png", "jpg", "jpeg", "tif", "tiff"])

        if rgb_file is not None and nir_file is not None:
            rgb = prepare_rgb(read_uploaded_image(rgb_file))
            nir = read_uploaded_image(nir_file)
            ndvi = calculate_ndvi(rgb, nir)

except Exception as exc:
    st.error(f"Input error: {exc}")
    st.stop()

if rgb is None or ndvi is None:
    st.warning("Upload imagery to start landslide prediction.")
    st.stop()

# Ensure NDVI and RGB sizes match.
if ndvi.shape != rgb.shape[:2]:
    ndvi = cv2.resize(ndvi, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

with st.spinner("Running landslide segmentation..."):
    probability_map, mask = predict_landslide(rgb, ndvi, model_type, threshold)
    overlay = create_overlay(rgb, mask, overlay_alpha)
    metrics = calculate_metrics(mask, pixel_resolution_m)

# -----------------------------------------------------------------------------
# Visual layout
# -----------------------------------------------------------------------------
st.subheader("🛰️ Visual Prediction")
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**Original RGB Image**")
    st.image(rgb, use_container_width=True)

with right_col:
    st.markdown("**Predicted Landslide Mask Overlay**")
    st.image(overlay, use_container_width=True)

mask_col, ndvi_col = st.columns(2)
with mask_col:
    st.markdown("**Binary Segmentation Mask**")
    st.image(mask, clamp=True, use_container_width=True)
with ndvi_col:
    st.markdown("**NDVI Visualization**")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

# -----------------------------------------------------------------------------
# Metrics dashboard
# -----------------------------------------------------------------------------
st.subheader("📊 Analytics & Risk Metrics")
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Total Area", f"{metrics.total_area_km2:.4f} km²")
metric_2.metric("Landslide Area", f"{metrics.landslide_area_km2:.4f} km²")
metric_3.metric("Coverage", f"{metrics.landslide_percentage:.2f}%")
metric_4.metric("Severity", metrics.severity)

if metrics.severity == "High Risk":
    st.error("🚨 High Risk: Large landslide-affected region detected.")
elif metrics.severity == "Medium Risk":
    st.warning("⚠️ Medium Risk: Noticeable landslide-affected region detected.")
else:
    st.success("✅ Low Risk: Limited landslide-affected region detected.")

chart_col, table_col = st.columns([1, 1])
with chart_col:
    st.plotly_chart(plot_area_chart(metrics), use_container_width=True)
with table_col:
    st.markdown("**Prediction Summary**")
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Model",
                "Threshold",
                "Pixel Resolution",
                "Total Pixels",
                "Landslide Pixels",
                "Total Area km²",
                "Landslide Area km²",
                "Landslide Coverage %",
                "Severity",
            ],
            "Value": [
                model_type,
                f"{threshold:.2f}",
                f"{metrics.pixel_resolution_m:.2f} m/pixel",
                f"{metrics.total_pixels:,}",
                f"{metrics.landslide_pixels:,}",
                f"{metrics.total_area_km2:.6f}",
                f"{metrics.landslide_area_km2:.6f}",
                f"{metrics.landslide_percentage:.2f}",
                metrics.severity,
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------
st.subheader("⬇️ Export Results")
d1, d2, d3, d4 = st.columns(4)

with d1:
    st.download_button(
        "Download Mask PNG",
        data=create_mask_png(mask),
        file_name="landslide_mask.png",
        mime="image/png",
    )

with d2:
    st.download_button(
        "Download Mask TIFF",
        data=create_geotiff_bytes(mask),
        file_name="landslide_mask.tiff",
        mime="image/tiff",
    )

with d3:
    st.download_button(
        "Download CSV Report",
        data=create_csv_report(metrics, model_type, threshold),
        file_name="landslide_report.csv",
        mime="text/csv",
    )

with d4:
    st.download_button(
        "Download PDF Report",
        data=create_pdf_report(metrics, model_type, threshold),
        file_name="landslide_report.pdf",
        mime="application/pdf",
    )

st.download_button(
    "Download All Outputs as ZIP",
    data=create_download_zip(mask, overlay, metrics, model_type, threshold),
    file_name="landslide_detection_outputs.zip",
    mime="application/zip",
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# Developer notes
# -----------------------------------------------------------------------------
with st.expander("Developer notes: where to add real U-Net / FCN inference"):
    st.code(
        """
# Replace predict_landslide() with something like this:

@st.cache_resource
def load_pytorch_model(model_type):
    import torch
    if model_type == "U-Net":
        model = UNet(in_channels=4, out_channels=1)
        model.load_state_dict(torch.load("models/unet_model.pth", map_location="cpu"))
    else:
        model = FCN(in_channels=4, out_channels=1)
        model.load_state_dict(torch.load("models/fcn_model.pth", map_location="cpu"))
    model.eval()
    return model

def predict_landslide(rgb, ndvi, model_type, threshold):
    import torch
    model_input = resize_for_model(rgb, ndvi, target_size=256)  # H,W,4
    tensor = torch.from_numpy(model_input).permute(2,0,1).unsqueeze(0).float()
    model = load_pytorch_model(model_type)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    prob = cv2.resize(prob, (rgb.shape[1], rgb.shape[0]))
    mask = (prob >= threshold).astype(np.uint8) * 255
    return prob, mask
        """,
        language="python",
    )
