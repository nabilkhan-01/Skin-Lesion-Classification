"""
Skin Lesion Classification — Streamlit Application

Clinical & Research AI Demonstration Interface for the fine-tuned EfficientNetB0
dermoscopic image classifier with Grad-CAM explainability.
Pure clean medical white design.
"""

import sys
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_model, load_image, predict, CLASS_NAMES, IMAGE_SIZE
from src.gradcam_utils import generate_gradcam, resize_heatmap, get_top_predictions

# ---------------------------------------------------------------------------
# Constants & Clinical Mappings
# ---------------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_finetuned_best.keras"

CLASS_LABELS = {
    "akiec": "Solar or actinic keratosis",
    "bcc": "Basal cell carcinoma",
    "bkl": "Pigmented benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma, NOS",
    "nv": "Nevus",
    "scc": "Squamous cell carcinoma, NOS",
    "vasc": "Vascular lesion",
}

CLASS_TYPES = {
    "akiec": ("Pre-malignant / Keratinocytic", "warning"),
    "bcc": ("Malignant / Non-Melanoma", "danger"),
    "bkl": ("Benign Keratinocytic", "success"),
    "df": ("Benign Fibrous Histiocytoma", "success"),
    "mel": ("Malignant Melanocytic", "danger"),
    "nv": ("Benign Melanocytic", "success"),
    "scc": ("Malignant / Non-Melanoma", "danger"),
    "vasc": ("Benign Vascular Proliferation", "success"),
}

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
from streamlit.runtime import exists
if exists():
    st.set_page_config(
        page_title="Skin Lesion Classification",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

# ---------------------------------------------------------------------------
# Helper: render unindented HTML to prevent Markdown code block parsing
# ---------------------------------------------------------------------------
def render_html(html_str: str):
    """Strip leading spaces from every line to guarantee markdown never creates a <pre><code> block."""
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CSS Styling — Pure Clinical White Medical Theme
# ---------------------------------------------------------------------------
render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&family=Noto+Sans:wght@400;500;600;700&display=swap');

    /* Global White Clinical Canvas */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    .block-container {
        max-width: 1120px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    /* All base typography defaults to deep readable slate */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #0F172A !important;
    }

    /* Preserve native icon font ligatures */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* File Uploader Dropzone: Clean White Card */
    [data-testid="stFileUploaderDropzone"],
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #0F172A !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background-color: #E2E8F0 !important;
    }
    [data-testid="stFileUploaderDropzone"] small {
        color: #64748B !important;
    }

    /* Hero Panel */
    .med-hero {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .clinical-brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #ECFEFF !important;
        color: #0891B2 !important;
        border: 1px solid #A5F3FC !important;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        margin-bottom: 0.6rem;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #059669;
        border-radius: 50%;
        display: inline-block;
    }
    .med-title {
        font-family: 'Figtree', sans-serif !important;
        font-size: 2.15rem;
        font-weight: 800;
        color: #0F172A !important;
        margin: 0 0 0.35rem 0;
        line-height: 1.15;
    }
    .med-subtitle {
        font-family: 'Figtree', sans-serif !important;
        font-size: 1.05rem;
        font-weight: 600;
        color: #0891B2 !important;
        margin: 0 0 0.75rem 0;
    }
    .med-desc {
        font-family: 'Noto Sans', sans-serif !important;
        font-size: 0.95rem;
        color: #475569 !important;
        line-height: 1.6;
        margin: 0;
    }

    /* Section Headers */
    .med-section-header {
        font-family: 'Figtree', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A !important;
        margin-top: 1.8rem;
        margin-bottom: 0.85rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #E2E8F0 !important;
    }

    /* Primary Prediction Card */
    .pred-card-box {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }
    .pred-card-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0891B2 0%, #06B6D4 100%);
    }
    .pred-kicker {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B !important;
        margin-bottom: 0.25rem;
    }
    .pred-name {
        font-family: 'Figtree', sans-serif !important;
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A !important;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .pred-tags {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .code-pill {
        background: #F1F5F9 !important;
        color: #0F172A !important;
        font-family: ui-monospace, monospace;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
    }
    .type-pill {
        font-size: 0.74rem;
        font-weight: 600;
        padding: 0.18rem 0.6rem;
        border-radius: 9999px;
    }
    .type-pill.danger {
        background: #FEF2F2 !important;
        color: #DC2626 !important;
        border: 1px solid #FECACA !important;
    }
    .type-pill.warning {
        background: #FFFBEB !important;
        color: #D97706 !important;
        border: 1px solid #FDE68A !important;
    }
    .type-pill.success {
        background: #ECFDF5 !important;
        color: #059669 !important;
        border: 1px solid #A7F3D0 !important;
    }
    .conf-metric-row {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px;
        padding: 0.85rem 1.2rem;
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
    }
    .conf-number {
        font-family: 'Figtree', sans-serif !important;
        font-size: 2.1rem;
        font-weight: 800;
        color: #0891B2 !important;
        line-height: 1;
    }
    .conf-caption {
        font-size: 0.82rem;
        color: #64748B !important;
        font-weight: 500;
    }

    /* Top 3 Predictions */
    .top3-container {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .top3-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 0.9rem;
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px;
    }
    .top3-rank-pill {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: #0891B2;
        color: #FFFFFF;
        font-size: 0.74rem;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 0.5rem;
    }
    .top3-rank-pill.rank-2 { background: #64748B; }
    .top3-rank-pill.rank-3 { background: #94A3B8; }
    .top3-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #1E293B !important;
    }
    .top3-code {
        font-size: 0.76rem;
        color: #64748B !important;
        font-family: ui-monospace, monospace;
    }
    .top3-pct {
        font-family: 'Figtree', sans-serif !important;
        font-size: 0.96rem;
        font-weight: 700;
        color: #0F172A !important;
    }

    /* Class Probability Bars */
    .prob-track-box {
        margin-bottom: 0.65rem;
    }
    .prob-label-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.83rem;
        margin-bottom: 0.25rem;
    }
    .prob-text {
        font-weight: 600;
        color: #334155 !important;
    }
    .prob-val-text {
        font-family: 'Figtree', sans-serif !important;
        font-weight: 700;
        color: #0F172A !important;
    }
    .prob-bar-rail {
        height: 8px;
        width: 100%;
        background: #F1F5F9 !important;
        border-radius: 9999px;
        overflow: hidden;
    }
    .prob-bar-progress {
        height: 100%;
        border-radius: 9999px;
        background: #CBD5E1;
        transition: width 0.3s ease;
    }
    .prob-bar-progress.highlight {
        background: linear-gradient(90deg, #0891B2 0%, #06B6D4 100%);
    }

    /* Explanation Callout */
    .med-explanation {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 4px solid #0891B2 !important;
        border-radius: 10px;
        padding: 1.15rem 1.4rem;
        margin-top: 0.75rem;
        margin-bottom: 1.25rem;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #334155 !important;
    }
    .med-explanation p {
        margin: 0 0 0.5rem 0;
        color: #334155 !important;
    }
    .med-explanation strong {
        color: #0F172A !important;
    }

    /* Medical Disclaimer */
    .med-disclaimer {
        background: #FFFBEB !important;
        border: 1px solid #FDE68A !important;
        border-radius: 12px;
        padding: 1.1rem 1.35rem;
        margin-top: 2.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.85rem;
    }
    .disclaimer-icon {
        color: #D97706 !important;
        font-size: 1.3rem;
        line-height: 1;
        margin-top: 2px;
    }
    .disclaimer-header {
        font-family: 'Figtree', sans-serif !important;
        font-size: 0.92rem;
        font-weight: 700;
        color: #92400E !important;
        margin-bottom: 0.2rem;
    }
    .disclaimer-body {
        font-size: 0.85rem;
        color: #B45309 !important;
        line-height: 1.5;
        margin: 0;
    }

    /* Native Expanders */
    [data-testid="stExpander"], details {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary, details summary {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] summary svg, details summary svg {
        fill: #0F172A !important;
    }
    table, th, td {
        color: #0F172A !important;
        border-color: #E2E8F0 !important;
    }
    </style>
    """
)


# ---------------------------------------------------------------------------
# Cached Model Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading classification model (EfficientNetB0)...")
def cached_load_model():
    """Load the final trained model once with Streamlit resource caching."""
    if not MODEL_PATH.exists():
        st.error(f"Target model file not found at: {MODEL_PATH}")
        st.stop()
    return load_model()


# ---------------------------------------------------------------------------
# Grad-CAM Visualization Helper
# ---------------------------------------------------------------------------
def build_gradcam_images(model, image_tensor, class_index=None):
    """
    Generate Grad-CAM heatmap and overlay using Matplotlib colormaps.
    Returns: (heatmap_rgb, overlay_rgb, original_np)
    """
    heatmap_raw, _, _ = generate_gradcam(model, image_tensor, class_index=class_index)
    heatmap_resized = resize_heatmap(heatmap_raw, IMAGE_SIZE)

    colormap = matplotlib.colormaps["jet"]
    heatmap_coloured = colormap(heatmap_resized)[:, :, :3]
    heatmap_rgb = (heatmap_coloured * 255).astype(np.uint8)

    original_np = np.clip(image_tensor.numpy(), 0, 255).astype(np.uint8)

    alpha = 0.40
    overlay = np.clip(
        original_np.astype(np.float32) * (1.0 - alpha) + heatmap_rgb.astype(np.float32) * alpha,
        0,
        255
    ).astype(np.uint8)

    return heatmap_rgb, overlay, original_np


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    # Hero Title Panel
    render_html(
        """
        <div class="med-hero">
            <div class="clinical-brand-badge">
                <span class="pulse-dot"></span>
                Dermatology AI Research · Fine-Tuned EfficientNetB0
            </div>
            <h1 class="med-title">Skin Lesion Classification</h1>
            <div class="med-subtitle">Deep Learning-Based Dermoscopic Image Classification with Explainable AI</div>
            <p class="med-desc">
                Upload a dermoscopic image to obtain a predicted lesion category, class probabilities,
                and a Grad-CAM visualization showing the image regions that contributed to the model prediction.
            </p>
        </div>
        """
    )

    # Load Model
    model = cached_load_model()

    # Image Upload Section
    render_html('<div class="med-section-header">Image Upload</div>')

    uploaded_file = st.file_uploader(
        "Upload a dermoscopic image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Accepted formats: JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("Please upload a dermoscopic image to view the automated classification.")
        render_model_info()
        render_dataset_info()
        render_disclaimer()
        return

    # Image decoding
    try:
        image_bytes = uploaded_file.getvalue()
        image_tensor, _ = load_image(image_bytes)
    except Exception as exc:
        st.error(f"Failed to decode image file. Please verify it is a valid JPG, JPEG, or PNG.\n\nError: {exc}")
        return

    # Model inference
    try:
        pred_index, pred_class, confidence, probabilities = predict(model, image_tensor)
    except Exception as exc:
        st.error(f"Inference error during model prediction:\n\n{exc}")
        return

    human_label = CLASS_LABELS.get(pred_class, pred_class)
    class_category, risk_tag = CLASS_TYPES.get(pred_class, ("Lesion", "warning"))
    top3 = get_top_predictions(probabilities, top_k=3)

    # -----------------------------------------------------------------------
    # Prediction Section & Uploaded Image
    # -----------------------------------------------------------------------
    render_html('<div class="med-section-header">Prediction & Source Image</div>')

    img_col, pred_col = st.columns([1.0, 1.25], gap="large")

    with img_col:
        st.markdown("**Uploaded Dermoscopic Image**")
        st.image(image_bytes, width="stretch")
        st.caption(f"Uploaded file: {uploaded_file.name}")

    with pred_col:
        render_html(
            f"""
            <div class="pred-card-box">
                <div class="pred-kicker">Predicted Class</div>
                <div class="pred-name">{human_label}</div>
                <div class="pred-tags">
                    <span class="code-pill">({pred_class})</span>
                    <span class="type-pill {risk_tag}">{class_category}</span>
                </div>
                <div class="conf-metric-row">
                    <div>
                        <div class="pred-kicker">Confidence</div>
                        <div class="conf-number">{confidence * 100:.1f}%</div>
                    </div>
                    <div class="conf-caption">
                        {"Dominant prediction probability" if confidence >= 0.70 else "Distributed probability across competing classes"}
                    </div>
                </div>
            </div>
            """
        )

        # Top 3 Predictions
        st.markdown("**Top 3 Predictions**")
        top3_html = '<div class="top3-container">'
        for rank, item in enumerate(top3, 1):
            rank_class = f"rank-{rank}" if rank in [2, 3] else ""
            c_name = CLASS_LABELS.get(item["class"], item["class"])
            c_pct = item["confidence"] * 100
            top3_html += f"""
            <div class="top3-row">
                <div style="display:flex; align-items:center;">
                    <span class="top3-rank-pill {rank_class}">{rank}</span>
                    <div>
                        <span class="top3-title">{c_name}</span>
                        <span class="top3-code">({item['class']})</span>
                    </div>
                </div>
                <span class="top3-pct">{c_pct:.1f}%</span>
            </div>
            """
        top3_html += "</div>"
        render_html(top3_html)

    # -----------------------------------------------------------------------
    # Class Probabilities (All 8 Classes)
    # -----------------------------------------------------------------------
    render_html('<div class="med-section-header">Class Probabilities</div>')

    sorted_indices = np.argsort(probabilities)[::-1]
    max_prob = float(probabilities[sorted_indices[0]]) if len(sorted_indices) > 0 else 1.0

    p_col1, p_col2 = st.columns(2, gap="medium")
    for i, idx in enumerate(sorted_indices):
        cls_id = CLASS_NAMES[idx]
        cls_title = CLASS_LABELS.get(cls_id, cls_id)
        p_val = float(probabilities[idx])
        p_pct = p_val * 100
        bar_w = (p_val / max_prob * 100) if max_prob > 0 else 0
        is_winner = (i == 0)
        h_class = "highlight" if is_winner else ""

        target_col = p_col1 if (i % 2 == 0) else p_col2
        with target_col:
            render_html(
                f"""
                <div class="prob-track-box">
                    <div class="prob-label-row">
                        <span class="prob-text">{'★ ' if is_winner else ''}{cls_title} <span style="color:#64748B; font-weight:400;">({cls_id})</span></span>
                        <span class="prob-val-text">{p_pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-rail">
                        <div class="prob-bar-progress {h_class}" style="width:{bar_w:.1f}%;"></div>
                    </div>
                </div>
                """
            )

    # -----------------------------------------------------------------------
    # Model Explainability — Grad-CAM
    # -----------------------------------------------------------------------
    render_html('<div class="med-section-header">Model Explainability — Grad-CAM</div>')

    try:
        heatmap_rgb, overlay_rgb, original_np = build_gradcam_images(
            model, image_tensor, class_index=pred_index
        )

        g1, g2, g3 = st.columns(3, gap="medium")
        with g1:
            st.markdown("**1. Original Image**")
            st.image(original_np, use_container_width=True)
        with g2:
            st.markdown("**2. Grad-CAM Heatmap**")
            st.image(heatmap_rgb, use_container_width=True)
        with g3:
            st.markdown("**3. Grad-CAM Overlay**")
            st.image(overlay_rgb, use_container_width=True)

        st.caption(
            "Grad-CAM highlights image regions with greater attribution to the selected model prediction. "
            "The visualization describes model attention and does not provide a clinical explanation or diagnosis."
        )
    except Exception as exc:
        st.warning(f"Grad-CAM visualization could not be generated:\n\n{exc}")

    # -----------------------------------------------------------------------
    # Explanation Section
    # -----------------------------------------------------------------------
    render_html('<div class="med-section-header">Why did the model make this prediction?</div>')
    render_careful_explanation(top3, confidence, human_label, pred_class)

    # -----------------------------------------------------------------------
    # About the Model & Dataset (Expandable)
    # -----------------------------------------------------------------------
    render_model_info()
    render_dataset_info()

    # -----------------------------------------------------------------------
    # Medical Disclaimer
    # -----------------------------------------------------------------------
    render_disclaimer()


# ---------------------------------------------------------------------------
# Scientific Explanation Renderer
# ---------------------------------------------------------------------------
def render_careful_explanation(top3, confidence, human_label, pred_class):
    """Generate objective, non-diagnostic explanation based strictly on model outputs."""
    p1 = (
        f"The model assigned the highest probability to <strong>{human_label}</strong> ({pred_class}) "
        f"with a confidence of <strong>{confidence * 100:.1f}%</strong>."
    )

    p2 = ""
    if len(top3) >= 2:
        second = top3[1]
        second_name = CLASS_LABELS.get(second["class"], second["class"])
        diff = confidence - second["confidence"]

        if diff < 0.10:
            p2 = (
                f"The model was relatively uncertain because the probabilities of the top classes were close. "
                f"The second most likely class, <strong>{second_name}</strong> ({second['class']}), "
                f"received <strong>{second['confidence'] * 100:.1f}%</strong>."
            )
        else:
            p2 = (
                f"The model showed relatively high confidence in this prediction. "
                f"The prediction probability was higher than the next most likely class, "
                f"<strong>{second_name}</strong> ({second['class']}) at <strong>{second['confidence'] * 100:.1f}%</strong>."
            )

    p3 = (
        "Grad-CAM highlights image regions with greater attribution to the selected model prediction. "
        "The visualization describes model attention and does not provide a clinical explanation or diagnosis."
    )

    render_html(
        f"""
        <div class="med-explanation">
            <p>{p1}</p>
            {f'<p>{p2}</p>' if p2 else ''}
            <p>{p3}</p>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Expandable: About the Model
# ---------------------------------------------------------------------------
def render_model_info():
    with st.expander("About the Model"):
        st.markdown(
            """
| Model Property | Specification |
|:---|:---|
| **Model Architecture** | Fine-tuned EfficientNetB0 |
| **Input Dimensions** | 224 × 224 pixels (RGB) |
| **Number of Classes** | 8 categories |
| **Training Approach** | ImageNet transfer learning + fine-tuning |
| **Final Test Accuracy** | 75.34% |
| **Weighted Precision** | 77.66% |
| **Weighted Recall** | 75.34% |
| **Weighted F1** | 76.25% |
| **Macro F1** | 52.70% |

**Per-Class Test F1 Breakdown:**
- `akiec` (Solar / Actinic Keratosis) = **0.37**
- `bcc` (Basal Cell Carcinoma) = **0.57**
- `bkl` (Pigmented Benign Keratosis) = **0.56**
- `df` (Dermatofibroma) = **0.42**
- `mel` (Melanoma, NOS) = **0.55**
- `nv` (Nevus) = **0.88**
- `scc` (Squamous Cell Carcinoma, NOS) = **0.21**
- `vasc` (Vascular Lesion) = **0.65**

The model was fine-tuned on the project's dermoscopic image dataset using pre-trained ImageNet feature weights.
The classification head and top convolutional blocks were trained for target dermatological categorization.
This model has not undergone clinical validation.
            """
        )


# ---------------------------------------------------------------------------
# Expandable: About the Dataset
# ---------------------------------------------------------------------------
def render_dataset_info():
    with st.expander("About the Dataset"):
        st.markdown(
            """
- **11,720 dermoscopic images** utilized in this research project
- **8 classification categories:** Solar or actinic keratosis (`akiec`), Basal cell carcinoma (`bcc`), Pigmented benign keratosis (`bkl`), Dermatofibroma (`df`), Melanoma, NOS (`mel`), Nevus (`nv`), Squamous cell carcinoma, NOS (`scc`), and Vascular lesion (`vasc`)
- **Lesion-level stratified train/validation/test splitting** applied to ensure patient-independent evaluation
- **Data augmentation** applied during training (rotation, flip, zoom, contrast variation)
- **Class imbalance mitigation** handled using moderated class weighting during loss optimization

The dataset is retained for academic research purposes and is not distributed through this application.
            """
        )


# ---------------------------------------------------------------------------
# Medical Regulatory Disclaimer
# ---------------------------------------------------------------------------
def render_disclaimer():
    render_html(
        """
        <div class="med-disclaimer">
            <div class="disclaimer-icon">⚠</div>
            <div>
                <div class="disclaimer-header">Research & Educational Use Only</div>
                <p class="disclaimer-body">
                    Research/Educational Use Only — This system is not a medical diagnostic tool and should not be used for clinical decision-making.
                </p>
            </div>
        </div>
        """
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from streamlit.runtime import exists
    if not exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(stcli.main())
    else:
        main()
