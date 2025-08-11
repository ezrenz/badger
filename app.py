import io, os, math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import cv2
import streamlit as st

# ---- Face detection (MediaPipe) ----
import mediapipe as mp
mp_fd = mp.solutions.face_detection

# ---- Background removal (rembg) ----
from rembg import remove as rembg_remove


# =========================
# Utility helpers
# =========================
def pil_to_cv_rgb(img: Image.Image) -> np.ndarray:
    """PIL RGB -> numpy RGB"""
    return np.array(img)

def cv_to_pil(img_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(img_rgb)

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def clamp(v, lo, hi): return max(lo, min(hi, v))


# =========================
# Auto crop to 3:4 by face
# =========================
@dataclass
class CropParams:
    x: int; y: int; w: int; h: int

def find_face_crop(pil_img: Image.Image,
                   out_size: Tuple[int, int],
                   head_ratio: float = 0.64,
                   top_headroom: float = 0.20) -> CropParams:
    """
    Use MediaPipe face detection to compute a crop such that
    face height ~= head_ratio * out_height, centered with
    a bit of space above the head.
    Fallback: center 3:4 crop.
    """
    img = pil_to_cv_rgb(pil_img)
    h, w, _ = img.shape

    # Detect face
    with mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        results = fd.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    det = None
    if results.detections:
        # choose the highest score box
        det = max(results.detections, key=lambda d: d.score[0])
        bb = det.location_data.relative_bounding_box
        fx1 = int(bb.xmin * w)
        fy1 = int(bb.ymin * h)
        fw  = int(bb.width * w)
        fh  = int(bb.height * h)
        fx2 = fx1 + fw
        fy2 = fy1 + fh
    else:
        fx1 = fy1 = fw = fh = None

    out_w, out_h = out_size

    if det and fh > 0:
        # scale so face height -> head_ratio * out_h
        wanted_head_px = head_ratio * out_h
        scale = wanted_head_px / float(fh)
        crop_w = int(round(out_w / scale))
        crop_h = int(round(out_h / scale))

        fcx = fx1 + fw // 2
        cy = int(round(fy1 - top_headroom * crop_h))
        cx = int(round(fcx - crop_w / 2))

        cx = clamp(cx, 0, w - crop_w)
        cy = clamp(cy, 0, h - crop_h)
        crop_w = clamp(crop_w, 1, w)
        crop_h = clamp(crop_h, 1, h)
        return CropParams(cx, cy, crop_w, crop_h)
    else:
        # center 3:4 crop
        want_w = min(w, int(h * 3 / 4))
        want_h = int(want_w * 4 / 3)
        cx = (w - want_w) // 2
        cy = (h - want_h) // 2
        return CropParams(cx, cy, want_w, want_h)


# =========================
# Background removal + compose
# =========================
def remove_bg_and_compose(pil_img_3x4: Image.Image,
                          bg_hex: str = "#58B4FF",
                          feather_px: int = 8) -> Image.Image:
    """
    Run rembg on the 3:4 image, feather the alpha, and
    composite onto a solid background color.
    """
    # rembg expects bytes
    rgba_bytes = rembg_remove(pil_img_3x4.convert("RGBA"))
    rgba = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")

    # feather alpha
    if feather_px > 0:
        # PIL doesn't have Gaussian blur just for alpha; split and blur alpha channel
        r, g, b, a = rgba.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=feather_px))
        rgba = Image.merge("RGBA", (r, g, b, a))

    bg = Image.new("RGBA", rgba.size, hex_to_rgb(bg_hex) + (255,))
    out = Image.alpha_composite(bg, rgba)
    return out.convert("RGB")


# =========================
# Enhancement (levels + unsharp)
# =========================
def auto_levels_contrast(img_rgb: np.ndarray) -> np.ndarray:
    """Clip 1% tails, slight gamma, +10% contrast."""
    p = img_rgb.astype(np.float32)
    lum = (0.2126*p[...,0] + 0.7152*p[...,1] + 0.0722*p[...,2]).astype(np.uint8)
    hist = np.bincount(lum.flatten(), minlength=256)
    n = lum.size
    clip = int(n * 0.01)
    acc = 0; low = 0
    for i in range(256):
        acc += hist[i]; 
        if acc > clip: low = i; break
    acc = 0; high = 255
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc > clip: high = i; break
    if high <= low: low, high = 0, 255
    scale = 255.0 / (high - low)

    gamma = 0.98
    contrast = 1.10

    for c in range(3):
        chan = (p[...,c] - low) * scale
        chan = np.clip(chan, 0, 255)
        chan = 255.0 * np.power(chan/255.0, gamma)
        chan = (chan - 128.0) * contrast + 128.0
        p[...,c] = np.clip(chan, 0, 255)

    return p.astype(np.uint8)

def unsharp_mask(img_rgb: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    blur = cv2.GaussianBlur(img_rgb, (0,0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_DEFAULT)
    sharp = cv2.addWeighted(img_rgb, 1 + amount, blur, -amount, 0)
    return sharp

def enhance(pil_img: Image.Image, do_enhance: bool = True) -> Image.Image:
    if not do_enhance:
        return pil_img
    arr = pil_to_cv_rgb(pil_img)
    arr = auto_levels_contrast(arr)
    arr = unsharp_mask(arr, amount=0.5, sigma=1.0)
    return cv_to_pil(arr)


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Badge Photo — Python", page_icon="📷", layout="wide")
st.title("📷 Badge Photo — Capture ➜ Background ➜ Enhance (Python)")

left, right = st.columns(2, gap="large")

# ---- LEFT: Step 1 - Capture/Upload ----
with left:
    st.subheader("Step 1: Capture / Upload")
    cam = st.camera_input("Use camera (or upload below)", key="cam", help="Grant permission if prompted.")
    uploaded = st.file_uploader("Or upload a photo", type=["jpg","jpeg","png"], key="uploader")

    colA, colB, colC = st.columns(3)
    out_size_label = colA.selectbox("Output size (3:4)", ["600×800","900×1200","1200×1600"], index=1)
    out_w, out_h = map(int, out_size_label.replace("×","x").split("x"))
    head_ratio = 0.64  # fixed, common passport-style
    do_enhance = colB.checkbox("Auto-enhance", True)
    mirror_preview = colC.checkbox("Mirror preview (camera)", True)

    working_img: Optional[Image.Image] = None
    src_label = None

    if cam is not None:
        # Streamlit camera returns a file-like object
        working_img = Image.open(cam).convert("RGB")
        if mirror_preview:
            working_img = ImageOps.mirror(working_img)
        src_label = "Camera"
    elif uploaded is not None:
        working_img = Image.open(uploaded).convert("RGB")
        src_label = "Upload"

    if working_img is not None:
        st.image(working_img, caption=f"{src_label} preview", use_column_width=True)
        if st.button("➡️ Capture ➜ Send to Work Area", use_container_width=True):
            st.session_state["captured"] = working_img
            st.success("Captured and sent to the right panel!")


# ---- RIGHT: Step 2 - Work Area ----
with right:
    st.subheader("Step 2: Work Area — Background & Color")
    if "captured" not in st.session_state:
        st.info("Capture or upload on the left, then click **Capture ➜ Send to Work Area**.")
    else:
        src: Image.Image = st.session_state["captured"].copy()

        # Auto crop to 3:4 by face; fallback center crop
        crop = find_face_crop(src, (out_w, out_h), head_ratio=head_ratio)
        src_np = pil_to_cv_rgb(src)
        crop_np = src_np[crop.y:crop.y+crop.h, crop.x:crop.x+crop.w, :]
        crop_pil = cv_to_pil(crop_np).resize((out_w, out_h), Image.LANCZOS)

        col1, col2 = st.columns([1,1])
        with col1:
            st.caption("Cropped 3:4 (auto)")
            st.image(crop_pil, use_column_width=True)

        # Controls
        bg_color = st.color_picker("Background color", "#58B4FF")
        feather = st.slider("Edge feather (px)", 0, 30, 10)
        do_remove = st.checkbox("Remove background", True)

        final_img = crop_pil
        if do_remove:
            # rembg expects the image size you're composing at (already out size)
            # Use a small feather in PIL after rembg
            from PIL import ImageFilter
            rgba_bytes = rembg_remove(crop_pil.convert("RGBA"))
            rgba = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")
            if feather > 0:
                r,g,b,a = rgba.split()
                a = a.filter(ImageFilter.GaussianBlur(radius=feather))
                rgba = Image.merge("RGBA", (r,g,b,a))
            bg = Image.new("RGBA", rgba.size, hex_to_rgb(bg_color)+(255,))
            final_img = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            # Just put the cropped image onto a solid background
            bg = Image.new("RGB", (out_w, out_h), hex_to_rgb(bg_color))
            final_img = Image.composite(crop_pil, bg, Image.new("L", (out_w, out_h), 255))

        # Enhance
        final_img = enhance(final_img, do_enhance=do_enhance)

        with col2:
            st.caption("Work result")
            st.image(final_img, use_column_width=True)

        # Download
        fmt = st.selectbox("Download format", ["JPEG","PNG"], index=0)
        buf = io.BytesIO()
        if fmt == "PNG":
            final_img.save(buf, format="PNG")
            mime = "image/png"
            ext = "png"
        else:
            final_img.save(buf, format="JPEG", quality=95, subsampling=0)
            mime = "image/jpeg"
            ext = "jpg"
        st.download_button(
            label="⬇️ Download final",
            data=buf.getvalue(),
            file_name=f"badge_{out_w}x{out_h}.{ext}",
            mime=mime,
            use_container_width=True
        )
