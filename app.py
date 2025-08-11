import io, os, math, pathlib, urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import streamlit as st

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
# YuNet model helper (no mediapipe)
# =========================
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_DIR = pathlib.Path("./models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"

def ensure_yunet_model() -> str:
    if not YUNET_PATH.exists():
        # download once at startup
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH.as_posix())
    return YUNET_PATH.as_posix()

def create_yunet_detector(img_w: int, img_h: int):
    """Create YuNet detector for given input size."""
    model_path = ensure_yunet_model()
    # OpenCV changed API name across versions: try new, then old
    if hasattr(cv2, "FaceDetectorYN") and hasattr(cv2.FaceDetectorYN, "create"):
        det = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(img_w, img_h),
            score_threshold=0.7,
            nms_threshold=0.3,
            top_k=5000
        )
    elif hasattr(cv2, "FaceDetectorYN_create"):
        det = cv2.FaceDetectorYN_create(
            model_path, "", (img_w, img_h), 0.7, 0.3, 5000
        )
    else:
        raise RuntimeError("This OpenCV build lacks FaceDetectorYN (YuNet).")
    return det


# =========================
# Auto crop to 3:4 by face (YuNet)
# =========================
@dataclass
class CropParams:
    x: int; y: int; w: int; h: int

def find_face_crop(pil_img: Image.Image,
                   out_size: Tuple[int, int],
                   head_ratio: float = 0.64,
                   top_headroom: float = 0.20) -> CropParams:
    """
    Use YuNet face detector to compute a crop such that
    face height ~= head_ratio * out_height, centered with
    a bit of space above the head.
    Fallback: center 3:4 crop.
    """
    img_rgb = pil_to_cv_rgb(pil_img)
    h, w, _ = img_rgb.shape
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
        detector = create_yunet_detector(w, h)
        # For some OpenCV builds, must call setInputSize before detect:
        if hasattr(detector, "setInputSize"):
            detector.setInputSize((w, h))
        _, faces = detector.detect(img_bgr)
    except Exception:
        faces = None

    out_w, out_h = out_size

    if faces is not None and len(faces) > 0:
        # faces: Nx15 (x, y, w, h, landmarks..., score)
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, fw, fh = map(int, faces[0][:4])

        # scale so face height -> head_ratio * out_h
        wanted_head_px = head_ratio * out_h
        scale = wanted_head_px / float(max(fh, 1))
        crop_w = int(round(out_w / scale))
        crop_h = int(round(out_h / scale))

        fcx = x + fw // 2
        cy = int(round(y - top_headroom * crop_h))
        cx = int(round(fcx - crop_w / 2))

        cx = clamp(cx, 0, max(0, w - crop_w))
        cy = clamp(cy, 0, max(0, h - crop_h))
        crop_w = clamp(crop_w, 1, w)
        crop_h = clamp(crop_h, 1, h)
        return CropParams(cx, cy, crop_w, crop_h)
    else:
        # center 3:4 crop
        want_w = min(w, int(h * 3 / 4))
        want_h = int(round(want_w * 4 / 3))
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
    rgba_bytes = rembg_remove(pil_img_3x4.convert("RGBA"))
    rgba = Image.open(io.BytesIO(rgba_bytes)).convert("RGBA")

    # feather alpha
    if feather_px > 0:
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
        acc += hist[i]
        if acc > clip:
            low = i; break
    acc = 0; high = 255
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc > clip:
            high = i; break
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

    if cam
