"""Image-enhancement layer for OCR on hard scans (low-contrast grayscale,
shaded columns, small dense text). Targets the failure modes seen on real
scanned financial tables: gray-shaded cells misread, faint/small digits, skew.

Composable steps (all operate on a grayscale plane, return a 3-channel PIL so
downstream TATR/OCR get an RGB image):
  deskew       - straighten small rotation (minAreaRect on foreground)
  denoise      - fastNlMeans, removes scan speckle
  clahe        - adaptive histogram equalization, lifts faint text
  adaptive_thresh - binarize per-region, kills background shading (aggressive:
                    great for OCR, can hurt TATR detection -> off by default)
  upscale      - cubic upscale, helps small dense digits

Note: enhancement improves OCR *reading* accuracy; it does NOT fix structural
issues (column detection, merged cells, wrapped headers).
"""
import cv2
import numpy as np
from PIL import Image


def _to_gray(pil):
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2GRAY)


def _to_pil(gray):
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def deskew(gray, max_angle=15.0, min_angle=0.3):
    """Estimate and correct small page skew from the foreground pixel cloud.
    Ignores tiny (<min_angle) or implausible (>max_angle) estimates."""
    thr = cv2.threshold(cv2.bitwise_not(gray), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 50:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < min_angle or abs(angle) > max_angle:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def clahe(gray, clip=2.0, grid=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)


def denoise(gray, h=7):
    return cv2.fastNlMeansDenoising(gray, None, h, 7, 21)


def adaptive_thresh(gray, block=31, c=10):
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, c)


def upscale(gray, factor=2.0):
    if not factor or factor == 1.0:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def enhance(pil, *, do_deskew=True, do_denoise=True, do_clahe=True,
            do_thresh=False, scale=2.0):
    """Run the enhancement pipeline on a PIL image, return an enhanced PIL image.
    Conservative default (deskew + denoise + CLAHE + 2x upscale, no binarization)
    helps OCR while staying safe for TATR detection. Turn on do_thresh for a
    hard binarization when background shading is the problem."""
    gray = _to_gray(pil)
    if do_deskew:
        gray = deskew(gray)
    if do_denoise:
        gray = denoise(gray)
    if do_clahe:
        gray = clahe(gray)
    if do_thresh:
        gray = adaptive_thresh(gray)
    gray = upscale(gray, scale)
    return _to_pil(gray)
