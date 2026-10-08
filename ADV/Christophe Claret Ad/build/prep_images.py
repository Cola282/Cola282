"""Prepare official Claret renders for the dark stage.

Each image: Lanczos upscale (2x) + light unsharp, then an alpha mask that keys
out the black studio background (closed + blurred so dark PVD parts of the watch
stay opaque) and a soft elliptical feather so nothing has a hard edge.
Run with: python3 -I prep_images.py <raw_dir> <out_dir>
"""
import sys, os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

RAW, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# name -> (source, crop box in source px or None, key threshold, feather)
JOBS = {
    "xtrem_front":   ("ablogtowatch/Christophe-Claret-X-TREM-1-StingHD-7.jpg", None, 0.06, 0.16),
    "xtrem_macro":   ("ablogtowatch/Christophe-Claret-X-TREM-1-StingHD-10.jpg", None, 0.05, 0.10),
    "xtrem_angle":   ("ablogtowatch/Christophe-Claret-X-TREM-1-StingHD-5.jpg", None, 0.05, 0.10),
    "xtrem_tourb":   ("ablogtowatch/Christophe-Claret-X-TREM-1-StingHD-8.jpg", None, 0.05, 0.10),
    "margot_full":   ("jewelleryeditor/ChristopheClarotMargot002.jpg", None, 0.10, 0.14),
    "margot_cu":     ("jewelleryeditor/MargotWatchCU.jpg", None, 0.10, 0.10),
    "m_poker":       ("chronopassion/christophe_claret_poker_71b614b5a8.jpg", None, 0.07, 0.14),
    "m_maestro":     ("chronopassion/christophe_claret_maestro_green_manba_43080acfca.jpg", None, 0.07, 0.14),
    "m_maestro2":    ("chronopassion/maestro_face_ccf9d8f5a6.jpg", None, 0.07, 0.14),
    "m_marguerite":  ("chronopassion/marguerite_291941645d.jpg", None, 0.07, 0.14),
    "m_allegro":     ("ablogtowatch/Christophe-Claret-Allegro-4.jpg", None, 0.05, 0.10),
    "m_baccara":     ("monochrome/Christophe-Claret-baccara-2.jpg", None, 0.07, 0.12),
    "movement":      ("chronopassion/christopheclaret_angelico_mtr_dtc08_0020_030_mvt_a4_72dpi_rvb_9ccd8e5274.jpg", None, 0.05, 0.12),
}


BLACK = {"margot_full": 0.16, "margot_cu": 0.06, "xtrem_front": 0.06, "m_poker": 0.08, "m_maestro": 0.06, "m_maestro2": 0.06, "m_marguerite": 0.08}


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


for name, (src, crop, thr, feather) in JOBS.items():
    im = Image.open(os.path.join(RAW, src)).convert("RGB")
    if crop:
        im = im.crop(crop)
    w, h = im.size
    im = im.resize((w * 2, h * 2), Image.LANCZOS)
    im = im.filter(ImageFilter.UnsharpMask(radius=2.2, percent=70, threshold=2))
    a = np.asarray(im).astype(np.float32) / 255
    bp_ = BLACK.get(name, 0.03)  # crush the studio-grey halo into the stage black
    a = np.clip((a - bp_) / (1 - bp_), 0, 1)
    lum = a.max(axis=2)
    key = (lum > thr).astype(np.uint8) * 255
    # close holes so dark parts inside the watch stay solid
    k = max(9, (min(key.shape) // 60) | 1)
    km = ndimage.maximum_filter(key, size=k)
    km = ndimage.minimum_filter(km, size=k)
    km = ndimage.binary_fill_holes(km > 0).astype(np.float32)
    m = ndimage.gaussian_filter(km, min(key.shape) / 140)
    soft = smoothstep(thr * 0.4, thr * 1.6, lum)
    m = np.maximum(m * 0.92, soft) * np.clip(m * 1.4, 0, 1)
    # elliptical feather to the frame edge
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx - W / 2) / (W / 2)
    dy = (yy - H / 2) / (H / 2)
    r = np.sqrt(dx ** 2 + dy ** 2)
    edge = 1 - smoothstep(1 - feather * 2.2, 1.0 + feather * 0.2, r)
    edge *= smoothstep(0, feather, np.minimum.reduce([xx / W, (W - xx) / W, yy / H, (H - yy) / H]))
    alpha = np.clip(m * edge, 0, 1)
    rgba = np.dstack([a, alpha])
    Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(os.path.join(OUT, name + ".png"), compress_level=3)
    print(name, im.size)
