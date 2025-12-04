import rasterio
import numpy as np
import matplotlib.pyplot as plt

def show_rgb(path, title):
    # MultiSenGE S2 band order: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
    with rasterio.open(path) as src:
        b2 = src.read(1)   # B02
        b3 = src.read(2)   # B03
        b4 = src.read(3)   # B04
    rgb = np.stack([b4, b3, b2], axis=-1).astype(float)

    # simple contrast stretch (2–98th percentiles)
    lo, hi = np.percentile(rgb[~np.isnan(rgb)], (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)

    plt.figure(figsize=(5,5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(title)
    plt.show()

show_rgb("data/raw/MultiSenGE/s2/31TFN_20200731_S2_4626_514.tif", "Pre (2020‑07‑31)")
show_rgb("data/raw/MultiSenGE/s2/31TFN_20201128_S2_4626_514.tif", "Post (2020‑11‑28)")