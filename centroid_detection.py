import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
import re

IMAGE_FOLDER = Path("/Users/bananyamani/Desktop/PSI/50ms_cropped_clean")
OUTPUT_DIR = Path("figures")

N_IMAGES = 200
THRESHOLD = 8 #intenstiy threshold

MIN_COMPONENT_AREA = 3
MAX_COMPONENT_AREA = 10000
ROI_PAD = 4

BIC_IMPROVEMENT = 10 #require 2-Gaussian BIC to be lower by at least this much to accept 2-spot model
MIN_TWO_SPOT_SEP = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(IMAGE_FOLDER.glob("clean_frame_*.tif"))[:N_IMAGES]
if not files:
    raise RuntimeError("No files found matching clean_frame_*.tif")

def extract_exposure_label(image_folder, files):
    folder_name = image_folder.name.lower()

    match = re.search(r"(\d+(?:ms|s))", folder_name)
    if match:
        return match.group(1)

    for f in files:
        match = re.search(r"(\d+(?:ms|s))", f.name.lower())
        if match:
            return match.group(1)

    return "unknown"

EXPOSURE_LABEL = extract_exposure_label(IMAGE_FOLDER, files)
print("exposure label:", EXPOSURE_LABEL)

stack = np.zeros_like(tiff.imread(files[0]), dtype=np.float32)
for f in files:
    stack += tiff.imread(f).astype(np.float32)

img = stack / len(files)
img_min, img_max = float(img.min()), float(img.max())

print("frames used:", len(files))
print("stack min/max:", img_min, img_max)
print("saving figures to:", OUTPUT_DIR.resolve())

mask = img > THRESHOLD
cc_labels, n_cc = ndi.label(mask)
areas = np.bincount(cc_labels.ravel())

print("connected components found:", n_cc)

def gauss2d_1(coords, A, x0, y0, s, B):
    x, y = coords
    return (A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * s**2)) + B).ravel()

def gauss2d_2(coords, A1, x1, y1, s1, A2, x2, y2, s2, B):
    x, y = coords
    g1 = A1 * np.exp(-((x - x1)**2 + (y - y1)**2) / (2 * s1**2))
    g2 = A2 * np.exp(-((x - x2)**2 + (y - y2)**2) / (2 * s2**2))
    return (g1 + g2 + B).ravel()

def compute_bic(rss, n, k):
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + k * np.log(n)

def fit_roi(img, x1, x2, y1, y2):
    roi = img[y1:y2, x1:x2].astype(float)

    yy, xx = np.indices(roi.shape)
    xdata, ydata = xx.ravel(), yy.ravel()
    zdata = roi.ravel()

    B0 = float(np.median(roi))
    A0 = float(roi.max() - B0)
    iy, ix = np.unravel_index(np.argmax(roi), roi.shape)

    p0_1 = [A0, ix, iy, 2.5, B0]
    bounds_1 = ([0, 0, 0, 0.5, 0], [1e6, roi.shape[1], roi.shape[0], 10, 1e6])

    popt1, _ = curve_fit(
        gauss2d_1,
        (xdata, ydata),
        zdata,
        p0=p0_1,
        bounds=bounds_1,
        maxfev=20000
    )

    fit1 = gauss2d_1((xx, yy), *popt1).reshape(roi.shape)
    rss1 = np.sum((roi - fit1) ** 2)
    bic1 = compute_bic(rss1, roi.size, 5)

    p0_2 = [
        0.7 * A0, ix, max(0, iy - 1), 2.5,
        0.5 * A0, ix, min(roi.shape[0] - 1, iy + 1), 2.5,
        B0
    ]

    bounds_2 = (
        [0, 0, 0, 0.5, 0, 0, 0, 0.5, 0],
        [1e6, roi.shape[1], roi.shape[0], 10, 1e6, roi.shape[1], roi.shape[0], 10, 1e6]
    )

    popt2, _ = curve_fit(
        gauss2d_2,
        (xdata, ydata),
        zdata,
        p0=p0_2,
        bounds=bounds_2,
        maxfev=40000
    )

    fit2 = gauss2d_2((xx, yy), *popt2).reshape(roi.shape)
    rss2 = np.sum((roi - fit2) ** 2)
    bic2 = compute_bic(rss2, roi.size, 9)

    x1g, y1g = x1 + popt1[1], y1 + popt1[2]

    xA, yA = x1 + popt2[1], y1 + popt2[2]
    xB, yB = x1 + popt2[5], y1 + popt2[6]

    sep = np.hypot(xB - xA, yB - yA)
    use_two = (bic2 < bic1 - BIC_IMPROVEMENT) and (sep >= MIN_TWO_SPOT_SEP)

    if use_two:
        centers = [(yA, xA), (yB, xB)]
        model = 2
    else:
        centers = [(y1g, x1g)]
        model = 1

    return {
        "model": model,
        "centers": centers,
        "bic1": bic1,
        "bic2": bic2,
        "rss1": rss1,
        "rss2": rss2,
        "sep": sep,
        "roi": (x1, x2, y1, y2)
    }

results = []
centroids = []

objects = ndi.find_objects(cc_labels)

for comp_id, slc in enumerate(objects, start=1):
    if slc is None:
        continue

    area = areas[comp_id]
    if not (MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA):
        continue

    ys, xs = slc
    y1 = max(0, ys.start - ROI_PAD)
    y2 = min(img.shape[0], ys.stop + ROI_PAD)
    x1 = max(0, xs.start - ROI_PAD)
    x2 = min(img.shape[1], xs.stop + ROI_PAD)

    try:
        result = fit_roi(img, x1, x2, y1, y2)
        result.update({"component_id": comp_id, "area": int(area)})
        results.append(result)
        centroids.extend(result["centers"])
    except Exception as e:
        print(f"component {comp_id} fit failed: {e}")

print("components successfully fit:", len(results))
print("final spots:", len(centroids))

def spot_delta(centroids, i, j):
    y1, x1 = centroids[i - 1]
    y2, x2 = centroids[j - 1]
    dx, dy = x2 - x1, y2 - y1
    return dx, dy, np.hypot(dx, dy)

for i, j in [(15, 16), (18, 19)]:
    dx, dy, d = spot_delta(centroids, i, j)
    print(f"{i}-{j}: dx={dx:.2f}, dy={dy:.2f}, d={d:.2f}")

def plot_image(img, centroids=None, title="", filename=None, show=True):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        img,
        cmap="viridis",
        origin="upper",
        vmin=img_min,
        vmax=img_max
    )

    if centroids is not None:
        for i, (y, x) in enumerate(centroids, start=1):
            ax.scatter(x, y, s=40,color="red", marker="x")
            ax.text(x + 2, y - 2, str(i), fontsize=10, color="white")

    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    if filename is not None:
        png_path = OUTPUT_DIR / f"{filename}.png"
        pdf_path = OUTPUT_DIR / f"{filename}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print("saved:", png_path.resolve())
        print("saved:", pdf_path.resolve())

    if show:
        plt.show()

    plt.close(fig)

plot_image(
    img,
    title="Average Stacked Image",
    filename=f"{EXPOSURE_LABEL}_stacked_image",
    show=True
)

plot_image(
    img,
    centroids=centroids,
    title=f"Detected Spots | N={len(centroids)}, {EXPOSURE_LABEL}",
    filename=f"{EXPOSURE_LABEL}_stacked_image_with_spots",
    show=True
)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(mask, cmap="gray", origin="upper")
ax.set_title("mask = img > threshold")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

mask_png = OUTPUT_DIR / "mask.png"
mask_pdf = OUTPUT_DIR / "mask.pdf"

fig.savefig(mask_png, dpi=300, bbox_inches="tight")
fig.savefig(mask_pdf, bbox_inches="tight")

print("saved:", mask_pdf.resolve())

plt.show()
plt.close(fig)

for r in results:
    print(
        f"comp {r['component_id']:2d} | area={r['area']:4d} | "
        f"model={r['model']} | bic1={r['bic1']:.2f} | "
        f"bic2={r['bic2']:.2f} | sep={r['sep']:.2f}"
    )