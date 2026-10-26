import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

image_folder = "/Users/bananyamani/Desktop/PSI/50ms_cropped_clean"
n_images = 200
threshold = 8

files = sorted(Path(image_folder).glob("clean_frame_*.tif"))[:n_images]
if len(files) == 0:
    raise RuntimeError("No files found matching clean_frame_*.tif")

stack = None

for f in files:
    frame = tiff.imread(f).astype(np.float32)
    if stack is None:
        stack = frame.copy()
    else:
        stack += frame

stack /= float(len(files))
img = stack.astype(np.float32)

img_min = float(img.min())
img_max = float(img.max())

print("frames used:", len(files))
print("stack min/max:", img_min, img_max)

# ---------------------------
# DETECTION
# ---------------------------
mask = img > threshold
cc_labels, n_cc = ndi.label(mask)
distance = ndi.distance_transform_edt(mask)

coords = peak_local_max(
    img,
    min_distance=4,
    threshold_abs=threshold,
    labels=mask
)

r_in = 3
r_out = 10

def peak_prominence(image, y, x, r_in=3, r_out=10):
    y0, y1 = max(0, y - r_out), min(image.shape[0], y + r_out + 1)
    x0, x1 = max(0, x - r_out), min(image.shape[1], x + r_out + 1)
    patch = image[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    cy, cx = y - y0, x - x0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = patch[(rr >= r_in) & (rr <= r_out)]
    if ring.size == 0:
        return 0.0
    return float(image[y, x] - np.median(ring))

areas = np.bincount(cc_labels.ravel())
by_comp = {}

for (y, x) in coords:
    comp = int(cc_labels[y, x])
    if comp == 0:
        continue
    p = peak_prominence(img, int(y), int(x), r_in=r_in, r_out=r_out)
    by_comp.setdefault(comp, []).append((int(y), int(x), p, float(img[y, x])))

small_component_area_px = 250
k_small = 2
k_large = 1

coords_filt = []
for comp, lst in by_comp.items():
    k = k_small if areas[comp] <= small_component_area_px else k_large
    lst_sorted = sorted(lst, key=lambda t: (t[2], t[3]), reverse=True)
    coords_filt.extend([(y, x) for (y, x, p, I) in lst_sorted[:k]])

coords_filt = np.array(coords_filt, dtype=np.int32) 


markers = np.zeros_like(img, dtype=np.int32)
for k, (y, x) in enumerate(coords_filt, start=1):
    markers[y, x] = k

ws = watershed(-distance, markers, mask=mask)

region_ids = np.unique(ws)
region_ids = region_ids[region_ids != 0]
centroids = ndi.center_of_mass(img, labels=ws, index=region_ids)
centroids = list(centroids)

centroids = sorted(
    centroids,
    key=lambda c: img[int(c[0]), int(c[1])],
    reverse=True
)

print("connected components:", n_cc)
print("peaks detected:", len(coords))
print("markers after ranking:", len(coords_filt))
print("final watershed regions:", len(region_ids))

def spot_delta(centroids, i, j):
    """
    Compute dx, dy and Euclidean distance between two detected spots.

    Parameters
    ----------
    centroids : list or array
        Output from ndi.center_of_mass()
    i : int
        first spot number (1-based indexing)
    j : int
        second spot number (1-based indexing)

    Returns
    -------
    dx : float
        x displacement (pixels)
    dy : float
        y displacement (pixels)
    d : float
        Euclidean distance (pixels)
    """

    y1, x1 = centroids[i - 1]
    y2, x2 = centroids[j - 1]

    dx = x2 - x1
    dy = y2 - y1

    d = np.sqrt(dx**2 + dy**2)

    return dx, dy, d


pairs = [(4, 7), (8, 14),(10,13)]

for i, j in pairs:
    dx, dy, d = spot_delta(centroids, i, j)
    print(f"{i}-{j}: dx={dx:.2f}, dy={dy:.2f}, d={d:.2f}")


fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    img,
    cmap="viridis",
    origin="upper",
    vmin=img_min,
    vmax=img_max
)
fig.colorbar(im, ax=ax)
ax.set_title("Average Stacked Image")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
fig.savefig("stacked_image.png", dpi=300, bbox_inches="tight")
plt.close(fig)


fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    img,
    cmap="viridis",
    origin="upper",
    vmin=img_min,
    vmax=img_max
)

for i, (y, x) in enumerate(centroids, start=1):
    ax.scatter(x, y, color="red", s=40)
    ax.text(x + 2, y - 2, str(i), color="white", fontsize=10)

fig.colorbar(im, ax=ax)
ax.set_title(f"Detected Spots | N={len(region_ids)}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")

fig.savefig("stacked_image_with_spots.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.imshow(mask, cmap="gray", origin="upper")
plt.title("mask = img > threshold")
plt.show()

#scalefactor is approximately 0.33
# 5.5 µm / 17 px ≈ 0.32 
# 3.6 µm / 9.7 px ≈ 0.37
# distance for set 3, 8-? ≈ 5.43 px which is about 2um
