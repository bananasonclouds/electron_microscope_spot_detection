# -*- coding: utf-8 -*-
"""
Script for bead detection, BIC analysis, empirical FWHM calculation, and Visual Report
"""

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi
from scipy.optimize import curve_fit

# ================= CONFIGURATION =================
IMAGE_FOLDER = Path(r"/Users/bananyamani/Desktop/PSI/50ms_cropped_clean")
PATH_TEM = r"/Users/bananyamani/Desktop/PSI/003_e_close.tif"
OUTPUT_DIR = Path("figures")

N_IMAGES = 20
THRESHOLD = 8 # Intensity threshold

MIN_COMPONENT_AREA = 3
MAX_COMPONENT_AREA = 10000
ROI_PAD = 2

BIC_IMPROVEMENT = 10 
# Set this to your theoretical diffraction limit or the previously measured FWHM
MIN_TWO_SPOT_SEP_UM = 2.8 

# Optical Parameters (Projected pixel size in um)
PIXEL_SIZE_X = 0.437  
PIXEL_SIZE_Y = 0.404  

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(IMAGE_FOLDER.glob("clean_frame_*.tif"))[:N_IMAGES]
if not files:
    raise RuntimeError("No files found matching clean_frame_*.tif")

stack = np.zeros_like(tiff.imread(files[0]), dtype=np.float32)
for f in files:
    stack += tiff.imread(f).astype(np.float32)

img = stack / len(files)
img_min, img_max = float(img.min()), float(img.max())

mask = img > THRESHOLD
cc_labels, n_cc = ndi.label(mask)
areas = np.bincount(cc_labels.ravel())

# ================= BASE FITTING FUNCTIONS =================
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
    popt1, _ = curve_fit(gauss2d_1, (xdata, ydata), zdata, p0=p0_1, bounds=bounds_1, maxfev=20000)
    
    fit1 = gauss2d_1((xx, yy), *popt1).reshape(roi.shape)
    rss1 = np.sum((roi - fit1) ** 2)
    bic1 = compute_bic(rss1, roi.size, 5)

    p0_2 = [0.7 * A0, ix, max(0, iy - 1), 2.5, 0.5 * A0, ix, min(roi.shape[0] - 1, iy + 1), 2.5, B0]
    bounds_2 = ([0, 0, 0, 0.5, 0, 0, 0, 0.5, 0], [1e6, roi.shape[1], roi.shape[0], 10, 1e6, roi.shape[1], roi.shape[0], 10, 1e6])
    popt2, _ = curve_fit(gauss2d_2, (xdata, ydata), zdata, p0=p0_2, bounds=bounds_2, maxfev=40000)
    
    fit2 = gauss2d_2((xx, yy), *popt2).reshape(roi.shape)
    rss2 = np.sum((roi - fit2) ** 2)
    bic2 = compute_bic(rss2, roi.size, 9)

    x1g, y1g = x1 + popt1[1], y1 + popt1[2]
    xA, yA = x1 + popt2[1], y1 + popt2[2]
    xB, yB = x1 + popt2[5], y1 + popt2[6]

    dx_um = (xB - xA) * PIXEL_SIZE_X
    dy_um = (yB - yA) * PIXEL_SIZE_Y
    sep_um = np.hypot(dx_um, dy_um)

    use_two = (bic2 < bic1 - BIC_IMPROVEMENT) and (sep_um >= MIN_TWO_SPOT_SEP_UM)

    if use_two:
        centers = [(yA, xA), (yB, xB)]
        model = 2
    else:
        centers = [(y1g, x1g)]
        model = 1

    return {
        "model": model, "centers": centers,
        "bic1": bic1, "bic2": bic2,
        "sep_um": sep_um, "roi": (x1, x2, y1, y2)
    }

# ================= CENTROID SEARCH EXECUTION =================
results = []
centroids = []
objects = ndi.find_objects(cc_labels)

for comp_id, slc in enumerate(objects, start=1):
    if slc is None: continue
    area = areas[comp_id]
    if not (MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA): continue

    ys, xs = slc
    y1, y2 = max(0, ys.start - ROI_PAD), min(img.shape[0], ys.stop + ROI_PAD)
    x1, x2 = max(0, xs.start - ROI_PAD), min(img.shape[1], xs.stop + ROI_PAD)

    try:
        res = fit_roi(img, x1, x2, y1, y2)
        res.update({"component_id": comp_id, "area": int(area)})
        results.append(res)
        centroids.extend(res["centers"])
    except Exception:
        pass

# ================= ELLIPTICAL FITTING FUNCTIONS (FWHM) =================
def gauss_ellittica(coords, A, xo, yo, sx, sy, B):
    x, y = coords
    return (A * np.exp(-((x - xo)**2 / (2 * sx**2) + (y - yo)**2 / (2 * sy**2))) + B).ravel()

def fit_fwhm_ellittica(img, cy, cx):
    """Returns X/Y FWHM, popt, and ROI for the visual report"""
    rad = 13
    h, w = img.shape
    y1, y2 = max(0, int(cy) - rad), min(h, int(cy) + rad + 1)
    x1, x2 = max(0, int(cx) - rad), min(w, int(cx) + rad + 1)
    roi = img[y1:y2, x1:x2].astype(float)
    
    if roi.shape[0] < 5 or roi.shape[1] < 5: return None, None, None, None
    
    yy, xx = np.indices(roi.shape)
    B0 = np.median(roi)
    A0 = roi.max() - B0
    p0 = [A0, roi.shape[1]//2, roi.shape[0]//2, 2.0, 2.0, B0]
    bounds = ([0, 0, 0, 0.1, 0.1, 0], [np.inf, roi.shape[1], roi.shape[0], 10, 10, np.inf])
    
    try:
        popt, _ = curve_fit(gauss_ellittica, (xx.ravel(), yy.ravel()), roi.ravel(), p0=p0, bounds=bounds)
        fwhm_factor = 2 * np.sqrt(2 * np.log(2))
        fwhm_x_um = (popt[3] * fwhm_factor) * PIXEL_SIZE_X
        fwhm_y_um = (popt[4] * fwhm_factor) * PIXEL_SIZE_Y
        return fwhm_x_um, fwhm_y_um, popt, roi
    except:
        return None, None, None, None

def find_nearest_centroid(click_x, click_y, centroids):
    distances = [np.hypot(cx - click_x, cy - click_y) for cy, cx in centroids]
    return centroids[np.argmin(distances)]

# ================= INTERACTIVE INTERFACE =================
try: plt.switch_backend('Qt5Agg') 
except: pass

try:
    img_tem = plt.imread(PATH_TEM)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    ax_fluo = axes[0]
    axes[1].imshow(img_tem, cmap='gray')
    axes[1].set_title("TEM (Reference)")
    axes[1].axis('off')
except FileNotFoundError:
    fig, ax_fluo = plt.subplots(1, 1, figsize=(8, 8))

vmax_disp = np.percentile(img, 99.5)
ax_fluo.imshow(img, cmap="viridis", origin="upper", vmin=img_min, vmax=vmax_disp)

for i, (y, x) in enumerate(centroids, start=1):
    ax_fluo.scatter(x, y, s=40, color="red", marker="x")
    ax_fluo.text(x + 2, y - 2, str(i), fontsize=10, color="white")

ax_fluo.set_title("Click near the X of single beads.\nPress ENTER when finished.")
clicks = plt.ginput(-1, timeout=-1)
plt.close(fig)

# ================= STATISTICS CALCULATION AND REPORT CREATION =================
report_data = []

if clicks:
    fwhm_x_list, fwhm_y_list = [], []
    for idx, (click_x, click_y) in enumerate(clicks):
        cy, cx = find_nearest_centroid(click_x, click_y, centroids)
        fx, fy, popt, roi = fit_fwhm_ellittica(img, cy, cx)
        
        if fx is not None:
            fwhm_x_list.append(fx)
            fwhm_y_list.append(fy)
            report_data.append({
                'id': idx + 1, 'cx': cx, 'cy': cy,
                'fx': fx, 'fy': fy, 'popt': popt, 'roi': roi
            })

    # Console Print
    print("\n=== DATA ===")
    print(f"Analyzed beads: {len(fwhm_x_list)}")
    print(f"Resolution X: {np.mean(fwhm_x_list):.2f} ± {np.std(fwhm_x_list):.2f} µm")
    print(f"Resolution Y: {np.mean(fwhm_y_list):.2f} ± {np.std(fwhm_y_list):.2f} µm")

    # Generate Report Figures
    n_beads = len(report_data)
    if n_beads > 0:
        print("\nGenerating visual report...")
        fig_rep, axes_rep = plt.subplots(n_beads, 3, figsize=(15, 4 * n_beads))
        
        # Safely handle axes if only one bead is selected (1D instead of 2D array)
        if n_beads == 1: axes_rep = np.expand_dims(axes_rep, axis=0)

        for i, data in enumerate(report_data):
            roi, popt = data['roi'], data['popt']
            A, xo, yo, sx, sy, B = popt
            
            # 1. 2D ROI Image
            ax2d = axes_rep[i, 0]
            ax2d.imshow(roi, cmap='viridis')
            ax2d.plot(xo, yo, 'r+', markersize=12, markeredgewidth=2) # Center found by the fit
            ax2d.set_title(f"Selected Bead {data['id']}\nCenter: [{int(data['cx'])}, {int(data['cy'])}]")
            ax2d.axis('off')

            # 2. X Profile (Horizontal slice at center)
            axX = axes_rep[i, 1]
            y_idx = int(round(yo))
            x_profile = roi[y_idx, :]
            x_axis = np.arange(roi.shape[1])
            fit_x = A * np.exp(-((x_axis - xo)**2 / (2 * sx**2))) + B
            
            axX.plot(x_axis, x_profile, 'o', color='gray', label='Raw Data')
            axX.plot(x_axis, fit_x, 'r-', linewidth=2, label='Gaussian Fit')
            axX.set_title(f"X-axis Profile\nFWHM = {data['fx']:.2f} µm")
            axX.legend()
            axX.grid(True, linestyle='--', alpha=0.6)

            # 3. Y Profile (Vertical slice at center)
            axY = axes_rep[i, 2]
            x_idx = int(round(xo))
            y_profile = roi[:, x_idx]
            y_axis = np.arange(roi.shape[0])
            fit_y = A * np.exp(-((y_axis - yo)**2 / (2 * sy**2))) + B
            
            axY.plot(y_axis, y_profile, 'o', color='gray', label='Raw Data')
            axY.plot(y_axis, fit_y, 'b-', linewidth=2, label='Gaussian Fit')
            axY.set_title(f"Y-axis Profile\nFWHM = {data['fy']:.2f} µm")
            axY.legend()
            axY.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        report_path = OUTPUT_DIR / "FWHM_Calibration_Report.png"
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"Visual report saved to: {report_path.resolve()}")
        plt.show()

else:
    print("No beads selected for FWHM calibration.")