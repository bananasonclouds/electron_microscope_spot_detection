import numpy as np
import tifffile as tiff
from pathlib import Path

# -------- USER SETTINGS --------

image_folder = "/Users/bananyamani/Desktop/PSI/50ms_cropped_clean"
output_file = "/Users/bananyamani/Desktop/PSI/stack_mean.tif"

n_images = 200   # change this to stack fewer images

# --------------------------------

files = sorted(Path(image_folder).glob("clean_frame_*.tif"))[:n_images]

stack = None

for i,f in enumerate(files):
    
    img = tiff.imread(f).astype(np.float32)
    
    if stack is None:
        stack = img
    else:
        stack += img

stack /= len(files)

tiff.imwrite(output_file, stack)