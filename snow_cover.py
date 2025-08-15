# Adonis Engel July 2025
# Script designed to give a relative snow cover value for the images provided

import cv2
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS


# Manually select dates and sites
DATE_FILTERS = ["2023-04-15"]
#SITE_FILTERS = ["GP1", "GP2", "GP3", "SL", "TL"] ## This site set is all the sites that are within bounds on the GEOTIFFs
SITE_FILTERS = ["GP3_2023", "FT_2023", "SG_2023", "FF_2023", "GG_2023", "GT_2023", "GP2_2023", "GP1_2023", "SL_2023", "TL_2023", "TW_2023" "SW_2023"]


# Load site info from Excel
site_info_df = pd.read_excel("1854TA_SnowStation_Locations.xlsx", usecols="B:D", header=None)
site_info_df.columns = ["site_prefix", "lat", "lon"]

site_lookup = {
    str(row['site_prefix']).upper().strip(): {
        "site_name": str(row['site_prefix']).upper().strip(),
        "lat": row['lat'],
        "lon": row['lon']
    }
    for _, row in site_info_df.iterrows()
}

# EXIF timestamp extraction
def extract_exif_timestamp(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return ""
        for tag, value in exif.items():
            if TAGS.get(tag) == "DateTimeOriginal":
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"[!] EXIF read failed for {image_path}: {e}")
    return ""

# Path setup
input_root = Path(r"D:\Full Data Set\2022-2023")

# Create an output folder next to the images
output_root = input_root / "snow_analysis_results"
output_root.mkdir(parents=True, exist_ok=True)

output_mask_folder = output_root / "snow_masks"
output_mask_folder.mkdir(parents=True, exist_ok=True)

output_csv = output_root / "snow_mask_summaryV2.csv"

image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
results = []

image_list = [p for p in input_root.rglob("*") if p.suffix.lower() in image_extensions]

# Process images ## Add PATH check in 67-68 to increase speed
start_time = time.time()
for image_path in tqdm(image_list, desc="Processing images"):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[!] Could not read {image_path}")
        continue

    img = cv2.resize(img, (640, 480))
    h = img.shape[0]
    top = int(h * 0.25)
    bottom = int(h * 0.9)
    roi = img[top:bottom, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_snow = np.array([0, 0, 170])
    upper_snow = np.array([180, 40, 255])
    snow_mask = cv2.inRange(hsv, lower_snow, upper_snow)

    # Dynamic area filtering
    min_area = (snow_mask.shape[0] * snow_mask.shape[1]) * 0.001  # 0.1% of ROI
    contours, _ = cv2.findContours(snow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:          
       if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(snow_mask, [cnt], -1, 0, -1)

    # Morph Smoothing
    kernel = np.ones((5, 5), np.uint8)
    snow_mask = cv2.morphologyEx(snow_mask, cv2.MORPH_CLOSE, kernel)

    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[top:bottom, :] = snow_mask

    snow_pixels = np.sum(snow_mask > 0)
    total_pixels = snow_mask.size
    snow_percent = (snow_pixels / total_pixels) * 100

    relative_path = image_path.relative_to(input_root)
    mask_name = relative_path.with_suffix('').as_posix().replace('/', '_') + "_mask.png"
    mask_path = output_mask_folder / mask_name
    cv2.imwrite(str(mask_path), full_mask)

    timestamp = extract_exif_timestamp(image_path)

    if not timestamp or timestamp.strftime("%Y-%m-%d") not in DATE_FILTERS:
        continue

    parent_folder = image_path.parts[-2]
    prefix = parent_folder.upper()

    if SITE_FILTERS and prefix not in SITE_FILTERS:
        continue

    site_data = site_lookup.get(prefix, {"site_name": "", "lat": "", "lon": ""})

    results.append({
        "filename": str(relative_path),
        "timestamp": timestamp,
        "site_code": prefix,
        "site_name": site_data['site_name'],
        "lat": site_data['lat'],
        "lon": site_data['lon'],
        "snow_percent": round(snow_percent, 2),
        "mask_path": str(mask_path)
    })

######
def create_site_overlays(results, site_code, out_dir):
    """Create one overlay PNG per date for a single site."""
    site_results = [r for r in results if r['site_code'] == site_code]
    if not site_results:
        print(f"[!] No images found for site {site_code}")
        return
    
    for date in DATE_FILTERS:
        for r in site_results:
            if pd.to_datetime(r['timestamp']).strftime("%Y-%m-%d") == date:
                # Load original & mask
                img = cv2.cvtColor(cv2.imread(str(input_root / r['filename'])), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(r['mask_path']), cv2.IMREAD_GRAYSCALE)
                mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img, 0.7, mask_color, 0.3, 0)

                out_path = Path(out_dir) / f"{site_code}_{date}.png"
                cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                print(f"[INFO] Overlay saved: {out_path}")
                break

# Example: Generate separate overlays for GP1
if results:
    overlay_dir = output_root / "poster_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    create_site_overlays(results, "GP1", overlay_dir)
######

# Save results + final readout
if results:
    pd.DataFrame(results).to_csv(output_csv, index=False)
    elapsed = time.time() - start_time
    print(f"   Images processed: {len(results)}")
    print(f"   Output CSV: snow_mask_summary.csv")
    print(f"   Mask images saved to: {output_mask_folder.resolve()}")
    print(f"   Time elapsed: {elapsed:.2f} seconds")
else:
    print("\n No images processed. Check your paths or file extensions.")
