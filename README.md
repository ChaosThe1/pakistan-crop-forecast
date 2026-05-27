Pakistan Cotton & Wheat Yield Classifier
Predicts below-baseline harvests for 7 Punjab districts (Bahawalpur, Rahim Yar Khan, Bahawalnagar, Multan, Vehari, Lodhran, Khanewal) using Sentinel-2 NDVI and ERA5-Land weather. Ground truth is Punjab Crop Reporting Service final estimates.

Walk-forward 2021-2024: ~87% accuracy, 80% below-recall, AUC 0.80.

Setup
You need your own Google Earth Engine service account. The credentials in yield_classifier.py are placeholders and won't authenticate.

Create a Google Cloud project and enable the Earth Engine API.
Register the project at https://signup.earthengine.google.com/.
In GCP IAM, create a service account and download its JSON key.
Edit the top of yield_classifier.py:
SERVICE_ACCOUNT = 'your-sa@your-project.iam.gserviceaccount.com'
KEY_PATH = 'path/to/your-key.json'
PROJECT = 'your-gee-project'
Full guide: https://developers.google.com/earth-engine/guides/service_account

The .gitignore excludes *-key.json so your key won't be committed by accident — if you name yours differently, update it.

Run
pip install earthengine-api pandas numpy scikit-learn matplotlib
python yield_classifier.py
On a fresh checkout, uncomment pull_all_satellite_data() near the bottom to fetch NDVI + ERA5 and write the four CSVs (~15-30 min, EE quota permitting). After that the script reads from the cached CSVs and re-runs in seconds.

Data
NDVI — Sentinel-2 SR Harmonized, 10-day median composites, masked to ESA WorldCover cropland (class 40)
Weather — ERA5-Land daily aggregates rolled up to monthly: max/mean temperature, total precipitation, soil moisture (top layer)
Yields — Punjab CRS final estimates, maund/acre. Cotton kharif 2017-2024, wheat rabi 2018-2024.
Labels
For each (district, crop, year), the baseline is the mean of all prior years for that district+crop (no leakage). A year is labeled below if (yield − baseline) / baseline < −0.10. The classifier predicts below vs not-below.
