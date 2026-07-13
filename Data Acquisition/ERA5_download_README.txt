ERA5-LAND CDS API DOWNLOAD SCRIPT
================================

File
----
download_era5_land_monthly.py

Purpose
-------
Downloads monthly averaged ERA5-Land data for:
- 2 m temperature
- surface runoff
- total precipitation

Default period
--------------
January 2019 to December 2023

Default area
------------
[27.22, 85.81, 21.42, 89.88]
CDS order: [North, West, South, East]

Installation
------------
pip install cdsapi

Authentication
--------------
Configure your own CDS API credentials before running the script.
Do not upload API keys, tokens, or personal authentication files to GitHub.

Run
---
python download_era5_land_monthly.py

Output
------
data/era5/era5_land_monthly_2019_2023.zip

Repository note
---------------
The script contains only the public data-acquisition workflow. It does not
include downloaded ERA5 files or any water-quality observations.
