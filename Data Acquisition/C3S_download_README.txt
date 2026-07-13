C3S SYSTEM 51 CDS API DOWNLOAD SCRIPT
====================================

File
----
download_c3s_system51_monthly.py

Purpose
-------
Downloads C3S seasonal monthly single-level forecasts for:

- 2 m temperature
- mean surface runoff rate
- total precipitation

Configuration
-------------
Originating centre: ECMWF
Forecast system: 51
Product type: ensemble mean
Lead months: 1-6
Initialization period: January 2018 to December 2023
Output format: NetCDF

Default area
------------
[27.22, 85.81, 21.42, 89.88]

The CDS area order is:
[North, West, South, East]

Why 2018 is included
--------------------
Initialization months from 2018 can be retained when lead-time forecasts are
needed to cover target months at the beginning of the 2019 study period.

Installation
------------
pip install cdsapi

Authentication
--------------
Configure your own CDS API credentials before running the script.
Do not upload API keys, tokens, or personal authentication files to GitHub.

Run
---
python download_c3s_system51_monthly.py

Output
------
One NetCDF file is written for each initialization month under:

data/c3s_system51/

Example:
c3s_ecmwf_system51_ensemble_mean_2019_01_leads_1_6.nc

Re-running
----------
Existing non-empty files are skipped by default, allowing interrupted
downloads to be resumed safely.

Repository note
---------------
The script contains only the public data-acquisition workflow. It does not
include downloaded C3S files or any water-quality observations.
