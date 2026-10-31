# data_settings.py

# ==========================================
# COORDINATE REFERENCE SYSTEMS (CRS)
# ==========================================
DEG_CRS = '4326'
DEG_NAD_CRS = '4269'
METER_NAD_CRS = '3347'

# ==========================================
# DIAGNOSIS SETTINGS
# ==========================================
BANDS_FEATURES = [
        's2_b02', 
        's2_b03', 
        's2_b04', 
        's2_b08', 
        's2_b11', 
        's2_b12'
    ]
DYNAMIC_FEATURES = ['tmax', 'rh', 'ws', 'prec', 'evi', 'ndvi', 'u10', 'v10'] + BANDS_FEATURES
STATIC_FEATURES = ['dem', 'slope', 'aspect', 
                   'biomass', 'closure', 'prcb', 'prcc', 
                   'height',
                   'prc_balsam_fir', 'prc_black_spruce', 'prc_douglas_fir', 'prc_jack_pine',
                   'prc_lodgepole_pine', 'prc_ponderosa_pine', 
                   'prc_tamarack', 'prc_white_red_pine']

LANDCOVER_BASE_YEARS = ['2015', '2020']


# ==========================================
# VIIRS SETTINGS
# ==========================================
# Fuel
VIIRS_CRS_TRANSFORM = [0.004166666666666667, 0, -141.0, 0, -0.004166666666666667, 83.0]
# API / Fetching
VIIRS_MAX_RETRIES = 5
VIIRS_WAIT_TIME_SECONDS = 2
# GEE settings
VIIRS_GEE_PROJECT = 'widlfires-vegetation'
VIIRS_GEE_IMAGE = 'NASA/VIIRS/002/VNP13A1'

# ==========================================
# SENTINEL SATELLITE SETTINGS
# ==========================================
# Collection
SENTINEL_COLLECTION = 'sentinel-2-l2a'
# API / Fetching
SENTINEL_API_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1'
SENTINEL_MAX_LOOKBACK_DAYS = 30
SENTINEL_MAX_RETRIES = 5
SENTINEL_MAX_DOWNLOAD_RETRIES = 3
SENTINEL_WORKERS = 8
# Thresholds
SENTINEL_CLOUD_COVERAGE_THRESHOLDS = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
# Bands
SENTINEL_BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
SENTINEL_VISUAL = ['visual']