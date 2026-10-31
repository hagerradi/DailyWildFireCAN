# samples_settings.py

# SEED
RANDOM_SEED = 42

# SIMPLE/TIMESERIES 

DYNAMIC_FEATURES = ['tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', # WEATHER
                     's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12' # SENTINEL-2
                    ]

STATIC_FEATURES = ['dem', 'slope', 'aspect', #TOPOGRAPHY
                   # SCANFI
                   'biomass', 'closure', 'prcb', 'prcc', 
                   'height',
                   'prc_balsam_fir', 'prc_black_spruce', 'prc_douglas_fir', 
                   'prc_jack_pine','prc_lodgepole_pine', 'prc_ponderosa_pine', 
                   'prc_tamarack', 'prc_white_red_pine',
                   # CCRS LANDCOVER
                   'ccrs_landcover',
                   # HUMAN INFLUENCE INDEX
                   'hii',
                   # CANLAD DISTURBANCE
                   'annual_disturbance'
                   ]

TARGET_YEARS =  ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']

# TIMESERIES
SEQUENCE_LENGTH = 3

# TOGGLES
REMOVE_MISSING_DATA = True

# GEE CONFIGURATION
GEE_PROJECT = 'widlfires-vegetation'
ALPHA_EARTH_GEE_IMAGE = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'

# SPLITS
HOLDOUT_YEAR = '2024'
N_CLUSTERS = 10