"""Channel layout of the released DailyWildfireCAN samples.

Every sample ``x`` has 50 channels (``(50, 256, 256)`` for mono-temporal samples,
``(T, 50, 256, 256)`` for multi-temporal ones), in the order below. The same order is
documented on the Hugging Face dataset card.
"""

DYNAMIC_FEATURES = [
    'tmax', 'rh', 'ws', 'prec', 'u10', 'v10',               # ERA5-Land weather (z-score)
    'evi', 'ndvi',                                          # MODIS vegetation indices
    's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12',  # Sentinel-2 reflectance
]

STATIC_FEATURES = [
    'dem', 'slope', 'aspect_sin', 'aspect_cos',
    'biomass', 'closure', 'prcb', 'prcc', 'height',
    'prc_balsam_fir', 'prc_black_spruce', 'prc_douglas_fir', 'prc_jack_pine',
    'prc_lodgepole_pine', 'prc_ponderosa_pine', 'prc_tamarack', 'prc_white_red_pine',
    # CCRS land cover, one-hot encoded
    'ccrs_1_needleleaf', 'ccrs_2_taiga_needleleaf', 'ccrs_5_broadleaf',
    'ccrs_6_mixed_forest', 'ccrs_8_shrubland', 'ccrs_10_grassland',
    'ccrs_11_polar_shrubland', 'ccrs_12_polar_grassland', 'ccrs_13_polar_barren',
    'ccrs_14_wetland', 'ccrs_15_cropland', 'ccrs_16_barren', 'ccrs_17_urban',
    'ccrs_18_water', 'ccrs_19_snow',
    'hii',                     # log(1 + Human Influence Index)
    'nbac_wildfire_history',   # binary: burned (NBAC) in the years before the prediction year
]

STATE_FEATURES = [
    'fire_mask',           # cumulative burned area up to day T (binary)
    'fire_arrival_time',   # days since fire start / 100 on burned pixels, 0 elsewhere
]

FEATURE_NAMES = DYNAMIC_FEATURES + STATIC_FEATURES + STATE_FEATURES
NUM_CHANNELS = len(FEATURE_NAMES)

FIRE_MASK_CHANNEL = FEATURE_NAMES.index('fire_mask')
RGB_FEATURES = ['s2_b02', 's2_b03', 's2_b04']
SENTINEL2_FEATURES = [f for f in DYNAMIC_FEATURES if f.startswith('s2_')]

assert NUM_CHANNELS == 50 and FIRE_MASK_CHANNEL == 48
