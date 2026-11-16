from pathlib import Path

GRID_SIZE = 256
PIXEL_SIZE = 90
X_COL = 'easting'
Y_COL = 'northing'
LON_COL = 'lon'
LAT_COL = 'lat'

SUBSET_FEATURE_LIST = ['ID', 'lon', 'lat', 'easting', 'northing', 'fireday', 'year', 'DOB', 'firearea', 'cumuarea', 'prec', 'tmax', 'ws', 'rh', 
                        'dem', 'slope', 'aspect', 'Biomass', 'Closure', 'prcB', 'prcC']

# To be replaced with the absolute paths
PROJECT_FOLDER = 'project_folder'
DATA_FOLDER = 'data_folder'
OUTPUT_FOLDER = 'output_folder'

H5_OUTPUT_FOLDER = f'{OUTPUT_FOLDER}/Fires_H5'

# SAMPLES (train, val, test_easy, test_medium, test_hard), as released on Hugging Face
SAMPLE_FOLDER = f'{OUTPUT_FOLDER}/Samples'
TIMESERIES_SAMPLE_FOLDER = f'{OUTPUT_FOLDER}/Timeseries_Samples'

# LEGACY SAMPLES (train, val, and a single test set), used only by legacy/
LEGACY_SAMPLE_FOLDER = f'{OUTPUT_FOLDER}/Samples_Legacy'
LEGACY_TIMESERIES_SAMPLE_FOLDER = f'{OUTPUT_FOLDER}/Timeseries_Samples_Legacy'

# CLUSTERED SAMPLES (organized as clusters), used only by legacy/
SAMPLE_FOLDER_CLUSTERS = f'{OUTPUT_FOLDER}/Samples_Clusters'
TIMESERIES_SAMPLE_FOLDER_CLUSTERS = f'{OUTPUT_FOLDER}/Timeseries_Samples_Clusters'

METADATA_FOLDER = f'{PROJECT_FOLDER}/fires_metadata'
SATELLITE_STATUS_FOLDER = f'{PROJECT_FOLDER}/Satellite_Status'

# CFSDS
BASE_FOLDER = f'{DATA_FOLDER}/Fire_growth_points'
# ERA5
ERA5_FOLDER = f'{DATA_FOLDER}/ERA5'
# SCANFI
SCANFI_FOLDER = f'{DATA_FOLDER}/SCANFI'
SCANFI_ORG_FOLDER = f'{DATA_FOLDER}/SCANFI_ORG'
# LANDCOVER
LANDCOVER_FOLDER = f'{DATA_FOLDER}/LANDCOVER'
LANDCOVER_ORG_FOLDER = f'{DATA_FOLDER}/LANDCOVER_ORG'
# DISTURBANCE
DISTURBANCE_FOLDER = f'{DATA_FOLDER}/DISTURBANCE'
DISTURBANCE_ORG_FOLDER = f'{DATA_FOLDER}/DISTURBANCE_ORG'
# HII
HII_ORG_FOLDER = f'{DATA_FOLDER}/HII_ORG'
HII_FOLDER = f'{DATA_FOLDER}/HII'

# DEM/Topography Folders
DEM_FOLDER = f'{DATA_FOLDER}/DEM_API'
DEM_TILES_FOLDER = f'{DATA_FOLDER}/DEM_API/DEM_Tiles'
ELEVATION_FOLDER = f'{DATA_FOLDER}/DEM_API/Elevation'
ELEVATION_AVG_FOLDER = f'{DATA_FOLDER}/DEM_API/Elevation_avg'
SLOPE_FOLDER = f'{DATA_FOLDER}/DEM_API/Slope'
ASPECT_FOLDER = f'{DATA_FOLDER}/DEM_API/Aspect'


# ==========================================
# AUTOMATIC DIRECTORY CREATION
# ==========================================
DIRS_TO_CREATE = [
    OUTPUT_FOLDER,
    H5_OUTPUT_FOLDER,
    SAMPLE_FOLDER,
    TIMESERIES_SAMPLE_FOLDER,
    METADATA_FOLDER,
    SATELLITE_STATUS_FOLDER,
    DEM_FOLDER,
    ELEVATION_FOLDER,
    ELEVATION_AVG_FOLDER,
    SLOPE_FOLDER,
    ASPECT_FOLDER
]

for d in DIRS_TO_CREATE:
    Path(d).mkdir(parents=True, exist_ok=True)