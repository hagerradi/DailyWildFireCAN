from pathlib import Path
import numpy as np
import rioxarray
import xarray as xr
import h5py

SCANFI_VARS = {
    # Original Variables
    'Biomass': 'SCANFI_att_biomass',
    'Closure': 'SCANFI_att_closure',
    'prcC': 'SCANFI_spsCC_otherConiferous',
    'prcB': 'SCANFI_spsCC_broadleaf',
    
    # New Continuous Variables
    'age': 'SCANFI_age_median',
    'height': 'SCANFI_att_height',
    
    # New Species Crown Closures
    'prc_balsam_fir': 'SCANFI_spsCC_balsamFir',
    'prc_black_spruce': 'SCANFI_spsCC_blackSpruce',
    'prc_douglas_fir': 'SCANFI_spsCC_douglasFir',
    'prc_jack_pine': 'SCANFI_spsCC_jackPine',
    'prc_lodgepole_pine': 'SCANFI_spsCC_lodgepolePine',
    'prc_ponderosa_pine': 'SCANFI_spsCC_ponderosaPine',
    'prc_tamarack': 'SCANFI_spsCC_tamarack',
    'prc_white_red_pine': 'SCANFI_spsCC_whiteRedPine'
}

def extract_scanfi_for_grid(tif_path, easting_grid, northing_grid):
    """Extracts 2D grid data from a 90m TIFF file directly.
    NO TRANSFORMATION APPLIED. Assumes grids and TIF share the exact same CRS.

    Args:
      tif_path: path of the TIF file for a specific feature
      easting_grid: the easting coordinates 2D grid
      northing_grid: the northing coordinates 2D grid

    Returns: the feature grid

    """
    with rioxarray.open_rasterio(tif_path) as da:
        
        # Crop the raster to the fire's extent using the raw coordinates
        buffer = 200
        min_x, max_x = easting_grid.min() - buffer, easting_grid.max() + buffer
        min_y, max_y = northing_grid.min() - buffer, northing_grid.max() + buffer
        
        da_cropped = da.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
        
        # Create 2D Xarray coordinates directly from the EPSG:3347 grids
        x_coords = xr.DataArray(easting_grid, dims=("y", "x"))
        y_coords = xr.DataArray(northing_grid, dims=("y", "x"))
        
        # Sample the cropped raster at the exact grid points
        sampled = da_cropped.sel(x=x_coords, y=y_coords, method="nearest").compute()
        
        # Extract the numpy array
        if sampled.ndim == 3 and sampled.shape[0] == 1:
            return sampled.values[0]
        
        return sampled.values
    

def run_single_fire_scanfi(h5_path, scanfi_folder, current_year="2020"):
    """Tile-Based Architecture.
    Processes static SCANFI fuel data from the PREVIOUS year (T-1) using pre-projected 90m TIFs.

    Args:
      h5_path: the file of the fire's H5 file
      scanfi_folder: the folder containing the scanfi TIFs
      current_year: the year of the scanfi maps

    Returns:

    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return
    
    year_folder = Path(scanfi_folder) / current_year
    
    print(f"\n{'='*60}")
    print(f"Loading pre-fire SCANFI fuel data from {current_year}...")

    if not year_folder.exists():
        print(f"CRITICAL ERROR: SCANFI folder for {current_year} not found at {year_folder}")
        return

    with h5py.File(h5_path, "a") as f:
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        
        if not tile_ids:
            print("No tiles found in this H5 file. Skipping.")
            return

        for key, file_prefix in SCANFI_VARS.items():
            var_name = key.lower()
            
            # Find the 90m TIF
            all_files = list(year_folder.glob(f"*{file_prefix}*_90m.tif"))
            matching_files = all_files
            
            if not matching_files:
                print(f"Skipping {var_name}: No valid 90m TIF found for '{file_prefix}' in {year_folder}.")
                continue
            
            tif_path = matching_files[0]
            print(f"--- Extracting {var_name} from {tif_path.name} ---")

            for tid in tile_ids:
                tile_grp = f[tid]
                if 'static_features' not in tile_grp:
                    static_grp = tile_grp.create_group('static_features')
                else:
                    static_grp = tile_grp['static_features']

                # Grab the EPSG:3347 coordinates
                easting_grid = tile_grp['coords/theoretical_easting'][:]  
                northing_grid = tile_grp['coords/theoretical_northing'][:] 

                # Extract the 2D array
                grid_data = extract_scanfi_for_grid(tif_path, easting_grid, northing_grid)
                
                # Safecty check: Convert to float32
                grid_data = grid_data.astype(np.float32)

                # Save to HDF5
                if var_name in static_grp:
                    del static_grp[var_name]
                
                static_grp.create_dataset(var_name, data=grid_data, compression="lzf")

    print(f"Successfully processed all SCANFI tiles for {h5_path.name}")