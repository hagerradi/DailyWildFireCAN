from pathlib import Path
import numpy as np
import rioxarray
import xarray as xr
import h5py

DISTURBANCE_VARS = {
    'annual_disturbance': 'canlad_annual'
}

def extract_annual_disturbance_for_grid(tif_path, easting_grid, northing_grid):
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
    

def run_single_fire_annual_disturbance(h5_path, annual_disturbance_folder, disturbance_year="2020"):
    """Tile-Based Architecture.
    Processes static DISRTURBANCE fuel data from the PREVIOUS year using pre-projected 90m TIFs.

    Args:
      h5_path: the file of the fire's H5 file
      annual_disturbance_folder: the folder containing the annual disturbance TIFs
      disturbance_year: the year of the DISTURBANCE maps
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return
    
    annual_disturbance_folder = Path(annual_disturbance_folder)
    
    print(f"\n{'='*60}")
    print(f"Loading pre-fire DISTURBANCE fuel data from {disturbance_year}...")
    
    if not annual_disturbance_folder.exists():
        print(f"CRITICAL ERROR: ANNUAL DISTURBANCE folder for {disturbance_year} not found at {annual_disturbance_folder}")
        return

    with h5py.File(h5_path, "a") as f:
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        
        if not tile_ids:
            print("No tiles found in this H5 file. Skipping.")
            return

        for key, file_prefix in DISTURBANCE_VARS.items():
            var_name = key.lower()
            
            # Find the 90m TIF

            # Example of file name : canlad_annual_2016_v1_90m.tif
            all_files = list(annual_disturbance_folder.glob(f"{file_prefix}_{disturbance_year}_v1_90m.tif"))
            
            matching_files = all_files
            
            if not matching_files:
                print(f"Skipping {var_name}: No valid 90m TIF found for '{file_prefix}'.")
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
                grid_data = extract_annual_disturbance_for_grid(tif_path, easting_grid, northing_grid)

                # Group all Defoliation Severities (7 and 8) into Class 6
                grid_data = np.where((grid_data == 7) | (grid_data == 8), 6, grid_data)

                # Save to HDF5
                if var_name in static_grp:
                    del static_grp[var_name]
                
                static_grp.create_dataset(var_name, data=grid_data, compression="lzf")

    print(f"Successfully processed all DISTURBANCE tiles for {h5_path.name}")