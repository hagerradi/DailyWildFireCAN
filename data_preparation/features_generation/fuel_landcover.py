from pathlib import Path
import rioxarray
import xarray as xr
import h5py

LANDCOVER_VARS = {
    'ccrs_landcover': 'landcover'
}

def extract_landcover_for_grid(tif_path, easting_grid, northing_grid):
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
    

def run_single_fire_landcover(h5_path, landcover_folder, current_year="2020"):
    """Tile-Based Architecture.
    Processes static LANDCOVER fuel data from the PREVIOUS year using pre-projected 90m TIFs.

    Args:
      h5_path: the file of the fire's H5 file
      landcover_folder: the folder containing the landcover TIFs
      current_year: the year of the LANDCOVER maps
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return
    
    landcover_folder = Path(landcover_folder)
    
    print(f"\n{'='*60}")
    print(f"Loading pre-fire LANDCOVER fuel data from {current_year}...")

    if not landcover_folder.exists():
        print(f"CRITICAL ERROR: LANDCOVER folder for {current_year} not found at {landcover_folder}")
        return

    with h5py.File(h5_path, "a") as f:
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        
        if not tile_ids:
            print("No tiles found in this H5 file. Skipping.")
            return

        for key, file_prefix in LANDCOVER_VARS.items():
            var_name = key.lower()
            
            # Find the 90m TIF
            # Example of file name : landcover-2015-classification_90m.tif
            all_files = list(landcover_folder.glob(f"{file_prefix}-{current_year}-classification_90m.tif"))
            matching_files = all_files
            
            if not matching_files:
                print(f"Skipping {var_name}: No valid 90m TIF found for '{file_prefix}' in {landcover_folder}.")
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
                grid_data = extract_landcover_for_grid(tif_path, easting_grid, northing_grid)

                # Save to HDF5
                if var_name in static_grp:
                    del static_grp[var_name]
                
                static_grp.create_dataset(var_name, data=grid_data, compression="lzf")

    print(f"Successfully processed all LANDCOVER tiles for {h5_path.name}")