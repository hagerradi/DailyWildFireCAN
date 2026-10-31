from pathlib import Path
import h5py
import numpy as np
import xarray as xr
import rioxarray

from data_preparation.data_configs.data_settings import LANDCOVER_BASE_YEARS


DISTURBANCE_VARS = {
    'annual_disturbance_nbac': 'nbac_disturbance_maps'
}

def extract_annual_nbac_disturbance_for_grid(tif_path, easting_grid, northing_grid):
    """Extracts 2D grid data from a 90m TIFF file.
    Assumes grids and TIF share the exact same CRS.

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
    

def run_single_fire_annual_nbac_disturbance(h5_path, annual_disturbance_folder, disturbance_year="2020"):
    """Tile-Based Architecture.
    Processes static disturbance fuel data from the previous years up until the base year.

    Args:
      h5_path: the file of the fire's H5 file
      annual_disturbance_folder: the folder containing the annual disturbance TIFs
      disturbance_year: the year of the disturbance maps

    Returns:

    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return
    
    annual_disturbance_folder = Path(annual_disturbance_folder)
    
    disturbance_year = int(disturbance_year)
    
    print(f"\n{'='*60}")
    print(f"Loading pre-fire DISTURBANCE fuel data from {disturbance_year}...")
    
    if not annual_disturbance_folder.exists():
        print(f"CRITICAL ERROR: ANNUAL DISTURBANCE folder for {disturbance_year} not found at {annual_disturbance_folder}")
        return
    
    if str(disturbance_year) in LANDCOVER_BASE_YEARS:
        disturbance_years = []
    else:
        previous_base_year = max(
            int(year) for year in LANDCOVER_BASE_YEARS
            if int(year) < disturbance_year
        )
        disturbance_years = list(range(previous_base_year, disturbance_year + 1))

    with h5py.File(h5_path, "a") as f:
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        
        if not tile_ids:
            print("No tiles found in this H5 file. Skipping.")
            return

        for key, file_prefix in DISTURBANCE_VARS.items():
            
            var_name = key.lower()
            
            tif_paths = [
                annual_disturbance_folder / f"{file_prefix}_{year}_90m.tif"
                for year in disturbance_years
            ]

            if not disturbance_years:
                print(f"Base year {disturbance_year} filled with zeros")
            else:
                print(f"Considering disturbance years: {disturbance_years}")
                print("Files:")
                for tif_path in tif_paths:
                    print(f"  - {tif_path.name}")

            for tid in tile_ids:
                tile_grp = f[tid]
                if 'static_features' not in tile_grp:
                    static_grp = tile_grp.create_group('static_features')
                else:
                    static_grp = tile_grp['static_features']

                # Grab the EPSG:3347 coordinates
                easting_grid = tile_grp['coords/theoretical_easting'][:]  
                northing_grid = tile_grp['coords/theoretical_northing'][:] 
                
                if not disturbance_years:
                    grid_data = np.zeros_like(easting_grid)
                    
                else:
                    yearly_grids = []
                    
                    for tif_path in tif_paths:
                        grid = extract_annual_nbac_disturbance_for_grid(
                            tif_path,
                            easting_grid,
                            northing_grid
                        )
                        yearly_grids.append(grid)

                    grid_data = np.min(np.stack(yearly_grids), axis=0)
                    
                    # Group all Defoliation Severities (7 and 8) into Class 6
                    grid_data = np.where((grid_data == 7) | (grid_data == 8), 6, grid_data)

                # Save to HDF5
                if var_name in static_grp:
                    del static_grp[var_name]
                
                static_grp.create_dataset(var_name, data=grid_data, compression="lzf")

    print(f"Successfully processed all DISTURBANCE tiles for {h5_path.name}")