from pathlib import Path
import numpy as np
import rioxarray
import xarray as xr
import h5py

HII_VARS = {'hii': 'hii'}

def extract_hii_for_grid(tif_path, lon_grid, lat_grid):
    """Extracts 2D grid data from a HII original TIFF file directly.

    Args:
      tif_path: path of the TIF file for a specific feature
      lon_grid: the longitude coordinates 2D grid
      lat_grid: the latitude coordinates 2D grid

    Returns: the feature grid

    """
    with rioxarray.open_rasterio(tif_path) as da:

        # Dynamically extract the NoData value from the raster metadata
        nodata_val = da.rio.nodata
        
        # Crop the raster to the fire's extent using the raw coordinates
        buffer = 0.05
        min_x, max_x = lon_grid.min() - buffer, lon_grid.max() + buffer
        min_y, max_y = lat_grid.min() - buffer, lat_grid.max() + buffer
        
        da_cropped = da.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)
        
        # Create 2D Xarray coordinates directly from the EPSG:4326 grids
        x_coords = xr.DataArray(lon_grid, dims=("y", "x"))
        y_coords = xr.DataArray(lat_grid, dims=("y", "x"))
        
        # Sample the cropped raster at the exact grid points
        sampled = da_cropped.sel(x=x_coords, y=y_coords, method="nearest").compute()
        
        # Extract the numpy array
        if sampled.ndim == 3 and sampled.shape[0] == 1:
            return sampled.values[0], nodata_val
        
        return sampled.values, nodata_val
    

def run_single_fire_hii(h5_path, hii_folder, hii_year="2019"):
    """Tile-Based Architecture.
    Processes static HII data from the most recent previous year.

    Args:
      h5_path: the file of the fire's H5 file
      hii_folder: the folder containing the human influence index (HII) TIFs
      current_year: the year of the HII maps

    Returns:

    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} does not exist.")
        return
    
    hii_folder = Path(hii_folder)
    
    print(f"\n{'='*60}")
    print(f"Loading pre-fire HII fuel data from {hii_year}...")

    with h5py.File(h5_path, "a") as f:
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        
        if not tile_ids:
            print("No tiles found in this H5 file. Skipping.")
            return

        for key, file_prefix in HII_VARS.items():
            var_name = key.lower()
            
            # Find the TIF
            all_files = list(hii_folder.glob(f"{file_prefix}_{hii_year}-01-01.tif"))
            
            matching_files = all_files
            
            if not matching_files:
                print(f"Skipping {var_name}: No valid native TIF found for '{file_prefix}'.")
                continue
            
            tif_path = matching_files[0]
            print(f"--- Extracting {var_name} from {tif_path.name} ---")

            for tid in tile_ids:
                tile_grp = f[tid]
                if 'static_features' not in tile_grp:
                    static_grp = tile_grp.create_group('static_features')
                else:
                    static_grp = tile_grp['static_features']

                # Grab the EPSG:4326 coordinates
                lon_grid = tile_grp['coords/theoretical_lon'][:]  
                lat_grid = tile_grp['coords/theoretical_lat'][:] 

                # Extract the 2D array
                grid_data, nodata_val = extract_hii_for_grid(tif_path, lon_grid, lat_grid)
                
                # Convert to float32
                grid_data = grid_data.astype(np.float32)

                # Since the empty pixels are water pixels, set them to 0.0
                if nodata_val is not None:
                    print(f"--- No Data Value {nodata_val}. ---")
                    grid_data = np.where(grid_data == nodata_val, 0.0, grid_data)

                # Save to HDF5
                if var_name in static_grp:
                    del static_grp[var_name]
                
                static_grp.create_dataset(var_name, data=grid_data, compression="lzf")

    print(f"Successfully processed all HII tiles for {h5_path.name}")