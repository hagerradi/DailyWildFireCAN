from pathlib import Path
import ee
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import requests
from rasterio.io import MemoryFile
import rioxarray
import xarray as xr
from tqdm import tqdm
import h5py
from data_preparation.data_configs.data_settings import DEG_CRS, VIIRS_CRS_TRANSFORM, VIIRS_MAX_RETRIES, VIIRS_WAIT_TIME_SECONDS, VIIRS_GEE_IMAGE, VIIRS_GEE_PROJECT

try:
    ee.Initialize(project=VIIRS_GEE_PROJECT)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=VIIRS_GEE_PROJECT)

def get_gee_url_with_retry(ee_image, roi, max_retries=VIIRS_MAX_RETRIES):
    """Requests a download URL from GEE with exponential backoff.
    Injects the exact EPSG:4326 grid transform for perfect pixel alignment.

    Args:
      ee_image: the GEE image corresponding to the VIIRS product
      roi: the bounding box
      max_retries:  the maximum retries for fetching data from GEE (Default value = 5)

    """
    wait_time = VIIRS_WAIT_TIME_SECONDS
    
    for attempt in range(max_retries):
        try:
            url = ee_image.getDownloadURL({
                'crs': f'EPSG:{DEG_CRS}', 
                'crs_transform': VIIRS_CRS_TRANSFORM,
                'region': roi,
                'format': 'GEO_TIFF'
            })
            return url
            
        except ee.EEException as e:
            error_msg = str(e).lower()
            if "too many" in error_msg or "capacity" in error_msg or "quota" in error_msg:
                print(f"GEE busy. Retrying in {wait_time}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
                wait_time *= 2  
            else:
                raise e
                
    print("Failed to get GEE URL after max retries.")
    
    return None

def fetch_in_memory_raster(url):
    """Downloads GeoTIFF bytes and loads them into rioxarray in RAM.

    Args:
        url: The direct web URL of the GeoTIFF file.

    Returns:
        da: The loaded raster data (or None if the download fails).
        memfile: The in-memory file object.
        src: The opened dataset reader.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"HTTP Error {response.status_code} during download.")
        return None, None, None
        
    memfile = MemoryFile(response.content)
    src = memfile.open()
    da = rioxarray.open_rasterio(src)
    return da, memfile, src

def run_daily_viirs_pipeline(h5_path):
    """Updated for Tile-Based Architecture.
    Fetches VIIRS vegetation data for the entire fire extent and distributes it
    into the individual tiles and days.

    Args:
      h5_path: the fire's H5 file path
    """
    fire_name = Path(h5_path).stem
    print(f"\n{'='*60}\nRunning Tile-Based VIIRS Pipeline: {fire_name}")
    
    with h5py.File(h5_path, "a") as f:
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]
        if not tile_ids: return

        # FIND GLOBAL BOUNDS ACROSS ALL TILES
        global_lon_min, global_lon_max = float('inf'), float('-inf')
        global_lat_min, global_lat_max = float('inf'), float('-inf')
        all_doys = []

        for tid in tile_ids:
            lon_grid = f[f"{tid}/coords/theoretical_lon"][:]
            lat_grid = f[f"{tid}/coords/theoretical_lat"][:]
            global_lon_min = min(global_lon_min, lon_grid.min())
            global_lon_max = max(global_lon_max, lon_grid.max())
            global_lat_min = min(global_lat_min, lat_grid.min())
            global_lat_max = max(global_lat_max, lat_grid.max())
            
            # Collect DOYs
            days_grp = f[f"{tid}/days"]
            for d_key in days_grp.keys():
                all_doys.append(int(days_grp[d_key].attrs['DOB']))

        # Define GEE ROI for the entire fire
        buffer = 0.05
        roi = ee.Geometry.Rectangle([global_lon_min - buffer, global_lat_min - buffer, 
                                     global_lon_max + buffer, global_lat_max + buffer])
        
        # CALCULATE TIME WINDOW
        fire_year = int(f.attrs['year'])
        unique_doys = sorted(list(set(all_doys)))
        
        start_date_ee = ee.Date.fromYMD(fire_year, 1, 1).advance(min(unique_doys) - 33, 'day')
        end_date_ee = ee.Date.fromYMD(fire_year, 1, 1).advance(max(unique_doys), 'day')

        # PRE-FETCH SCHEDULE
        print("Fetching VIIRS composite schedule...")
        viirs_times_ms = ee.ImageCollection("NASA/VIIRS/002/VNP13A1") \
            .filterDate(start_date_ee, end_date_ee) \
            .aggregate_array('system:time_start') \
            .getInfo()
        viirs_times_ms = sorted(viirs_times_ms)

        # Cache variables for the loop
        cached_time_ms = None
        da_ndvi, da_evi = None, None
        memfile, src = None, None

        # LOOP THROUGH UNIQUE DOYS (Process all tiles for a given day at once)
        for current_doy in tqdm(unique_doys):
            
            # Find the composite for this DOY
            current_date_py = datetime(fire_year, 1, 1, tzinfo=timezone.utc) + timedelta(days=current_doy - 1)
            target_limit_ms = int(current_date_py.timestamp() * 1000)
            sixteen_days_ms = 16 * 24 * 60 * 60 * 1000
            
            valid_times = [t for t in viirs_times_ms if (t + sixteen_days_ms) <= target_limit_ms]
            if not valid_times: continue
            target_time_ms = valid_times[-1]

            # Fetch new image if the 16-day window changed
            if target_time_ms != cached_time_ms:
                if memfile: 
                    src.close(); memfile.close()
                
                img = (ee.ImageCollection(VIIRS_GEE_IMAGE)
                        .filter(ee.Filter.eq('system:time_start', target_time_ms))
                        .select(['NDVI', 'EVI']).first().clip(roi))
                
                url = get_gee_url_with_retry(img, roi)
                if not url: continue
                
                da, memfile, src = fetch_in_memory_raster(url)
                if da is None: continue
                da_ndvi, da_evi = da.sel(band=1), da.sel(band=2)
                cached_time_ms = target_time_ms

            # DISTRIBUTE TO TILES
            for tid in tile_ids:
                target_day_key = None
                for k in f[f"{tid}/days"].keys():
                    if int(f[f"{tid}/days/{k}"].attrs['DOB']) == current_doy:
                        target_day_key = k
                        break
                
                if not target_day_key: continue

                # Get tile coords
                lon_grid = f[f"{tid}/coords/theoretical_lon"][:]
                lat_grid = f[f"{tid}/coords/theoretical_lat"][:]
                x_coords = xr.DataArray(lon_grid, dims=("y", "x"))
                y_coords = xr.DataArray(lat_grid, dims=("y", "x"))

                # Interpolate
                ndvi_vals = da_ndvi.sel(x=x_coords, y=y_coords, method="nearest").compute().values.astype(np.float32)
                evi_vals = da_evi.sel(x=x_coords, y=y_coords, method="nearest").compute().values.astype(np.float32)

                # Save to features
                feat_grp = f[f"{tid}/days/{target_day_key}/features"]
                
                for name, data in [("ndvi", ndvi_vals), ("evi", evi_vals)]:
                    if name in feat_grp:
                        del feat_grp[name]
                        
                    # Create a fresh float32 container
                    feat_grp.create_dataset(name, data=data, compression="lzf")

        # Cleanup
        if memfile: src.close(); memfile.close()