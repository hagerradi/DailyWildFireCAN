import h5py
import odc.stac
import planetary_computer
import pystac_client
import numpy as np
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar
import json
import planetary_computer
from odc.geo.geobox import GeoBox
from affine import Affine

from data_preparation.data_configs.data_settings import SENTINEL_API_URL, SENTINEL_MAX_LOOKBACK_DAYS, SENTINEL_MAX_RETRIES, SENTINEL_MAX_DOWNLOAD_RETRIES, SENTINEL_CLOUD_COVERAGE_THRESHOLDS, SENTINEL_COLLECTION, SENTINEL_WORKERS

def get_tile_master_cube(h5_path, tile_id, bands):
    """
    Downloads the satellite history for a specific tile into a 4D NumPy array.
    Includes the 404 overhead fallback to drop broken STAC scenes.
    """

    # ---------------------------------------------------------
    # FRESH AUTHENTICATION (Prevents Token Timeouts)
    # ---------------------------------------------------------
    print(f"[{tile_id}] Connecting to Planetary Computer for fresh tokens...")
    catalog = pystac_client.Client.open(
        SENTINEL_API_URL,
        modifier=planetary_computer.sign_inplace,
    )

    with h5py.File(h5_path, 'r') as f:
        tile_grp = f[tile_id]
        
        # Setup the GeoBox
        e = tile_grp['coords/theoretical_easting'][:]
        n = tile_grp['coords/theoretical_northing'][:]
        lons = tile_grp['coords/theoretical_lon'][:]
        lats = tile_grp['coords/theoretical_lat'][:]
        
        transform = Affine.translation(e.min(), n.max()) * Affine.scale(90, -90)
        height, width = e.shape
        geobox = GeoBox(shape=(height, width), affine=transform, crs="EPSG:3347")
        
        # Setup Temporal Bounds
        days_grp = tile_grp['days']
        day_keys = sorted(days_grp.keys())
        if not day_keys: return None, {}
        
        fire_year = int(f.attrs['year'])
        start_dob = int(days_grp[day_keys[0]].attrs['DOB'])
        end_dob = int(days_grp[day_keys[-1]].attrs['DOB'])

    max_lookback_days = 30
    start_date = datetime(fire_year, 1, 1) + timedelta(days=start_dob - max_lookback_days)
    end_date = datetime(fire_year, 1, 1) + timedelta(days=end_dob + 1)

    # ---------------------------------------------------------
    # STAC API Retry Loop
    # ---------------------------------------------------------
    items = []
    max_api_retries = SENTINEL_MAX_RETRIES
    for attempt in range(max_api_retries):
        try:
            search = catalog.search(
                collections=[SENTINEL_COLLECTION],
                bbox=[lons.min() - 0.1, lats.min() - 0.1, lons.max() + 0.1, lats.max() + 0.1],
                datetime=f"{start_date.date()}/{end_date.date()}"
            )
            items = list(search.items())
            break  
        except Exception as e:
            if attempt < max_api_retries - 1: 
                time.sleep((attempt + 1) * 10)
            else:
                print(f"[{tile_id}] STAC API Failed.")
                return None, {}

    if not items: return None, {}

    # ---------------------------------------------------------
    # Initial Lazy Load Setup
    # ---------------------------------------------------------
    dataset = odc.stac.load(
        items,
        bands=bands,
        geobox=geobox,
        patch_url=planetary_computer.sign,
        groupby="id",
        chunks={'time': 1, 'x': width, 'y': height}
    )
    stack_da = dataset.to_array(dim="band").transpose("time", "band", "y", "x")

    print(f"Estimated Size in RAM: {stack_da.nbytes / (1024**2):.2f} MB")
    print(f"Downloading Master Cube: {stack_da.shape} (Time, Bands, H, W)...")
    print("Object Type:", type(stack_da))
    print("Data Type:  ", stack_da.dtype)
    
    # ---------------------------------------------------------
    # Dask Threading (Download Compute with Retries)
    # ---------------------------------------------------------
    max_retries = SENTINEL_MAX_DOWNLOAD_RETRIES
    cube = None
    download_success = False
    
    with dask.config.set(scheduler='threads', num_workers=SENTINEL_WORKERS):
        for attempt in range(max_retries):
            try:
                print(f"[{tile_id}] Download Attempt {attempt + 1}...")
                with ProgressBar():
                    cube = stack_da.compute()            
                
                if cube is not None:
                    print(f"[{tile_id}] Download successful!")
                    download_success = True
                    break 
                    
            except Exception as e:
                print(f"\n[!] Attempt {attempt + 1} failed with error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1) 
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("[!] Max optimistic retries exhausted. Preparing for fallback...")

        # ---------------------------------------------------------
        # THE 404 OVERHEAD FALLBACK
        # ---------------------------------------------------------
        if not download_success:
            print(f"\n[!] Standard download failed for {tile_id}. Initiating 404 overhead scan to drop broken scenes...")
            
            valid_items = []
            for item in tqdm(items, desc="Checking Items"):
                try:
                    # 1x1 pixel slice to force odc-stac to check file existence
                    test_ds = odc.stac.load(
                        [item], 
                        bands=bands, 
                        geobox=geobox, 
                        patch_url=planetary_computer.sign,
                        chunks={} 
                    ).isel(x=slice(0, 1), y=slice(0, 1)).compute()
                    test_ds.compute()
                    valid_items.append(item)
                except Exception as e:
                    print(f"\n[!] SKIPPING BROKEN ITEM: {item.id}")
            
            if not valid_items:
                print("❌ All items in this search returned 404 or errors.")
                return None, {}
            
            # Rebuild the dataset and stack with ONLY healthy items
            items = valid_items
            print(f"Rebuilding lazy cube with {len(items)} healthy items...")
            
            dataset_clean = odc.stac.load(
                items,
                bands=bands,
                geobox=geobox,
                patch_url=planetary_computer.sign,
                groupby="id",
                chunks={'time': 1, 'x': width, 'y': height}
            )
            stack_da_clean = dataset_clean.to_array(dim="band").transpose("time", "band", "y", "x")
            
            # One final desperate compute
            print("Attempting to download cleaned Master Cube...")
            try:
                with ProgressBar():
                    cube = stack_da_clean.compute()
                print("Cleaned download successful!")
            except Exception as e:
                print("❌ Cleaned download also failed. Max retries exhausted.")
                raise e # Crash the task, because it's genuinely broken.

    # ---------------------------------------------------------
    # Map Metadata
    # ---------------------------------------------------------
    metadata = {
        "dates": [np.datetime64(i.datetime.date(), 'D') for i in items],
        "cloud_covers": np.array([i.properties.get("eo:cloud_cover", 100) for i in items]),
        "item_ids": np.array([i.id for i in items])
    }

    return cube.values if hasattr(cube, 'values') else cube, metadata

def process_tile_h5_from_cube(h5_path, tile_id, cube, metadata, bands):
    """
    Slices the pre-downloaded cube using the Tiered Single-Day Fallback strategy
    and saves strictly to the dedicated 'satellite' group with cloud cover metadata.
    """
    if cube is None: return
    
    cube_dates = metadata["dates"]
    cube_ccs = metadata["cloud_covers"]
    max_lookback_days = SENTINEL_MAX_LOOKBACK_DAYS
    threshold_tiers = SENTINEL_CLOUD_COVERAGE_THRESHOLDS

    with h5py.File(h5_path, 'a') as f:
        fire_year = int(f.attrs['year'])
        days_grp = f[f"{tile_id}/days"]
        
        for day_key in tqdm(sorted(days_grp.keys()), desc=f"Slicing {tile_id}"):
            day_grp = days_grp[day_key]
            current_dob = int(day_grp.attrs['DOB'])
            target_dt = np.datetime64(datetime(fire_year, 1, 1) + timedelta(days=current_dob - 1), 'D')
            
            found_valid_mosaic = False
            final_mosaic = None
            used_day_indices = []
            idx_first_valid = None

            for current_threshold in threshold_tiers:
                for lookback in range(max_lookback_days):
                    check_dt = target_dt - np.timedelta64(lookback, 'D')
                    
                    day_indices = [
                        i for i, d in enumerate(cube_dates) 
                        if d == check_dt and cube_ccs[i] <= current_threshold
                    ]
                    if not day_indices: continue

                    day_indices = sorted(day_indices, key=lambda idx: cube_ccs[idx])
                    daily_stack = cube[day_indices]
                    
                    mask = ~np.isnan(daily_stack)
                    idx_first_valid = np.argmax(mask, axis=0) 
                    
                    rows, cols = np.indices(daily_stack.shape[2:])
                    # final_mosaic = np.zeros(daily_stack.shape[1:], dtype=daily_stack.dtype)
                    final_mosaic = np.full(
                        daily_stack.shape[1:],
                        np.nan,
                        dtype=daily_stack.dtype
                    )
                    for b in range(daily_stack.shape[1]):
                        final_mosaic[b] = daily_stack[idx_first_valid[b], b, rows, cols]

                    check_indices = [i for i, b_name in enumerate(bands) if b_name != "SCL"]
                    check_mosaic = final_mosaic[check_indices]
                    
                    if np.isfinite(check_mosaic).all() and not np.all(check_mosaic == 0):   
                        found_valid_mosaic = True
                        used_day_indices = day_indices
                        break
                
                if found_valid_mosaic: break

            if not found_valid_mosaic:
                if 'satellite' in day_grp:
                    del day_grp['satellite']
                continue

            # ---------------------------------------------------------
            # Calculate stats for the successful day
            # ---------------------------------------------------------
            stats_list = []
            total_pix = cube.shape[2] * cube.shape[3]
            window_ccs = cube_ccs[used_day_indices]
            selected_ccs = window_ccs[idx_first_valid[0]] 
            
            u_ccs, counts = np.unique(selected_ccs, return_counts=True)
            for cc, count in zip(u_ccs, counts):
                local_idx = np.where(window_ccs == cc)[0][0]
                global_date_idx = used_day_indices[local_idx]
                
                stats_list.append({
                    "pixels_pct": round((count / total_pix) * 100, 2),
                    "cloud_cover_pct": round(float(cc), 2),
                    "acquisition_date": str(cube_dates[global_date_idx])
                })

            # ---------------------------------------------------------
            # Save to H5 inside the dedicated 'satellite' group
            # ---------------------------------------------------------
            if 'satellite' in day_grp:
                del day_grp['satellite']
            sat_grp = day_grp.require_group('satellite')
            
            sat_grp.attrs["cloud_cover_stats"] = json.dumps(stats_list)
            
            for i, b_name in enumerate(bands):
                d_name = f"s2_{b_name.lower()}"
                if d_name in sat_grp: del sat_grp[d_name]
                
                if b_name == "SCL":
                    sat_grp.create_dataset(d_name, data=final_mosaic[i].astype(np.uint8), compression="lzf")
                else:
                    sat_grp.create_dataset(d_name, data=final_mosaic[i], compression="lzf")

def run_s2_h5_pipeline(h5_path, bands):
    """
    Coordinates downloading and slicing S2 data for all tiles in an H5 file.
    """
    
    pipeline_bands = list(bands)
    # if "SCL" not in pipeline_bands:
    #     pipeline_bands.append("SCL")

    print(f"\n{'='*60}\nRunning Tile-Based Sentinel-2 for: {h5_path.name}")

    with h5py.File(h5_path, 'r') as f:
        tile_ids = [k for k in f.keys() if k.startswith('tile_')]

    for tile_id in tile_ids:
        print(f"\n--- Processing {tile_id} ---")
        
        # Download
        cube, metadata = get_tile_master_cube(h5_path, tile_id, pipeline_bands)
        
        # Slice and Save
        if cube is not None:
            process_tile_h5_from_cube(h5_path, tile_id, cube, metadata, pipeline_bands)