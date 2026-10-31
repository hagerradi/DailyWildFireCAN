import pandas as pd
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import json
import os
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

def create_stratified_splits(df, train_split, random_state=42, overlap_mapper=None):
    """Takes a dataframe of fire coordinates, strictly groups overlapping fires to prevent
    data leakage, calculates combined bounding boxes/centroids, stratifies by size
    and geography, and returns train, val, and test ID lists.

    Args:
      df: the CFSD dataframe
      train_split: the size of the training subset in percentage
      random_state: the random seed (Default value = 42)
      overlap_mapper: the mapper defining the overlapping fires (Default value = None)

    Returns: the three lists of train, val and test IDs 

    """
    df_copy = df.copy()
    
    # ---------------------------------------------------------
    # UNION-FIND: Group overlapping fires into "Super-Fires"
    # ---------------------------------------------------------
    unique_fires = df_copy['ID'].astype(str).unique()
    
    # Initialize Union-Find dictionary (every fire is its own parent initially)
    parent = {f: f for f in unique_fires}
    
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Use the mapper to link fires that share a tile
    if overlap_mapper:
        for fires_list in overlap_mapper.values():
            fire_ids = [str(f['fire_id']) for f in fires_list if str(f['fire_id']) in parent]
            if len(fire_ids) > 1:
                first = fire_ids[0]
                for other in fire_ids[1:]:
                    union(first, other)

    # Assign every fire to its root "Super-Fire" ID
    df_copy['super_id'] = df_copy['ID'].astype(str).apply(lambda x: find(x))
    
    num_super_fires = df_copy['super_id'].nunique()
    print(f"Grouped {len(unique_fires)} individual fires into {num_super_fires} isolated Super-Fires to prevent data leaks.")

    # ---------------------------------------------------------
    # CALCULATE STRATIFICATION STATS ON SUPER-FIRES
    # ---------------------------------------------------------
    # Group by super_id so overlapping fires form one giant bounding box
    fire_stats = df_copy.groupby('super_id').agg(
        min_lat=('lat', 'min'),
        max_lat=('lat', 'max'),
        min_lon=('lon', 'min'),
        max_lon=('lon', 'max'),
        center_lat=('lat', 'mean'),  
        center_lon=('lon', 'mean')   
    ).reset_index()

    # Prevent 0-area calculations
    fire_stats['fire_size'] = (fire_stats['max_lat'] - fire_stats['min_lat'] + 1e-5) * \
                              (fire_stats['max_lon'] - fire_stats['min_lon'] + 1e-5)

    # Add 'Mega' to the Size Bins
    fire_stats['size_bin'] = pd.qcut(
        fire_stats['fire_size'], 
        q=[0, 0.50, 0.80, 0.95, 1.0], 
        labels=['Small', 'Med', 'Large', 'Mega'], 
        duplicates='drop'
    )

    # Create Geographic Bins
    fire_stats['lat_bin'] = pd.qcut(fire_stats['center_lat'], q=2, labels=['South', 'North'])
    fire_stats['lon_bin'] = pd.qcut(fire_stats['center_lon'], q=2, labels=['West', 'East'])

    # Combine them into a single stratify label
    fire_stats['stratify_group'] = fire_stats['size_bin'].astype(str) + "_" + \
                                   fire_stats['lat_bin'].astype(str) + "_" + \
                                   fire_stats['lon_bin'].astype(str)

    # A group needs at least 15 members to safely survive a 70/15/15 split.
    counts = fire_stats['stratify_group'].value_counts()
    rare_mask = fire_stats['stratify_group'].isin(counts[counts < 15].index)
    fire_stats.loc[rare_mask, 'stratify_group'] = fire_stats.loc[rare_mask, 'size_bin'].astype(str)

    # If the pure size group still has < 15 fires, force it to 'Large'
    final_counts = fire_stats['stratify_group'].value_counts()
    very_rare_mask = fire_stats['stratify_group'].isin(final_counts[final_counts < 15].index)
    fire_stats.loc[very_rare_mask, 'stratify_group'] = 'Large'

    # ---------------------------------------------------------
    # SPLIT THE SUPER-FIRES
    # ---------------------------------------------------------
    # First Split: Train vs Temp
    train_df, temp_df = train_test_split(
        fire_stats, 
        test_size=round(1.0 - train_split, 4),
        random_state=random_state, 
        stratify=fire_stats['stratify_group']
    )

    # If temp_df ends up with only 1 member for a specific group
    # Drop stratification for the second split.
    temp_counts = temp_df['stratify_group'].value_counts()
    if (temp_counts < 2).any():
        print("Warning: Some strata have < 2 members in the validation/test pool. Falling back to unstratified random split for Val/Test.")
        stratify_col = None
    else:
        stratify_col = temp_df['stratify_group']

    # Second Split: Val vs Test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=random_state, 
        stratify=stratify_col
    )

    # ---------------------------------------------------------
    # UNPACK BACK TO INDIVIDUAL FIRE IDs
    # ---------------------------------------------------------
    train_super_ids = train_df['super_id'].tolist()
    val_super_ids = val_df['super_id'].tolist()
    test_super_ids = test_df['super_id'].tolist()

    train_ids = df_copy[df_copy['super_id'].isin(train_super_ids)]['ID'].unique().tolist()
    val_ids = df_copy[df_copy['super_id'].isin(val_super_ids)]['ID'].unique().tolist()
    test_ids = df_copy[df_copy['super_id'].isin(test_super_ids)]['ID'].unique().tolist()

    print(f"Final Individual Fire Split -> Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    return train_ids, val_ids, test_ids

def create_splits_by_years(df, train_years, val_years, test_years, year_col='year'):
    """Splits fires strictly based on provided lists of years.
    
    Args:
      df: the CFSD dataframe.
      train_years: list of years for the training subset (e.g., [2018, 2019]).
      val_years: list of years for the validation subset (e.g., [2020]).
      test_years: list of years for the testing subset (e.g., [2021]).
      year_col: the name of the column containing the fire year (Default = 'year').

    Returns: 
      train_ids, val_ids, test_ids: three lists of individual fire IDs.
    """

    # Force the lists to be integers
    train_years = [int(y) for y in train_years]
    val_years = [int(y) for y in val_years]
    test_years = [int(y) for y in test_years]
    
    # Force the dataframe column to be integers during the check
    train_ids = df[df[year_col].astype(int).isin(train_years)]['ID'].unique().tolist()
    val_ids = df[df[year_col].astype(int).isin(val_years)]['ID'].unique().tolist()
    test_ids = df[df[year_col].astype(int).isin(test_years)]['ID'].unique().tolist()

    print(f"Strict Temporal Split -> Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    return train_ids, val_ids, test_ids

def create_rotational_splits(
        df, 
        n_train_months=3, 
        n_val_months=1, 
        n_test_months=1,
        id_col='ID', 
        year_col='year', 
        dob_col='DOB'
    ):
    """
    Splits fires chronologically using a rotating month sequence (e.g., 3 train, 1 val, 1 test).
    
    Rules:
      - Skips months with zero fire starts (inactive months do not count in the rotation).
      - Fire split is determined strictly by its start month (earliest record for that ID).
    
    Args:
      df: Pandas DataFrame containing fire data.
      n_train_months: Number of active months for training in each cycle (default 3).
      n_val_months: Number of active months for validation in each cycle (default 1).
      n_test_months: Number of active months for testing in each cycle (default 1).
      id_col: Name of the column containing unique Fire IDs (default 'ID').
      year_col: Name of the column containing the fire year (default 'year').
      dob_col: Name of the column containing Day of Burning / Day of Year (default 'DOB').
      
    Returns:
      train_ids, val_ids, test_ids: Three lists of unique fire IDs.
    """
    df_copy = df.copy()
    
    # Convert year + DOB to a full datetime, then to YYYY-MM Period
    # %Y%j parses Year + 3-digit Day of Year (e.g., 2021 + 045)
    df_copy['dates'] = pd.to_datetime(
        df_copy[year_col].astype(int).astype(str) + 
        df_copy[dob_col].astype(int).astype(str).str.zfill(3), 
        format='%Y%j'
    )
    df_copy['year_month'] = df_copy['dates'].dt.to_period('M')
    
    # Get the START month for every fire ID (earliest record per ID)
    fire_starts = df_copy.groupby(id_col)['year_month'].min().reset_index()
    
    # Extract ONLY active months (months where at least 1 fire started), sorted chronologically
    active_months = sorted(fire_starts['year_month'].unique())
    
    # Create the rotational pattern array (e.g., ['train', 'train', 'train', 'val', 'test'])
    rotation_pattern = (
        ['train'] * n_train_months + 
        ['val'] * n_val_months + 
        ['test'] * n_test_months
    )
    pattern_len = len(rotation_pattern)
    
    # Map each active month to a split using modulo indexing
    month_to_split = {
        month: rotation_pattern[idx % pattern_len] 
        for idx, month in enumerate(active_months)
    }
    
    # Assign split to each fire based on its start month
    fire_starts['split'] = fire_starts['year_month'].map(month_to_split)
    
    # Extract IDs for each split
    train_ids = fire_starts[fire_starts['split'] == 'train'][id_col].unique().tolist()
    val_ids = fire_starts[fire_starts['split'] == 'val'][id_col].unique().tolist()
    test_ids = fire_starts[fire_starts['split'] == 'test'][id_col].unique().tolist()
    
    # Print diagnostic overview
    print("="*60)
    print(f"Rotational Split Summary ({n_train_months} Train | {n_val_months} Val | {n_test_months} Test):")
    print(f"Total Active Months Processed: {len(active_months)}")
    print(f"Train Fires: {len(train_ids):,}")
    print(f"Val Fires:   {len(val_ids):,}")
    print(f"Test Fires:  {len(test_ids):,}")
    print("="*60)
    
    return train_ids, val_ids, test_ids

def _update_tallies(stats, feat_name, valid_pixels):
    """Helper function to cleanly update running sums (using float64 to prevent overflow).

    Args:
      stats: the stats dictionary
      feat_name: the list of features
      valid_pixels: the mask of valid pixels

    """
    # Force float64 math
    stats[feat_name]['sum'] += np.sum(valid_pixels, dtype=np.float64)
    stats[feat_name]['sum_sq'] += np.sum(valid_pixels ** 2, dtype=np.float64)
    stats[feat_name]['count'] += valid_pixels.size
    
    current_min = np.min(valid_pixels)
    current_max = np.max(valid_pixels)
    if current_min < stats[feat_name]['min']:
        stats[feat_name]['min'] = current_min
    if current_max > stats[feat_name]['max']:
        stats[feat_name]['max'] = current_max

def calculate_h5_statistics(h5_dir, mapper, train_ids, patch_size=256, stats_filename='dataset_stats.json', remove_missing_data=True):
    """Scans through H5 files based on UNIQUE (Tile, DOB) pairs to compute global stats.
    - Prevents double-counting overlapping fires.
    - Explicitly skips Satellite, NDVI, and EVI for stats generation.

    Args:
      h5_dir: the folder of H5 files.
      mapper: the mapper of overlapping fires
      train_ids: the list of training ids
      patch_size: the target grid size (Default value = 256)
      stats_filename: the json file's name to store the stats (Default value = 'dataset_stats.json')
      remove_missing_data: toggle to skip samples with missing data

    Returns:
      statistics dictionnary
    """
    
    stats = defaultdict(lambda: {
        'sum': 0.0, 
        'sum_sq': 0.0, 
        'count': 0, 
        'min': float('inf'), 
        'max': float('-inf')
    })

    # Track which static tiles we have already processed to avoid double counting
    processed_static_tiles = set()
    
    # Filter the mapper to ONLY include (Tile, DOB) keys that belong to training fires
    train_ids_set = set([str(x) for x in train_ids])
    
    valid_global_keys = []
    for global_key, fires_list in mapper.items():
        # If at least one fire in this tile/day is in the train set, we process the environment
        if any(str(f['fire_id']) in train_ids_set for f in fires_list):
            valid_global_keys.append((global_key, fires_list))

    print(f"Processing {len(valid_global_keys)} unique Tile-Day environments...")

    for global_key, fires_list in tqdm(valid_global_keys, desc="Calculating Stats"):
        
        # Unpack the global key (e.g., tile_200_126_DOB_2024_195)
        parts = global_key.split('_DOB_')
        tile_id = parts[0]
        
        # We just need to open ONE fire's H5 file to read the environment for this tile/day
        primary_fire = fires_list[0]
        fire_id = primary_fire['fire_id']
        day_key = primary_fire['day_key']
        
        file_path = os.path.join(h5_dir, f"fire_{fire_id}.h5")
        if not os.path.exists(file_path):
            continue
            
        # ========================================================
        # OPEN AND AUTO-CLOSE THE FILE
        # ========================================================
        try:
            with h5py.File(file_path, 'r', swmr=True) as f:
                
                # --- SIZE CHECK & QUALITY CHECK ---
                if f"{tile_id}/days/{day_key}" not in f:
                    continue
                    
                day_group = f[f"{tile_id}/days/{day_key}"]
                
                sample_feat = list(day_group['features'].keys())[0]
                h, w = day_group[f'features/{sample_feat}'].shape
                if h != patch_size or w != patch_size:
                    continue 
                    
                if remove_missing_data and "quality_mask" in day_group:
                    continue

                # --- PROCESS STATIC FEATURES (Run once per Tile) ---
                if tile_id not in processed_static_tiles and f"{tile_id}/static_features" in f:
                    static_grp = f[f"{tile_id}/static_features"]
                    for feat_name in static_grp.keys():
                        data = static_grp[feat_name][:]
                        valid_mask = (~np.isnan(data)) & (data != -9999.0)
                        
                        # Cast to float64
                        valid_pixels = data[valid_mask].astype(np.float64)
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)
                            
                    processed_static_tiles.add(tile_id)

                # --- PROCESS DYNAMIC FEATURES ---
                feat_grp = day_group.get('features')
                if feat_grp is not None:
                    for feat_name in feat_grp.keys():
                        
                        # SKIP NDVI AND EVI
                        if feat_name.lower() in ['ndvi', 'evi']:
                            continue
                            
                        data = feat_grp[feat_name][:]
                        valid_mask = ~np.isnan(data)
                        valid_pixels = data[valid_mask].astype(np.float64)
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)
                
                # --- PROCESS SATELLITE ---
                sat_grp = day_group.get('satellite')
                if sat_grp is not None:
                    
                    for feat_name in sat_grp.keys():

                        # if feat_name.lower() in ['s2_scl']:
                        if 'scl' in feat_name.lower():
                            
                            data = sat_grp[feat_name][:]
                            valid_mask = ~np.isnan(data)
                            valid_pixels = data[valid_mask].astype(np.float64)
                            
                            if len(valid_pixels) > 0:
                                _update_tallies(stats, feat_name, valid_pixels)
                        
                        # if 'visual' in feat_name.lower() and ADD_VISUAL_BAND:
                            
                        #     data = sat_grp[feat_name][:]
                        #     valid_mask = ~np.isnan(data)
                        #     valid_pixels = data[valid_mask].astype(np.float64)
                            
                        #     if len(valid_pixels) > 0:
                        #         _update_tallies(stats, feat_name, valid_pixels)
        
        except OSError:
            # Safely skips if a specific file happens to be corrupted
            continue

    # ==========================================
    # COMPUTE FINAL METRICS
    # ==========================================
    final_dict = {}
    for feat_name, tallies in stats.items():
        N = tallies['count']
        if N == 0:
            print(f"Warning: No valid data found for {feat_name}")
            continue
            
        mean = tallies['sum'] / N
        variance = (tallies['sum_sq'] / N) - (mean ** 2)
        std = np.sqrt(max(variance, 0.0))
        
        final_dict[feat_name] = {
            'mean': float(mean),
            'std': float(std + 1e-8), 
            'min': float(tallies['min']),
            'max': float(tallies['max']),
            'count': int(N)
        }

    with open(stats_filename, 'w') as out_file:
        json.dump(final_dict, out_file, indent=4)
        
    print(f"Done! Statistics saved to {stats_filename}.")
    return final_dict


def _get_expected_date(day_group) -> str:
    """Extracts the fire day date."""
    fire_datetime = day_group.attrs.get("noon_utc") or day_group.attrs.get("start_utc") or day_group.attrs.get("end_utc")
    if fire_datetime:
        if isinstance(fire_datetime, bytes):
            fire_datetime = fire_datetime.decode('utf-8')
        return fire_datetime.split("T")[0]
    return None

def _check_sat_quality(day_group, expected_date_str, max_cloud_cover_pct, max_sat_age_days) -> bool:
    """Validates that ALL satellite acquisitions meet thresholds."""
    if not expected_date_str or "satellite" not in day_group:
        return False
        
    sat_attrs = day_group["satellite"].attrs
    if "cloud_cover_stats" not in sat_attrs:
        return False
        
    stats_attr = sat_attrs["cloud_cover_stats"]
    if isinstance(stats_attr, bytes):
        stats_attr = stats_attr.decode('utf-8')
        
    try:
        stats_list = json.loads(stats_attr)
        if not stats_list:
            return False
            
        exp_dt = datetime.strptime(expected_date_str, "%Y-%m-%d")
        
        for stat in stats_list:
            cloud_pct = stat.get("cloud_cover_pct", 100.0)
            if cloud_pct > max_cloud_cover_pct:
                return False
                
            acq_date_str = stat.get("acquisition_date")
            if not acq_date_str:
                return False
                
            acq_dt = datetime.strptime(acq_date_str, "%Y-%m-%d")
            gap_days = (exp_dt - acq_dt).days
            
            if gap_days < 0 or gap_days > max_sat_age_days:
                return False
                
        return True
    except Exception:
        return False

def calculate_h5_statistics_filtered(h5_dir, mapper, 
                                     train_ids, 
                                     patch_size=256, 
                                     stats_filename='dataset_stats.json', 
                                     remove_missing_data=True,
                                     max_cloud_cover_pct=None,
                                     max_sat_age_days=None):
    
    """Scans through H5 files based on UNIQUE (Tile, DOB) pairs to compute global stats.
    - Prevents double-counting overlapping fires.
    - Explicitly skips Satellite, NDVI, and EVI for stats generation.

    Args:
      h5_dir: the folder of H5 files.
      mapper: the mapper of overlapping fires
      train_ids: the list of training ids
      patch_size: the target grid size (Default value = 256)
      stats_filename: the json file's name to store the stats (Default value = 'dataset_stats.json')
      remove_missing_data: toggle to skip samples with missing data

    Returns:
      statistics dictionnary
    """
    
    stats = defaultdict(lambda: {
        'sum': 0.0, 
        'sum_sq': 0.0, 
        'count': 0, 
        'min': float('inf'), 
        'max': float('-inf')
    })

    # Track which static tiles we have already processed to avoid double counting
    processed_static_tiles = set()
    
    # Filter the mapper to ONLY include (Tile, DOB) keys that belong to training fires
    train_ids_set = set([str(x) for x in train_ids])
    
    valid_global_keys = []
    for global_key, fires_list in mapper.items():
        # If at least one fire in this tile/day is in the train set, we process the environment
        if any(str(f['fire_id']) in train_ids_set for f in fires_list):
            valid_global_keys.append((global_key, fires_list))

    print(f"Processing {len(valid_global_keys)} unique Tile-Day environments...")

    for global_key, fires_list in tqdm(valid_global_keys, desc="Calculating Stats"):
        
        # Unpack the global key (e.g., tile_200_126_DOB_2024_195)
        parts = global_key.split('_DOB_')
        tile_id = parts[0]
        
        # We just need to open ONE fire's H5 file to read the environment for this tile/day
        primary_fire = fires_list[0]
        fire_id = primary_fire['fire_id']
        day_key = primary_fire['day_key']
        
        file_path = os.path.join(h5_dir, f"fire_{fire_id}.h5")
        if not os.path.exists(file_path):
            continue
            
        # ========================================================
        # OPEN AND AUTO-CLOSE THE FILE
        # ========================================================
        try:
            with h5py.File(file_path, 'r', swmr=True) as f:
                
                # --- SIZE CHECK & QUALITY CHECK ---
                if f"{tile_id}/days/{day_key}" not in f:
                    continue
                    
                day_group = f[f"{tile_id}/days/{day_key}"]
                
                sample_feat = list(day_group['features'].keys())[0]
                h, w = day_group[f'features/{sample_feat}'].shape
                if h != patch_size or w != patch_size:
                    continue 
                    
                if remove_missing_data and "quality_mask" in day_group:
                    continue

                # SATELLITE QUALITY CHECK
                if max_cloud_cover_pct is not None and max_sat_age_days is not None:
                    exp_date_str = _get_expected_date(day_group)
                    
                    # If it fails the check, we skip this tile-day entirely
                    if not _check_sat_quality(day_group, exp_date_str, max_cloud_cover_pct, max_sat_age_days):
                        continue

                # --- PROCESS STATIC FEATURES (Run once per Tile) ---
                if tile_id not in processed_static_tiles and f"{tile_id}/static_features" in f:
                    static_grp = f[f"{tile_id}/static_features"]
                    for feat_name in static_grp.keys():
                        data = static_grp[feat_name][:]
                        valid_mask = (~np.isnan(data)) & (data != -9999.0)
                        
                        # Cast to float64
                        valid_pixels = data[valid_mask].astype(np.float64)
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)
                            
                    processed_static_tiles.add(tile_id)

                # --- PROCESS DYNAMIC FEATURES ---
                feat_grp = day_group.get('features')
                if feat_grp is not None:
                    for feat_name in feat_grp.keys():
                        
                        # SKIP NDVI AND EVI
                        if feat_name.lower() in ['ndvi', 'evi']:
                            continue
                            
                        data = feat_grp[feat_name][:]
                        valid_mask = ~np.isnan(data)
                        valid_pixels = data[valid_mask].astype(np.float64)
                        
                        if len(valid_pixels) > 0:
                            _update_tallies(stats, feat_name, valid_pixels)
                
                # --- PROCESS SATELLITE ---
                sat_grp = day_group.get('satellite')
                if sat_grp is not None:
                    
                    for feat_name in sat_grp.keys():

                        # if feat_name.lower() in ['s2_scl']:
                        if 'scl' in feat_name.lower():
                            
                            data = sat_grp[feat_name][:]
                            valid_mask = ~np.isnan(data)
                            valid_pixels = data[valid_mask].astype(np.float64)
                            
                            if len(valid_pixels) > 0:
                                _update_tallies(stats, feat_name, valid_pixels)
                        
                        # if 'visual' in feat_name.lower() and ADD_VISUAL_BAND:
                            
                        #     data = sat_grp[feat_name][:]
                        #     valid_mask = ~np.isnan(data)
                        #     valid_pixels = data[valid_mask].astype(np.float64)
                            
                        #     if len(valid_pixels) > 0:
                        #         _update_tallies(stats, feat_name, valid_pixels)
        
        except OSError:
            # Safely skips if a specific file happens to be corrupted
            continue

    # ==========================================
    # COMPUTE FINAL METRICS
    # ==========================================
    final_dict = {}
    for feat_name, tallies in stats.items():
        N = tallies['count']
        if N == 0:
            print(f"Warning: No valid data found for {feat_name}")
            continue
            
        mean = tallies['sum'] / N
        variance = (tallies['sum_sq'] / N) - (mean ** 2)
        std = np.sqrt(max(variance, 0.0))
        
        final_dict[feat_name] = {
            'mean': float(mean),
            'std': float(std + 1e-8), 
            'min': float(tallies['min']),
            'max': float(tallies['max']),
            'count': int(N)
        }

    with open(stats_filename, 'w') as out_file:
        json.dump(final_dict, out_file, indent=4)
        
    print(f"Done! Statistics saved to {stats_filename}.")
    return final_dict