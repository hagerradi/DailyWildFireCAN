from __future__ import annotations

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import h5py
import numpy as np
import pandas as pd 
from tqdm import tqdm
import gc
import json
from datetime import datetime
from collections import OrderedDict

from src.config import Config
from samples_generation.data_generator_helper import calculate_h5_statistics
from samples_generation.data_generator_utils import lonlat_to_canada_lambert, append_tile_coordinates, assign_spatial_regions_by_fire_id, select_stratified_test_clusters, analyze_cluster_stats, apply_spatial_buffer, generate_experiment_splits

from configs import settings

from samples_generation.samples_configs.samples_settings import DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_YEARS, RANDOM_SEED, REMOVE_MISSING_DATA, HOLDOUT_YEAR, N_CLUSTERS

class H5FireSimpleDataset(Dataset):
    
    def __init__(self, h5_dir, id_list, mapper, dynamic_features, static_features=None, 
                 fire_feature='firearea', patch_size=256, 
                 use_cumuarea_prev=False, use_cumuarea=False, transform=None,
                 normalize=False, stats_dict=None, sat_nodata=0, 
                 use_cyclical_aspect=False, 
                 return_sat_age=False, 
                 return_coords=False,
                 return_date=True,
                 remove_missing_data=False):
        
        self.h5_dir = h5_dir
        self.id_list = [str(fid) for fid in id_list] 
        self.mapper = mapper
        
        self.dynamic_features = dynamic_features
        self.static_features = static_features if static_features else []
        self.fire_feature = fire_feature
        self.patch_size = patch_size
        
        # Accumulation flags
        self.use_cumuarea_prev = use_cumuarea_prev  
        self.use_cumuarea = use_cumuarea            
        self.transform = transform

        # Normalization params
        self.normalize = normalize
        self.stats_dict = stats_dict
        self.sat_nodata = sat_nodata
        
        self.use_cyclical_aspect = use_cyclical_aspect
        self.return_sat_age = return_sat_age
        self.return_date = return_date

        self.return_coords = return_coords
        self.center_idx = (self.patch_size // 2) - 1

        self.remove_missing_data = remove_missing_data
        
        # Safety check
        if self.normalize and self.stats_dict is None:
            raise ValueError("If normalize=True, you must provide the stats_dict!")
        
        # +2 to account for BOTH the binary fire mask AND the fireday grid
        base_channels = len(self.dynamic_features) + len(self.static_features) + 2
        
        # If we encode aspect, 1 channel becomes 2 (sin, cos), so we add 1 net channel
        if self.use_cyclical_aspect and 'aspect' in self.static_features:
            base_channels += 1

        # SCANFI: 1 string becomes 8 channels, so we add 7 net channels
        if 'landcover' in self.static_features:
            base_channels += 7
            
        # CCRS: 1 string becomes 15 channels, so we add 14 net channels
        if 'ccrs_landcover' in self.static_features:
            base_channels += 14

        # Annual CCRS: 1 string becomes 15 channels, so we add 14 net channels
        if 'annual_ccrs_landcover' in self.static_features:
            base_channels += 14
        
        # Annual Disturbance: 1 string becomes 6 channels (classes 1-6, dropping 0), so add 5 net channels
        if 'annual_disturbance' in self.static_features:
            base_channels += 5

        self.num_channels = base_channels

        # Maps the raw CCRS values (1-19) to continuous indices (0-14).
        # We map everything else (including 0 NoData) to index 15.
        self.ccrs_mapping = torch.full((256,), 15, dtype=torch.long)
        valid_ccrs_classes = [1, 2, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        for idx, class_val in enumerate(valid_ccrs_classes):
            self.ccrs_mapping[class_val] = idx
        
        self.samples = [] 
        self.open_h5_handles = OrderedDict()
        self.max_open_files = 700
        
        self._index_files()

    # Helper Methods
    def _get_expected_date(self, day_group) -> str | None:
        """Extracts the fire day date.

        Args:
          day_group: the key of the day in the H5 file

        Returns:
            the fire's date
        """
        fire_datetime = day_group.attrs.get("noon_utc") or day_group.attrs.get("start_utc") or day_group.attrs.get("end_utc")
        if fire_datetime:
            if isinstance(fire_datetime, bytes):
                fire_datetime = fire_datetime.decode('utf-8')
            return fire_datetime.split("T")[0]
        return None

    def _get_sat_age_days(self, day_group, expected_date_str) -> float:
        """Calculates the gap between fire day and satellite acquisition.

        Args:
          day_group: the key of the day in the H5 file
          expected_date_str: the date of the fireday

        Returns:
            the difference in days between the fireday and the day of the satellite image
        """
        if not expected_date_str or "satellite" not in day_group:
            return 0.0
            
        sat_attrs = day_group["satellite"].attrs
        if "cloud_cover_stats" not in sat_attrs:
            return 0.0
            
        stats_attr = sat_attrs["cloud_cover_stats"]
        if isinstance(stats_attr, bytes):
            stats_attr = stats_attr.decode('utf-8')
            
        try:
            stats_list = json.loads(stats_attr)
            exp_dt = datetime.strptime(expected_date_str, "%Y-%m-%d")
            
            min_gap = float('inf')
            found_valid = False
            
            for stat in stats_list:
                acq_date_str = stat.get("acquisition_date")
                if acq_date_str:
                    acq_dt = datetime.strptime(acq_date_str, "%Y-%m-%d")
                    gap_days = (exp_dt - acq_dt).days
                    
                    if gap_days >= 0 and gap_days < min_gap:
                        min_gap = gap_days
                        found_valid = True
            
            return float(min_gap) if found_valid else 0.0
            
        except Exception:
            return 0.0

    def _get_h5_handle(self, fire_id):
        """Helper to manage persistent file handles with an LRU capacity limit.

        Args:
          fire_id: the fire's ID

        Returns: the fire's file handle

        """
        file_path = os.path.join(self.h5_dir, f"fire_{fire_id}.h5")
        
        # If it's already open, mark it as 'recently used' and return it
        if file_path in self.open_h5_handles:
            self.open_h5_handles.move_to_end(file_path)
            return self.open_h5_handles[file_path]
            
        # If the cache is full, close and remove the oldest file
        if len(self.open_h5_handles) >= self.max_open_files:
            oldest_path, oldest_handle = self.open_h5_handles.popitem(last=False)
            try:
                oldest_handle.close()
            except Exception:
                pass
                
        # Open the new file, add it to the cache, and return it
        new_handle = h5py.File(file_path, 'r', swmr=True)
        self.open_h5_handles[file_path] = new_handle
        return new_handle
        
    def __del__(self):
        """Clean up all open file handles."""
        if hasattr(self, 'open_h5_handles'):
            for f in self.open_h5_handles.values():
                try:
                    f.close()
                except:
                    pass

    def _index_files(self):
        """Builds dataset samples directly from the offline mapper JSON."""
        
        for global_key, curr_fires_raw in tqdm(self.mapper.items(), desc="Indexing Tile Days from Mapper"):
            
            # Extract identifiers from the global key: tile_X_Y_DOB_YYYY_DDD
            parts = global_key.split('_DOB_')
            tile_id = parts[0]
            year_dob = parts[1].split('_')
            year = int(year_dob[0])
            dob = int(year_dob[1])
            
            # Filter the fires to ensure they are in our active split (train/val/test)
            curr_fires = [f for f in curr_fires_raw if f['fire_id'] in self.id_list]
            if not curr_fires:
                continue
                
            # Automatically infer Tomorrow's global key
            next_global_key = f"{tile_id}_DOB_{year}_{dob + 1}"
            next_fires_raw = self.mapper.get(next_global_key, [])
            next_fires = [f for f in next_fires_raw if f['fire_id'] in self.id_list]
            
            if not next_fires:
                continue # We skip if there's no ground truth label available for tomorrow
                
            # We open the H5 file for the FIRST fire in the list strictly to run our size/quality checks
            primary_fire = curr_fires[0]
            file_path = os.path.join(self.h5_dir, f"fire_{primary_fire['fire_id']}.h5")
            
            try:
                with h5py.File(file_path, 'r') as f:
                    curr_day_group = f[f"{tile_id}/days/{primary_fire['day_key']}"]
                    
                    # SIZE CHECK
                    sample_feat = list(curr_day_group['features'].keys())[0]
                    h, w = curr_day_group[f'features/{sample_feat}'].shape
                    if h != self.patch_size or w != self.patch_size:
                        continue 
                        
                    # QUALITY MASK CHECK
                    if self.remove_missing_data and "quality_mask" in curr_day_group:
                        continue

                    # EXTRACT THE FIRE DATE
                    exp_date_str = self._get_expected_date(curr_day_group)

                    date_arr = [0, 0, 0] # Default fallback
                    if exp_date_str:
                        try:
                            dt = datetime.strptime(exp_date_str, "%Y-%m-%d")
                            date_arr = [dt.year, dt.month, dt.day]
                        except Exception:
                            pass

                    # CALCULATE SAT AGE
                    delta_t_days = 0.0
                    if self.return_sat_age:
                        # exp_date_str = self._get_expected_date(curr_day_group)
                        delta_t_days = self._get_sat_age_days(curr_day_group, exp_date_str)

                    # EXTRACT COORDS 
                    center_lon, center_lat = 0.0, 0.0
                    if self.return_coords:
                        if f"{tile_id}/coords/theoretical_lon" in f:
                            center_lon = f[f"{tile_id}/coords/theoretical_lon"][self.center_idx, self.center_idx]
                            center_lat = f[f"{tile_id}/coords/theoretical_lat"][self.center_idx, self.center_idx]
                        
                    fireday = curr_day_group.attrs.get('fireday', -1)
                        
            except Exception as e:
                # File is missing or corrupted, skip
                continue

            # Pre-resolve paths deterministically based on prefix
            feature_paths, is_sat_flags = [], []
            for feat in self.dynamic_features:
                if feat.startswith('s2_'):
                    feature_paths.append(f"satellite/{feat}")
                    is_sat_flags.append(True)
                else:
                    feature_paths.append(f"features/{feat}")
                    is_sat_flags.append(False)
                    
            self.samples.append({
                'tile_id': tile_id,
                'dob': dob,
                'year': year,
                'fire_date': date_arr,
                'curr_fires': curr_fires,      # List of all overlapping fires for TODAY
                'next_fires': next_fires,      # List of all overlapping fires for TOMORROW
                'primary_fire_id': primary_fire['fire_id'],
                'primary_curr_key': primary_fire['day_key'],
                'primary_next_key': next_fires[0]['day_key'],
                'fireday': fireday,
                'feature_paths': feature_paths, 
                'is_sat_flags': is_sat_flags,
                'delta_t': delta_t_days,
                'center_lon': center_lon,
                'center_lat': center_lat
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tile_id = s['tile_id']
        
        # Pre-allocate input tensor with zeros safely
        x_tensor = torch.zeros((self.num_channels, self.patch_size, self.patch_size), dtype=torch.float32)
        
        # Load Environment from PRIMARY Fire (Because Topo/Weather/Sat are identical)
        f_primary = self._get_h5_handle(s['primary_fire_id'])
        curr_day_group = f_primary[f"{tile_id}/days/{s['primary_curr_key']}"]
        channel_idx = 0
        
        # Load Dynamics & Satellite
        for path, is_sat in zip(s['feature_paths'], s['is_sat_flags']):
            
            feat_name = path.split('/')[-1]
                    
            arr_patch = curr_day_group[path][:].astype(np.float32)
            
            if is_sat:
                
                if np.isnan(self.sat_nodata):
                    valid_mask = ~np.isnan(arr_patch)
                else:
                    valid_mask = (arr_patch != self.sat_nodata)

                if 'scl' not in feat_name.lower() and 'visual' not in feat_name.lower():
                    arr_patch = arr_patch * 0.0001
            else:
                valid_mask = ~np.isnan(arr_patch)
            
            # NORMALIZATION LOGIC
            if self.normalize and feat_name in self.stats_dict and feat_name != self.fire_feature and not is_sat:
                if feat_name not in ['ndvi', 'evi']:
                    mean = self.stats_dict[feat_name]['mean']
                    std = self.stats_dict[feat_name]['std']
                    arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std
            
            # NORMALIZATION LOGIC (SCL and visual)
            if self.normalize and feat_name in self.stats_dict and feat_name != self.fire_feature and is_sat:
                
                if 'scl' in feat_name.lower():
                    feat_max = self.stats_dict[feat_name]['max']
                    feat_min = self.stats_dict[feat_name]['min']
                    
                    # Prevent division by zero if an array is completely uniform
                    if feat_max > feat_min:
                        arr_patch[valid_mask] = (arr_patch[valid_mask] - feat_min) / (feat_max - feat_min)
                    else:
                        arr_patch[valid_mask] = 0.0

                if 'visual' in feat_name.lower():
                    feat_max = self.stats_dict[feat_name]['max']
                    feat_min = self.stats_dict[feat_name]['min']
                    
                    # Prevent division by zero if an array is completely uniform
                    if feat_max > feat_min:
                        arr_patch[valid_mask] = (arr_patch[valid_mask] - feat_min) / (feat_max - feat_min)
                    else:
                        arr_patch[valid_mask] = 0.0
                
            arr_patch[~valid_mask] = 0.0
            x_tensor[channel_idx] = torch.from_numpy(arr_patch)
            channel_idx += 1
            
        # Load Statics
        if self.static_features and f"{tile_id}/static_features" in f_primary:
            
            static_group = f_primary[f"{tile_id}/static_features"]
            
            for feature in self.static_features:

                # --- SCANFI LANDCOVER ONE-HOT ENCODING ---
                if feature == 'landcover':
                    # Load as Long (int64) which is required for PyTorch one_hot
                    arr_patch = static_group[feature][:].astype(np.int64)
                    arr_tensor = torch.from_numpy(arr_patch)
                    
                    # Map valid classes (1-8) to (0-7). Map 255 to temporary index 8.
                    mask_255 = (arr_tensor == 255)
                    mapped_tensor = torch.where(mask_255, torch.tensor(8), arr_tensor - 1)
                    
                    # One-hot encode into 9 channels (0 to 8). Shape becomes (H, W, 9)
                    one_hot = F.one_hot(mapped_tensor, num_classes=9)
                    
                    # Rearrange shape to (Channels, H, W) and drop the 9th channel (the 255s)
                    one_hot = one_hot[..., :8].permute(2, 0, 1).float()
                    
                    # Inject the 8 channels into the main x_tensor
                    x_tensor[channel_idx : channel_idx + 8] = one_hot
                    channel_idx += 8
                    continue
                # ---------------------------------------

                # --- CCRS LANDCOVER ONE-HOT ENCODING ---
                # elif feature == 'ccrs_landcover':
                elif feature in ['ccrs_landcover', 'annual_ccrs_landcover']:

                    arr_patch = static_group[feature][:].astype(np.int64)
                    arr_tensor = torch.from_numpy(arr_patch)
                    
                    # Instantly map the jumping values using our pre-built lookup tensor.
                    mapped_tensor = self.ccrs_mapping[arr_tensor]
                    
                    # One-hot encode into 16 channels, slice off the 16th (NoData)
                    one_hot = F.one_hot(mapped_tensor, num_classes=16)
                    one_hot = one_hot[..., :15].permute(2, 0, 1).float()
                    
                    x_tensor[channel_idx : channel_idx + 15] = one_hot
                    channel_idx += 15
                    continue
                # ---------------------------------------
                
                # --- ANNUAL DISTURBANCE ONE-HOT ENCODING ---
                elif feature == 'annual_disturbance':
                    arr_patch = static_group[feature][:].astype(np.int64)
                    arr_tensor = torch.from_numpy(arr_patch)
                    
                    # Map the NoData/Background (255) to index 6
                    # Shift active classes (1 to 6) down to indices (0 to 5)
                    mask_255 = (arr_tensor == 255)
                    mapped_tensor = torch.where(mask_255, torch.tensor(6), arr_tensor - 1)
                    
                    # One-hot encode into 7 channels (indices 0 through 6)
                    one_hot = F.one_hot(mapped_tensor, num_classes=7)
                    
                    # Drop the LAST channel (index 6) which holds the 255 background
                    # This leaves exactly 6 channels representing classes 1 through 6
                    one_hot = one_hot[..., :6].permute(2, 0, 1).float()
                    
                    x_tensor[channel_idx : channel_idx + 6] = one_hot
                    channel_idx += 6
                    continue
                # ---------------------------------------

                arr_patch = static_group[feature][:].astype(np.float32)

                # --- HUMAN INFLUENCE INDEX (HII) ---
                if feature == 'hii':
                    # Scale down the saved integer by 100 to get true 0-64 range
                    arr_patch = arr_patch / 100.0
                    
                    # Apply log1p transformation to squash the right-skew
                    arr_patch = np.log1p(arr_patch)
                        
                    x_tensor[channel_idx] = torch.from_numpy(arr_patch)
                    channel_idx += 1
                    continue
                # ---------------------------------------

                if feature == 'aspect' and self.use_cyclical_aspect:
                    aspect_rad = arr_patch * (np.pi / 180.0)
                    aspect_sin = np.sin(aspect_rad)
                    aspect_cos = np.cos(aspect_rad)
                    
                    x_tensor[channel_idx] = torch.from_numpy(aspect_sin)
                    channel_idx += 1
                    x_tensor[channel_idx] = torch.from_numpy(aspect_cos)
                    channel_idx += 1
                    continue

                # Percentage / Zero-Inflated Variables (Min-Max Scaling)
                elif feature.startswith('prc') or feature == 'closure':
                    valid_mask = ~np.isnan(arr_patch)
                    if self.normalize and feature in self.stats_dict:
                        f_min = self.stats_dict[feature]['min']
                        f_max = self.stats_dict[feature]['max']
                        
                        if f_max > f_min:
                            arr_patch[valid_mask] = (arr_patch[valid_mask] - f_min) / (f_max - f_min)
                        else:
                            arr_patch[valid_mask] = 0.0
                
                # Continuous Variables like Dem, Slope, Height, Biomass (Z-Score)
                elif self.normalize and feature in self.stats_dict and feature != self.fire_feature:
                    valid_mask = ~np.isnan(arr_patch)
                    mean = self.stats_dict[feature]['mean']
                    std = self.stats_dict[feature]['std']
                    arr_patch[valid_mask] = (arr_patch[valid_mask] - mean) / std

                
                x_tensor[channel_idx] = torch.from_numpy(arr_patch)
                channel_idx += 1

        # Process Fire Masks (Dynamic Aggregation of All Overlaps)
        curr_fire_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.bool)
        fireday_grid = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)
        
        for fire_dict in s['curr_fires']:
            fire_id = fire_dict['fire_id']
            curr_key = fire_dict['day_key']
            f_fire = self._get_h5_handle(fire_id)
            
            if self.use_cumuarea_prev:
                # Dynamically retrieve history specifically for this fire
                day_keys = sorted(list(f_fire[f"{tile_id}/days"].keys()))
                curr_idx = day_keys.index(curr_key)
                h_keys = day_keys[:curr_idx + 1]
                
                for h_key in h_keys:
                    h_group = f_fire[f"{tile_id}/days/{h_key}"]
                    h_arr = h_group[f"features/{self.fire_feature}"][:]
                    day_mask = (torch.from_numpy(h_arr) > 0)
                    
                    h_fireday = h_group.attrs.get('fireday', -1)
                    if h_fireday > 0:
                        new_burn_pixels = day_mask & ~curr_fire_mask
                        fireday_grid[new_burn_pixels] = float(h_fireday) / 100.0
                        
                    curr_fire_mask |= day_mask
            else:
                curr_arr = f_fire[f"{tile_id}/days/{curr_key}/features/{self.fire_feature}"][:]
                day_mask = (torch.from_numpy(curr_arr) > 0)
                
                c_fireday = f_fire[f"{tile_id}/days/{curr_key}"].attrs.get('fireday', -1)
                if c_fireday > 0:
                    new_burn_pixels = day_mask & ~curr_fire_mask
                    fireday_grid[new_burn_pixels] = float(c_fireday) / 100.0
                    
                curr_fire_mask |= day_mask
                
        x_tensor[-2] = curr_fire_mask.float()
        x_tensor[-1] = fireday_grid

        # Determine NEXT DAY mask (Output Label)
        raw_next_mask = torch.zeros((self.patch_size, self.patch_size), dtype=torch.bool)
        
        for fire_dict in s['next_fires']:
            fire_id = fire_dict['fire_id']
            next_key = fire_dict['day_key']
            
            f_fire = self._get_h5_handle(fire_id)
            next_arr = f_fire[f"{tile_id}/days/{next_key}/features/{self.fire_feature}"][:]
            raw_next_mask |= (torch.from_numpy(next_arr) > 0)
        
        y_tensor = torch.zeros((self.patch_size, self.patch_size), dtype=torch.float32)
        new_growth_mask = raw_next_mask & ~curr_fire_mask

        if self.use_cumuarea:
             y_tensor[curr_fire_mask] = 1.0 
             y_tensor[new_growth_mask] = 2.0
        else:
             y_tensor[new_growth_mask] = 1.0

        if self.transform:
            x_tensor = self.transform(x_tensor)

        out = [x_tensor, y_tensor]

        if self.return_sat_age:
            delta_t_tensor = torch.tensor(s['delta_t'], dtype=torch.float32)
            out.append(delta_t_tensor)
        
        if self.return_coords:
            # Shape: (2,) -> [Longitude, Latitude]
            coords_tensor = torch.tensor([s['center_lon'], s['center_lat']], dtype=torch.float32)
            out.append(coords_tensor)
        
        if self.return_date:
            # Shape: (3,) -> [Year, Month, Day]
            date_tensor = torch.tensor(s['fire_date'], dtype=torch.int32)
            out.append(date_tensor)
        
        return tuple(out)

def save_dataset_to_disk(dataset, output_base_folder, split_name):
    """Iterates through the dataset and saves each sample as a .pt file.
    Creates the necessary subfolders automatically.

    Args:
      dataset: the samples dataset
      output_base_folder: the output folder of the samples
      split_name: train, val, or test
    """
    # Create the specific subfolder
    split_folder = os.path.join(output_base_folder, split_name)
    os.makedirs(split_folder, exist_ok=True)
    
    print(f"\nSaving {split_name.upper()} split to {split_folder}...")
    
    # Iterate directly through the dataset and save
    for idx in tqdm(range(len(dataset)), desc=f"Generating {split_name}"):
        
        sample_tuple = dataset[idx]

        x_np = sample_tuple[0].numpy().astype(np.float16)
        y_np = sample_tuple[1].numpy().astype(np.uint8)

        save_kwargs = {'x': x_np, 'y': y_np}
        tuple_idx = 2

        if dataset.return_sat_age:
            save_kwargs['delta_t'] = sample_tuple[tuple_idx].numpy().astype(np.uint8)
            tuple_idx += 1

        # Keep coords as 32-bit float to preserve decimal precision
        if dataset.return_coords:
            save_kwargs['coords'] = sample_tuple[tuple_idx].numpy().astype(np.float32)
            tuple_idx += 1
        
        # Save date as standard 32-bit integers
        if dataset.return_date:
            save_kwargs['fire_date'] = sample_tuple[tuple_idx].numpy().astype(np.int32)
            tuple_idx += 1

        file_path = os.path.join(split_folder, f"sample_{idx}.npz")
        
        # np.savez(file_path, **save_kwargs)
        np.savez_compressed(file_path, **save_kwargs)

def get_retained_fire_ids(dataset):
    retained_ids = set()
    for s in dataset.samples:
        for f_dict in s['curr_fires']:
            retained_ids.add(f_dict['fire_id'])
        for f_dict in s['next_fires']:
            retained_ids.add(f_dict['fire_id'])
    retained_ids = sorted(list(retained_ids))
    return retained_ids

def generate_simple_offline_data(config: Config) -> None:
    """

    Args:
      config: Config: the config parameters
    """
    
    tc = config.training

    is_sat_age = True
    is_coords = True
    is_fire_date = True

    dynamic_feats = DYNAMIC_FEATURES
    static_feats = STATIC_FEATURES
    target_years = TARGET_YEARS

    print('-------- INFO (ADVANCED SAMPLES) --------')
    print(dynamic_feats)
    print(static_feats)
    print(target_years)
    print('-----------------------')
    
    # ---------------------------------------------------------
    # LOAD MAPPERS & DATAFRAMES
    # ---------------------------------------------------------
    print('Loading Offline Tile Mappers...')
    master_mapper = {}
    for year in target_years:
        mapper_path = os.path.join(settings.METADATA_FOLDER, f"tile_dob_mapper_{year}.json")
        if os.path.exists(mapper_path):
            with open(mapper_path, 'r') as f:
                master_mapper.update(json.load(f))

    print('Reading dataframes dynamically...')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    dfs = [pd.read_csv(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv', usecols=columns_to_use) 
           for year in target_years if os.path.exists(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv')]
    
    fire_growth_combined = pd.concat(dfs, ignore_index=True)
    valid_ids = fire_growth_combined['ID'].unique()
    print(len(valid_ids))

    retained_ids = valid_ids
    
    fire_growth_filtered = fire_growth_combined[fire_growth_combined['ID'].astype(str).isin(retained_ids)]

    print('lonlat_to_canada_lambert')
    fire_growth_filtered, _ = lonlat_to_canada_lambert(fire_growth_filtered, 
                                                       lon_col=settings.LON_COL, 
                                                       lat_col=settings.LAT_COL)
    print('\n')

    print('append_tile_coordinates')
    fire_growth_filtered = append_tile_coordinates(fire_growth_filtered, 
                                                   grid_size=settings.GRID_SIZE, 
                                                   pixel_size=settings.PIXEL_SIZE, 
                                                   x_col=settings.X_COL, 
                                                   y_col=settings.Y_COL)
    print('\n')

    print('assign_spatial_regions_by_fire_id')
    fire_growth_filtered =  assign_spatial_regions_by_fire_id(fire_growth_filtered, 
                                                              fire_id_col='ID', 
                                                              n_regions=N_CLUSTERS)
    print('\n')

    print('analyze_cluster_stats')
    cluster_summary = analyze_cluster_stats(fire_growth_filtered, holdout_year=2024)
    print('\n')

    print('select_stratified_test_clusters')
    test_clusters = select_stratified_test_clusters(cluster_summary, 
                                                    fire_growth_filtered, 
                                                    holdout_year=int(HOLDOUT_YEAR), 
                                                    fire_id_col='ID', 
                                                    min_fires=10, 
                                                    min_holdout_fires=30)
    print('\n')

    print('apply_spatial_buffer')
    fire_growth_filtered = apply_spatial_buffer(fire_growth_filtered, test_clusters, fire_id_col='ID', buffer_tiles=2)
    print('\n')

    print('generate_experiment_splits')
    splits_dict = generate_experiment_splits(fire_growth_filtered, 
                                             holdout_year=int(HOLDOUT_YEAR), 
                                             holdout_clusters=test_clusters)
    print('\n')

    train_ids = splits_dict['train_ids']
    val_ids = splits_dict['val_ids']
    test_space_ids = splits_dict['test_space_ids']
    test_time_ids = splits_dict['test_time_ids']
    test_spacetime_ids = splits_dict['test_spacetime_ids']

    del dfs, fire_growth_combined, fire_growth_filtered
    gc.collect()

    # Pretrain dataset
    pretrain_dataset = H5FireSimpleDataset(
            h5_dir=settings.H5_OUTPUT_FOLDER, 
            id_list=train_ids, 
            mapper=master_mapper,
            dynamic_features=dynamic_feats, 
            static_features=static_feats,
            fire_feature="cumuarea", 
            patch_size=settings.GRID_SIZE,
            use_cumuarea=tc.use_cumuarea, 
            use_cumuarea_prev=tc.use_cumuarea_prev,
            normalize=False, 
            stats_dict=None, 
            sat_nodata=np.nan,
            remove_missing_data=REMOVE_MISSING_DATA,
            return_coords=is_coords,
            return_date=is_fire_date
        )
    pretrain_ids = get_retained_fire_ids(pretrain_dataset)
    print(f'Pre-train IDS : {len(pretrain_ids)}')

    # ---------------------------------------------------------
    # CALCULATE STATS
    # ---------------------------------------------------------
    stats_dict = calculate_h5_statistics(
        settings.H5_OUTPUT_FOLDER, 
        master_mapper,
        # train_ids,
        pretrain_ids,
        stats_filename="simple_dataset_stats_8_years_reb_v1.json",
        remove_missing_data=REMOVE_MISSING_DATA
    )
    # stats_filepath = "simple_dataset_stats_8_years_reb_v1.json"
    # print(f"Loading statistics directly from {stats_filepath}...")
    # with open(stats_filepath, "r") as f:
    #     stats_dict = json.load(f)

    common_kwargs = {
        'h5_dir': settings.H5_OUTPUT_FOLDER, 
        'mapper': master_mapper,
        'dynamic_features': dynamic_feats, 
        'static_features': static_feats,
        'fire_feature': "cumuarea", 
        'patch_size': settings.GRID_SIZE,
        'use_cumuarea': tc.use_cumuarea, 
        'use_cumuarea_prev': tc.use_cumuarea_prev,
        'normalize': True, 
        'stats_dict': stats_dict, 
        'sat_nodata': np.nan,
        'use_cyclical_aspect': tc.use_cyclical_aspect, 
        'return_sat_age': is_sat_age,
        'remove_missing_data': REMOVE_MISSING_DATA,
        'return_coords': is_coords,
        'return_date': is_fire_date
    }

    # ---------------------------------------------------------
    # INSTANTIATE DATASETS AND SAVE TO DISK
    # ---------------------------------------------------------
    train_dataset = H5FireSimpleDataset(id_list=train_ids, **common_kwargs)
    print(f'Number of training samples {len(train_dataset)}')
    print(f'Trains IDS : {len(get_retained_fire_ids(train_dataset))}')

    val_dataset = H5FireSimpleDataset(id_list=val_ids, **common_kwargs)
    print(f'Number of validation samples {len(val_dataset)}')
    print(f'Val IDS : {len(get_retained_fire_ids(val_dataset))}')

    test_space_dataset = H5FireSimpleDataset(id_list=test_space_ids, **common_kwargs)
    print(f'Number of test space samples {len(test_space_dataset)}')
    print(f'Test-Space IDS : {len(get_retained_fire_ids(test_space_dataset))}')

    test_time_dataset = H5FireSimpleDataset(id_list=test_time_ids, **common_kwargs)
    print(f'Number of test time samples {len(test_time_dataset)}')
    print(f'Test-Time IDS : {len(get_retained_fire_ids(test_time_dataset))}')

    test_spacetime_dataset = H5FireSimpleDataset(id_list=test_spacetime_ids, **common_kwargs)
    print(f'Number of test spacetime samples {len(test_spacetime_dataset)}')
    print(f'Test-Space#Time IDS : {len(get_retained_fire_ids(test_spacetime_dataset))}')

    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------
    sample_out = test_spacetime_dataset[0]
    print(f'\nVerification:')
    print(f'Shape of X : {sample_out[0].shape}')
    print(f'Shape of Y : {sample_out[1].shape}')
    
    out_idx = 2
    if test_spacetime_dataset.return_sat_age:
        print(f'Satellite age : {sample_out[out_idx].item()} days')
        out_idx += 1
    
    if test_spacetime_dataset.return_coords:
        coords = sample_out[out_idx].numpy()
        print(f'Coordinates   : [Lon: {coords[0]:.4f}, Lat: {coords[1]:.4f}]')
        out_idx += 1
    
    if test_spacetime_dataset.return_date:
        f_date = sample_out[out_idx].numpy()
        print(f'Fire Date     : {f_date[0]}-{f_date[1]:02d}-{f_date[2]:02d}')

    # Save
    samples_folder = settings.SAMPLE_FOLDER_ADVANCED
    print(samples_folder)
    
    save_dataset_to_disk(train_dataset, samples_folder, "train")
    save_dataset_to_disk(val_dataset, samples_folder, "val")
    save_dataset_to_disk(test_space_dataset, samples_folder, "test_space")
    save_dataset_to_disk(test_time_dataset, samples_folder, "test_time")
    save_dataset_to_disk(test_spacetime_dataset, samples_folder, "test_spacetime")
    
    print("\nAll data successfully generated and saved to disk!")


def generate_metadata_single_dataset(dataset, split_name, output_folder, fire_sizes_dict):
    """
    Directly accesses dataset.samples to build the metadata catalog as a JSON.
    Includes both the primary fire ID, a list of ALL fires in that tile, and their terminal sizes.
    """
    metadata_list = []
    
    print(f"Extracting metadata for {split_name} (Total: {len(dataset.samples)} samples)...")
    
    for idx, s in enumerate(dataset.samples):
        
        # Format the date nicely
        year, month, day = s['fire_date']
        date_str = f"{year}-{month:02d}-{day:02d}"
        
        # Extract ALL fire IDs involved in this tile (current day + next day)
        curr_ids = [str(f['fire_id']) for f in s['curr_fires']]
        next_ids = [str(f['fire_id']) for f in s['next_fires']]
        all_fire_ids = sorted(list(set(curr_ids + next_ids)))
        
        # Look up the max size for each fire ID from the fast dictionary
        # Defaulting to 0.0 if for some reason it's missing
        all_fire_sizes = [float(fire_sizes_dict.get(fid, 0.0)) for fid in all_fire_ids]
        primary_fire_id = str(s['primary_fire_id'])
        
        metadata_list.append({
            "sample_index": idx,
            "filename": f"sample_{idx}.npz",
            "primary_fire_id": primary_fire_id,
            "primary_fire_size": float(fire_sizes_dict.get(primary_fire_id, 0.0)),
            "all_fire_ids": all_fire_ids,  
            "all_fire_sizes": all_fire_sizes, # <--- Parallel list of max cumuareas
            "year": int(s['year']),
            "date": date_str,
            "fireday": int(s['fireday']),
            "center_lon": float(s['center_lon']),
            "center_lat": float(s['center_lat']),
            "tile_id": str(s['tile_id']),
            "satellite_age_days": float(s['delta_t'])
        })
        
    os.makedirs(output_folder, exist_ok=True)
    
    json_path = os.path.join(output_folder, f"simple_{split_name}_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata_list, f, indent=4)
        
    print(f"Done! Saved JSON metadata to {json_path}")

def generate_metadata_all_datasets(config: Config):

    tc = config.training

    dynamic_feats = DYNAMIC_FEATURES
    static_feats = STATIC_FEATURES
    target_years = TARGET_YEARS

    # ---------------------------------------------------------
    # LOAD MAPPERS & DATAFRAMES
    # ---------------------------------------------------------
    print('Loading Offline Tile Mappers...')
    master_mapper = {}
    for year in target_years:
        mapper_path = os.path.join(settings.METADATA_FOLDER, f"tile_dob_mapper_{year}.json")
        if os.path.exists(mapper_path):
            with open(mapper_path, 'r') as f:
                master_mapper.update(json.load(f))

    print('Reading dataframes dynamically...')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    dfs = [pd.read_csv(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv', usecols=columns_to_use) 
           for year in target_years if os.path.exists(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv')]
    
    fire_growth_combined = pd.concat(dfs, ignore_index=True)
    valid_ids = fire_growth_combined['ID'].unique()
    print(len(valid_ids))


    retained_ids = valid_ids
    
    fire_growth_filtered = fire_growth_combined[fire_growth_combined['ID'].astype(str).isin(retained_ids)]

    print('lonlat_to_canada_lambert')
    fire_growth_filtered, _ = lonlat_to_canada_lambert(fire_growth_filtered, 
                                                       lon_col=settings.LON_COL, 
                                                       lat_col=settings.LAT_COL)
    print('\n')

    print('append_tile_coordinates')
    fire_growth_filtered = append_tile_coordinates(fire_growth_filtered, 
                                                   grid_size=settings.GRID_SIZE, 
                                                   pixel_size=settings.PIXEL_SIZE, 
                                                   x_col=settings.X_COL, 
                                                   y_col=settings.Y_COL)
    print('\n')

    print('assign_spatial_regions_by_fire_id')
    fire_growth_filtered =  assign_spatial_regions_by_fire_id(fire_growth_filtered, 
                                                              fire_id_col='ID', 
                                                              n_regions=N_CLUSTERS)
    print('\n')

    print('analyze_cluster_stats')
    cluster_summary = analyze_cluster_stats(fire_growth_filtered, holdout_year=2024)
    print('\n')

    print('select_stratified_test_clusters')
    test_clusters = select_stratified_test_clusters(cluster_summary, 
                                                    fire_growth_filtered, 
                                                    holdout_year=int(HOLDOUT_YEAR), 
                                                    fire_id_col='ID', 
                                                    min_fires=10, 
                                                    min_holdout_fires=30)
    print('\n')

    print('apply_spatial_buffer')
    fire_growth_filtered = apply_spatial_buffer(fire_growth_filtered, test_clusters, fire_id_col='ID', buffer_tiles=2)
    print('\n')

    print('generate_experiment_splits')
    splits_dict = generate_experiment_splits(fire_growth_filtered, 
                                             holdout_year=int(HOLDOUT_YEAR), 
                                             holdout_clusters=test_clusters)
    print('\n')

    train_ids = splits_dict['train_ids']
    val_ids = splits_dict['val_ids']
    test_space_ids = splits_dict['test_space_ids']
    test_time_ids = splits_dict['test_time_ids']
    test_spacetime_ids = splits_dict['test_spacetime_ids']

    print("Pre-computing max fire sizes for metadata...")
    # Group by ID, get the max cumuarea, and instantly turn it into a dict { '2024_188': 15000.5, ... }
    fire_sizes_dict = fire_growth_filtered.groupby('ID')['cumuarea'].max().to_dict()
    # Convert keys to strings to perfectly match the JSON formatting
    fire_sizes_dict = {str(k): float(v) for k, v in fire_sizes_dict.items()}
    # ------------------------------------

    del dfs, fire_growth_combined, fire_growth_filtered
    gc.collect()

    is_sat_age = True
    is_coords = True
    is_fire_date = True

    common_kwargs = {
        'h5_dir': settings.H5_OUTPUT_FOLDER, 
        'mapper': master_mapper,
        'dynamic_features': dynamic_feats, 
        'static_features': static_feats,
        'fire_feature': "cumuarea", 
        'patch_size': settings.GRID_SIZE,
        'use_cumuarea': tc.use_cumuarea, 
        'use_cumuarea_prev': tc.use_cumuarea_prev,
        'normalize': False, 
        'stats_dict': None, 
        'sat_nodata': np.nan,
        'use_cyclical_aspect': tc.use_cyclical_aspect, 
        'return_sat_age': is_sat_age,
        'remove_missing_data': REMOVE_MISSING_DATA,
        'return_coords': is_coords,
        'return_date': is_fire_date
    }

    # ---------------------------------------------------------
    # INSTANTIATE DATASETS AND SAVE TO DISK
    # ---------------------------------------------------------
    train_dataset = H5FireSimpleDataset(id_list=train_ids, **common_kwargs)
    print(f'Number of training samples {len(train_dataset)}')
    print(f'Trains IDS : {len(get_retained_fire_ids(train_dataset))}')
    generate_metadata_single_dataset(train_dataset, 'train', settings.METADATA_FOLDER, fire_sizes_dict)

    val_dataset = H5FireSimpleDataset(id_list=val_ids, **common_kwargs)
    print(f'Number of validation samples {len(val_dataset)}')
    print(f'Val IDS : {len(get_retained_fire_ids(val_dataset))}')
    generate_metadata_single_dataset(val_dataset, 'val', settings.METADATA_FOLDER, fire_sizes_dict)

    test_space_dataset = H5FireSimpleDataset(id_list=test_space_ids, **common_kwargs)
    print(f'Number of test space samples {len(test_space_dataset)}')
    print(f'Test-Space IDS : {len(get_retained_fire_ids(test_space_dataset))}')
    generate_metadata_single_dataset(test_space_dataset, 'test_space', settings.METADATA_FOLDER, fire_sizes_dict)

    test_time_dataset = H5FireSimpleDataset(id_list=test_time_ids, **common_kwargs)
    print(f'Number of test time samples {len(test_time_dataset)}')
    print(f'Test-Time IDS : {len(get_retained_fire_ids(test_time_dataset))}')
    generate_metadata_single_dataset(test_time_dataset, 'test_time', settings.METADATA_FOLDER, fire_sizes_dict)

    test_spacetime_dataset = H5FireSimpleDataset(id_list=test_spacetime_ids, **common_kwargs)
    print(f'Number of test spacetime samples {len(test_spacetime_dataset)}')
    print(f'Test-Space-Time IDS : {len(get_retained_fire_ids(test_spacetime_dataset))}')
    generate_metadata_single_dataset(test_spacetime_dataset, 'test_spacetime', settings.METADATA_FOLDER, fire_sizes_dict)

def generate_raw_temporal_labels(dataset, output_base_folder, split_name):
    """
    Creates a SEPARATE folder containing the pure, single-day labels for temporal uncertainty,
    PLUS a spatially dilated version of the standard target (Day T+1) for spatial uncertainty.
    """
    target_folder = os.path.join(output_base_folder, f"{split_name}_temporal_labels")
    os.makedirs(target_folder, exist_ok=True)

    print(f"\nGenerating labels (T, T+1, T+2, and Dilated T+1) for {split_name}...")

    # Helper function to get ONLY the raw mask for ONE specific day
    def get_single_day_raw_mask(tile_id, year, target_dob):
        mask = torch.zeros((dataset.patch_size, dataset.patch_size), dtype=torch.bool)
        day_key = f"{tile_id}_DOB_{year}_{target_dob}"
        
        # Get fires for this exact day
        fires = dataset.mapper.get(day_key, [])
        valid_fires = [f for f in fires if f['fire_id'] in dataset.id_list]

        # Read only this day's array from H5
        for f_dict in valid_fires:
            f_fire = dataset._get_h5_handle(f_dict['fire_id'])
            c_key = f_dict['day_key']
            c_arr = f_fire[f"{tile_id}/days/{c_key}/features/{dataset.fire_feature}"][:]
            mask |= (torch.from_numpy(c_arr) > 0)
            
        return mask.numpy().astype(np.uint8)

    for idx, s in enumerate(tqdm(dataset.samples, desc=f"{split_name} uncertainty labels")):
        tile_id = s['tile_id']
        year = s['year']
        dob = s['dob']

        # 1. Day T (minus 1 day)
        y_minus1 = get_single_day_raw_mask(tile_id, year, dob)
        
        # 2. Day T+1 (Standard Target)
        y_standard = get_single_day_raw_mask(tile_id, year, dob + 1)
        
        # 3. Day T+2 (plus 1 day)
        y_plus1 = get_single_day_raw_mask(tile_id, year, dob + 2)

        # ---------------------------------------------------------
        # 4. SPATIAL UNCERTAINTY: Dilate y_standard by 1 pixel
        # ---------------------------------------------------------
        # Convert to tensor, add batch/channel dims: (1, 1, H, W)
        y_std_tensor = torch.from_numpy(y_standard).float().unsqueeze(0).unsqueeze(0)
        
        # A 3x3 max pool with stride 1 dilates the active pixels by exactly 1 pixel
        y_dilated_tensor = F.max_pool2d(y_std_tensor, kernel_size=3, stride=1, padding=1)
        
        # Squeeze back to (H, W) and convert to uint8 numpy array
        y_dilated = y_dilated_tensor.squeeze().numpy().astype(np.uint8)

        # Save all FOUR in one separate file
        file_path = os.path.join(target_folder, f"sample_{idx}.npz")
        np.savez(
            file_path, 
            y_minus1=y_minus1,
            y=y_standard,
            y_plus1=y_plus1,
            y_dilated=y_dilated
        )

def generate_labels_all_datasets(config: Config):

    tc = config.training

    dynamic_feats = DYNAMIC_FEATURES
    static_feats = STATIC_FEATURES
    target_years = TARGET_YEARS

    # ---------------------------------------------------------
    # LOAD MAPPERS & DATAFRAMES
    # ---------------------------------------------------------
    print('Loading Offline Tile Mappers...')
    master_mapper = {}
    for year in target_years:
        mapper_path = os.path.join(settings.METADATA_FOLDER, f"tile_dob_mapper_{year}.json")
        if os.path.exists(mapper_path):
            with open(mapper_path, 'r') as f:
                master_mapper.update(json.load(f))

    print('Reading dataframes dynamically...')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    dfs = [pd.read_csv(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv', usecols=columns_to_use) 
           for year in target_years if os.path.exists(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv')]
    
    fire_growth_combined = pd.concat(dfs, ignore_index=True)
    valid_ids = fire_growth_combined['ID'].unique()
    print(len(valid_ids))


    retained_ids = valid_ids
    
    fire_growth_filtered = fire_growth_combined[fire_growth_combined['ID'].astype(str).isin(retained_ids)]

    print('lonlat_to_canada_lambert')
    fire_growth_filtered, _ = lonlat_to_canada_lambert(fire_growth_filtered, 
                                                       lon_col=settings.LON_COL, 
                                                       lat_col=settings.LAT_COL)
    print('\n')

    print('append_tile_coordinates')
    fire_growth_filtered = append_tile_coordinates(fire_growth_filtered, 
                                                   grid_size=settings.GRID_SIZE, 
                                                   pixel_size=settings.PIXEL_SIZE, 
                                                   x_col=settings.X_COL, 
                                                   y_col=settings.Y_COL)
    print('\n')

    print('assign_spatial_regions_by_fire_id')
    fire_growth_filtered =  assign_spatial_regions_by_fire_id(fire_growth_filtered, 
                                                              fire_id_col='ID', 
                                                              n_regions=N_CLUSTERS)
    print('\n')

    print('analyze_cluster_stats')
    cluster_summary = analyze_cluster_stats(fire_growth_filtered, holdout_year=2024)
    print('\n')

    print('select_stratified_test_clusters')
    test_clusters = select_stratified_test_clusters(cluster_summary, 
                                                    fire_growth_filtered, 
                                                    holdout_year=int(HOLDOUT_YEAR), 
                                                    fire_id_col='ID', 
                                                    min_fires=10, 
                                                    min_holdout_fires=30)
    print('\n')

    print('apply_spatial_buffer')
    fire_growth_filtered = apply_spatial_buffer(fire_growth_filtered, test_clusters, fire_id_col='ID', buffer_tiles=2)
    print('\n')

    print('generate_experiment_splits')
    splits_dict = generate_experiment_splits(fire_growth_filtered, 
                                             holdout_year=int(HOLDOUT_YEAR), 
                                             holdout_clusters=test_clusters)
    print('\n')

    train_ids = splits_dict['train_ids']
    val_ids = splits_dict['val_ids']
    test_space_ids = splits_dict['test_space_ids']
    test_time_ids = splits_dict['test_time_ids']
    test_spacetime_ids = splits_dict['test_spacetime_ids']

    del dfs, fire_growth_combined, fire_growth_filtered
    gc.collect()

    is_sat_age = True
    is_coords = True
    is_fire_date = True

    common_kwargs = {
        'h5_dir': settings.H5_OUTPUT_FOLDER, 
        'mapper': master_mapper,
        'dynamic_features': dynamic_feats, 
        'static_features': static_feats,
        'fire_feature': "cumuarea", 
        'patch_size': settings.GRID_SIZE,
        'use_cumuarea': tc.use_cumuarea, 
        'use_cumuarea_prev': tc.use_cumuarea_prev,
        'normalize': False, 
        'stats_dict': None, 
        'sat_nodata': np.nan,
        'use_cyclical_aspect': tc.use_cyclical_aspect, 
        'return_sat_age': is_sat_age,
        'remove_missing_data': REMOVE_MISSING_DATA,
        'return_coords': is_coords,
        'return_date': is_fire_date
    }

    print('Generating labels')

    samples_folder = settings.SAMPLE_FOLDER_ADVANCED
    print(samples_folder)

    # ---------------------------------------------------------
    # INSTANTIATE DATASETS AND SAVE TO DISK
    # ---------------------------------------------------------
    train_dataset = H5FireSimpleDataset(id_list=train_ids, **common_kwargs)
    print(f'Number of training samples {len(train_dataset)}')
    print(f'Trains IDS : {len(get_retained_fire_ids(train_dataset))}')
    generate_raw_temporal_labels(train_dataset, samples_folder, 'train')

    val_dataset = H5FireSimpleDataset(id_list=val_ids, **common_kwargs)
    print(f'Number of validation samples {len(val_dataset)}')
    print(f'Val IDS : {len(get_retained_fire_ids(val_dataset))}')
    generate_raw_temporal_labels(val_dataset, samples_folder, 'val')

    test_space_dataset = H5FireSimpleDataset(id_list=test_space_ids, **common_kwargs)
    print(f'Number of test space samples {len(test_space_dataset)}')
    print(f'Test-Space IDS : {len(get_retained_fire_ids(test_space_dataset))}')
    generate_raw_temporal_labels(test_space_dataset, samples_folder, 'test_space')

    test_time_dataset = H5FireSimpleDataset(id_list=test_time_ids, **common_kwargs)
    print(f'Number of test time samples {len(test_time_dataset)}')
    print(f'Test-Time IDS : {len(get_retained_fire_ids(test_time_dataset))}')
    generate_raw_temporal_labels(test_time_dataset, samples_folder, 'test_time')

    test_spacetime_dataset = H5FireSimpleDataset(id_list=test_spacetime_ids, **common_kwargs)
    print(f'Number of test spacetime samples {len(test_spacetime_dataset)}')
    print(f'Test-Space-Time IDS : {len(get_retained_fire_ids(test_spacetime_dataset))}')
    generate_raw_temporal_labels(test_spacetime_dataset, samples_folder, 'test_spacetime')