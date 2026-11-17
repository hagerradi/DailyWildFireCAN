from __future__ import annotations

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import glob
import h5py
import numpy as np
import pandas as pd 
from tqdm import tqdm
import gc
import json
from datetime import datetime
from collections import OrderedDict

from src.config import Config
from samples_generation.data_generator_utils import lonlat_to_canada_lambert, append_tile_coordinates, assign_spatial_regions_by_fire_id

from configs import settings

from samples_generation.samples_configs.samples_settings import DYNAMIC_FEATURES, STATIC_FEATURES, TARGET_YEARS, REMOVE_MISSING_DATA, HOLDOUT_YEAR, N_CLUSTERS

class H5FireSimpleDataset(Dataset):
    
    def __init__(self, h5_dir, id_list, mapper, dynamic_features, static_features=None, 
                 fire_feature='firearea', patch_size=256, 
                 use_cumuarea_prev=False, use_cumuarea=False, transform=None,
                 normalize=False, stats_dict=None, sat_nodata=0, 
                 use_cyclical_aspect=False, 
                 return_sat_age=False, 
                 return_coords=False,
                 return_date=True,
                 remove_missing_data=False,
                 cluster_map=None):
        
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

        self.cluster_map = cluster_map if cluster_map else {}
        
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

            primary_fire_id_str = str(primary_fire['fire_id'])
                    
            self.samples.append({
                'tile_id': tile_id,
                'dob': dob,
                'year': year,
                'cluster_id': self.cluster_map.get(primary_fire_id_str, -1),
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

def save_dataset_to_disk(dataset, output_base_folder):
    """
    Iterates through the dataset and saves each sample as a .npz file.
    Dynamically routes samples to 'cluster_X_YYYY' subfolders and saves all fire IDs.
    Ensures filenames are perfectly sequential inside each subfolder.
    """
    print(f"\nSaving ALL raw data to {output_base_folder} dynamically...")
    
    # NEW: Dictionary to track the sequential index for each specific folder
    folder_counters = {}
    
    # Iterate directly through the dataset and save
    for idx in tqdm(range(len(dataset)), desc="Generating Samples"):
        
        sample_tuple = dataset[idx]
        s_meta = dataset.samples[idx]
        
        # Extract routing information
        cluster_id = s_meta['cluster_id']
        year = s_meta['year']
        tile_id = str(s_meta['tile_id'])
        primary_fire_id = str(s_meta['primary_fire_id'])
        
        curr_ids = np.array(list(set([str(f['fire_id']) for f in s_meta['curr_fires']])), dtype=str)
        next_ids = np.array(list(set([str(f['fire_id']) for f in s_meta['next_fires']])), dtype=str)

        # Dynamically create the subfolder
        folder_name = f"cluster_{cluster_id}_{year}"
        split_folder = os.path.join(output_base_folder, folder_name)
        os.makedirs(split_folder, exist_ok=True)

        # Get the current local index for this folder, then increment it
        if folder_name not in folder_counters:
            folder_counters[folder_name] = 0
            
        local_idx = folder_counters[folder_name]
        folder_counters[folder_name] += 1

        # Prepare data tensors
        x_np = sample_tuple[0].numpy().astype(np.float16)
        y_np = sample_tuple[1].numpy().astype(np.uint8)

        save_kwargs = {
            'x': x_np, 
            'y': y_np,
            'tile_id': tile_id,
            'primary_fire_id': primary_fire_id,
            'curr_fire_ids': curr_ids,
            'next_fire_ids': next_ids
        }
        
        tuple_idx = 2

        if dataset.return_sat_age:
            save_kwargs['delta_t'] = sample_tuple[tuple_idx].numpy().astype(np.uint8)
            tuple_idx += 1

        if dataset.return_coords:
            save_kwargs['coords'] = sample_tuple[tuple_idx].numpy().astype(np.float32)
            tuple_idx += 1
        
        if dataset.return_date:
            save_kwargs['fire_date'] = sample_tuple[tuple_idx].numpy().astype(np.int32)
            tuple_idx += 1

        # Save using the sequential local_idx
        file_path = os.path.join(split_folder, f"sample_{local_idx}.npz")
        
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

def generate_simple_offline_data(config: Config, target_year: int) -> None:
    """
    Generates raw, unnormalized samples for a specific target_year,
    but clusters ALL years globally to ensure spatial consistency.
    """
    tc = config.training

    is_sat_age = True
    is_coords = True
    is_fire_date = True

    dynamic_feats = DYNAMIC_FEATURES
    static_feats = STATIC_FEATURES
    target_years = TARGET_YEARS

    print(f'-------- INFO (SAMPLES REBUTTAL - {target_year}) --------')
    print(dynamic_feats)
    print(static_feats)
    print('-----------------------')
    
    # ---------------------------------------------------------
    # LOAD GLOBAL MAPPERS & DATAFRAMES (FOR CONSISTENT CLUSTERING)
    # ---------------------------------------------------------
    print('Loading Offline Tile Mappers...')
    master_mapper = {}
    for year in target_years:
        mapper_path = os.path.join(settings.METADATA_FOLDER, f"tile_dob_mapper_{year}.json")
        if os.path.exists(mapper_path):
            with open(mapper_path, 'r') as f:
                master_mapper.update(json.load(f))

    print('Reading dataframes dynamically for global clustering...')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    dfs = [pd.read_csv(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv', usecols=columns_to_use) 
           for year in target_years if os.path.exists(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv')]
    
    fire_growth_combined = pd.concat(dfs, ignore_index=True)
    
    # ---------------------------------------------------------
    # GLOBAL SPATIAL PROCESSING
    # ---------------------------------------------------------
    print('lonlat_to_canada_lambert')
    fire_growth_combined, _ = lonlat_to_canada_lambert(fire_growth_combined, 
                                                       lon_col=settings.LON_COL, 
                                                       lat_col=settings.LAT_COL)
    
    print('append_tile_coordinates')
    fire_growth_combined = append_tile_coordinates(fire_growth_combined, 
                                                   grid_size=settings.GRID_SIZE, 
                                                   pixel_size=settings.PIXEL_SIZE, 
                                                   x_col=settings.X_COL, 
                                                   y_col=settings.Y_COL)

    print('assign_spatial_regions_by_fire_id (Global Clustering)')
    fire_growth_combined = assign_spatial_regions_by_fire_id(fire_growth_combined, 
                                                             fire_id_col='ID', 
                                                             n_regions=N_CLUSTERS)
    
    # Create a global mapping dictionary { 'fire_id': cluster_id }
    cluster_col_name = 'cluster_id' if 'cluster_id' in fire_growth_combined.columns else 'region_id'
    cluster_mapping = dict(zip(fire_growth_combined['ID'].astype(str), fire_growth_combined[cluster_col_name]))

    # ---------------------------------------------------------
    # FILTER FOR PARALLEL TARGET YEAR
    # ---------------------------------------------------------
    # Get exactly which IDs belong to the year this script is currently processing
    target_df = pd.read_csv(f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{target_year}/Firegrowth_pts_v1_1_{target_year}.csv', usecols=['ID'])
    target_year_ids = target_df['ID'].astype(str).unique()
    print(f"\nProcessing {len(target_year_ids)} unique fires for the year {target_year}...")

    # Free up memory before data generation
    del dfs, fire_growth_combined, target_df
    gc.collect()

    # ---------------------------------------------------------
    # INSTANTIATE MASTER DATASET (RAW / UNNORMALIZED)
    # ---------------------------------------------------------
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
        'return_date': is_fire_date,
        'cluster_map': cluster_mapping
    }

    print(f'\nInstantiating Master Dataset for {target_year}...')
    master_dataset = H5FireSimpleDataset(id_list=target_year_ids, **common_kwargs)
    print(f'Number of generated samples: {len(master_dataset)}')

    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------
    sample_out = master_dataset[0]
    print(f'\nVerification:')
    print(f'Shape of X : {sample_out[0].shape}')
    print(f'Shape of Y : {sample_out[1].shape}')
    
    out_idx = 2
    if master_dataset.return_sat_age:
        print(f'Satellite age : {sample_out[out_idx].item()} days')
        out_idx += 1
    
    if master_dataset.return_coords:
        coords = sample_out[out_idx].numpy()
        print(f'Coordinates   : [Lon: {coords[0]:.4f}, Lat: {coords[1]:.4f}]')
        out_idx += 1
    
    if master_dataset.return_date:
        f_date = sample_out[out_idx].numpy()
        print(f'Fire Date     : {f_date[0]}-{f_date[1]:02d}-{f_date[2]:02d}')

    # ---------------------------------------------------------
    # SAVE DYNAMICALLY TO DISK
    # ---------------------------------------------------------
    samples_folder = settings.SAMPLE_FOLDER_CLUSTERS
    save_dataset_to_disk(master_dataset, samples_folder)
    
    print(f"\nAll raw data for {target_year} successfully generated and saved!")

def get_fold_split(samples_base_dir, n_clusters, target_years, fold_id):
    """
    Generates the train, val, and test file lists for a specific fold_id.
    
    Args:
        samples_base_dir (str): Path to the generated sample folders (e.g., Cluster_v1).
        n_clusters (int): Total number of clusters (e.g., 10).
        target_years (list): The years to include in the cross-validation (e.g., [2017, ..., 2023]).
        fold_id (int): The specific fold to process (0 to n_clusters-1).
        
    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' file lists.
        int, int: The IDs of the val_cluster and test_cluster.
    """
    # 1. Ensure fold_id is valid (0 to 9)
    if not (0 <= fold_id < n_clusters):
        raise ValueError(f"fold_id must be between 0 and {n_clusters - 1}")
    
    # 2. Define cluster roles for this specific fold
    val_cluster = fold_id
    test_cluster = (fold_id + 1) % n_clusters
    train_clusters = [c for c in range(n_clusters) if c not in [val_cluster, test_cluster]]
    
    fold_files = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # 3. Stream through the generic target_years and route files
    for year in target_years:
        for c in range(n_clusters):
            folder_path = os.path.join(samples_base_dir, f"cluster_{c}_{year}")
            
            if os.path.exists(folder_path):
                # Grab all .npz files and sort them so everything is fully deterministic
                files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
                
                if c in train_clusters:
                    fold_files["train"].extend(files)
                elif c == val_cluster:
                    fold_files["val"].extend(files)
                elif c == test_cluster:
                    fold_files["test"].extend(files)
                    
    return fold_files, val_cluster, test_cluster

def get_channel_names(dynamic_features, static_features, use_cyclical_aspect=True):
    """
    Reconstructs the exact sequence of channel names present in the saved 'x' tensor.
    """
    channel_names = []
    
    # 1. Dynamic Features
    for feat in dynamic_features:
        channel_names.append(feat)
        
    # 2. Static Features
    for feat in static_features:
        if feat == 'landcover':
            # SCANFI: 8 one-hot channels (classes 1 to 8)
            for c in range(1, 9):
                channel_names.append(f"landcover_class_{c}")
                
        elif feat in ['ccrs_landcover', 'annual_ccrs_landcover']:
            # CCRS: 15 one-hot channels
            for c in range(1, 16):
                channel_names.append(f"{feat}_class_{c}")
                
        elif feat == 'annual_disturbance':
            # Disturbance: 6 one-hot channels (classes 1 to 6)
            for c in range(1, 7):
                channel_names.append(f"annual_disturbance_class_{c}")
                
        elif feat == 'aspect' and use_cyclical_aspect:
            channel_names.append("aspect_sin")
            channel_names.append("aspect_cos")
            
        else:
            # Standard scalar static features (dem, slope, hii, closure, prc_*, etc.)
            channel_names.append(feat)
            
    # Accumulated Fire Masks (Always added at the end)
    channel_names.append("curr_fire_mask")
    channel_names.append("fireday_grid")
    
    return channel_names


def calculate_fold_stats(
    train_files, 
    dynamic_features, 
    static_features, 
    use_cyclical_aspect=True
):
    """
    Computes leak-free statistics (mean, std, min, max) for a specific fold's training set.
    Keys the dictionary by explicit feature name and tags channels that require Z-score normalization.
    """
    channel_names = get_channel_names(dynamic_features, static_features, use_cyclical_aspect)
    num_channels = len(channel_names)
    
    # Pre-identify channels that do NOT require standard Z-score normalization
    skip_normalization_keywords = [
        '_class_',          # One-hot encoded features
        'aspect_sin',       # Cyclical features bounded in [-1, 1]
        'aspect_cos', 
        'curr_fire_mask',   # Binary mask [0, 1]
        'fireday_grid'      # Pre-scaled fireday grid
    ]
    
    # Initialize statistical accumulators (using float64 to avoid overflow)
    channel_sum = np.zeros(num_channels, dtype=np.float64)
    channel_sq_sum = np.zeros(num_channels, dtype=np.float64)
    total_pixels = 0
    
    ch_min = np.full(num_channels, np.inf, dtype=np.float64)
    ch_max = np.full(num_channels, -np.inf, dtype=np.float64)
    
    print(f"\nComputing statistics across {len(train_files)} training samples ({num_channels} channels)...")
    
    # Stream through numpy arrays
    for filepath in tqdm(train_files, desc="Calculating Stats"):
        # Load sample directly
        sample = np.load(filepath)
        x = sample['x'].astype(np.float32)  # Shape: (C, H, W)
        
        # Ensure channel count matches expected names
        if x.shape[0] != num_channels:
            raise ValueError(f"Mismatch: File has {x.shape[0]} channels, but expected {num_channels} from config!")
            
        # Flatten spatial dimensions per channel: (C, H*W)
        x_flat = x.reshape(num_channels, -1)
        
        # Accumulate sums and bounds
        channel_sum += x_flat.sum(axis=1)
        channel_sq_sum += (x_flat ** 2).sum(axis=1)
        total_pixels += x_flat.shape[1]
        
        ch_min = np.minimum(ch_min, x_flat.min(axis=1))
        ch_max = np.maximum(ch_max, x_flat.max(axis=1))
        
    # Compute final Mean and Standard Deviation
    means = channel_sum / total_pixels
    # Variance = E[X^2] - (E[X])^2
    variances = np.maximum(0.0, (channel_sq_sum / total_pixels) - (means ** 2))
    stds = np.sqrt(variances)
    
    # Protect against zero standard deviation (e.g., constant feature channels)
    stds[stds == 0.0] = 1.0
    
    # Build output metadata dictionary
    stats_dict = {}
    for i, name in enumerate(channel_names):
        # Determine whether this channel should be normalized in the DataLoader
        should_normalize = not any(kw in name for kw in skip_normalization_keywords)
        
        stats_dict[name] = {
            "channel_idx": i,
            "mean": float(means[i]),
            "std": float(stds[i]),
            "min": float(ch_min[i]),
            "max": float(ch_max[i]),
            "normalize": should_normalize
        }
        
    return stats_dict

def get_buffered_fire_ids(df_clustered, val_cluster, test_cluster, fire_id_col='ID', buffer_tiles=2):
    """
    Computes the buffer using the global dataframe and returns a set of Fire IDs 
    that are too close to the Val/Test sets and MUST be dropped from Train.
    """
    print(f"Calculating strict tile-level buffer zone (radius = {buffer_tiles} tiles)...")
    
    test_clusters = [val_cluster, test_cluster]
    
    # Extract unique physical tiles for Holdout (Val/Test) vs Train regions
    unique_tiles = df_clustered[['tile_col', 'tile_row', 'cluster_id', fire_id_col]].drop_duplicates(subset=['tile_col', 'tile_row'])
    
    test_tiles = unique_tiles[unique_tiles['cluster_id'].isin(test_clusters)]
    train_tiles = unique_tiles[~unique_tiles['cluster_id'].isin(test_clusters)]
    
    if test_tiles.empty or train_tiles.empty:
        return set()

    test_coords = test_tiles[['tile_col', 'tile_row']].values  
    train_coords = train_tiles[['tile_col', 'tile_row']].values  
    
    # Compute exact coordinate differences (Δcol, Δrow) using Chebyshev distance
    diffs = np.abs(train_coords[:, np.newaxis, :] - test_coords[np.newaxis, :, :])
    grid_dists = diffs.max(axis=2)
    
    # Find minimum tile distance from each train tile to ANY test/val tile
    min_dist_per_train_tile = grid_dists.min(axis=1)
    
    # Identify training tiles that fall within the buffer radius
    buffered_train_tiles = train_tiles[min_dist_per_train_tile <= buffer_tiles]
    
    # Get all fire IDs that touch these buffered tiles
    buffered_fire_ids = set(buffered_train_tiles[fire_id_col].astype(str).unique())
    
    print(f" -> Buffer identified {len(buffered_fire_ids)} fires to be dropped from Training.")
    
    return buffered_fire_ids

from tqdm import tqdm

def apply_buffer_to_train_files(train_files, buffered_fire_ids):
    """
    Scans the train .npz files and drops any that belong to the buffered fire IDs.
    """
    filtered_train_files = []
    
    print(f"Filtering {len(train_files)} training files against the buffer...")
    for filepath in tqdm(train_files, desc="Applying Buffer"):
        # We only load the metadata keys, which is extremely fast
        with np.load(filepath) as data:
            # Convert to string to match the set
            fire_id = str(data['primary_fire_id'])
            
        if fire_id not in buffered_fire_ids:
            filtered_train_files.append(filepath)
            
    dropped_count = len(train_files) - len(filtered_train_files)
    print(f" -> Dropped {dropped_count} samples due to 50km spatial overlap.")
    print(f" -> Final secure training set: {len(filtered_train_files)} samples.")
    
    return filtered_train_files

def generate_fold_metadata(fold_id):
    """
    Generates leak-free statistics and spatial buffer configurations for a specific fold.
    """
    # Isolate Cross-Validation Years (Remove Holdout Year)
    cv_years = [y for y in TARGET_YEARS if y != HOLDOUT_YEAR]
    
    print(f'-------- INFO (METADATA GENERATION - FOLD {fold_id}) --------')
    print(f"Cross-Validation Years: {cv_years}")
    print(f"Holdout Year Excluded: {HOLDOUT_YEAR}")
    print('------------------------------------------------------')
    
    # ---------------------------------------------------------
    # GLOBAL SPATIAL PROCESSING (To calculate buffer)
    # ---------------------------------------------------------
    print('Reading dataframes dynamically for global clustering...')
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    
    dfs = []
    for year in cv_years:
        csv_path = f'{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv'
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path, usecols=columns_to_use))
            
    fire_growth_combined = pd.concat(dfs, ignore_index=True)
    
    print('Applying: lonlat_to_canada_lambert')
    fire_growth_combined, _ = lonlat_to_canada_lambert(
        fire_growth_combined, lon_col=settings.LON_COL, lat_col=settings.LAT_COL
    )
    
    print('Applying: append_tile_coordinates')
    fire_growth_combined = append_tile_coordinates(
        fire_growth_combined, grid_size=settings.GRID_SIZE, 
        pixel_size=settings.PIXEL_SIZE, x_col=settings.X_COL, y_col=settings.Y_COL
    )

    print('Applying: assign_spatial_regions_by_fire_id (Global Clustering)')
    fire_growth_combined = assign_spatial_regions_by_fire_id(
        fire_growth_combined, fire_id_col='ID', n_regions=N_CLUSTERS
    )
    
    # ---------------------------------------------------------
    # FOLD SPLITS & BUFFERING
    # ---------------------------------------------------------
    print(f'\nExtracting files for Fold {fold_id}...')
    fold_files, val_cluster, test_cluster = get_fold_split(
        samples_base_dir=settings.SAMPLE_FOLDER_CLUSTERS, 
        n_clusters=N_CLUSTERS, 
        target_years=cv_years, 
        fold_id=fold_id
    )
    
    print(f"Fold {fold_id} -> Val Cluster: {val_cluster} | Test Cluster: {test_cluster}")
    
    buffered_fire_ids = get_buffered_fire_ids(
        df_clustered=fire_growth_combined, 
        val_cluster=val_cluster, 
        test_cluster=test_cluster, 
        fire_id_col='ID', 
        buffer_tiles=2
    )
    
    # Free up memory as the global dataframe is no longer needed
    del dfs, fire_growth_combined
    gc.collect()

    # ---------------------------------------------------------
    # FILTERING & STATISTICS
    # ---------------------------------------------------------
    filtered_train_files = apply_buffer_to_train_files(
        fold_files['train'], buffered_fire_ids
    )
    
    stats_dict = calculate_fold_stats(
        filtered_train_files, 
        dynamic_features=DYNAMIC_FEATURES, 
        static_features=STATIC_FEATURES
    )

    print("\n================ FINAL FOLD COUNTS ================")
    print(f"Train samples (after buffer): {len(filtered_train_files):,}")
    print(f"Val samples:                  {len(fold_files['val']):,}")
    print(f"Test samples:                 {len(fold_files['test']):,}")
    print("===================================================\n")
    
    # ---------------------------------------------------------
    # SAVE METADATA JSON
    # ---------------------------------------------------------
    metadata = {
        "fold_id": fold_id,
        "val_cluster": val_cluster,
        "test_cluster": test_cluster,
        "sample_counts": {
            "train": len(filtered_train_files),
            "val": len(fold_files['val']),
            "test": len(fold_files['test'])
        },
        "buffered_fire_ids": list(buffered_fire_ids),  # Convert set to list for JSON serialization
        "stats": stats_dict
    }
    
    os.makedirs(settings.METADATA_FOLDER, exist_ok=True)
    output_path = os.path.join(settings.METADATA_FOLDER, f"fold_{fold_id}_metadata.json")
    
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n Successfully saved fold {fold_id} metadata to {output_path}")