import os
import glob
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from skimage import morphology

from configs import settings

from samples_generation.samples_configs.samples_settings import HOLDOUT_YEAR, TARGET_YEARS, N_CLUSTERS, STATIC_FEATURES, DYNAMIC_FEATURES
from src.config import Config

def get_channel_names(dynamic_features, static_features, use_cyclical_aspect=True):
    """Dynamically reconstructs the channel order based on the config features lists."""
    channel_names = []
    
    # Dynamic Features
    for feat in dynamic_features:
        channel_names.append(feat)
        
    # Static Features
    for feat in static_features:
        if feat == 'landcover':
            for c in range(1, 9): channel_names.append(f"landcover_class_{c}")
        elif feat in ['ccrs_landcover', 'annual_ccrs_landcover']:
            for c in range(1, 16): channel_names.append(f"{feat}_class_{c}")
        elif feat == 'annual_disturbance':
            for c in range(1, 7): channel_names.append(f"annual_disturbance_class_{c}")
        elif feat == 'aspect' and use_cyclical_aspect:
            channel_names.append("aspect_sin")
            channel_names.append("aspect_cos")
        else:
            # This naturally includes 'hii' as long as it is in your config!
            channel_names.append(feat)
            
    # Fire Masks
    channel_names.append("curr_fire_mask")
    channel_names.append("fireday_grid")
    
    return channel_names


class CrossValFireDataset(Dataset):
    def __init__(self, 
                 files: list,
                 dynamic_features: list,
                 static_features: list,
                 stats_dict: dict = None,
                 return_sat_age: bool = False, 
                 return_coords: bool = False,
                 return_loc_emb: bool = False,
                 return_olmo_emb: bool = False,
                 return_alpha_emb: bool = False):
        
        self.files = files
        self.stats_dict = stats_dict
        
        self.return_sat_age = return_sat_age
        self.return_coords = return_coords
        self.return_loc_emb = return_loc_emb
        self.return_olmo_emb = return_olmo_emb
        self.return_alpha_emb = return_alpha_emb

        # Generate the dynamic list exactly based on config (includes hii)
        self.channel_names = get_channel_names(dynamic_features, static_features, use_cyclical_aspect=True)
        
        # TEMP HARDCODING: removinh HII
        drop_feature_names = ['hii']
        self.keep_indices = [
            i for i, name in enumerate(self.channel_names) 
            if name not in drop_feature_names
        ]

    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = self.files[idx]
        
        has_delta, has_coords, has_emb, has_olmo, has_alpha = False, False, False, False, False

        with np.load(file_path) as data:
            x_np = data['x'].astype(np.float32)
            y_np = data['y'] 

            if self.return_sat_age and 'delta_t' in data:
                delta_t_np = data['delta_t']
                has_delta = True
            if self.return_coords and 'coords' in data:
                coords_np = data['coords']
                has_coords = True
            if self.return_loc_emb and 'loc_emb' in data:
                loc_emb_np = data['loc_emb']
                has_emb = True
            if self.return_olmo_emb and 'olmo_emb' in data:
                olmo_emb_np = data['olmo_emb']
                has_olmo = True
            if self.return_alpha_emb and 'alpha_emb' in data:
                alpha_emb_np = data['alpha_emb']
                has_alpha = True

        # ---------------------------------------------------------
        # APPLY NORMALIZATION
        # ---------------------------------------------------------
        if self.stats_dict is not None:
            for c_idx, feat_name in enumerate(self.channel_names):
                
                if feat_name not in self.stats_dict:
                    continue

                if feat_name.startswith('s2_'):
                    if 'scl' in feat_name.lower() or 'visual' in feat_name.lower():
                        f_min = self.stats_dict[feat_name]['min']
                        f_max = self.stats_dict[feat_name]['max']
                        if f_max > f_min:
                            x_np[c_idx] = (x_np[c_idx] - f_min) / (f_max - f_min)
                        else:
                            x_np[c_idx] = 0.0
                    else:
                        pass
                
                elif feat_name in ['ndvi', 'evi']:
                    pass
                
                elif feat_name.startswith('prc') or feat_name == 'closure':
                    f_min = self.stats_dict[feat_name]['min']
                    f_max = self.stats_dict[feat_name]['max']
                    if f_max > f_min:
                        x_np[c_idx] = (x_np[c_idx] - f_min) / (f_max - f_min)
                    else:
                        x_np[c_idx] = 0.0
                
                elif any(k in feat_name for k in ['_class_', 'aspect_', 'mask', 'hii', 'fireday']):
                    pass # HII already got log1p scaling in .npz generation

                else:
                    mean = self.stats_dict[feat_name]['mean']
                    std = self.stats_dict[feat_name]['std']
                    x_np[c_idx] = (x_np[c_idx] - mean) / std

        # TEMP HARDCODING: removing HII
        x_np = x_np[self.keep_indices, :, :]
        
        # ---------------------------------------------------------
        # MASK HOLE-FILLING
        # ---------------------------------------------------------
        y_bool = y_np > 0 
        y_filled_np = morphology.remove_small_holes(y_bool, area_threshold=1)
        y_filled_np = y_filled_np.astype(y_np.dtype)

        x_tensor = torch.from_numpy(x_np).float() 
        y_tensor = torch.from_numpy(y_filled_np).float()

        sample = {
            "input_grids": x_tensor,
            "label": y_tensor
        }
        
        if has_delta: sample["satellite_age"] = torch.from_numpy(delta_t_np).float()
        if has_coords: sample["coords"] = torch.from_numpy(coords_np).float()
        if has_emb: sample["loc_emb"] = torch.from_numpy(loc_emb_np).float()
        if has_olmo: sample["olmo_emb"] = torch.from_numpy(olmo_emb_np).float()
        if has_alpha: sample["alpha_emb"] = torch.from_numpy(alpha_emb_np).float()
            
        return sample


def get_dataloaders_clusters(config: Config, 
                             fold_id: int,
                             is_sat_age: bool, 
                             is_coords: bool = False, 
                             is_loc_emb: bool = False, 
                             is_olmo_emb: bool = False,
                             is_alpha_emb: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiates the Cross-Validation DataLoaders for a specific fold."""
    
    tc = config.training
    target_years = [y for y in TARGET_YEARS if y != HOLDOUT_YEAR]
    n_clusters = N_CLUSTERS
    
    samples_base_dir = settings.SAMPLE_FOLDER_CLUSTERS
    print(f'Samples base path : {samples_base_dir}')

    # if is_olmo_emb:
    #     samples_base_dir = settings.CLUSTER_FOLDER.replace("v1", "v2")
    #     print(f'Samples path switched to : {samples_base_dir}')

    json_path = os.path.join(settings.METADATA_FOLDER, f"fold_{fold_id}_metadata.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
        
    print(f"Loading Fold {fold_id} Configuration from {json_path}...")
    with open(json_path, 'r') as f:
        meta = json.load(f)

    val_cluster = meta['val_cluster']
    test_cluster = meta['test_cluster']
    buffered_fire_ids = set(meta['buffered_fire_ids'])
    stats_dict = meta['stats']

    import glob
    raw_train_files, val_files, test_files = [], [], []
    train_clusters = [c for c in range(n_clusters) if c not in [val_cluster, test_cluster]]

    for year in target_years:
        for c in range(n_clusters):
            folder_path = os.path.join(samples_base_dir, f"cluster_{c}_{year}")
            if os.path.exists(folder_path):
                files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
                if c in train_clusters:
                    raw_train_files.extend(files)
                elif c == val_cluster:
                    val_files.extend(files)
                elif c == test_cluster:
                    test_files.extend(files)

    train_files = []
    print(f"Applying spatial buffer filtering to {len(raw_train_files)} train files...")
    
    for filepath in tqdm(raw_train_files, desc="Dropping buffered samples"):
        with np.load(filepath, allow_pickle=True) as data:
            
            # Extract both current and next day fire IDs
            curr_ids = data['curr_fire_ids']
            next_ids = data['next_fire_ids']
            
            # Combine them and convert to strings
            sample_fire_ids = [str(fid) for fid in curr_ids] + [str(fid) for fid in next_ids]
            
            # Check if ANY of this sample's involved fires touch the buffer zone
            has_buffer_overlap = any(fid in buffered_fire_ids for fid in sample_fire_ids)
            
            # Only keep the file if NONE of its fires touch the buffer
            if not has_buffer_overlap:
                train_files.append(filepath)

    print(f'Number of offline CV Training samples   :: {len(train_files)}')
    print(f'Number of offline CV Validation samples :: {len(val_files)}')
    print(f'Number of offline CV Test samples       :: {len(test_files)}')

    dataset_kwargs = {
        "dynamic_features": DYNAMIC_FEATURES,
        "static_features": STATIC_FEATURES,
        "stats_dict": stats_dict,
        "return_sat_age": is_sat_age,
        "return_coords": is_coords,
        "return_loc_emb": is_loc_emb,
        "return_olmo_emb": is_olmo_emb,
        "return_alpha_emb": is_alpha_emb
    }

    train_dataset = CrossValFireDataset(train_files, **dataset_kwargs)
    val_dataset = CrossValFireDataset(val_files, **dataset_kwargs)
    test_dataset = CrossValFireDataset(test_files, **dataset_kwargs)

    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------
    if len(test_dataset) > 0:
        print('\n--- Verification of Sample 0 ---')
        
        sample_data = test_dataset[0]
        x = sample_data['input_grids']
        print(f'Input Tensor Shape : {x.shape}')

        if is_sat_age and 'satellite_age' in sample_data:
            sat_age = sample_data['satellite_age']
            print(f'Satellite Age Tensor : {sat_age}')

        if is_coords and 'coords' in sample_data:
            coords = sample_data['coords']
            print(f'Coordinates Tensor : {coords.tolist()}')
        
        if is_loc_emb:
            loc_emb = sample_data.get('loc_emb', None)
            if loc_emb is not None: print(f"Location Embedding Tensor Shape: {loc_emb.shape}")
        
        if is_olmo_emb:
            olmo_emb = sample_data.get('olmo_emb', None)
            if olmo_emb is not None: print(f"Olmo Embedding Tensor Shape: {olmo_emb.shape}")

        if is_alpha_emb:
            alpha_emb = sample_data.get('alpha_emb', None)
            if alpha_emb is not None: print(f"Alpha Embedding Tensor Shape: {alpha_emb.shape}")

        print("Channel Ranges (Min -> Max) for Sample 0:")
        for i in range(x.shape[0]):
            chan_min = x[i].min().item()
            chan_max = x[i].max().item()
            print(f"  Channel [{i:02d}]: {chan_min:>8.4f}  ->  {chan_max:>8.4f}")
        print('--------------------------------\n')

    train_loader = DataLoader(train_dataset, batch_size=tc.batch_size, shuffle=True, num_workers=tc.num_workers, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers, persistent_workers=False)
    test_loader = DataLoader(test_dataset, batch_size=tc.batch_size, shuffle=False, num_workers=tc.num_workers, persistent_workers=False)
    
    return train_loader, val_loader, test_loader

def get_2024_eval_dataloader(config: Config, 
                             fold_id: int,
                             is_sat_age: bool = False, 
                             is_coords: bool = False, 
                             is_loc_emb: bool = False, 
                             is_olmo_emb: bool = False,
                             is_alpha_emb: bool = False) -> DataLoader:
    
    """Instantiates the Holdout (2024) DataLoader using a specific fold's normalization stats."""
    
    tc = config.training
    samples_base_dir = settings.SAMPLE_FOLDER_CLUSTERS

    # if is_olmo_emb:
    #     samples_base_dir = settings.SAMPLE_FOLDER_CLUSTERS.replace("v1", "v2")

    # Load the stats for this specific fold
    json_path = os.path.join(settings.METADATA_FOLDER, f"fold_{fold_id}_metadata.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")
        
    with open(json_path, 'r') as f:
        meta = json.load(f)
    stats_dict = meta['stats']

    # Gather all files for the HOLDOUT_YEAR (2024) across all clusters
    eval_files = []
    
    for c in range(N_CLUSTERS):
        folder_path = os.path.join(samples_base_dir, f"cluster_{c}_{HOLDOUT_YEAR}")
        if os.path.exists(folder_path):
            files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))
            eval_files.extend(files)

    print(f"Fold {fold_id} Eval: Loaded {len(eval_files)} samples for year {HOLDOUT_YEAR}.")

    # Instantiate your exact existing Dataset
    dataset_kwargs = {
        "dynamic_features": DYNAMIC_FEATURES,
        "static_features": STATIC_FEATURES,
        "stats_dict": stats_dict,
        "return_sat_age": is_sat_age,
        "return_coords": is_coords,
        "return_loc_emb": is_loc_emb,
        "return_olmo_emb": is_olmo_emb,
        "return_alpha_emb": is_alpha_emb
    }

    eval_dataset = CrossValFireDataset(eval_files, **dataset_kwargs)
    
    # Return DataLoader
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False, 
        num_workers=tc.num_workers, 
        persistent_workers=False
    )
    
    return eval_loader

if __name__ == "__main__":
    import time
    from tqdm import tqdm
    from src.config import Config

    print("=== STARTING DATALOADER SPEED TEST ===")
    
    # Load config
    config = Config.from_yaml("configs/default.yaml")
    
    # Set the fold you want to test
    fold_id = 0
    
    # Measure instantiation time (Metadata loading + File filtering)
    start_time = time.time()
    print(f"\n[1] Instantiating DataLoaders for Fold {fold_id}...")
    
    train_loader, val_loader, test_loader = get_dataloaders_clusters(
        config=config,
        fold_id=fold_id,
        is_sat_age=False,
        is_coords=False,
        is_loc_emb=False,
        is_olmo_emb=False,
        is_alpha_emb=False
    )
    
    init_time = time.time() - start_time
    print(f"\n-> Instantiation completed in {init_time:.2f} seconds.")
    print(f"-> Train loader contains {len(train_loader)} batches (batch size: {config.training.batch_size}).")
    
    # Measure iteration time (Data loading & Normalization)
    print("\n[2] Testing Train Loader iteration speed (First 50 batches)...")
    
    start_iter_time = time.time()
    
    for i, batch in enumerate(tqdm(train_loader, desc="Iterating Train Loader")):
        # Unpack to ensure getitem runs completely and pushes to RAM
        x = batch['input_grids']
        y = batch['label']
        
        # Print shape on the very first batch just to verify
        if i == 0:
            tqdm.write(f"First batch loaded! X: {x.shape} | Y: {y.shape}")
            
    end_iter_time = time.time()
    total_iter_time = end_iter_time - start_iter_time
    avg_batch_time = total_iter_time / len(train_loader)
    
    print(f"\n-> Iterated {len(train_loader)} batches in {total_iter_time:.2f} seconds.")
    print(f"-> Average time per batch: {avg_batch_time:.4f} seconds.")
    print("=== TEST COMPLETE ===")