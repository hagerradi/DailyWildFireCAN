import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
from skimage import morphology
import numpy as np

class DailyFireDataset(Dataset):
    def __init__(self, data_dir: str, 
                 all_feature_names: list = None,
                 rgb_feature_names: list = None,
                 env_feature_names: list = None,
                 return_sat_age: bool = False, 
                 return_coords: bool = False,
                 return_loc_emb: bool = False,
                 return_olmo_emb: bool = False,
                 return_alpha_emb: bool = False):
        """
            data_dir: Path to the specific split folder (e.g., settings.SAMPLE_FOLDER + "/train")
        """
        
        self.data_dir = data_dir
        self.return_sat_age = return_sat_age
        self.return_coords = return_coords
        self.return_loc_emb = return_loc_emb
        self.return_olmo_emb = return_olmo_emb
        self.return_alpha_emb = return_alpha_emb

        # AUTOMATIC INDEX EXTRACTION FROM FEATURE NAMES
        if all_feature_names is not None:
            # Map RGB names -> integer indices
            if rgb_feature_names is not None:
                self.rgb_indices = [all_feature_names.index(f) for f in rgb_feature_names]

            # Map Env names -> integer indices
            if env_feature_names is not None:
                self.env_indices = [all_feature_names.index(f) for f in env_feature_names]
            else:
                # If env_feature_names isn't provided, use all remaining features
                self.env_indices = [i for i in range(len(all_feature_names)) if i not in self.rgb_indices]
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.files[idx])

        has_delta = False
        has_coords = False
        has_emb = False
        has_olmo = False
        has_alpha = False

        with np.load(file_path) as data:
            x_np = data['x']
            y_np = data['y'] 

            # Conditionally load to prevent KeyErrors on older datasets
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
        
        # APPYING MAX_SIZE=1 HOLE FILLING TO THE MASK (remove noise pixels from projection)
        y_bool = y_np > 0 
        # Fill the 1-pixel projection gaps
        y_filled_np = morphology.remove_small_holes(y_bool, area_threshold=1)
        # Convert back to its original integer type (uint8)
        y_filled_np = y_filled_np.astype(y_np.dtype)

        # Convert everything to PyTorch Tensors
        x_tensor = torch.from_numpy(x_np).float() 
        y_tensor = torch.from_numpy(y_filled_np).float()

        # Extract the two branches
        rgb_tensor = x_tensor[self.rgb_indices, :, :]
        env_tensor = x_tensor[self.env_indices, :, :]

        # Build sample dictionary
        sample = {
            "input_rgb": rgb_tensor,
            "input_env": env_tensor,
            "label": y_tensor
        }
        
        if has_delta:
            delta_t_tensor = torch.from_numpy(delta_t_np).float()
            sample["satellite_age"] = delta_t_tensor
            
        if has_coords:
            coords_tensor = torch.from_numpy(coords_np).float()
            sample["coords"] = coords_tensor
        
        if has_emb:
            sample["loc_emb"] = torch.from_numpy(loc_emb_np).float()
            
        if has_olmo:
            sample["olmo_emb"] = torch.from_numpy(olmo_emb_np).float()
        
        if has_alpha:
            sample["alpha_emb"] = torch.from_numpy(alpha_emb_np).float()
            
        return sample


def get_advanced_dual_dataloaders(config: Config, 
                    all_features,
                    rgb_features,
                    env_features,        
                    is_sat_age: bool, 
                    is_coords: bool = False, 
                    is_loc_emb: bool = False, 
                    is_olmo_emb: bool = False,
                    is_alpha_emb: bool = False) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Instantiates the offline datasets and wraps them in PyTorch DataLoaders.

    Args:
      config: Config: config parameters
      is_sat_age: bool: toggle to choose if the satellite age gap is included in the sample
      is_coords: bool: toggle to choose if the coordinates are included in the sample
      is_loc_emb: toggle to choose if the location embeddings are included in the sample

    Returns: the train, val, and test loaders

    """
    tc = config.training
    
    base_data_path = settings.SAMPLE_FOLDER_ADVANCED
    print(f'Samples path : {base_data_path}')

    # Define the paths to the precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_space_dir = os.path.join(base_data_path, "test_space")
    test_time_dir = os.path.join(base_data_path, "test_time")
    test_spacetime_dir = os.path.join(base_data_path, "test_spacetime")

    # Refactor dataset arguments
    dataset_kwargs = {
        "all_feature_names": all_features,
        "rgb_feature_names": rgb_features,
        "env_feature_names": env_features,
        "return_sat_age": is_sat_age,
        "return_coords": is_coords,
        "return_loc_emb": is_loc_emb,
        "return_olmo_emb": is_olmo_emb,
        "return_alpha_emb": is_alpha_emb
    }

    # Instantiate the datasets, passing the sat_age flag
    train_dataset = DailyFireDataset(train_dir, **dataset_kwargs)
    val_dataset = DailyFireDataset(val_dir, **dataset_kwargs)
    test_space_dataset = DailyFireDataset(test_space_dir, **dataset_kwargs)
    test_time_dataset = DailyFireDataset(test_time_dir, **dataset_kwargs)
    test_spacetime_dataset = DailyFireDataset(test_spacetime_dir, **dataset_kwargs)

    print(f'Number of offline training samples   :: {len(train_dataset)}')
    print(f'Number of offline validation samples :: {len(val_dataset)}')
    print(f'Number of offline test-space samples       :: {len(test_space_dataset)}')
    print(f'Number of offline test-time samples       :: {len(test_time_dataset)}')
    print(f'Number of offline test-spacetime samples       :: {len(test_spacetime_dataset)}')

    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------
    print('\n--- Verification of Sample 0 ---')
    
    # Safely grab x
    sample_data = test_spacetime_dataset[0]
    x_rgb = sample_data['input_rgb']
    x_env = sample_data['input_env']
    
    print(f'RGB Tensor Shape : {x_rgb.shape}')
    print(f'Env Tensor Shape : {x_env.shape}')

    if is_sat_age:
        sat_age = sample_data['satellite_age']
        print(f'Satellite Age Tensor : {sat_age}')

    if is_coords:
        coords = sample_data['coords']
        print(f'Coordinates Tensor : {coords.tolist()}')
    
    if is_loc_emb:
        loc_emb = sample_data.get('loc_emb', None)
        if loc_emb is not None:
            print(f"Location Embedding Tensor Shape: {loc_emb.shape}")
        else:
            print(f"[!] Warning: loc_emb requested but missing from sample 0.")
    
    if is_olmo_emb:
        olmo_emb = sample_data.get('olmo_emb', None)
        if olmo_emb is not None:
            print(f"Olmo Embedding Tensor Shape: {olmo_emb.shape}")
        else:
            print(f"[!] Warning: olmo_emb requested but missing from sample 0.")

    if is_alpha_emb:
        alpha_emb = sample_data.get('alpha_emb', None)
        if alpha_emb is not None:
            print(f"Alpha Embedding Tensor Shape: {alpha_emb.shape}")
        else:
            print(f"[!] Warning: alpha_emb requested but missing from sample 0.")

    print("\nChannel Ranges (Min -> Max) for RGB:")
    for i in range(x_rgb.shape[0]):
        chan_min = x_rgb[i].min().item()
        chan_max = x_rgb[i].max().item()
        print(f"  RGB Channel [{i:02d}]: {chan_min:>8.4f}  ->  {chan_max:>8.4f}")

    print("\nChannel Ranges (Min -> Max) for Env:")
    for i in range(x_env.shape[0]):
        chan_min = x_env[i].min().item()
        chan_max = x_env[i].max().item()
        print(f"  Env Channel [{i:02d}]: {chan_min:>8.4f}  ->  {chan_max:>8.4f}")
    
    print('--------------------------------\n')

    # Wrap them in DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=tc.batch_size, 
        shuffle=True, 
        num_workers=tc.num_workers,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False, 
        num_workers=tc.num_workers,
        persistent_workers=False
    )
    
    test_space_loader = DataLoader(
        test_space_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False,
        num_workers=tc.num_workers,
        persistent_workers=False   
    )

    test_time_loader = DataLoader(
        test_time_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False,
        num_workers=tc.num_workers,
        persistent_workers=False   
    )

    test_spacetime_loader = DataLoader(
        test_spacetime_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False,
        num_workers=tc.num_workers,
        persistent_workers=False   
    )
    
    return train_loader, val_loader, test_space_loader, test_time_loader, test_spacetime_loader

if __name__ == "__main__":
    
    data_dir = os.path.join(settings.SAMPLE_FOLDER_ADVANCED, "train")
    
    print("--- Running Dataset Flag Verification ---")
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"[!] Warning: Data directory {data_dir} is empty or missing.")
    else:
        # TEST 1: All Flags ON
        print("\nTest 1: sat_age=True, coords=True, loc_emb=True, olmo_emb=True, alpha_emb=True")
        ds_all = DailyFireDataset(data_dir, return_sat_age=True, return_coords=True, return_loc_emb=True, return_olmo_emb=True, return_alpha_emb=True)
        out_all = ds_all[0]
        print(f"  Key Count: {len(out_all)}")
        print(f"  X Shape: {out_all['input_grids'].shape}")
        print(f"  Y Shape: {out_all['label'].shape}")
        if 'satellite_age' in out_all: print(f"  Sat Age: {out_all['satellite_age'].item()}")
        if 'coords' in out_all: print(f"  Coords: {out_all['coords'].tolist()}")
        if 'loc_emb' in out_all: print(f"  Loc Emb Shape: {out_all['loc_emb'].shape}")
        if 'olmo_emb' in out_all: print(f"  Olmo Emb Shape: {out_all['olmo_emb'].shape}")
        if 'alpha_emb' in out_all: print(f"  Alpha Emb Shape: {out_all['alpha_emb'].shape}")

        # TEST 2: Only Embeddings ON
        print("\nTest 2: Only Embeddings ON (loc, olmo, alpha)")
        ds_emb = DailyFireDataset(data_dir, return_sat_age=False, return_coords=False, return_loc_emb=True, return_olmo_emb=True, return_alpha_emb=True)
        out_emb = ds_emb[0]
        print(f"  Key Count: {len(out_emb)}")
        print(f"  X Shape: {out_emb['input_grids'].shape}")
        if 'alpha_emb' in out_emb: print(f"  Alpha Emb Shape: {out_emb['alpha_emb'].shape}")
        print(f"  Contains 'coords' key?: {'coords' in out_emb}")

        # TEST 3: All Flags OFF
        print("\nTest 3: All Flags OFF")
        ds_none = DailyFireDataset(data_dir, return_sat_age=False, return_coords=False, return_loc_emb=False, return_olmo_emb=False, return_alpha_emb=False)
        out_none = ds_none[0]
        print(f"  Key Count: {len(out_none)}")
        print(f"  X Shape: {out_none['input_grids'].shape}")
        print(f"  Contains 'alpha_emb' key?: {'alpha_emb' in out_none}")