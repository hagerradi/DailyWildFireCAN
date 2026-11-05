import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
from skimage import morphology
import numpy as np

class DailyFireDataset(Dataset):
    def __init__(self, data_dir: str, 
                 return_sat_age: bool = False, 
                 return_coords: bool = False,
                 return_loc_emb: bool = False,
                 return_olmo_emb: bool = False,
                 return_alpha_emb: bool = False,
                 uncertainty_mode: str = None):
        """
            data_dir: Path to the specific split folder (e.g., settings.SAMPLE_FOLDER + "/train")
        """
        
        self.data_dir = data_dir
        self.return_sat_age = return_sat_age
        self.return_coords = return_coords
        self.return_loc_emb = return_loc_emb
        self.return_olmo_emb = return_olmo_emb
        self.return_alpha_emb = return_alpha_emb

        self.uncertainty_mode = uncertainty_mode
        
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

            # Conditionally load
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

        #
        if self.uncertainty_mode in ['spatial', 'temporal']:
            
            parent_dir = os.path.dirname(self.data_dir)
            split_name = os.path.basename(self.data_dir)

            label_parent_dir = parent_dir
            
            # Guarantee this only happens for test sets
            if 'test' in split_name:
                
                label_file = os.path.join(label_parent_dir, f"{split_name}_temporal_labels", self.files[idx])
                
                with np.load(label_file) as unc_data:
                    if self.uncertainty_mode == 'spatial':
                        y_np = unc_data['y_dilated']
                    elif self.uncertainty_mode == 'temporal':
                        # Merge T-1, T, and T+1 into a single 2D mask
                        y_np = np.maximum.reduce([unc_data['y_minus1'], y_np, unc_data['y_plus1']])

        
        # APPYING MAX_SIZE=1 HOLE FILLING TO THE MASK (remove noise pixels from projection)
        y_bool = y_np > 0 
        # Fill the 1-pixel projection gaps
        y_filled_np = morphology.remove_small_holes(y_bool, area_threshold=1)
        # Convert back to its original integer type (uint8)
        y_filled_np = y_filled_np.astype(y_np.dtype)

        # Convert everything to PyTorch Tensors
        x_tensor = torch.from_numpy(x_np).float() 
        y_tensor = torch.from_numpy(y_filled_np).float()

        # Build sample
        sample = {
            "input_grids": x_tensor,
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


def get_dataloaders_advanced_labels(config: Config, 
                    is_sat_age: bool, 
                    is_coords: bool = False, 
                    is_loc_emb: bool = False, 
                    is_olmo_emb: bool = False,
                    is_alpha_emb: bool = False,
                    uncertainty_mode: str = None) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Instantiates the offline datasets and wraps them in PyTorch DataLoaders.

    Args:
      config: Config: config parameters
      is_sat_age: bool: toggle to choose if the satellite age gap is included in the sample
      is_coords: bool: toggle to choose if the coordinates are included in the sample
      is_loc_emb: toggle to choose if the location embeddings are included in the sample

    Returns: the train, val, and test loaders

    """
    tc = config.training

    print('Uncertainty Data Loader')
    
    base_data_path = settings.SAMPLE_FOLDER_ADVANCED
    print(f'Samples path : {base_data_path}')

    # if is_olmo_emb:
    #     base_data_path = settings.SAMPLE_FOLDER_REB + "_OLMO"
    #     print(f'Samples path Instead : {base_data_path}')

    # Define the paths to the precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_time_dir = os.path.join(base_data_path, "test_easy")
    test_space_dir = os.path.join(base_data_path, "test_medium")
    test_spacetime_dir = os.path.join(base_data_path, "test_hard")
    
    # test_space_dir = os.path.join(base_data_path, "test_space")
    # test_time_dir = os.path.join(base_data_path, "test_time")
    # test_spacetime_dir = os.path.join(base_data_path, "test_spacetime")

    # Refactor dataset arguments
    dataset_kwargs = {
        "return_sat_age": is_sat_age,
        "return_coords": is_coords,
        "return_loc_emb": is_loc_emb,
        "return_olmo_emb": is_olmo_emb,
        "return_alpha_emb": is_alpha_emb
    }

    # Instantiate the datasets, passing the sat_age flag
    train_dataset = DailyFireDataset(train_dir, **dataset_kwargs, uncertainty_mode=None)
    val_dataset = DailyFireDataset(val_dir, **dataset_kwargs, uncertainty_mode=None)
    test_space_dataset = DailyFireDataset(test_space_dir, **dataset_kwargs, uncertainty_mode=uncertainty_mode)
    test_time_dataset = DailyFireDataset(test_time_dir, **dataset_kwargs, uncertainty_mode=uncertainty_mode)
    test_spacetime_dataset = DailyFireDataset(test_spacetime_dir, **dataset_kwargs, uncertainty_mode=uncertainty_mode)

    print(f'Number of offline training samples   :: {len(train_dataset)}')
    print(f'Number of offline validation samples :: {len(val_dataset)}')
    print(f'Number of offline test-space samples       :: {len(test_space_dataset)}')
    print(f'Number of offline test-time samples       :: {len(test_time_dataset)}')
    print(f'Number of offline test-spacetime samples       :: {len(test_spacetime_dataset)}')


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