import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
from skimage import morphology
import numpy as np

class TimeSeriesFireDataset(Dataset):
    def __init__(self, data_dir: str, 
                 return_positions: bool = False,
                 return_sat_age: bool = False, 
                 return_coords: bool = False,
                 return_loc_emb: bool = False):
        """
            data_dir: Path to the specific split folder (e.g., settings.TIMESERIES_SAMPLE_FOLDER + "/train")
        """
        self.data_dir = data_dir
        self.return_positions = return_positions
        self.return_sat_age = return_sat_age
        self.return_coords = return_coords
        self.return_loc_emb = return_loc_emb
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.files[idx])

        has_pos = False
        has_delta = False
        has_coords = False
        has_emb = False

        with np.load(file_path) as data:
            
            x_np = data['x']
            y_np = data['y']
            
            # Conditionally load variables
            if self.return_positions and 'positions' in data:
                positions_np = data['positions']
                has_pos = True
                
            if self.return_sat_age and 'delta_t' in data:
                delta_t_np = data['delta_t']
                has_delta = True
            
            if self.return_coords and 'coords' in data:
                coords_np = data['coords']
                has_coords = True
            
            if self.return_loc_emb and 'loc_emb' in data:
                loc_emb_np = data['loc_emb']
                has_emb = True

        # APPYING MAX_SIZE=1 HOLE FILLING TO THE MASK (remove noise pixels from projection)
        y_bool = y_np > 0 
        # Fill the 1-pixel projection gaps
        y_filled_np = morphology.remove_small_holes(y_bool, area_threshold=1)
        # Convert back to its original integer type (uint8)
        y_filled_np = y_filled_np.astype(y_np.dtype)

        # Build sample dictionary
        sample = {
            "input_grids": torch.from_numpy(x_np).float(),
            "label": torch.from_numpy(y_filled_np).float()
        }

        # Add optional items
        if has_pos:
            sample["positions"] = torch.from_numpy(positions_np).float()
        if has_delta:
            sample["satellite_age"] = torch.from_numpy(delta_t_np).float()
        if has_coords:
            sample["coords"] = torch.from_numpy(coords_np).float()
        if has_emb:
            sample["loc_emb"] = torch.from_numpy(loc_emb_np).float()

        return sample


def get_timeseries_dataloaders(
    config: Config,
    return_positions: bool = False,
    return_sat_age: bool = False, 
    return_coords: bool = False, 
    return_loc_emb: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Instantiates the offline time-series datasets and wraps them in PyTorch DataLoaders.

        return data['x'], data['y']

    """
    tc = config.training

    base_data_path = settings.TIMESERIES_SAMPLE_FOLDER

    # Define the paths to your precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_dir = os.path.join(base_data_path, "test")

    # Instantiate the datasets
    train_dataset = TimeSeriesFireDataset(train_dir, 
                                          return_positions=return_positions,
                                          return_sat_age=return_sat_age,
                                          return_coords=return_coords,
                                          return_loc_emb=return_loc_emb)
    
    val_dataset = TimeSeriesFireDataset(val_dir, 
                                          return_positions=return_positions,
                                          return_sat_age=return_sat_age,
                                          return_coords=return_coords,
                                          return_loc_emb=return_loc_emb)
    
    test_dataset = TimeSeriesFireDataset(test_dir,
                                          return_positions=return_positions,
                                          return_sat_age=return_sat_age,
                                          return_coords=return_coords,
                                          return_loc_emb=return_loc_emb)

    print(f'Number of offline TS training samples   :: {len(train_dataset)}')
    print(f'Number of offline TS validation samples :: {len(val_dataset)}')
    print(f'Number of offline TS test samples       :: {len(test_dataset)}')

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
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=tc.batch_size, 
        shuffle=False,
        num_workers=tc.num_workers,
        persistent_workers=False   
    )

    # ---------------------------------------------------------
    # DATALOADER BATCH VERIFICATION
    # ---------------------------------------------------------
    if len(test_loader) > 0:
        print('\n--- Verification of Test Loader Batch ---')
        
        # Grab the very first batch
        batch = next(iter(test_loader))
        
        print(f"Batched Input Grids Shape : {batch['input_grids'].shape}  <- Expected: (Batch, Seq, Channels, H, W)")
        print(f"Batched Target Shape      : {batch['label'].shape}  <- Expected: (Batch, H, W)")
        
        if 'positions' in batch:
            print(f"Batched Positions Shape   : {batch['positions'].shape}  <- Expected: (Batch, Seq)")
        if 'satellite_age' in batch:
            print(f"Batched Sat Age Shape     : {batch['satellite_age'].shape}  <- Expected: (Batch, Seq)")
        if 'coords' in batch:
            print(f"Batched Coords Shape      : {batch['coords'].shape}  <- Expected: (Batch, 2)")
        if 'loc_emb' in batch:
            print(f"Batched Loc Emb Shape     : {batch['loc_emb'].shape}")
            
        print('-----------------------------------------\n')

    return train_loader, val_loader, test_loader