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
                 return_coords: bool = False):
        """
            data_dir: Path to the specific split folder (e.g., settings.TIMESERIES_SAMPLE_FOLDER + "/train")
        """
        self.data_dir = data_dir
        self.return_positions = return_positions
        self.return_sat_age = return_sat_age
        self.return_coords = return_coords

        # # --- FEATURE CHANNEL SELECTION (Dropping 'hii') ---
        # all_feature_names = [
        #     # Dynamic (14)
        #     'tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
        #     's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12',
        #     # Static (8)
        #     'dem', 'slope', 'aspect (sin)', 'aspect (cos)', 
        #     'biomass', 'closure', 'prcb', 'prcc',
        #     # Static between landcovers (9)
        #     'height',
        #     'prc_balsam_fir', 'prc_black_spruce', 'prc_douglas_fir', 'prc_jack_pine',
        #     'prc_lodgepole_pine', 'prc_ponderosa_pine', 'prc_tamarack', 'prc_white_red_pine',
        #     # CCRS Landcover (15)
        #     'ccrs_1_needleleaf', 'ccrs_2_taiga_needleleaf', 'ccrs_5_broadleaf', 
        #     'ccrs_6_mixed_forest', 'ccrs_8_shrubland', 'ccrs_10_grassland', 
        #     'ccrs_11_polar_shrubland', 'ccrs_12_polar_grassland', 'ccrs_13_polar_barren', 
        #     'ccrs_14_wetland', 'ccrs_15_cropland', 'ccrs_16_barren', 
        #     'ccrs_17_urban', 'ccrs_18_water', 'ccrs_19_snow',
        #     # Human Influence Index (1) -> DROPPED
        #     'hii',
        #     # Annual Disturbance (6)
        #     'dist_1_wildfire', 'dist_2_harvesting', 'dist_3_other', 
        #     'dist_4_water', 'dist_5_defoliation_harvest', 'dist_6_defoliation_all',
        #     # Fire Masks (2)
        #     'accumulated_mask', 'scaled_accumulated_mask'
        # ]

        # drop_feature_names = ['hii']
        
        # # Indices of channels to keep
        # self.keep_indices = [
        #     i for i, name in enumerate(all_feature_names) 
        #     if name not in drop_feature_names
        # ]
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.files[idx])

        has_pos = False
        has_delta = False
        has_coords = False

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

        # # SLICE OUT DROPPED CHANNELS ('hii')
        # # Checks whether x is 4D (Seq, Channels, H, W) or 3D (Channels, H, W)
        # if x_np.ndim == 4:
        #     x_np = x_np[:, self.keep_indices, :, :]
        # elif x_np.ndim == 3:
        #     x_np = x_np[self.keep_indices, :, :]

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

        return sample


def get_advanced_timeseries_dataloaders(
    config: Config,
    return_positions: bool = False,
    return_sat_age: bool = False, 
    return_coords: bool = False
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Instantiates the offline time-series datasets and wraps them in PyTorch DataLoaders.

        return data['x'], data['y']

    """
    tc = config.training

    base_data_path = settings.TIMESERIES_SAMPLE_FOLDER_ADVANCED

    # Define the paths to your precomputed splits
    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_space_dir = os.path.join(base_data_path, "test_space")
    test_time_dir = os.path.join(base_data_path, "test_time")
    test_spacetime_dir = os.path.join(base_data_path, "test_spacetime")

    dataset_kwargs = {
        "return_positions": return_positions,
        "return_sat_age": return_sat_age,
        "return_coords": return_coords,
    }

    train_dataset = TimeSeriesFireDataset(train_dir, **dataset_kwargs)
    val_dataset = TimeSeriesFireDataset(val_dir, **dataset_kwargs)
    test_space_dataset = TimeSeriesFireDataset(test_space_dir, **dataset_kwargs)
    test_time_dataset = TimeSeriesFireDataset(test_time_dir, **dataset_kwargs)
    test_spacetime_dataset = TimeSeriesFireDataset(test_spacetime_dir, **dataset_kwargs)

    print(f'Number of offline TS training samples   :: {len(train_dataset)}')
    print(f'Number of offline TS validation samples :: {len(val_dataset)}')
    print(f'Number of offline TS test-space samples       :: {len(test_space_dataset)}')
    print(f'Number of offline TS test-time samples       :: {len(test_time_dataset)}')
    print(f'Number of offline TS test-spacetime samples       :: {len(test_spacetime_dataset)}')

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

    # ---------------------------------------------------------
    # DATALOADER BATCH VERIFICATION
    # ---------------------------------------------------------
    if len(test_spacetime_loader) > 0:
        print('\n--- Verification of Test Loader Batch ---')
        
        # Grab the very first batch
        batch = next(iter(test_spacetime_loader))
        
        print(f"Batched Input Grids Shape : {batch['input_grids'].shape}  <- Expected: (Batch, Seq, Channels, H, W)")
        print(f"Batched Target Shape      : {batch['label'].shape}  <- Expected: (Batch, H, W)")
        
        if 'positions' in batch:
            print(f"Batched Positions Shape   : {batch['positions'].shape}  <- Expected: (Batch, Seq)")
        if 'satellite_age' in batch:
            print(f"Batched Sat Age Shape     : {batch['satellite_age'].shape}  <- Expected: (Batch, Seq)")
        if 'coords' in batch:
            print(f"Batched Coords Shape      : {batch['coords'].shape}  <- Expected: (Batch, 2)")
            
        print('-----------------------------------------\n')

    return train_loader, val_loader, test_space_loader, test_time_loader, test_spacetime_loader

if __name__ == "__main__":
    print("--- Running Dataloader Verification on Real Train Dataset ---")
    
    # Instantiate project config
    config = Config()
    
    # Call your actual function with flags enabled
    train_loader, val_loader, _, _, _ = get_timeseries_dataloaders(
        config=config,
        return_positions=True,
        return_sat_age=False,
        return_coords=False
    )
    
    # Test a single batch directly from train_loader
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print("Success! First training batch loaded:")
        print(f"  Input Grids Shape : {batch['input_grids'].shape}")
        print(f"  Label Shape       : {batch['label'].shape}")
        if 'positions' in batch: print(f"  Positions Shape   : {batch['positions'].shape}")
        if 'satellite_age' in batch: print(f"  Sat Age Shape     : {batch['satellite_age'].shape}")
        if 'coords' in batch: print(f"  Coords Shape      : {batch['coords'].shape}")
        print("\n🎉 REAL TRAIN DATALOADER PASSED ALL CHECKS!")
    else:
        print(f"[!] Warning: No train files found at: {os.path.join(settings.TIMESERIES_SAMPLE_FOLDER_ADVANCED, 'train')}")