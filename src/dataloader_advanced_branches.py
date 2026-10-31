import os
import torch
from torch.utils.data import Dataset, DataLoader
from configs import settings
from src.config import Config
from skimage import morphology
import numpy as np

class FireDatasetWithBranches(Dataset):
    def __init__(self, data_dir: str, 
                 full_features: list, 
                 state_features: list,
                 dynamic_features: list,
                 static_features: list,
                 return_sat_age: bool = False, 
                 return_coords: bool = False,
                 return_loc_emb: bool = False):
        """
            data_dir: Path to the specific split folder
        """
        
        self.data_dir = data_dir
        self.return_sat_age = return_sat_age
        self.return_coords = return_coords
        self.return_loc_emb = return_loc_emb

        # Map the requested features to their exact channel indices
        self.state_indices = self._get_indices(full_features, state_features, "state")
        self.dynamic_indices = self._get_indices(full_features, dynamic_features, "dynamic")
        self.static_indices = self._get_indices(full_features, static_features, "static")
        
        # Sort files to ensure consistent, reproducible ordering across runs
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

    def _get_indices(self, full_features, target_features, block_name):
        """Helper to safely map feature names to channel indices."""
        indices = []
        for feat in target_features:
            if feat in full_features:
                indices.append(full_features.index(feat))
            else:
                raise ValueError(f"[!] Feature '{feat}' requested in {block_name}_features but not found in full_features.")
        return indices
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        
        file_path = os.path.join(self.data_dir, self.files[idx])

        has_delta = False
        has_coords = False
        has_emb = False

        with np.load(file_path) as data:
            x_np = data['x']
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

        # MULTIMODAL SLICING
        # Check if data has a Time dimension: (Time, Channels, H, W)
        if x_np.ndim == 4:
            x_state = x_np[:, self.state_indices, :, :]
            x_dynamic = x_np[:, self.dynamic_indices, :, :]
            # Static data doesn't change over time, so we just grab the first time step (t=0)
            x_static = x_np[0, self.static_indices, :, :]
            
        # If data is just a spatial snapshot: (Channels, H, W)
        elif x_np.ndim == 3:
            # We add a dummy time dimension of 1 so the ConvLSTM doesn't crash
            x_state = np.expand_dims(x_np[self.state_indices, :, :], axis=0)
            x_dynamic = np.expand_dims(x_np[self.dynamic_indices, :, :], axis=0)
            x_static = x_np[self.static_indices, :, :]

        # APPYING MAX_SIZE=1 HOLE FILLING TO THE MASK
        y_bool = y_np > 0 
        y_filled_np = morphology.remove_small_holes(y_bool, area_threshold=1)
        y_filled_np = y_filled_np.astype(y_np.dtype)

        # Build the structured sample
        sample = {
            "state": torch.from_numpy(x_state).float(),
            "dynamic": torch.from_numpy(x_dynamic).float(),
            "constant": torch.from_numpy(x_static).float(),
            "label": torch.from_numpy(y_filled_np).float()
        }
        
        if has_delta:
            sample["satellite_age"] = torch.from_numpy(delta_t_np).float()
        if has_coords:
            sample["coords"] = torch.from_numpy(coords_np).float()
        if has_emb:
            sample["loc_emb"] = torch.from_numpy(loc_emb_np).float()
            
        return sample


def get_advanced_dataloaders_with_branches(config: Config, 
                             full_features: list, 
                             state_features: list,
                             dynamic_features: list,
                             static_features: list,
                             is_sat_age: bool, 
                             is_coords: bool = False,
                             is_loc_emb: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    """Instantiates the offline datasets and wraps them in PyTorch DataLoaders."""
    
    tc = config.training

    print('--- MULTIMODAL ABLATIONS ---')
    base_data_path = settings.TIMESERIES_SAMPLE_FOLDER_ADVANCED
    print(f'Samples path : {base_data_path}')

    train_dir = os.path.join(base_data_path, "train")
    val_dir = os.path.join(base_data_path, "val")
    test_space_dir = os.path.join(base_data_path, "test_space")
    test_time_dir = os.path.join(base_data_path, "test_time")
    test_spacetime_dir = os.path.join(base_data_path, "test_spacetime")

    # Pass the specialized feature lists into the dataset
    common_kwargs = {
        "full_features": full_features,
        "state_features": state_features,
        "dynamic_features": dynamic_features,
        "static_features": static_features,
        "return_sat_age": is_sat_age,
        "return_coords": is_coords,
        "return_loc_emb": is_loc_emb
    }

    train_dataset = FireDatasetWithBranches(train_dir, **common_kwargs)
    val_dataset = FireDatasetWithBranches(val_dir, **common_kwargs)
    test_space_dataset = FireDatasetWithBranches(test_space_dir, **common_kwargs)
    test_time_dataset = FireDatasetWithBranches(test_time_dir, **common_kwargs)
    test_spacetime_dataset = FireDatasetWithBranches(test_spacetime_dir, **common_kwargs)

    print(f'Number of offline training samples   :: {len(train_dataset)}')
    print(f'Number of offline validation samples :: {len(val_dataset)}')
    print(f'Number of offline test-space samples :: {len(test_space_dataset)}')
    print(f'Number of offline test-time samples  :: {len(test_time_dataset)}')
    print(f'Number of offline test-spacetime samples :: {len(test_spacetime_dataset)}')

    # ---------------------------------------------------------
    # VERIFICATION
    # ---------------------------------------------------------
    print('\n--- Verification of Sample 0 (Multimodal) ---')
    
    sample_data = test_space_dataset[0]
    print(f"State Block Shape   (x1) : {sample_data['state'].shape}")
    print(f"Dynamic Block Shape (x2) : {sample_data['dynamic'].shape}")
    print(f"Static Block Shape  (x3) : {sample_data['constant'].shape}")
    print(f"Label Shape              : {sample_data['label'].shape}")

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
    
    from types import SimpleNamespace

    # Create a dummy config to bypass needing your full training setup
    class DummyConfig:
        training = SimpleNamespace(batch_size=2, num_workers=0)
    
    config = DummyConfig()

    # Full Features list
    FULL_FEATURES = [
        # --- DYNAMIC FEATURES (14) ---
        'tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
        's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12',
        
        # --- STATIC FEATURES (before SCANFI landcover) (8) ---
        'dem', 'slope', 'aspect (sin)', 'aspect (cos)', 
        'biomass', 'closure', 'prcb', 'prcc',
        
        # --- STATIC FEATURES (between landcovers) (9) ---
        'height',
        'prc_balsam_fir', 'prc_black_spruce', 'prc_douglas_fir', 'prc_jack_pine',
        'prc_lodgepole_pine', 'prc_ponderosa_pine', 'prc_tamarack', 'prc_white_red_pine',
        
        # --- CCRS LANDCOVER (15) ---
        'ccrs_1_needleleaf', 'ccrs_2_taiga_needleleaf', 'ccrs_5_broadleaf', 
        'ccrs_6_mixed_forest', 'ccrs_8_shrubland', 'ccrs_10_grassland', 
        'ccrs_11_polar_shrubland', 'ccrs_12_polar_grassland', 'ccrs_13_polar_barren', 
        'ccrs_14_wetland', 'ccrs_15_cropland', 'ccrs_16_barren', 'ccrs_17_urban', 
        'ccrs_18_water', 'ccrs_19_snow',

        # --- HUMAN INFLUENCE INDEX (1) ---
        'hii',

        # --- ANNUAL DISTURBANCE (6) ---
        'dist_1_wildfire', 'dist_2_harvesting', 'dist_3_other', 
        'dist_4_water', 'dist_5_defoliation_harvest', 'dist_6_defoliation_all',
        
        # --- FIRE MASKS (2) ---
        'accumulated_mask', 'scaled_accumulated_mask'
    ]

    # The sub-lists
    STATE_FEATURES = ['accumulated_mask', 'scaled_accumulated_mask']
    
    DYNAMIC_FEATURES = [
        'tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
        's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12'
    ]
    
    # Dynamically grab all remaining features for the Static block
    STATIC_FEATURES = [f for f in FULL_FEATURES if f not in STATE_FEATURES and f not in DYNAMIC_FEATURES]
    # STATIC_FEATURES = [f for f in FULL_FEATURES if f not in STATE_FEATURES and f not in DYNAMIC_FEATURES and f not in ['hii']]

    # Initialize DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_space_loader, test_time_loader, test_spacetime_loader = get_advanced_dataloaders_with_branches(
        config=config,
        full_features=FULL_FEATURES,
        state_features=STATE_FEATURES,
        dynamic_features=DYNAMIC_FEATURES,
        static_features=STATIC_FEATURES,
        is_sat_age=False
    )

    # Fetch a single batch to test
    batch = next(iter(test_space_loader))
    
    state = batch['state']       # Expected: [Batch, Time, Channels, H, W]
    dynamic = batch['dynamic']   # Expected: [Batch, Time, Channels, H, W]
    constant = batch['constant'] # Expected: [Batch, Channels, H, W]
    label = batch['label']       # Expected: [Batch, H, W]

    print("\n=========================================")
    print("        BATCH SHAPES VERIFICATION        ")
    print("=========================================")
    print(f"State Block   (Fire History): {state.shape}")
    print(f"Dynamic Block (Weather/Sat) : {dynamic.shape}")
    print(f"Static Block  (Topography)  : {constant.shape}")
    print(f"Target Label                : {label.shape}")
    
    print("\n=========================================")
    print("      CHANNEL RANGES (MIN -> MAX)        ")
    print("=========================================")

    # Helper function to print ranges per channel
    def print_block_ranges(tensor, feature_names, block_name):
        print(f"\n--- {block_name.upper()} BLOCK ---")
        
        # If the tensor has a Time dimension: [Batch, Time, C, H, W]
        if tensor.dim() == 5:
            for c, feat in enumerate(feature_names):
                c_min = tensor[:, :, c, :, :].min().item()
                c_max = tensor[:, :, c, :, :].max().item()
                print(f"{feat:>25} | Min: {c_min:>8.4f} | Max: {c_max:>8.4f}")
                
        # If the tensor is Static: [Batch, C, H, W]
        elif tensor.dim() == 4:
            for c, feat in enumerate(feature_names):
                c_min = tensor[:, c, :, :].min().item()
                c_max = tensor[:, c, :, :].max().item()
                print(f"{feat:>25} | Min: {c_min:>8.4f} | Max: {c_max:>8.4f}")

    # Print the ranges
    print_block_ranges(state, STATE_FEATURES, "State")
    print_block_ranges(dynamic, DYNAMIC_FEATURES, "Dynamic")
    print_block_ranges(constant, STATIC_FEATURES, "Constant")