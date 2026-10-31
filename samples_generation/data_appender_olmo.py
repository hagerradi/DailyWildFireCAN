import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime, timedelta

from configs import settings
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

# ---------------------------------------------------------
# FEATURE MAPPING CONFIGURATION
# ---------------------------------------------------------
FULL_FEATURES = [
    # --- DYNAMIC FEATURES (14) ---
    'tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
    's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12',
    
    # --- STATIC FEATURES (Topography) (4) ---
    'dem', 'slope', 'aspect (sin)', 'aspect (cos)',     
    
    # --- STATIC FEATURES (SCANFI) (13) ---
    'biomass', 'closure', 'prcb', 'prcc',
    'height',
    'prc_balsam_fir', 'prc_black_spruce', 'prc_douglas_fir', 'prc_jack_pine',
    'prc_lodgepole_pine', 'prc_ponderosa_pine', 'prc_tamarack', 'prc_white_red_pine',
    
    # --- CCRS LANDCOVER (15) ---
    'ccrs_1_needleleaf', 
    'ccrs_2_taiga_needleleaf', 
    'ccrs_5_broadleaf', 
    'ccrs_6_mixed_forest', 
    'ccrs_8_shrubland', 
    'ccrs_10_grassland', 
    'ccrs_11_polar_shrubland', 
    'ccrs_12_polar_grassland', 
    'ccrs_13_polar_barren', 
    'ccrs_14_wetland', 
    'ccrs_15_cropland', 
    'ccrs_16_barren', 
    'ccrs_17_urban', 
    'ccrs_18_water', 
    'ccrs_19_snow',

    # --- HUMAN INFLUENCE INDEX (1) ---
    'hii',

    # --- ANNUAL DISTURBANCE (6) ---
    'dist_1_wildfire', 
    'dist_2_harvesting', 
    'dist_3_other', 
    'dist_4_water', 
    'dist_5_defoliation_harvest', 
    'dist_6_defoliation_all',
    
    # --- FIRE MASKS (2) ---
    'accumulated_mask', 'scaled_accumulated_mask'
]

# Represents the 6 Sentinel-2 channels available in the dataset
SATELLITE_FEATURES = ['s2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12']

# The 12-band order OlmoEarth v1.2 expects
OLMO_S2_ORDER = [
    's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b05', 's2_b06', 
    's2_b07', 's2_b8a', 's2_b11', 's2_b12', 's2_b01', 's2_b09'
]

def generate_offline_olmo_embeddings(data_folder: str):
    """
    Reads existing .npz files in place, extracts the 6 satellite channels, 
    zero-pads to a 12-band tensor, generates OlmoEarth embeddings, and appends them back to the same file.
    """
    # Setup Device and Load OlmoEarth v1.2 Base
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading OlmoEarth-v1.2-Base to {device}...")
    
    model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE)
    model = model.to(device)
    model.eval()
    
    # Process each split
    # splits = ['train', 'val', 'test']
    splits = ['train', 'val', 'test_space', 'test_time', 'test_spacetime']
    
    for split in splits:
        split_dir = os.path.join(data_folder, split)
        
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist. Skipping.")
            continue
            
        # Find all .npz files in the current split
        npz_files = glob.glob(os.path.join(split_dir, "*.npz"))
        
        if not npz_files:
            print(f"No files found in {split_dir}. Skipping.")
            continue
            
        print(f"\nProcessing {split.upper()} split ({len(npz_files)} files)...")
        
        # Disable gradients since we are just doing inference
        with torch.no_grad():
            for file_path in tqdm(npz_files, desc=f"Olmo Embedding {split}"):
                
                # Load existing data
                with np.load(file_path) as data:
                    file_data = {key: data[key] for key in data.files}
                
                if 'x' not in file_data:
                    print(f"Warning: No 'x' tensor found in {file_path}. Skipping.")
                    continue
                
                x_tensor = file_data['x']  # Expected shape: (Channels, 256, 256)
                
                # Reconstruct the 12 bands via Zero-Padding
                s2_12_bands = []
                for band in OLMO_S2_ORDER:
                    if band in SATELLITE_FEATURES and band in FULL_FEATURES:
                        idx = FULL_FEATURES.index(band)
                        band_data = x_tensor[idx] * 10000.0 
                        s2_12_bands.append(band_data)
                    else:
                        s2_12_bands.append(np.zeros((256, 256), dtype=np.float32))
                
                # Stack to shape: (12, 256, 256)
                s2_np = np.stack(s2_12_bands)
                
                # Reshape to meet Olmo's exact wrapper needs: (B, H, W, T, C)
                s2_np = np.transpose(s2_np, (1, 2, 0))          # (256, 256, 12)
                s2_np = np.expand_dims(s2_np, axis=(0, 3))       # (1, 256, 256, 1, 12)
                s2_tensor = torch.from_numpy(s2_np).float().to(device)
                
                # Dynamic Date & Mask
                # Extract the saved date and delta_t to find the exact Satellite Date
                if 'fire_date' in file_data and 'delta_t' in file_data:
                    f_date = file_data['fire_date']
                    
                    # Directly cast delta_t to an integer
                    delta_days = int(file_data['delta_t'])
                    
                    # Create a datetime object of the fire (using the standard 1-indexed month)
                    fire_dt = datetime(year=int(f_date[0]), month=int(f_date[1]), day=int(f_date[2]))
                    
                    # Subtract the age of the satellite image
                    sat_dt = fire_dt - timedelta(days=delta_days)
                    
                    # Extract values for OlmoEarth (converting month to 0-indexed)
                    year = sat_dt.year
                    month = sat_dt.month - 1  
                    day = sat_dt.day
                    
                    timestamps = torch.tensor([[[day, month, year]]], dtype=torch.long, device=device)
                else:
                    # Fallback just in case a sample is missing the date or delta_t
                    timestamps = torch.tensor([[[15, 6, 2024]]], dtype=torch.long, device=device)

                dummy_mask = torch.ones((1, 256, 256, 1), dtype=torch.long, device=device) * MaskValue.ONLINE_ENCODER.value
                
                sample = MaskedOlmoEarthSample(
                    sentinel2_l2a=s2_tensor,
                    sentinel2_l2a_mask=dummy_mask,
                    timestamps=timestamps,
                )
                
                # Run the Vision Transformer Forward Pass using standard 16-patch sizing
                output = model.encoder(sample, fast_pass=True, patch_size=16)
                features = output["tokens_and_masks"].sentinel2_l2a  # Shape: (1, 16, 16, 1, 1, 768)
                
                # Clean up the dimensions to match standard PyTorch channel format: (C, H, W)
                features = features.squeeze(0).squeeze(-2).squeeze(-2)  # Shape: (16, 16, 768)
                embedding_grid = features.permute(2, 0, 1)              # Shape: (768, 16, 16)
                
                # Append to dictionary as half-precision float16 to save space
                file_data['olmo_emb'] = embedding_grid.cpu().numpy().astype(np.float16)
                
                # Overwrite the exact same file in place
                np.savez_compressed(file_path, **file_data)

    print("\nOffline OlmoEarth embedding generation complete!")

if __name__ == "__main__":
    
    TARGET_FOLDER = settings.SAMPLE_FOLDER_ADVANCED.replace("v1", "v2")
    print(f'Target folder : {TARGET_FOLDER}')
    
    generate_offline_olmo_embeddings(TARGET_FOLDER)