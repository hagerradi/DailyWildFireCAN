import os
import glob
import h5py
import numpy as np
from tqdm import tqdm

from configs import settings
from data_preparation.data_configs.data_settings import DYNAMIC_FEATURES, STATIC_FEATURES

def apply_quality_masks(h5_folder, dynamic_feats, static_feats):
    """Creates a binary quality mask (1 = NaN/Inf, 0 = Clean) per day, per tile.
    Only saves the mask to the H5 file if there is at least one bad pixel.

    Args:
      h5_folder: folder containing the H5 files
      dynamic_feats: dynamic features list
      static_feats: static features list

    """
    h5_files = glob.glob(os.path.join(h5_folder, "*.h5"))
    
    if not h5_files:
        print(f"No .h5 files found in {h5_folder}")
        return

    days_with_masks = 0
    clean_days = 0

    for file_path in tqdm(h5_files, desc="Generating Quality Masks"):
        try:
            # Open in append mode to modify the file
            with h5py.File(file_path, "a") as f:
                
                # Find all tile groups in the file
                tile_ids = [k for k in f.keys() if k.startswith('tile_')]
                if not tile_ids:
                    continue
                
                for tile_id in tile_ids:
                    tile_grp = f[tile_id]
                
                    # ---------------------------------------------------------
                    # PRE-CALCULATE STATIC MASK (It doesn't change day-to-day)
                    # ---------------------------------------------------------
                    base_shape = None
                    
                    # Try Statics first
                    if "static_features" in tile_grp:
                        for feat in static_feats:
                            path = f"{tile_id}/static_features/{feat}"
                            if path in f:
                                base_shape = f[path].shape
                                break
                                
                    # Try Dynamics if Statics are missing entirely
                    if base_shape is None and "days" in tile_grp:
                        for day_key in tile_grp["days"].keys():
                            for feat in dynamic_feats:
                                path = f"{tile_id}/days/{day_key}/satellite/{feat}" if feat.startswith('s2_') else f"{tile_id}/days/{day_key}/features/{feat}"
                                if path in f:
                                    base_shape = f[path].shape
                                    break
                            if base_shape is not None:
                                break

                    if base_shape is None:
                        continue # The tile is completely empty, safely skip it
                    
                    static_mask = np.zeros(base_shape, dtype=bool)
                    
                    if "static_features" in tile_grp:
                        for feat in static_feats:
                            path = f"{tile_id}/static_features/{feat}"
                            if path in f:
                                data = f[path][:]
                                static_mask = static_mask | ~np.isfinite(data)
                            else:
                                # MISSING FEATURE: The entire mask becomes True (bad)
                                static_mask[:] = True 
                                break # No need to check the rest
                    else:
                        static_mask[:] = True # Missing the whole group

                    # ---------------------------------------------------------
                    # CALCULATE DYNAMIC MASK PER DAY
                    # ---------------------------------------------------------
                    if "days" not in tile_grp:
                        continue
                        
                    for day_key in tile_grp["days"].keys():
                        day_mask = np.copy(static_mask) # Start with the static errors
                        
                        for feat in dynamic_feats:
                            # Route path dynamically
                            if feat.startswith('s2_'):
                                path = f"{tile_id}/days/{day_key}/satellite/{feat}"
                            else:
                                path = f"{tile_id}/days/{day_key}/features/{feat}"
                            
                            if path in f:
                                data = f[path][:]
                                day_mask = day_mask | ~np.isfinite(data)
                            else:
                                # MISSING FEATURE: The whole day is ruined. Mark all pixels as bad.
                                day_mask[:] = True
                                break # Skip checking the other features for this day
                                
                        # ---------------------------------------------------------
                        # SAVE LOGIC
                        # ---------------------------------------------------------
                        mask_path = f"{tile_id}/days/{day_key}/quality_mask"
                        
                        # If we already generated a mask on a previous run, delete it to refresh
                        if mask_path in f:
                            del f[mask_path]
                            
                        if np.any(day_mask):
                            # Problems found
                            # 1 means bad pixel, 0 means good pixel
                            f.create_dataset(
                                mask_path, 
                                data=day_mask.astype(np.uint8), 
                                compression="lzf"
                            )
                            days_with_masks += 1
                        else:
                            # Perfectly clean day. Do not save anything
                            clean_days += 1

        except Exception as e:
            print(f"\n[!] Failed to process {file_path}: {e}")

    print("\n=== Quality Mask Summary ===")
    print(f"Perfectly Clean Tile-Days (No Mask Saved): {clean_days}")
    print(f"Problematic Tile-Days (Mask Saved): {days_with_masks}")

if __name__ == "__main__":
    
    folder_path = settings.H5_OUTPUT_FOLDER

    dynamic_feats = DYNAMIC_FEATURES
    static_feats = STATIC_FEATURES
    
    apply_quality_masks(
        h5_folder=folder_path, 
        dynamic_feats=dynamic_feats, 
        static_feats=static_feats
    )