import argparse
import pandas as pd
import json
from tqdm import tqdm
import os
from pathlib import Path

from configs import settings
from data_preparation.features_generation.grid_generation import append_tile_coordinates
from data_preparation.features_generation.helpers import lonlat_to_canada_lambert

def generate_mapper_for_year(year, output_folder):
    """
    Saves the json mapper for the specified year

    Args:
      year: the target year 
      output_folder: the output folder to store the mappers

    """
    
    # Resolve Path dynamically based on the year
    csv_path = f"{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{year}/Firegrowth_pts_v1_1_{year}.csv"
    
    if not os.path.exists(csv_path):
        print(f"[!] Error: Could not find CSV for {year} at {csv_path}")
        return
        
    print(f"Reading {os.path.basename(csv_path)}...")
    columns_to_use = [col for col in settings.SUBSET_FEATURE_LIST if col not in ['easting', 'northing']]
    df = pd.read_csv(csv_path, usecols=columns_to_use)
    df, _ = lonlat_to_canada_lambert(df)
    
    print("Calculating absolute tile coordinates...")
    # Apply the exact grid math
    df_mapped = append_tile_coordinates(df, 
                                        grid_size=settings.GRID_SIZE, 
                                        pixel_size=settings.PIXEL_SIZE, 
                                        x_col=settings.X_COL, 
                                        y_col=settings.Y_COL)
    
    print(f"Building mapper dictionary for {year}...")
    mapper = {}
    
    # Group the Global Anchors (Tile ID and DOB)
    grouped = df_mapped.groupby(['tile_id', 'DOB'])
    
    for (tile_id, dob), group in tqdm(grouped, desc=f"Processing {year} Tile/DOB Groups"):
        
        dob_int = int(float(dob))
        year_int = int(year)
        
        # Create a unique global key
        global_key = f"{tile_id}_DOB_{year_int}_{dob_int}"
        
        # Extract only the unique combinations of Fire ID and Fireday for this tile
        unique_fires = group[['ID', 'fireday']].drop_duplicates()
        
        fires_list = []
        for _, row in unique_fires.iterrows():
            fire_id = str(row['ID'])
            fireday = int(row['fireday'])
            
            # Construct the exact HDF5 group name
            day_key = f"day_{fireday:03d}" 
            
            fires_list.append({
                "fire_id": fire_id,
                "day_key": day_key
            })
            
        mapper[global_key] = fires_list

    # Print Stats
    total_unique_combinations = len(mapper)
    overlapping_combinations = sum(1 for fires in mapper.values() if len(fires) > 1)
    max_overlap = max((len(fires) for fires in mapper.values()), default=0)
    
    print("\n" + "="*50)
    print(f" MAPPER GENERATION COMPLETE FOR {year}")
    print("="*50)
    print(f"Total Unique (Tile + Date) Combinations : {total_unique_combinations}")
    print(f"Combinations with Overlapping Fires     : {overlapping_combinations}")
    print(f"Maximum concurrent fires in single tile : {max_overlap}")
    print("="*50 + "\n")

    # Save to JSON
    output_json_path = os.path.join(output_folder, f"tile_dob_mapper_{year}.json")
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_json_path, 'w') as out_file:
        json.dump(mapper, out_file, indent=4)
        
    print(f"Saved successfully to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate offline tile mapper from raw data.")
    parser.add_argument("year", type=str, help="The target year to process (e.g., 2024)")
    
    args = parser.parse_args()

    out_dir = settings.METADATA_FOLDER

    print(f'Metadata folder : {out_dir}')
    print(f'Metadata year : {args.year}')
    
    generate_mapper_for_year(
        year=args.year,
        output_folder=out_dir
    )