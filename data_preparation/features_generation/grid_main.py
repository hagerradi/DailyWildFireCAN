import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from configs import settings
from data_preparation.features_generation.helpers import lonlat_to_canada_lambert
import data_preparation.features_generation.grid_generation as grid_generation
import data_preparation.features_generation.weather as weather
import data_preparation.features_generation.fuel_scanfi as fuel_scanfi
import data_preparation.features_generation.fuel_viirs as fuel_viirs
import data_preparation.features_generation.topography as topography
import data_preparation.features_generation.fuel_landcover as fuel_landcover
import data_preparation.features_generation.nbac_disturbance_cumulative as nbac_disturbance_cumulative
import data_preparation.features_generation.human_index as human_index

TOPO_VARS = {
    'dem': settings.ELEVATION_FOLDER,
    'slope': settings.SLOPE_FOLDER,
    'aspect': settings.ASPECT_FOLDER
}

def process_fire_pipeline(fire_id, 
                          fire_growth_pts, 
                          scanfi_year, 
                          landcover_year, 
                          disturbance_year,
                          hii_year):
    """Executes the complete data extraction pipeline for a single fire.

    Args:
      fire_id: the fire ID
      fire_growth_pts: the CFSD dataframe 
      scanfi_year: the year of the SCANFI data
      landcover_year: the year of the LANDCOVER data
      disturbance_year: the year of the DISTURBANCE data
      hii_year: the year of the HII data
    """
    print(f"\n--- Starting Fire: {fire_id} ---")
    
    # Filter DF for this specific fire
    fire_df = fire_growth_pts[fire_growth_pts['ID'] == fire_id].copy()
    fire_df, _ = lonlat_to_canada_lambert(fire_df)

    # Grid Construction
    grid_generation.generate_fire_h5(
        df=fire_df,
        fire_id=fire_id,
        output_folder=settings.H5_OUTPUT_FOLDER, 
        grid_size=settings.GRID_SIZE, 
        pixel_size=settings.PIXEL_SIZE, 
        x_col=settings.X_COL,
        y_col=settings.Y_COL,
        lon_col=settings.LON_COL,
        lat_col=settings.LAT_COL,
        fill_value=np.nan
    )

    h5_path = Path(settings.H5_OUTPUT_FOLDER) / f"fire_{fire_id}.h5"
    
    # Generate the environmental variables
    weather.run_single_h5_era5_pipeline(h5_path, settings.ERA5_FOLDER)
    topography.run_static_topography_pipeline(h5_path, TOPO_VARS)
    fuel_scanfi.run_single_fire_scanfi(h5_path, settings.SCANFI_FOLDER, scanfi_year)
    fuel_viirs.run_daily_viirs_pipeline(h5_path)
    fuel_landcover.run_single_fire_landcover(h5_path, settings.LANDCOVER_FOLDER, landcover_year)
    nbac_disturbance_cumulative.run_single_fire_annual_nbac_disturbance(h5_path, 
                                                          annual_disturbance_folder=settings.DISTURBANCE_FOLDER,
                                                          disturbance_year=disturbance_year)
    human_index.run_single_fire_hii(h5_path, 
                                        hii_folder=settings.HII_ORG_FOLDER,
                                        hii_year=hii_year)

    print(f"--- Finished Fire: {fire_id} ---")


def run_local(all_fire_ids, fire_growth_pts, scanfi_year, landcover_year, disturbance_year, hii_year):
    """Processes all fires sequentially for local/laptop execution.

    Args:
      all_fire_ids: list of fires IDs
      fire_growth_pts: the CFSD dataframe
      scanfi_year: the year of the SCANFI data
      landcover_year: the year of the CCRS LANDCOVER data
      disturbance_year: the year of the DISTURBANCE data
      hii_year: the year of the HUMAN INDEX data
    """
    total_fires = len(all_fire_ids)
    print(f"\n=== LOCAL MODE: Processing ALL {total_fires} fires sequentially ===")
    
    for i, fire_id in enumerate(all_fire_ids, start=1):
        print(f"\n[{i}/{total_fires}] Processing Fire ID: {fire_id}")
        process_fire_pipeline(fire_id, 
                              fire_growth_pts, 
                              scanfi_year=scanfi_year,
                              landcover_year=landcover_year,
                              disturbance_year=disturbance_year,
                              hii_year=hii_year)


def run_distributed(all_fire_ids, 
                    fire_growth_pts, 
                    scanfi_year, 
                    landcover_year,
                    disturbance_year,
                    hii_year,
                    task_id, 
                    chunk_size):
    """Processes a chunk of fires based on the SLURM Array Task ID.

    Args:
      all_fire_ids: the list fire IDs
      fire_growth_pts: the CFSD dataframe 
      scanfi_year: the year of the SCANFI data
      landcover_year: the year of the LANDCOVER data
      disturbance_year: the year of the DISTURBANCE data
      hii_year: the year of the HUMAN INDEX data
      task_id: the ID of the task for the distributed job
      chunk_size: the number of fires to process in the same task
    """
    total_fires = len(all_fire_ids)
    
    start_idx = task_id * chunk_size
    end_idx = start_idx + chunk_size
    
    # Safety check if the math pushes the start index beyond the list length
    if start_idx >= total_fires:
        print(f"[!] Task ID {task_id} starts at {start_idx}, which is out of bounds (max {total_fires}). Exiting gracefully.")
        sys.exit(0)
        
    my_fires = all_fire_ids[start_idx:end_idx]
    
    print(f"\n=== DISTRIBUTED MODE: Task {task_id} ===")
    print(f"Processing {len(my_fires)} fires (Indices {start_idx} to {start_idx + len(my_fires) - 1} out of {total_fires})")
    print(f"Assigned Fires: {my_fires}")
    
    for fire_id in my_fires:
        process_fire_pipeline(fire_id, 
                              fire_growth_pts, 
                              scanfi_year=scanfi_year, 
                              landcover_year=landcover_year,
                              disturbance_year=disturbance_year,
                              hii_year=hii_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Complete Feature H5 Files for Wildfires.")
    parser.add_argument("year", type=str, help="The target year to process (e.g., 2024)")
    parser.add_argument("--mode", type=str, choices=["local", "distributed"], default="local", 
                        help="Execution mode: 'local' (sequential) or 'distributed' (parallel array task)")
    parser.add_argument("--task-id", type=int, default=-1, 
                        help="SLURM Array Task ID (Required if mode is 'distributed')")
    parser.add_argument("--chunk-size", type=int, default=5,
                        help="Number of fires to process per SLURM task (default: 5)")
    
    args = parser.parse_args()

    print(f"Grid Size Configuration: {settings.GRID_SIZE}")

    target_year = args.year
    print(f"Target Year: {target_year}")

    # 5-year interval logic
    target_year_int = int(target_year)
    if target_year_int > 2020:
        scanfi_year = "2020"
        landcover_year = "2020"
        hii_year = "2020"
    else:
        scanfi_year = "2015"
        landcover_year = "2015"
        hii_year = str(target_year_int - 1)
    disturbance_year = str(target_year_int - 1)
        
    print(f"Using SCANFI data from year: {scanfi_year}")
    
    csv_path = f"{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{target_year}/Firegrowth_pts_v1_1_{target_year}.csv"
    try:
        fire_growth_pts = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not find CSV at {csv_path}")
        sys.exit(1)
        
    all_fire_ids = sorted(fire_growth_pts['ID'].unique())
    print(f"Number of unique fires: {len(all_fire_ids)}")

    # Route to the correct execution path
    if args.mode == "local":
        run_local(all_fire_ids, 
                  fire_growth_pts, 
                  scanfi_year=scanfi_year,
                  landcover_year=landcover_year,
                  disturbance_year=disturbance_year,
                  hii_year=hii_year)
    
    elif args.mode == "distributed":
        if args.task_id == -1:
            print("[!] Error: You must provide a --task-id when using distributed mode.")
            sys.exit(1)
        run_distributed(all_fire_ids, 
                        fire_growth_pts, 
                        scanfi_year=scanfi_year, 
                        landcover_year=landcover_year,
                        disturbance_year=disturbance_year,
                        hii_year=hii_year,
                        task_id=args.task_id, 
                        chunk_size=args.chunk_size)