import sys
import argparse
from pathlib import Path
import pandas as pd

from configs import settings
import data_preparation.features_generation.human_index as human_index

def process_fire_pipeline(fire_id, hii_year):
    """Executes the complete data extraction pipeline for a single fire.

    Args:
      fire_id: the fire ID
      hii_year: the year of the HII data
    """
    print(f"\n--- Starting Fire: {fire_id} ---")

    h5_path = Path(settings.H5_OUTPUT_FOLDER) / f"fire_{fire_id}.h5"
    
    # Generate the hii variable
    human_index.run_single_fire_hii(h5_path, 
                                    hii_folder=settings.HII_ORG_FOLDER,
                                    hii_year=hii_year)

    print(f"--- Finished Fire: {fire_id} ---")


def run_local(all_fire_ids, hii_year):
    """Processes all fires sequentially for local/laptop execution.

    Args:
      all_fire_ids: list of fires IDs
      hii_year: the year of the HII data
    """
    total_fires = len(all_fire_ids)
    print(f"\n=== LOCAL MODE: Processing ALL {total_fires} fires sequentially ===")
    
    for i, fire_id in enumerate(all_fire_ids, start=1):
        print(f"\n[{i}/{total_fires}] Processing Fire ID: {fire_id}")
        process_fire_pipeline(fire_id, hii_year)


def run_distributed(all_fire_ids, hii_year, task_id, chunk_size):
    """Processes a chunk of fires based on the SLURM Array Task ID.

    Args:
      all_fire_ids: the list fire IDs
      hii_year: the year of the HII data
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
        process_fire_pipeline(fire_id, hii_year)


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

    # (T-1)-year interval logic
    target_year_int = int(target_year)
    if target_year_int > 2020:
        hii_year = '2020'
    else:
        hii_year = str(target_year_int - 1)
        
    print(f"Using HII data from year: {hii_year}")
    
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
        run_local(all_fire_ids, hii_year)
    elif args.mode == "distributed":
        if args.task_id == -1:
            print("[!] Error: You must provide a --task-id when using distributed mode.")
            sys.exit(1)
        run_distributed(all_fire_ids, hii_year, args.task_id, args.chunk_size)