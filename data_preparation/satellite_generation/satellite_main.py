import sys
import argparse
from pathlib import Path
import pandas as pd
import gc

from configs import settings
import data_preparation.satellite_generation.satellite_sentinel as satellite_sentinel
from data_preparation.data_configs.data_settings import SENTINEL_BANDS


def process_fire_pipeline(fire_id, task_id="Local"):
    """Executes the satellite download and processing pipeline for a single fire.

    Args:
      fire_id: the target fire's ID
      task_id: the task id for the distributed job (Default value = "Local")

    Returns:

    """
    print(f"\n--- [Task {task_id}] Starting Satellite Pipeline for Fire: {fire_id} ---")
    
    # Define Paths and verify Grid H5 exists
    h5_path = Path(settings.H5_OUTPUT_FOLDER) / f"fire_{fire_id}.h5"
    
    if not h5_path.exists():
        print(f"!!! [Task {task_id}] Error: {h5_path} not found. Ensure Grid pipeline ran first.")
        return 

    # Setup Tracking Markers
    Path(settings.SATELLITE_STATUS_FOLDER).mkdir(parents=True, exist_ok=True)
    success_marker = Path(settings.SATELLITE_STATUS_FOLDER) / f"success_{fire_id}.txt"
    fail_marker = Path(settings.SATELLITE_STATUS_FOLDER) / f"fail_{fire_id}.txt"

    if success_marker.exists():
        print(f"--- [Task {task_id}] Skipping Fire {fire_id}: Already processed successfully ---")
        return
    
    if fail_marker.exists():
        print(f"--- [Task {task_id}] Clearing previous failure marker for Fire {fire_id} ---")
        fail_marker.unlink()

    selected_bands = SENTINEL_BANDS

    # Run the Pipeline
    try:
        satellite_sentinel.run_s2_h5_pipeline(
            h5_path=h5_path,
            # catalog=catalog,
            bands=selected_bands
        )
        success_marker.touch()
        print(f"--- [Task {task_id}] Successfully finished Fire: {fire_id} ---")
        
    except Exception as e:
        fail_marker.touch()
        print(f"!!! [Task {task_id}] Failed Fire {fire_id}: {str(e)}")
        # If we are on a distributed system, exit(1) so SLURM marks the job as FAILED
        if task_id != "Local":
            sys.exit(1)


def run_local(all_fire_ids):
    """Processes all fires sequentially for local/laptop execution.

    Args:
      all_fire_ids: the list of fires' IDs
    """
    total_fires = len(all_fire_ids)
    print(f"\n=== LOCAL MODE: Processing ALL {total_fires} fires sequentially ===")
    
    for i, fire_id in enumerate(all_fire_ids, start=1):
        print(f"\n[{i}/{total_fires}] Processing Fire ID: {fire_id}")
        process_fire_pipeline(fire_id, task_id="Local")


def run_distributed(all_fire_ids, task_id):
    """Processes a single fire based exactly on the SLURM Array Task ID.

    Args:
      all_fire_ids: the list of fires' IDs
      task_id: the task id for the distributed job

    """
    total_fires = len(all_fire_ids)
    
    if task_id < 0 or task_id >= total_fires:
        print(f"[!] Error: Task ID {task_id} is out of bounds (0 to {total_fires - 1}). Exiting.")
        sys.exit(1)
        
    target_fire_id = all_fire_ids[task_id]
    print(f"\n=== DISTRIBUTED MODE: Task {task_id} / {total_fires - 1} ===")
    
    process_fire_pipeline(target_fire_id, task_id=task_id)


def main():
    
    parser = argparse.ArgumentParser(description="Generate Satellite Images for Wildfires.")
    parser.add_argument("year", type=str, help="The target year to process (e.g., 2024)")
    parser.add_argument("--mode", type=str, choices=["local", "distributed"], default="local", 
                        help="Execution mode: 'local' (sequential) or 'distributed' (parallel array task)")
    parser.add_argument("--task-id", type=int, default=-1, 
                        help="SLURM Array Task ID (Required if mode is 'distributed')")
    
    args = parser.parse_args()

    target_year = args.year
    print(f"Target Year: {target_year}")
    
    csv_path = f"{settings.BASE_FOLDER}/Firegrowth_pts_v1_1_{target_year}/Firegrowth_pts_v1_1_{target_year}.csv"
    try:
        fire_growth_pts = pd.read_csv(csv_path, usecols=['ID'])
    except FileNotFoundError:
        print(f"[!] Error: Could not find CSV at {csv_path}")
        sys.exit(1)

    all_fire_ids = sorted(fire_growth_pts['ID'].unique())
    print(f"Number of unique fires: {len(all_fire_ids)}")
    # all_fire_ids = ['2024_160']

    del fire_growth_pts
    gc.collect() 
    print("DataFrame deleted to free up RAM.")

    # Route to the correct execution path
    if args.mode == "local":
        run_local(all_fire_ids)
    elif args.mode == "distributed":
        if args.task_id == -1:
            print("[!] Error: You must provide a --task-id when using distributed mode.")
            sys.exit(1)
        run_distributed(all_fire_ids, args.task_id)


if __name__ == "__main__":
    main()