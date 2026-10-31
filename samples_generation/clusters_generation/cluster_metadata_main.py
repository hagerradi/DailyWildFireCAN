import argparse
from src.config import Config

from samples_generation.clusters_generation.cluster_generator import generate_fold_metadata

def main():
    parser = argparse.ArgumentParser(description="Metadata Generator (Stats & Buffer) for Wildfire Folds")
    
    parser.add_argument(
        "--type", 
        type=str, 
        required=True, 
        choices=["simple", "timeseries"],
        help="Choose which dataset type to generate metadata for: 'simple' or 'timeseries'"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml", 
        help="Path to YAML config file."
    )

    # Argument for parallelizing by fold
    parser.add_argument(
        "--fold_id", 
        type=int, 
        required=True, 
        help="The specific fold ID to generate metadata for (e.g., 0 to 9)."
    )
    
    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    if args.type == "simple":
        print(f"Starting Metadata generation (Stats & Buffers) for SIMPLE dataset | Fold {args.fold_id}...")
        generate_fold_metadata(args.fold_id)
        
    elif args.type == "timeseries":
        print(f"Starting Metadata generation for TIME-SERIES dataset | Fold {args.fold_id}...")
        pass

if __name__ == "__main__":
    main()