import argparse

from src.config import Config
from samples_generation.clusters_generation.cluster_generator import generate_simple_offline_data

def main():
    parser = argparse.ArgumentParser(description="Offline Data Generator for Wildfire Preprocessing")
    
    parser.add_argument(
        "--type", 
        type=str, 
        required=True, 
        choices=["simple", "timeseries"],
        help="Choose which dataset type to generate: 'simple' or 'timeseries'"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml", 
        help="Path to YAML config file."
    )

    # Argument for parallelizing by year
    parser.add_argument(
        "--target_year", 
        type=int, 
        required=True, 
        help="The specific year to generate samples for (e.g., 2017)."
    )
    
    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    if args.type == "simple":
        print(f"Starting raw data generation for SIMPLE dataset for year {args.target_year}...")
        generate_simple_offline_data(config, args.target_year)
        
    elif args.type == "timeseries":
        print(f"Starting data generation for TIME-SERIES dataset for year {args.target_year}...")
        pass

if __name__ == "__main__":
    main()