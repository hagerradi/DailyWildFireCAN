import argparse

from src.config import Config
from samples_generation.data_generator_advanced import generate_metadata_all_datasets
from samples_generation.data_generator_advanced_timeseries import generate_timeseries_metadata_all_datasets
from samples_generation.samples_configs.samples_settings import SEQUENCE_LENGTH

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
    
    args = parser.parse_args()
    config = Config.from_yaml(args.config)

    if args.type == "simple":
        print("Starting metadata generation for SIMPLE dataset...")
        generate_metadata_all_datasets(config)
        
    elif args.type == "timeseries":
        print("Starting metadata generation for TIME-SERIES dataset...")
        generate_timeseries_metadata_all_datasets(config, SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()