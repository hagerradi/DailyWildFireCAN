import argparse

from src.config import Config
from samples_generation.data_generator_advanced import generate_simple_offline_data
from samples_generation.data_generator_advanced_timeseries import generate_timeseries_offline_data
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
        print("Starting offline data generation for SIMPLE dataset (with ROTATIONS)...")
        generate_simple_offline_data(config)
        
    elif args.type == "timeseries":
        print("Starting offline data generation for TIME-SERIES dataset (with ROTATIONS)...")
        generate_timeseries_offline_data(config, seq_length=SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()