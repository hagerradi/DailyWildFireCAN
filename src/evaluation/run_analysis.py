import argparse

from src.config import Config
from src.evaluation.generalization_analysis import run_generalization_analysis
from src.evaluation.spatial_analysis import run_spatial_analysis
from src.evaluation.variation_analysis import run_variation_analysis

def main() -> None:
    parser = argparse.ArgumentParser(description="Analysis — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    
    # Run the generalization analysis
    run_generalization_analysis(config)
    
    # Run the spatial analysis
    run_spatial_analysis(config)
    
    # # Run the spatial variation analysis
    # run_variation_analysis(config, uncertainty_mode='spatial')

    # # Run the temporal variation analysis
    # run_variation_analysis(config, uncertainty_mode='temporal')

if __name__ == "__main__":
    main()