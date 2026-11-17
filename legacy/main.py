import argparse

from src.config import Config
from legacy.train import train
from src.test import test
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire segmentation — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    trainer, test_loader = train(config)


    # Run Testing
    checkpoint_dir = Path(config.training.checkpoint_dir)
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
    
    if best_checkpoint.exists():
        print(f"\n--- Starting Final Test Evaluation with {best_checkpoint} ---")        
        # Run test
        test(trainer, best_checkpoint, test_loader)
    else:
        print("No best_checkpoint.pt found to test.")

if __name__ == "__main__":
    main()