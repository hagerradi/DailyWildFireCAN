import argparse

from src.config import Config
from src.train_advanced import train
from src.test import test
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire segmentation — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    trainer, test_space_loader, test_time_loader, test_spacetime_loader = train(config)

    # Run Testing
    checkpoint_dir = Path(config.training.checkpoint_dir)
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
    
    if best_checkpoint.exists():
        print(f"\n--- Starting Final Test Evaluation with {best_checkpoint} ---")        
        # Run test
        print('TEST-SPACE')
        test(trainer, best_checkpoint, test_space_loader, generate_image=False)
        print('\n')
        print('TEST-TIME')
        test(trainer, best_checkpoint, test_time_loader, generate_image=False)
        print('\n')
        print('TEST-SPACETIME')
        test(trainer, best_checkpoint, test_spacetime_loader, generate_image=True)
        print('\n')
    else:
        print("No best_checkpoint.pt found to test.")

if __name__ == "__main__":
    main()