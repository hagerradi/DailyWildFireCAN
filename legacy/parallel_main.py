import argparse
import os
from pathlib import Path

from src.config import Config
from legacy.train import train
from src.test import test

def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire segmentation — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    
    # --- Parallel & Experiment Arguments ---
    parser.add_argument("--run_id", type=int, default=1, help="Run ID (1, 2, or 3)")
    parser.add_argument("--arch", type=str, default=None, help="Override model architecture (e.g., unet_age)")
    
    args = parser.parse_args()

    # Load the dataclass-based config from YAML
    config = Config.from_yaml(args.config)
    
    # ARCHITECTURE OVERRIDE
    if args.arch is not None:
        config.model.architecture = args.arch

    # SEED MATH: (run_id * (fixed_seed + (run_id - 1))) % (2 ** 31 - 1)
    fixed_seed = config.seed 
    dynamic_seed = (args.run_id * (fixed_seed + (args.run_id - 1))) % (2**31 - 1)
    config.seed = dynamic_seed

    # CHECKPOINT DIRECTORY
    config.training.checkpoint_dir = f"checkpoints_{config.model.architecture}_run{args.run_id}"
    
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    # COMET LOGGING
    if config.comet.enabled:
        config.comet.experiment_name = f"{config.model.architecture}_run{args.run_id}"

    print(f"\n--- [EXPERIMENT BATCH INIT] ---")
    print(f"Architecture:  {config.model.architecture}")
    print(f"Run ID:        {args.run_id}")
    print(f"Generated Seed:{config.seed}")
    print(f"Output Path:   {config.training.checkpoint_dir}")
    print(f"-------------------------------\n")
    # =====================================================================

    # Start Training
    trainer, test_loader = train(config)

    # Run Testing
    checkpoint_dir = Path(config.training.checkpoint_dir)
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
    
    if best_checkpoint.exists():
        print(f"\n--- Starting Final Test Evaluation with {best_checkpoint} ---")        
        test(trainer, best_checkpoint, test_loader)
    else:
        print(f"No best_checkpoint.pt found in {checkpoint_dir}")

if __name__ == "__main__":
    main()