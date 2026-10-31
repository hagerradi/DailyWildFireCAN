import argparse
from pathlib import Path

from src.config import Config
from src.test import test
from src.train_advanced import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire segmentation — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--run_id", type=int, default=1, help="Run ID (e.g., 1, 2, or 3)")

    args = parser.parse_args()

    # Load configuration
    config = Config.from_yaml(args.config)

    # Calculate dynamic seed reproducibly
    fixed_seed = config.seed
    config.seed = (args.run_id * (fixed_seed + (args.run_id - 1))) % (2**31 - 1)

    # Nested Directory Structure: checkpoints_<arch>/run_<id>
    checkpoint_dir = Path(f"checkpoints_{config.model.architecture}") / f"run_{args.run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Pass path to config as string for compatibility
    config.training.checkpoint_dir = str(checkpoint_dir)

    # Comet logging setup
    if config.comet.enabled:
        config.comet.experiment_name = f"{config.model.architecture}_run{args.run_id}"

    print("\n--- [EXPERIMENT BATCH INIT] ---")
    print(f"Architecture:   {config.model.architecture}")
    print(f"Run ID:         {args.run_id}")
    print(f"Base Seed:      {fixed_seed}")
    print(f"Generated Seed: {config.seed}")
    print(f"Output Path:    {checkpoint_dir}")
    print("-------------------------------\n")

    # Start Training
    trainer, test_space_loader, test_time_loader, test_spacetime_loader = train(config)

    # Run Testing
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"

    if best_checkpoint.exists():
        print(f"\n--- Starting Final Test Evaluation with {best_checkpoint} ---")
        
        print("\n[TEST-SPACE]")
        test(trainer, best_checkpoint, test_space_loader, generate_image=False)

        print("\n[TEST-TIME]")
        test(trainer, best_checkpoint, test_time_loader, generate_image=False)

        print("\n[TEST-SPACETIME]")
        test(trainer, best_checkpoint, test_spacetime_loader, generate_image=True)
    else:
        print(f"WARNING: Best checkpoint not found at {best_checkpoint}")


if __name__ == "__main__":
    main()