import argparse
from pathlib import Path
from src.config import Config
from src.train_clusters import train
from src.test import test

def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire segmentation — CV Entry Point")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    parser.add_argument("--fold_id", type=int, required=True, help="Fold ID for cross-validation (0 to 9).")
    
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    fold_id = args.fold_id

    # ---------------------------------------------------------
    # DYNAMIC CHECKPOINT OVERRIDE PER FOLD
    # Result: checkpoints_model_cross_val/fold_0/
    # ---------------------------------------------------------
    base_checkpoint_dir = Path(f"{config.training.checkpoint_dir}_cross_val")
    fold_checkpoint_dir = base_checkpoint_dir / f"fold_{fold_id}"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    config.training.checkpoint_dir = str(fold_checkpoint_dir)
    print(f"Checkpoints for Fold {fold_id} -> {fold_checkpoint_dir}")

    # ---------------------------------------------------------
    # DYNAMIC COMET EXPERIMENT NAME OVERRIDE
    # Result: "model_fold_0"
    # ---------------------------------------------------------
    if hasattr(config, 'comet') and getattr(config.comet, 'enabled', False):
        base_exp_name = getattr(config.comet, 'experiment_name', 'experiment')
        config.comet.experiment_name = f"{base_exp_name}_fold_{fold_id}"
        
        # Optionally add the fold tag dynamically
        if hasattr(config.comet, 'experiment_tags') and config.comet.experiment_tags is not None:
            config.comet.experiment_tags.append(f"fold_{fold_id}")
            
        print(f"Comet Experiment Name -> {config.comet.experiment_name}")

    # ---------------------------------------------------------
    # TRAIN & TEST EXECUTION
    # ---------------------------------------------------------
    trainer, test_loader = train(config, fold_id=fold_id)

    # Run Testing on Best Checkpoint
    best_checkpoint = fold_checkpoint_dir / "best_checkpoint.pt"
    
    if best_checkpoint.exists():
        print(f"\n--- Starting Final Test Evaluation for Fold {fold_id} with {best_checkpoint} ---")        
        test(trainer, best_checkpoint, test_loader)
    else:
        print(f"No best_checkpoint.pt found in {fold_checkpoint_dir} to test.")

if __name__ == "__main__":
    main()