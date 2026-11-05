from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

from torch.utils.data import ConcatDataset, DataLoader

from src.config import Config

from configs import settings

# Dataloders
from src.dataloader_advancel_labels import get_dataloaders_advanced_labels

# Models
from src.models import unet
from src.models import unet_olmo_offline

# Evaluation
from src.evaluator import Evaluator
from src.utils import seed_everything
from src.focal_loss import FocalLoss 
from src.dice_loss import DiceLoss

class CombinedLoss(nn.Module):
    """Calculates a weighted combination of Focal Loss and Dice Loss for segmentation."""
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Add smooth=1.0 back to prevent gradient death on empty masks
        self.dice = DiceLoss(smooth=1.0, apply_sigmoid=True)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        """Computes the combined Focal and Dice loss.

        Args:
            inputs: The predicted logits from the model.
            targets: The ground truth target masks.

        Returns:
            The scalar tensor containing the final weighted loss.
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
            
        focal_l = self.focal(inputs, targets)
        dice_l = self.dice(inputs, targets)
        
        return (self.focal_weight * focal_l) + (self.dice_weight * dice_l)

def run_variation_analysis(config: Config, uncertainty_mode) -> None:
    """Sets up all components and executes the model evaluation loop.

    Initializes the dataloaders, model architecture, optimizer, schedulers, 
    and loss functions based on the provided configuration. Starts the 
    training process using the Trainer class and handles optional CometML logging.

    Args:
        config (Config): The global configuration object containing all hyperparameters 
            for the model, training loop, and logging.

    Returns:
        tuple: A tuple containing:
            - trainer (Trainer): The trainer instance after the training loop completes.
            - test_loader (DataLoader): The dataloader containing the test set split.
    """
    # Fix all random seeds (CPU, CUDA, CuDNN, Python)
    seed_everything(config.seed)

    # Model
    mc = config.model

    device = Evaluator._resolve_device(config.training.device)
    
    if mc.architecture == 'unet':

        train_loader, val_loader, test_space_loader, test_time_loader, test_spacetime_loader =  get_dataloaders_advanced_labels(config, 
                                                                            is_sat_age=False,
                                                                            uncertainty_mode=uncertainty_mode)

        model = unet.UNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )
    
    elif mc.architecture == 'unet_olmo_offline':
        
        train_loader, val_loader, test_space_loader, test_time_loader, test_spacetime_loader =  get_dataloaders_advanced_labels(config, 
                                                                    is_sat_age=False,
                                                                    is_coords=False,
                                                                    is_loc_emb=False,
                                                                    is_olmo_emb=True,
                                                                    uncertainty_mode=uncertainty_mode)
        model = unet_olmo_offline.UNetOlmo(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
            use_olmo=True,
            olmo_channels=768
        )
    
    else:
        raise ValueError(f"Unknown architecture specified in config: '{mc.architecture}'")

    if config.training.use_cumuarea:
        # --- 3-CLASS SETUP ---
        print("Initializing 3-Class CrossEntropy Loss...")
        weights = torch.tensor([1.0, 10.0, 50.0], dtype=torch.float32).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        # --- 2-CLASS SETUP (Using Custom Focal + Dice) ---
        # 0 = Background + Old Fire, 1 = New Fire
        print("Initializing 2-Class Combo Loss (Focal + Dice)...")
        # Higher gamma = harder focus on difficult pixels.
        loss_fn = CombinedLoss(alpha=0.85, gamma=2.0, focal_weight=0.5, dice_weight=0.5).to(device)

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn, 
        config=config.training, 
        logger=None
    )

    # Load model weights before evaluation
    checkpoint_dir = Path(config.training.checkpoint_dir)
    best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        print(f"Loading checkpoint from: {best_checkpoint}")
        loaded_epoch = evaluator.load_checkpoint(best_checkpoint)
        print(f"Successfully loaded checkpoint from epoch {loaded_epoch}")
    else:
        print("No best_checkpoint.pt found to test.")

    # ---------------------------------------------------------
    # MULTI-RUN EVALUATION & AVERAGING
    # ---------------------------------------------------------

    print(f"\nEvaluating on device: {evaluator.device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test-Space batches: {len(test_space_loader)} | Test-Time batches: {len(test_time_loader)} | Test-SpaceTime batches: {len(test_spacetime_loader)}")

    # Merge the PyTorch Datasets
    loaders_list = [test_space_loader, test_time_loader, test_spacetime_loader]
    datasets = [loader.dataset for loader in loaders_list]
    merged_dataset = ConcatDataset(datasets)
    merged_loader = DataLoader(
        merged_dataset,
        batch_size=config.training.batch_size, 
        shuffle=False, 
        num_workers=config.training.num_workers,
        persistent_workers=False
    )
        
    run_ids = [1, 2, 3]
    splits = {
        # "Test-Space": test_space_loader,
        # "Test-Time": test_time_loader,
        # "Test-SpaceTime": test_spacetime_loader
        "Merged-Tests": merged_loader
    }
    
    print("Space:", len(test_space_loader.dataset))
    print("Time:", len(test_time_loader.dataset))
    print("SpaceTime:", len(test_spacetime_loader.dataset))
    print("Dataset size:", len(merged_loader.dataset))
    print("Number of batches:", len(merged_loader))
    
    # Dictionary to hold all metrics across runs
    # Structure: all_results[split][metric] = [val_run1, val_run2, val_run3]
    all_results = {split: {} for split in splits.keys()}

    for run_id in run_ids:
        print(f"\n{'='*50}\nEVALUATING RUN {run_id}\n{'='*50}")
        
        # Build checkpoint path dynamically for each run
        checkpoint_dir = Path(f"checkpoints_{config.model.architecture}") / f"run_{run_id}"
        best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
        
        if not best_checkpoint.exists():
            print(f"[!] Warning: Checkpoint missing for run {run_id} at {best_checkpoint}. Skipping.")
            continue
            
        # Load weights for this specific run
        loaded_epoch = evaluator.load_checkpoint(best_checkpoint)
        print(f"Successfully loaded checkpoint from epoch {loaded_epoch}")
        
        # Evaluate all splits
        for split_name, loader in splits.items():
            loss, iou_metrics = evaluator.evaluate(loader)
            
            # Record Loss
            if "Loss" not in all_results[split_name]:
                all_results[split_name]["Loss"] = []
            all_results[split_name]["Loss"].append(loss)
            
            # Record all other metrics (IoU, F1, Recall, Precision, AUC-PR, etc.)
            for metric_name, value in iou_metrics.items():
                if metric_name not in all_results[split_name]:
                    all_results[split_name][metric_name] = []
                all_results[split_name][metric_name].append(value)
                
            print(f" [Run {run_id} | {split_name}] Loss: {loss:.4f} | Metrics: { {k: round(v, 4) for k, v in iou_metrics.items()} }")

    # CALCULATE AND PRINT FINAL AVERAGES
    print("\n" + "="*70)
    print(" FINAL AVERAGED RESULTS OVER 3 RUNS (MEAN ± STD) ")
    print("="*70)
    
    for split_name, metrics_dict in all_results.items():
        if not metrics_dict or len(metrics_dict.get("Loss", [])) == 0:
            continue # Skip if no runs were evaluated
            
        print(f"\n[Split: {split_name.upper()}]")
        
        # Sort keys
        metric_keys = list(metrics_dict.keys())
        keys_order = ["Test_AUC_PR", "Test_F1", "Test_Recall", "Test_Precision"]
        ordered_keys = [k for k in keys_order if k in metric_keys] + [k for k in metric_keys if k not in keys_order]
        
        for metric_name in ordered_keys:
            values_list = metrics_dict[metric_name]
            mean_val = np.mean(values_list)
            std_val = np.std(values_list)
            
            # Clean up the metric name for printing (e.g., "Test_AUC_PR" -> "AUC_PR")
            display_name = metric_name.replace("Test_", "")
            
            print(f"  -> {display_name.ljust(15)}: {mean_val:.4f} ± {std_val:.4f}")