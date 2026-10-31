from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import uuid

from src.trainer import Trainer


def test(trainer: Trainer, checkpoint_path: str, test_loader: DataLoader, generate_image: bool = True) -> float:
    """Load a checkpoint and evaluate on the test split.

    Args:
      trainer: Trainer used to train the model.
      checkpoint_path: Path to the model checkpoint file.
      test_loader: DataLoader containing the test dataset.

    Returns:
      : Average test loss.

    """
    
    epoch = trainer.load_checkpoint(checkpoint_path)

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Evaluating on device: {trainer.device}")

    test_loss, test_iou_metrics = trainer.validate(test_loader, "Test", epoch=None)
    
    print(f"Test loss: {test_loss:.4f}")

    for metric_name, value in test_iou_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    if trainer.logger is not None and generate_image:
        
        print("\nGetting test images for Comet visualization...")
        trainer.model.eval()
        
        images_logged = 0
        max_images = 300
        min_pixels = 50 # Minimum pixels
        
        # ==========================================
        # DYNAMIC SETUP
        # Check if we are in 3-class or 2-class mode
        # ==========================================
        is_3_class = trainer.config.use_cumuarea
        vmax_val = 2 if is_3_class else 1
        
        with torch.no_grad():

            for batch in test_loader:

                if images_logged >= max_images:
                    break # Stop if we hit max

                # ==========================================
                # UNIFIED FORWARD PASS
                # ==========================================
                labels = batch["label"].to(trainer.device)
                
                # Let the Trainer handle the messy routing!
                predictions = trainer._forward_pass(batch)

                # ==========================================
                # DYNAMIC PREVIOUS FIRE LABEL EXTRACTION
                # ==========================================
                # Find which tensor contains the environmental/state features
                if "input_grids" in batch:
                    env_features = batch["input_grids"]
                elif "input_env" in batch:
                    env_features = batch["input_env"]
                elif "state" in batch:
                    env_features = batch["state"]
                else:
                    raise ValueError("Unrecognized batch structure for visualization.")

                # Extract the previous fire mask (-2 index based on your original logic)
                if env_features.ndim == 5:
                    # Time-Series: (Batch, Time, Channel, H, W)
                    prev_fire_labels = env_features[:, -1, -2, :, :]
                elif env_features.ndim == 4:
                    # Spatial: (Batch, Channel, H, W)
                    prev_fire_labels = env_features[:, -2, :, :]
                else:
                    raise ValueError(f"Unexpected input tensor dimensions: {env_features.shape}")

                # ==========================================
                # PROBABILITY MAPS & PREDICTIONS
                # ==========================================
                if predictions.shape[1] == 1:
                    # Convert raw logits to probabilities
                    pred_probs = torch.sigmoid(predictions).squeeze(1)
                    # Apply the new aggressive threshold
                    pred_classes = (pred_probs > trainer.binary_classification_threshold).long()
                    prob_maps = pred_probs
                else:
                    pred_classes = torch.argmax(predictions, dim=1)
                    
                    # Convert logits to probabilities
                    pred_probs_all = torch.softmax(predictions, dim=1)
                    
                    # Extract just the "New Fire" probability layer
                    target_class_idx = 2 if is_3_class else 1
                    prob_maps = pred_probs_all[:, target_class_idx, :, :]

                # ==========================================
                # COARSE / CLEANED LABELS GENERATION
                # ==========================================
                downscale = 2

                # Cleaned GT
                raw_labels = labels.float().unsqueeze(1)
                dilated_labels = F.max_pool2d(raw_labels, kernel_size=3, stride=1, padding=1)
                closed_labels = -F.max_pool2d(-dilated_labels, kernel_size=3, stride=1, padding=1)
                coarse_labels = F.max_pool2d(closed_labels, kernel_size=downscale, stride=downscale).squeeze(1).long()
                
                # Cleaned Predictions
                pred_float = pred_classes.float().unsqueeze(1)
                dilated_preds = F.max_pool2d(pred_float, kernel_size=3, stride=1, padding=1)
                closed_preds = -F.max_pool2d(-dilated_preds, kernel_size=3, stride=1, padding=1)
                coarse_preds = F.max_pool2d(closed_preds, kernel_size=downscale, stride=downscale).squeeze(1).long()
                
                # ==========================================
                # IMAGE LOGGING LOOP
                # ==========================================
                # Check each individual image in this batch
                for i in range(labels.size(0)):
                    if images_logged >= max_images:
                        break
                        
                    true_label = labels[i]
                    
                    # DYNAMIC FILTERING LOGIC
                    if is_3_class:
                        # 3-Class: Need both Old Fire (1) and New Fire (2)
                        c1_count = (true_label == 1).sum().item()
                        c2_count = (true_label == 2).sum().item()
                        is_interesting = (c1_count >= min_pixels) and (c2_count >= min_pixels)
                    else:
                        # 2-Class: Just need enough New Fire (1)
                        c1_count = (true_label == 1).sum().item()
                        is_interesting = (c1_count >= min_pixels)
                    
                    # If it meets our criteria
                    if is_interesting:
                        true_np = true_label.cpu().numpy()
                        pred_np = pred_classes[i].cpu().numpy()
                        prob_np = prob_maps[i].cpu().numpy()
                        
                        # Get numpy array for the previous fire label
                        prev_fire_np = prev_fire_labels[i].cpu().numpy()

                        # Coarse numpy arrays
                        coarse_true_np = coarse_labels[i].cpu().numpy()
                        coarse_pred_np = coarse_preds[i].cpu().numpy()
                        
                        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
                        
                        # Previous Fire Label
                        axes[0].imshow(prev_fire_np, cmap='gray', vmin=0, vmax=1)
                        axes[0].set_title("Previous Fire Label")
                        axes[0].axis('off')
                        
                        # Ground Truth
                        axes[1].imshow(true_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[1].set_title("Ground Truth")
                        axes[1].axis('off')
                        
                        # Test Prediction
                        axes[2].imshow(pred_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[2].set_title("Test Prediction (Best Checkpoint)")
                        axes[2].axis('off')

                        # Probability Map
                        im = axes[3].imshow(prob_np, cmap='magma', vmin=0.0, vmax=1.0)
                        axes[3].set_title("New Fire Probability")
                        axes[3].axis('off')

                        # Ground Truth (Coarse)
                        axes[4].imshow(coarse_true_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[4].set_title(f"Ground Truth (Coarse {downscale}x)")
                        axes[4].axis('off')
                        
                        # Prediction (Coarse)
                        axes[5].imshow(coarse_pred_np, cmap='viridis', vmin=0, vmax=vmax_val)
                        axes[5].set_title(f"Prediction (Coarse {downscale}x)")
                        axes[5].axis('off')
                        
                        # The legend/colorbar on the right side
                        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
                        
                        # Save, log to Comet, and clean up
                        random_id = uuid.uuid4().hex[:6]
                        temp_img = f"test_vis_{images_logged}_{random_id}.png"
                        plt.savefig(temp_img, bbox_inches='tight')
                        plt.close(fig)
                        
                        trainer.logger.log_image(temp_img, name=f"Test_Results/Image_{images_logged}")
                        os.remove(temp_img)
                        
                        images_logged += 1
                        
        print(f"Successfully logged {images_logged} test visualizations to Comet!")

    return test_loss