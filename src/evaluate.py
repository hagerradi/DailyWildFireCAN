import os
import torch
import torch.nn as nn
import pandas as pd
from src.config import Config
import argparse

# Dataloders
from src.dataloader import get_dataloaders
from src.dataloader_timeseries import get_timeseries_dataloaders

# Models
from src.models import unet
from src.models import unet_time_gap
from src.models import unet_convlstm
from src.models import unet_attention
from src.models import unet_segformer
from src.models import utae

from src.evaluator import Evaluator
from src.utils import seed_everything
from src.focal_loss import FocalLoss 
from src.dice_loss import DiceLoss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        
        # Add smooth=1.0 back to prevent gradient death on empty masks
        self.dice = DiceLoss(smooth=1.0, apply_sigmoid=True)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
            
        focal_l = self.focal(inputs, targets)
        dice_l = self.dice(inputs, targets)
        
        return (self.focal_weight * focal_l) + (self.dice_weight * dice_l)

def setup_evaluation(config: Config, architecture: str):
    """
    Initializes the model, dataloaders, and trainer for a specific architecture.
    """

    seed_everything(config.seed)

    mc = config.model
    mc.architecture = architecture # Override config with current loop architecture
    
    # Setup Dataloaders & Model based on architecture
    if mc.architecture == 'unet':
        _, _, test_loader = get_dataloaders(config, is_sat_age=False)

        model = unet.UNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )
        
    elif mc.architecture == 'unet_attention':

        _, _, test_loader = get_dataloaders(config, is_sat_age=False)

        model = unet_attention.AttentionUNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
            use_attention=True
        )
        
    elif mc.architecture == 'unet_time_gap':

        _, _, test_loader = get_dataloaders(config, is_sat_age=True)

        model = unet_time_gap.UNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )
        
    elif mc.architecture == 'unet_convlstm':

        _, _, test_loader = get_timeseries_dataloaders(config, return_positions=False)

        model = unet_convlstm.SpatiotemporalUNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections
        )
        
    elif mc.architecture == 'unet_segformer':
        _, _, test_loader = get_dataloaders(config, is_sat_age=False)

        model = unet_segformer.UNetSegFormer(
            in_channels=mc.input_channels,
            out_classes=mc.num_classes
        )
    
    elif mc.architecture == 'utae':
        _, _, test_loader = get_timeseries_dataloaders(
            config,
            return_positions=True,
        )
        model = utae.UTAE(
            input_dim=mc.input_channels,
            num_classes=mc.num_classes,
            encoder_widths=mc.hidden_features,
            decoder_widths=mc.utae_decoder_widths,
            out_conv_channels=mc.utae_out_conv_channels,
            agg_mode=mc.utae_agg_mode,
            encoder_norm=mc.utae_encoder_norm,
            n_head=mc.utae_n_head,
            d_model=mc.utae_d_model,
            d_k=mc.utae_d_k,
            pad_value=mc.utae_pad_value,
        )

    else:
        raise ValueError(f"Unknown architecture: '{mc.architecture}'")

    # Setup Loss
    device = Evaluator._resolve_device(config.training.device)
    
    if config.training.use_cumuarea:
        # 3-CLASS SETUP
        print("Initializing 3-Class CrossEntropy Loss...")
        weights = torch.tensor([1.0, 10.0, 50.0], dtype=torch.float32).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        # 2-CLASS SETUP (Using Custom Focal + Dice)
        # 0 = Background + Old Fire, 1 = New Fire
        print("Initializing 2-Class Combo Loss (Focal + Dice)...")
        # Higher gamma = harder focus on difficult pixels
        loss_fn = CombinedLoss(alpha=0.85, gamma=2.0, focal_weight=0.5, dice_weight=0.5).to(device)

    # Evaluator
    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn, 
        config=config.training, 
        logger=None
    )
    
    return evaluator, test_loader

def main():

    parser = argparse.ArgumentParser(description="Wildfire segmentation evaluation — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    # ABLATION CONFIG PATHS HERE
    ENV_CONFIG_PATH = "configs/env.yaml" 
    SAT_CONFIG_PATH = "configs/sat.yaml"

    architectures = [
        'unet', 
        'unet_attention', 
        'unet_time_gap', 
        'unet_convlstm',  
        'unet_segformer',
        'utae'
    ]
    runs = [1, 2, 3]
    checkpoint_base_dir = "."
    
    all_results = []

    # Iterate over every model architecture
    for arch in architectures:
        print(f"\n{'='*40}\nEvaluating Architecture: {arch}\n{'='*40}")

        # CONFIG SWITCHING LOGIC
        if arch == 'unet_env':
            current_config_path = ENV_CONFIG_PATH
        elif arch == 'unet_sat':
            current_config_path = SAT_CONFIG_PATH
        else:
            current_config_path = args.config # Reverts to the default config from the argparse
            
        print(f"Loading config from: {current_config_path}")
        config = Config.from_yaml(current_config_path)
        
        # Setup the environment once per architecture
        evaluator, test_loader = setup_evaluation(config, arch)
        
        # Iterate over the 3 runs
        for run in runs:
            # Construct folder name: e.g., checkpoints_unet_run1
            folder_name = f"checkpoints_{arch}_run{run}"

            ckpt_path = os.path.join(checkpoint_base_dir, folder_name, "best_checkpoint.pt")
            
            if not os.path.exists(ckpt_path):
                print(f"[!] Warning: Checkpoint not found at {ckpt_path}. Skipping.")
                continue

            print(f'\n--- RUN {run} ---') 
            print(f"\n--- Loading {folder_name} ---")
            epoch = evaluator.load_checkpoint(ckpt_path)
            
            # Run Evaluation
            test_loss, test_metrics = evaluator.evaluate(test_loader)

            for metric_name, value in test_metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Store metrics
            result_dict = {
                "architecture": arch,
                "run": run,
                "epoch_loaded": epoch,
                "test_loss": test_loss,
            }
            # Unpack all metrics (e.g., IoU, F1) into the dict
            result_dict.update(test_metrics) 
            all_results.append(result_dict)

    # Save results to a clean CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = "evaluation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nAll evaluations complete! Results saved to {output_csv}")
        print(df.to_string())
    else:
        print("\nNo evaluations were successfully completed.")

if __name__ == "__main__":
    main()