from __future__ import annotations

import os
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import ConcatDataset, Subset, DataLoader

from src.config import Config
from configs import settings

from scipy.ndimage import distance_transform_edt, binary_erosion

# Dataloaders
from src.dataloader_advanced import get_advanced_dataloaders
from src.dataloader_advanced_timeseries import get_advanced_timeseries_dataloaders

# Models
from src.models import unet
from src.models import unet_olmo_offline
from src.models import utae

# Evaluation
from src.evaluator import Evaluator
from src.utils import seed_everything
from src.focal_loss import FocalLoss 
from src.dice_loss import DiceLoss

def run_spatial_error_breakdown(evaluator, loader, split_name, device, threshold=0.5):
    
    print(f"\n=======================================================")
    print(f"Running Advanced Spatial & Boundary Analysis: {split_name.upper()}")
    print(f"=======================================================")

    evaluator.model.eval()
    
    PIXEL_TO_HA = 0.81
    total_overpred_ha, total_underpred_ha, total_true_ha = 0.0, 0.0, 0.0

    # Boundary Accuracy Trackers
    boundary_stats = {
        '90m':  {'tp_p': 0, 'tp_t': 0, 'total_p': 0, 'total_t': 0},
        '180m': {'tp_p': 0, 'tp_t': 0, 'total_p': 0, 'total_t': 0},
        '270m': {'tp_p': 0, 'tp_t': 0, 'total_p': 0, 'total_t': 0}
    }

    # Spread-Distance Trackers
    spread_stats = {
        '0-90m':   {'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0},
        '90-180m': {'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0},
        '180-270m':{'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0},
        '270-360m':{'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0},
        '>360m':   {'tp': 0, 'fp': 0, 'fn': 0, 'n_gt': 0}
    }

    with torch.no_grad():
        for batch_data in loader:
            
            # Targets (label)
            if isinstance(batch_data, dict) and "label" in batch_data:
                targets = batch_data["label"].to(device)
            elif isinstance(batch_data, (list, tuple)):
                # Fallback for plain tuple datasets
                batch_data = {"input_grids": batch_data[0]} 
                targets = batch_data[1].to(device)
            else:
                targets = batch_data['label'].to(device)
                
            # Data construction
            kwargs = {}
            if "loc_emb" in batch_data: kwargs["loc_emb"] = batch_data["loc_emb"].to(device)
            if "satellite_age" in batch_data: kwargs["delta_t"] = batch_data["satellite_age"].to(device) 
            if "olmo_emb" in batch_data: kwargs["olmo_emb"] = batch_data["olmo_emb"].to(device)
            if "alpha_emb" in batch_data: kwargs["alpha_emb"] = batch_data["alpha_emb"].to(device)

            if "input_grids" in batch_data:
                outputs = evaluator.model(batch_data["input_grids"].to(device), **kwargs)
            elif "input_rgb" in batch_data and "input_env" in batch_data:
                outputs = evaluator.model(batch_data["input_rgb"].to(device), batch_data["input_env"].to(device), **kwargs)
            elif "state" in batch_data and "dynamic" in batch_data and "constant" in batch_data:
                outputs = evaluator.model(batch_data["state"].to(device), batch_data["dynamic"].to(device), batch_data["constant"].to(device))
            else:
                outputs = evaluator.model(batch_data['input'].to(device))

            # Current fire day
            # Automatically grabs the second to last channel [:, -2, :, :]
            if "input_grids" in batch_data:
                grids = batch_data["input_grids"]
                if grids.ndim == 4:       # UNet: [B, C, H, W]
                    current_fire_tensor = grids[:, -2, :, :]
                elif grids.ndim == 5:     # UTAE: [B, T, C, H, W]
                    current_fire_tensor = grids[:, -1, -2, :, :]
            elif "input_env" in batch_data:
                current_fire_tensor = batch_data["input_env"][:, -2, :, :] 
            elif "state" in batch_data:
                current_fire_tensor = batch_data["state"][:, -2, :, :]
            else:
                current_fire_tensor = torch.zeros_like(targets) 
                
            current_fire_np = current_fire_tensor.cpu().numpy().astype(bool)

            # Prep masks
            preds = (torch.sigmoid(outputs) > threshold).cpu().numpy().astype(bool)
            targets = targets.cpu().numpy().astype(bool)

            if preds.ndim == 4: preds = preds.squeeze(1)
            if targets.ndim == 4: targets = targets.squeeze(1)
            if current_fire_np.ndim == 4: current_fire_np = current_fire_np.squeeze(1)
            elif current_fire_np.ndim == 3 and targets.ndim == 3: pass # Shape matches

            # Process batch
            for p, t, c_fire in zip(preds, targets, current_fire_np):
                
                # Area Errors
                total_overpred_ha += (p & ~t).sum() * PIXEL_TO_HA
                total_underpred_ha += (~p & t).sum() * PIXEL_TO_HA
                total_true_ha += t.sum() * PIXEL_TO_HA

                # Boundary accuracy
                p_bound = p ^ binary_erosion(p)
                t_bound = t ^ binary_erosion(t)
                
                dist_T = distance_transform_edt(~t_bound) if t_bound.any() else np.full_like(t, 9999)
                dist_P = distance_transform_edt(~p_bound) if p_bound.any() else np.full_like(p, 9999)

                for px_dist, b_name in [(1, '90m'), (2, '180m'), (3, '270m')]:
                    boundary_stats[b_name]['tp_p'] += (p_bound & (dist_T <= px_dist)).sum()
                    boundary_stats[b_name]['total_p'] += p_bound.sum()
                    boundary_stats[b_name]['tp_t'] += (t_bound & (dist_P <= px_dist)).sum()
                    boundary_stats[b_name]['total_t'] += t_bound.sum()

                # Spread distance evaluation
                new_fire_gt = t & ~c_fire
                
                if c_fire.any():
                    dist_from_current = distance_transform_edt(~c_fire) * 90.0
                else:
                    dist_from_current = np.full_like(t, 9999.0)

                bands = {
                    '0-90m':   (dist_from_current > 0) & (dist_from_current <= 90),
                    '90-180m': (dist_from_current > 90) & (dist_from_current <= 180),
                    '180-270m':(dist_from_current > 180) & (dist_from_current <= 270),
                    '270-360m':(dist_from_current > 270) & (dist_from_current <= 360),
                    '>360m':   dist_from_current > 360,
                }

                for band_name, band_mask in bands.items():
                    band_mask = band_mask & ~c_fire
                    gt_band = new_fire_gt & band_mask
                    pred_band = p & band_mask

                    spread_stats[band_name]['tp'] += (pred_band & new_fire_gt).sum()
                    spread_stats[band_name]['fp'] += (pred_band & ~new_fire_gt).sum()
                    spread_stats[band_name]['fn'] += (gt_band & ~p).sum()
                    spread_stats[band_name]['n_gt'] += gt_band.sum()

    # Reporting
    print(f"\n[1. Area Analysis]")
    print(f"Total True Burn Area: {total_true_ha:,.1f} ha")
    print(f"Total Overprediction: {total_overpred_ha:,.1f} ha")
    print(f"Total Underprediction: {total_underpred_ha:,.1f} ha")

    print(f"\n[2. Boundary Accuracy]")
    for b_name in ['90m', '180m', '270m']:
        stats = boundary_stats[b_name]
        prec = stats['tp_p'] / (stats['total_p'] + 1e-8)
        rec = stats['tp_t'] / (stats['total_t'] + 1e-8)
        f1 = (2 * prec * rec) / (prec + rec + 1e-8)
        print(f"  {b_name} Tolerance -> Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

    print(f"\n[3. Error-by-Spread-Distance]")
    for band_name in ['0-90m', '90-180m', '180-270m', '270-360m', '>360m']:
        s = spread_stats[band_name]
        rec = s['tp'] / (s['tp'] + s['fn'] + 1e-8)
        prec = s['tp'] / (s['tp'] + s['fp'] + 1e-8)
        print(f"  {band_name} -> Recall: {rec:.3f} | Precision: {prec:.3f} | True Pixels in Band: {s['n_gt']}")

    results = {"Split": split_name}
    
    # Boundary Stats for the multi-run dataframe
    for b_name in ['90m', '180m', '270m']:
        stats = boundary_stats[b_name]
        results[f'Boundary_Prec_{b_name}'] = stats['tp_p'] / (stats['total_p'] + 1e-8)
        results[f'Boundary_Rec_{b_name}'] = stats['tp_t'] / (stats['total_t'] + 1e-8)
        results[f'Boundary_F1_{b_name}'] = (2 * results[f'Boundary_Prec_{b_name}'] * results[f'Boundary_Rec_{b_name}']) / (results[f'Boundary_Prec_{b_name}'] + results[f'Boundary_Rec_{b_name}'] + 1e-8)

    # Spread Stats for the multi-run dataframe
    for b_name in ['0-90m', '90-180m', '180-270m', '270-360m', '>360m']:
        s = spread_stats[b_name]
        results[f'Spread_Recall_{b_name}'] = s['tp'] / (s['tp'] + s['fn'] + 1e-8)
        results[f'Spread_Prec_{b_name}'] = s['tp'] / (s['tp'] + s['fp'] + 1e-8)

    return pd.DataFrame([results])

class CombinedLoss(nn.Module):
    """Calculates a weighted combination of Focal Loss and Dice Loss for segmentation."""
    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss(smooth=1.0, apply_sigmoid=True)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1).float()
            
        focal_l = self.focal(inputs, targets)
        dice_l = self.dice(inputs, targets)
        
        return (self.focal_weight * focal_l) + (self.dice_weight * dice_l)

def run_spatial_analysis(config: Config) -> None:
    
    seed_everything(config.seed)
    mc = config.model
    device = Evaluator._resolve_device(config.training.device)
    
    # LOAD DATALOADERS & INITIALIZE MODEL
    if mc.architecture == 'unet':
        print("Initializing Standard UNet...")
        _, _, test_space_loader, test_time_loader, test_spacetime_loader = get_advanced_dataloaders(config, is_sat_age=False)
        model = unet.UNet(input_channels=mc.input_channels, num_classes=mc.num_classes, hidden_features=mc.hidden_features, use_skip_connections=mc.use_skip_connections, use_activation_after_upsampling=mc.use_activation_after_upsampling)
        meta_prefix = "simple"
    
    elif mc.architecture == 'unet_olmo_offline':
        print("Initializing OlmoEarth...")
        _, _, test_space_loader, test_time_loader, test_spacetime_loader = get_advanced_dataloaders(config, 
            is_sat_age=False,
            is_coords=False,
            is_loc_emb=False,
            is_olmo_emb=True)
        model = unet_olmo_offline.UNetOlmo(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
            use_olmo=True,
            olmo_channels=768
        )
        meta_prefix = "simple"

    elif mc.architecture == 'utae':
        
        print("Initializing UTAE...")
        _, _, test_space_loader, test_time_loader, test_spacetime_loader = get_advanced_timeseries_dataloaders(
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
        meta_prefix = "timeseries"
        
    else:
        raise ValueError(f"Architecture '{mc.architecture}' is not supported.")

    model = model.to(device)
    loss_fn = CombinedLoss(alpha=0.85, gamma=2.0, focal_weight=0.5, dice_weight=0.5).to(device)

    evaluator = Evaluator(model=model, loss_fn=loss_fn, config=config.training, logger=None)

    splits_to_evaluate = [
        ("test_time", test_time_loader), # TEST-EASY
        ("test_space", test_space_loader), # TEST-MEDIUM
        ("test_spacetime", test_spacetime_loader) # TEST-HARD
    ]

    # ---------------------------------------------------------
    # MULTI-RUN AVERAGING LOGIC
    # ---------------------------------------------------------
    run_ids = [1, 2, 3]
    
    all_spatial_results = []

    for run_id in run_ids:
        print(f"\n{'='*50}\nEVALUATING RUN {run_id}\n{'='*50}")
        
        # Load specific run checkpoint
        checkpoint_dir = Path(f"checkpoints_{config.model.architecture}") / f"run_{run_id}"
        best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
        
        if not best_checkpoint.exists():
            print(f"[!] Warning: Checkpoint missing for run {run_id} at {best_checkpoint}. Skipping.")
            continue
            
        evaluator.load_checkpoint(best_checkpoint)

        for split_name, loader in splits_to_evaluate:
            meta_path = os.path.join(settings.METADATA_FOLDER, f"{meta_prefix}_{split_name}_metadata.json")
            
            # Run evaluations and collect dataframes            
            df_spatial = run_spatial_error_breakdown(evaluator, loader, split_name, device)
            df_spatial['Run_ID'] = run_id
            all_spatial_results.append(df_spatial)

    # ---------------------------------------------------------
    # AGGREGATE AND PRINT AVERAGES (MEAN ± STD)
    # ---------------------------------------------------------
    if len(all_spatial_results) > 0:
        
        combined_spatial = pd.concat(all_spatial_results, ignore_index=True)

        # Calculate Mean and Std for Spatial Analysis
        numeric_cols_spat = combined_spatial.select_dtypes(include='number').columns.drop('Run_ID', errors='ignore')
        spatial_mean = combined_spatial.groupby("Split")[numeric_cols_spat].mean().add_suffix('_Mean')
        spatial_std = combined_spatial.groupby("Split")[numeric_cols_spat].std().add_suffix('_Std')
        final_spatial = pd.concat([spatial_mean, spatial_std], axis=1).reset_index()

        print("\n" + "="*70)
        print(" FINAL AVERAGED RESULTS: SPATIAL ERROR PATTERNS (MEAN ± STD) ")
        print("="*70)
        for _, row in final_spatial.iterrows():
            print(f"\n[Split: {row['Split'].upper()}]")
            
            print("\n  [Boundary Accuracy (F1)]")
            print(f"    90m (1px):  {row.get('Boundary_F1_90m_Mean', 0):.3f} ± {row.get('Boundary_F1_90m_Std', 0):.3f}")
            print(f"    180m (2px): {row.get('Boundary_F1_180m_Mean', 0):.3f} ± {row.get('Boundary_F1_180m_Std', 0):.3f}")
            print(f"    270m (3px): {row.get('Boundary_F1_270m_Mean', 0):.3f} ± {row.get('Boundary_F1_270m_Std', 0):.3f}")

            print("\n  [Spread-Distance Detection (Recall)]")
            print(f"    0-90m:    {row.get('Spread_Recall_0-90m_Mean', 0):.3f} ± {row.get('Spread_Recall_0-90m_Std', 0):.3f}")
            print(f"    90-180m:  {row.get('Spread_Recall_90-180m_Mean', 0):.3f} ± {row.get('Spread_Recall_90-180m_Std', 0):.3f}")
            print(f"    180-360m: {row.get('Spread_Recall_180-360m_Mean', 0):.3f} ± {row.get('Spread_Recall_180-360m_Std', 0):.3f}")
            print(f"    >360m:    {row.get('Spread_Recall_>360m_Mean', 0):.3f} ± {row.get('Spread_Recall_>360m_Std', 0):.3f}")


    # ---------------------------------------------------------
    # MULTI-RUN AVERAGING FOR MERGED SPLITS (SUBGROUPS)
    # ---------------------------------------------------------

    print("\n" + "="*70)
    print(" RUNNING 3-RUN AVERAGING FOR MERGED SPLITS (SUBGROUPS) ")
    print("="*70)

    run_ids = [1, 2, 3]
    all_merged_spatial_results = []
    
    # Define meta_prefix based the model (using "simple" as standard)
    path_time = os.path.join(settings.METADATA_FOLDER, f"{meta_prefix}_test_time_metadata.json")
    path_spacetime = os.path.join(settings.METADATA_FOLDER, f"{meta_prefix}_test_spacetime_metadata.json")

    for run_id in run_ids:
        print(f"\n[MERGED SPLITS] Evaluating Run {run_id}...")
        
        # Load weights for this run
        checkpoint_dir = Path(f"checkpoints_{config.model.architecture}") / f"run_{run_id}"
        best_checkpoint = checkpoint_dir / "best_checkpoint.pt"
        
        if not best_checkpoint.exists():
            print(f"[!] Checkpoint missing for run {run_id}. Skipping.")
            continue
            
        evaluator.load_checkpoint(best_checkpoint)
                
        # Merge the datasets for spatial analysis
        loaders_list = [test_time_loader, test_space_loader, test_spacetime_loader]
        datasets = [loader.dataset for loader in loaders_list]
        merged_dataset = ConcatDataset(datasets)
        merged_loader = DataLoader(
            merged_dataset,
            batch_size=config.training.batch_size, 
            shuffle=False, 
            num_workers=config.training.num_workers,
            persistent_workers=False
        )
        
        # Run spatial analysis on merged splits
        df_spatial = run_spatial_error_breakdown(
            evaluator=evaluator,
            loader=merged_loader,
            split_name="test_merged",
            device=device
        )
        
        # Tag it with the run ID and save
        if df_spatial is not None and not df_spatial.empty:
            df_spatial['Run_ID'] = run_id
            all_merged_spatial_results.append(df_spatial)

    # CALCULATE AND PRINT MEAN and STD (SPATIAL)
    if len(all_merged_spatial_results) > 0:
        
        combined_spatial = pd.concat(
            all_merged_spatial_results,
            ignore_index=True
        )

        # Group by Split and calculate Mean and Std
        numeric_cols_spatial = combined_spatial.select_dtypes(
            include="number"
        ).columns.drop(["Run_ID"], errors="ignore")

        spatial_mean = (
            combined_spatial.groupby("Split")[numeric_cols_spatial]
            .mean()
            .add_suffix("_Mean")
        )
        spatial_std = (
            combined_spatial.groupby("Split")[numeric_cols_spatial]
            .std()
            .add_suffix("_Std")
        )

        final_spatial = pd.concat(
            [spatial_mean, spatial_std], axis=1
        ).reset_index()

        print("\n" + "=" * 70)
        print(" FINAL SPATIAL ANALYSIS: MERGED SPLITS (MEAN ± STD) ")
        print("=" * 70)

        for _, row in final_spatial.iterrows():
            print(f"\n[Split: {row['Split'].upper()}]")

            # Print each spatial metric
            for col in numeric_cols_spatial:
                mean = row.get(f"{col}_Mean", 0)
                std = row.get(f"{col}_Std", 0)

                print(f"  -> {col}: {mean:.3f} ± {std:.3f}")