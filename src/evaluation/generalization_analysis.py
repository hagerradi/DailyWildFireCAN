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

# Analysis functions
from src.evaluation.filters import (get_duration_indices, get_month_indices, 
                                    get_province_indices, get_size_indices, get_year_indices)


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

def get_test_suites(df):
    
    # Define the test conditions
    test_suites = {

        # ---------------------------------------------------------
        # Fire Age
        # ---------------------------------------------------------
        "Fireday: Early (1-3 days)": get_duration_indices(df, min_days=1, max_days=3),
        "Fireday: Developing (4-7 days)": get_duration_indices(df, min_days=4, max_days=7),
        "Fireday: Established (8-14 days)": get_duration_indices(df, min_days=8, max_days=14),
        "Fireday: Late Stage (>14 days)": get_duration_indices(df, min_days=15),

        # ---------------------------------------------------------
        # Fire Duration
        # ---------------------------------------------------------
        "Duration: Short (< 1 week)": get_duration_indices(df, min_days=1, max_days=7),
        "Duration: Medium (1 to 2 weeks)": get_duration_indices(df, min_days=8, max_days=14),
        "Duration: Long (15 to 30 days)": get_duration_indices(df, min_days=15, max_days=30),
        "Duration: Very Long (> 1 month)": get_duration_indices(df, min_days=31),

        # ---------------------------------------------------------
        # Fire Year
        # ---------------------------------------------------------
        "Year: 2023": get_year_indices(df, 2023),
        "Year: 2022": get_year_indices(df, 2022),
        "Year: 2021": get_year_indices(df, 2021),
        "Year: 2020": get_year_indices(df, 2020),
        "Year: 2019": get_year_indices(df, 2019),
        "Year: 2018": get_year_indices(df, 2018),
        "Year: 2017": get_year_indices(df, 2017),

        # ---------------------------------------------------------
        # Fire Size
        # ---------------------------------------------------------
        "Size: Extra Small (<600 ha)": get_size_indices(df, max_ha=600.0),
        "Size: Small (600 to 5000 ha)": get_size_indices(df, min_ha=600.0, max_ha=5000.0),
        "Size: Medium (5000 to 20000 ha)": get_size_indices(df, min_ha=5000.0, max_ha=20000.0),
        "Size: Large (>20000 ha)": get_size_indices(df, min_ha=20000.0),

        # ---------------------------------------------------------
        # Fire Months
        # ---------------------------------------------------------
        "Month: March": get_month_indices(df, 3),
        "Month: April": get_month_indices(df, 4),
        "Month: May": get_month_indices(df, 5),
        "Month: June": get_month_indices(df, 6),
        "Month: July": get_month_indices(df, 7),
        "Month: August": get_month_indices(df, 8),
        "Month: September": get_month_indices(df, 9),
        "Month: October": get_month_indices(df, 10),
        "Month: November": get_month_indices(df, 11),

        # ---------------------------------------------------------
        # Provincial Breakdown
        # ---------------------------------------------------------
        "Province: British Columbia (BC)": get_province_indices(df, 'BC'),
        "Province: Alberta (AB)": get_province_indices(df, 'AB'),
        "Province: Saskatchewan (SK)": get_province_indices(df, 'SK'),
        "Province: Manitoba (MB)": get_province_indices(df, 'MB'),
        "Province: Ontario (ON)": get_province_indices(df, 'ON'),
        "Province: Quebec (QC)": get_province_indices(df, 'QC'),
        "Territory: Northwest Territories (NWT)": get_province_indices(df, 'NWT'),
        "Territory: Yukon (YT)": get_province_indices(df, 'YT'),
    }
    
    return test_suites

def run_generalization_analysis_for_single_split(evaluator, base_loader, metadata_path, split_name, config):
    """
    Takes an existing dataloader and runs the Evaluator on each subset to analyze failure cases.
    """
    print(f"\n=======================================================")
    print(f"Running Generalization Analysis for: {split_name.upper()}")
    print(f"=======================================================")
    
    # Extract the underlying Dataset from the DataLoader
    dataset = base_loader.dataset
    
    # Load the JSON metadata
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata not found at {metadata_path}. Skipping.")
        return
        
    df = pd.read_json(metadata_path)
    
    # MULTI-FIRE LOGIC
    # For each sample, look at the list of all_fire_sizes.
    # Extract the maximum value to represent the true scale of the event in this tile.
    df['max_fire_size'] = df['all_fire_sizes'].apply(lambda sizes_list: max(sizes_list) if isinstance(sizes_list, list) and len(sizes_list) > 0 else 0.0)

    # Standardize and parse the date string into a true datetime object
    date_col = 'start_date' if 'start_date' in df.columns else 'date'
    df['parsed_date'] = pd.to_datetime(df[date_col])
    
    results = []
    
    test_suites = get_test_suites(df)
    
    # Loop through each condition, create a Subset, and evaluate
    for test_name, indices in test_suites.items():
        if len(indices) == 0:
            print(f"Skipping '{test_name}' - 0 samples found.")
            continue
            
        print(f"\nEvaluating: {test_name} ({len(indices)} samples)")
        
        # Create a dataset containing only these indices
        subset = Subset(dataset, indices)
        
        # Wrap it in a new dataloader
        filtered_loader = DataLoader(
            subset, 
            batch_size=config.training.batch_size, 
            shuffle=False, 
            num_workers=config.training.num_workers,
            persistent_workers=False
        )
        
        # Run the evaluator
        loss, metrics = evaluator.evaluate(filtered_loader)
        
        # Log results
        res = {
            "Split": split_name,
            "Test Condition": test_name,
            "Sample Count": len(indices),
            "Loss": loss
        }
        res.update(metrics)
        results.append(res)
        
        # Print quick summary
        print(f" -> Loss: {loss:.3f} | Metrics: { {k: round(v, 4) for k, v in metrics.items()} }")
        
    report_df = pd.DataFrame(results)

    return report_df

def run_generalization_analysis_for_merged_splits(evaluator, loaders_list, metadata_paths_list, merged_name, config):
    """
    Merges multiple DataLoaders and their corresponding JSON metadata files,
    then runs the generalization analysis over the combined dataset.
    """
    print(f"\n=======================================================")
    print(f"Running Failure Analysis for MERGED SPLITS: {merged_name.upper()}")
    print(f"=======================================================")
    
    # Merge the Datasets
    datasets = [loader.dataset for loader in loaders_list]
    merged_dataset = ConcatDataset(datasets)
    
    # Load and Merge the JSON metadata
    dfs = []
    for path in metadata_paths_list:
        if not os.path.exists(path):
            print(f"[!] Warning: Metadata not found at {path}. Skipping.")
        else:
            dfs.append(pd.read_json(path))
            
    if not dfs:
        print("[!] No valid metadata found. Aborting merged analysis.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Realign indices
    df['sample_index'] = df.index
    
    # Multi-fire logic
    df['max_fire_size'] = df['all_fire_sizes'].apply(lambda sizes_list: max(sizes_list) if isinstance(sizes_list, list) and len(sizes_list) > 0 else 0.0)

    # Standardize and parse the date string
    date_col = 'start_date' if 'start_date' in df.columns else 'date'
    df['parsed_date'] = pd.to_datetime(df[date_col])
    
    # Define the test conditions
    test_suites = get_test_suites(df)
    
    results = []
    
    # Loop through each condition, create a Subset, and evaluate
    for test_name, indices in test_suites.items():
        if len(indices) == 0:
            print(f"Skipping '{test_name}' - 0 samples found.")
            continue
            
        print(f"\nEvaluating: {test_name} ({len(indices)} samples)")
        
        subset = Subset(merged_dataset, indices)
        
        filtered_loader = DataLoader(
            subset, 
            batch_size=config.training.batch_size, 
            shuffle=False, 
            num_workers=config.training.num_workers,
            persistent_workers=False
        )
        
        loss, metrics = evaluator.evaluate(filtered_loader)
        
        res = {
            "Split": merged_name,
            "Test Condition": test_name,
            "Sample Count": len(indices),
            "Loss": loss
        }
        res.update(metrics)
        results.append(res)
        
        print(f" -> Loss: {loss:.3f} | Metrics: { {k: round(v, 4) for k, v in metrics.items()} }")
        
    report_df = pd.DataFrame(results)

    return report_df

def run_generalization_analysis(config: Config) -> None:
    
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
    
    all_failure_results = []

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
            df_failure = run_generalization_analysis_for_single_split(evaluator, loader, meta_path, split_name, config)
            df_failure['Run_ID'] = run_id  # Tag with run ID
            all_failure_results.append(df_failure)
            
    # ---------------------------------------------------------
    # AGGREGATE AND PRINT AVERAGES (MEAN ± STD)
    # ---------------------------------------------------------
    if len(all_failure_results) > 0:
        combined_failure = pd.concat(all_failure_results, ignore_index=True)
        
        # Calculate Mean and Std for Failure/Subgroup Analysis
        numeric_cols_fail = combined_failure.select_dtypes(include='number').columns.drop(['Run_ID', 'Sample Count'], errors='ignore')
        failure_mean = combined_failure.groupby(["Split", "Test Condition"])[numeric_cols_fail].mean().add_suffix('_Mean')
        failure_std = combined_failure.groupby(["Split", "Test Condition"])[numeric_cols_fail].std().add_suffix('_Std')
        final_failure = pd.concat([failure_mean, failure_std], axis=1).reset_index()

        print("\n" + "="*70)
        print(" FINAL AVERAGED RESULTS: SUBGROUPS & DURATION (MEAN ± STD) ")
        print("="*70)
        for _, row in final_failure.iterrows():
            print(f"\n[Split: {row['Split'].upper()}] Condition: {row['Test Condition']}")
            
            # Extract all metrics
            auc_pr_mean, auc_pr_std = row.get('Test_AUC_PR_Mean', 0), row.get('Test_AUC_PR_Std', 0)
            f1_mean, f1_std = row.get('Test_F1_Mean', 0), row.get('Test_F1_Std', 0)
            rec_mean, rec_std = row.get('Test_Recall_Mean', 0), row.get('Test_Recall_Std', 0)
            prec_mean, prec_std = row.get('Test_Precision_Mean', 0), row.get('Test_Precision_Std', 0)
            
            # Print
            print(f"  -> AUC-PR:  {auc_pr_mean:.3f} ± {auc_pr_std:.3f}")
            print(f"  -> F1:      {f1_mean:.3f} ± {f1_std:.3f}")
            print(f"  -> Recall:  {rec_mean:.3f} ± {rec_std:.3f}")
            print(f"  -> Prec:    {prec_mean:.3f} ± {prec_std:.3f}")


    # ---------------------------------------------------------
    # MULTI-RUN AVERAGING FOR MERGED SPLITS (SUBGROUPS)
    # ---------------------------------------------------------

    print("\n" + "="*70)
    print(" RUNNING 3-RUN AVERAGING FOR MERGED SPLITS (SUBGROUPS) ")
    print("="*70)

    run_ids = [1, 2, 3]
    all_merged_results = []
    
    # Define meta_prefix based on the model (using "simple" as standard)
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
        
        # Run the subgroup analysis (Fireday, Size, etc.) on the merged dataloader
        df_merged = run_generalization_analysis_for_merged_splits(
            evaluator=evaluator,
            loaders_list=[test_time_loader, test_spacetime_loader],
            metadata_paths_list=[path_time, path_spacetime],
            merged_name="test_time_merged",
            config=config
        )
        
        # Tag it with the run ID and save
        if df_merged is not None and not df_merged.empty:
            df_merged['Run_ID'] = run_id
            all_merged_results.append(df_merged)

    # ---------------------------------------------------------
    # CALCULATE AND PRINT MEAN ± STD (FAILURE)
    # ---------------------------------------------------------
    if len(all_merged_results) > 0:
        combined_merged = pd.concat(all_merged_results, ignore_index=True)
        
        # Group by the test condition and calculate Mean and Std
        numeric_cols_merged = combined_merged.select_dtypes(include='number').columns.drop(['Run_ID', 'Sample Count'], errors='ignore')
        merged_mean = combined_merged.groupby(["Split", "Test Condition"])[numeric_cols_merged].mean().add_suffix('_Mean')
        merged_std = combined_merged.groupby(["Split", "Test Condition"])[numeric_cols_merged].std().add_suffix('_Std')
        final_merged = pd.concat([merged_mean, merged_std], axis=1).reset_index()

        print("\n" + "="*70)
        print(" FINAL AVERAGED RESULTS: MERGED SPLIT SUBGROUPS (MEAN ± STD) ")
        print("="*70)
        
        for _, row in final_merged.iterrows():
            print(f"\n[Split: {row['Split'].upper()}] Condition: {row['Test Condition']}")
            
            # Extract metrics safely
            auc_pr_mean, auc_pr_std = row.get('Test_AUC_PR_Mean', 0), row.get('Test_AUC_PR_Std', 0)
            f1_mean, f1_std = row.get('Test_F1_Mean', 0), row.get('Test_F1_Std', 0)
            rec_mean, rec_std = row.get('Test_Recall_Mean', 0), row.get('Test_Recall_Std', 0)
            prec_mean, prec_std = row.get('Test_Precision_Mean', 0), row.get('Test_Precision_Std', 0)
            
            # Print
            print(f"  -> AUC-PR:  {auc_pr_mean:.3f} ± {auc_pr_std:.3f}")
            print(f"  -> Prec:    {prec_mean:.3f} ± {prec_std:.3f}")
            print(f"  -> Recall:  {rec_mean:.3f} ± {rec_std:.3f}")
            print(f"  -> F1:      {f1_mean:.3f} ± {f1_std:.3f}")