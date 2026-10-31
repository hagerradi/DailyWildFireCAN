import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse

from src.config import Config
from src.dataloader_advanced import get_advanced_dataloaders

class AlexandridisCA(nn.Module):
    def __init__(self, feature_names, stats_json, p_0=0.3, c_1=0.045, c_2=0.131, a_s=0.078):
        """
        Domain-specific physical baseline using Alexandridis et al. (2008) principles.
        """
        super().__init__()
        self.feature_names = feature_names
        self.stats = stats_json
        
        # Alexandridis empirical coefficients
        self.p_0 = p_0      # Base probability of vegetation catching fire
        self.c_1 = c_1      # Wind speed modifier
        self.c_2 = c_2      # Wind direction modifier
        self.a_s = a_s      # Slope modifier

        # Map feature names to their channel indices
        self.idx = {name: i for i, name in enumerate(feature_names)}

        # 8-neighbor directional unit vectors [dx, dy]
        # Order: E, NE, N, NW, W, SW, S, SE
        self.dirs = torch.tensor([
            [1, 0], [1, 1], [0, 1], [-1, 1],
            [-1, 0], [-1, -1], [0, -1], [1, -1]
        ], dtype=torch.float32)

        # 3x3 Conv kernel to find burning neighbors (1s in a ring, 0 in middle)
        self.neighbor_kernel = torch.tensor([
            [[1., 1., 1.],
             [1., 0., 1.],
             [1., 1., 1.]]
        ], dtype=torch.float32).unsqueeze(0) # Shape: (1, 1, 3, 3)

    def _unstandardize(self, tensor, name):
        """Reverses the Z-score normalization using your JSON stats."""
        mean = self.stats[name]['mean']
        std = self.stats[name]['std']
        # Feature shape is (B, H, W) -> we extract it using the index
        feat = tensor[:, self.idx[name], :, :]
        return (feat * std) + mean

    def forward(self, x, steps=1):
        """
        x: Input tensor from your dataloader (B, Channels, H, W)
        steps: How many CA iterations to project into the future.
        """
        device = x.device
        B, C, H, W = x.shape
        
        # Un-standardize physical features
        ws = self._unstandardize(x, 'ws')       # Wind speed
        u10 = self._unstandardize(x, 'u10')     # Wind vector X
        v10 = self._unstandardize(x, 'v10')     # Wind vector Y
        slope = self._unstandardize(x, 'slope') # Slope in degrees
        
        # Raw features (no un-standardization needed)
        ndvi = x[:, self.idx['ndvi'], :, :]
        
        # Get current binary fire mask and FORCE it to be clean binary (0 or 1) BEFORE the loop
        fire_state = x[:, self.idx['accumulated_mask'], :, :].unsqueeze(1).clone()
        fire_state = (fire_state > 0).float()
        
        # Save the initial core fire so we can subtract it at the very end
        initial_fire_state = fire_state.clone()
        
        # Build the "Unburnable" mask from CCRS
        unburnable = (
            x[:, self.idx['ccrs_18_water'], :, :] +
            x[:, self.idx['ccrs_17_urban'], :, :] +
            x[:, self.idx['ccrs_16_barren'], :, :] +
            x[:, self.idx['ccrs_19_snow'], :, :]
        )
        burnable_mask = (unburnable == 0).float().unsqueeze(1)

        # Wind direction angle (in radians)
        wind_dir = torch.atan2(v10, u10)

        self.neighbor_kernel = self.neighbor_kernel.to(device)
        self.dirs = self.dirs.to(device)

        # Step forward in time
        for _ in range(steps):
            
            # Find cells that have at least one burning neighbor
            active_neighbors = F.conv2d(fire_state, self.neighbor_kernel, padding=1)
            
            # Candidate cells: Must be currently unburned AND burnable AND have fire next to them
            candidates = (fire_state == 0) & (burnable_mask == 1) & (active_neighbors > 0)
            
            if not candidates.any():
                break # Fire stopped spreading!

            p_burn_total = torch.zeros((B, 1, H, W), device=device)

            # Check spread from all 8 directions
            for dx, dy in self.dirs:
                dir_angle = torch.atan2(dy, dx)
                
                # Wind Factor
                angle_diff = wind_dir - dir_angle
                ws_clamped = torch.clamp(ws, 0, 30) 
                v_w = torch.exp(self.c_1 * ws_clamped) * torch.exp(ws_clamped * self.c_2 * (torch.cos(angle_diff) - 1.0))
                
                # Slope Factor
                v_s = torch.exp(self.a_s * slope)

                # Base probability boosted slightly by NDVI
                p_base = torch.clamp(self.p_0 + (ndvi * 0.2), 0.1, 0.8)

                # Directional Probability
                p_dir = p_base * (1.0 + v_w) * (1.0 + v_s)
                
                shifted_fire = torch.roll(fire_state, shifts=(int(dy), int(dx)), dims=(2, 3))
                p_burn_total += (shifted_fire * p_dir.unsqueeze(1))

            # Threshold: Catch fire if probability > 0.5
            new_ignitions = (p_burn_total > 0.5).float() * candidates.float()
            
            # ACCUMULATE fire so the core doesn't disappear in step 2
            fire_state = torch.clamp(fire_state + new_ignitions, 0.0, 1.0)

        # Extract the new growth
        new_growth = fire_state - initial_fire_state

        # Return just the new growth (Batch, H, W)
        return new_growth.squeeze(1)


def run_ca_dummy_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Wildfire segmentation — entry point.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # Load your generated stats JSON
    stats_filepath = "simple_dataset_stats_8_years_reb_v1.json"
    with open(stats_filepath, "r") as f:
        stats_json = json.load(f)

    all_feature_names = [
            # --- DYNAMIC FEATURES (14) ---
            'tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
            's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12',
            
            # --- STATIC FEATURES (before SCANFI landcover) (17) ---
            'dem', 'slope', 'aspect (sin)', 'aspect (cos)', 
            'biomass', 'closure', 'prcb', 'prcc',
            'height',
            'prc_balsam_fir', 'prc_black_spruce', 'prc_douglas_fir', 'prc_jack_pine',
            'prc_lodgepole_pine', 'prc_ponderosa_pine', 'prc_tamarack', 'prc_white_red_pine',
            
            # --- CCRS LANDCOVER (15) ---
            'ccrs_1_needleleaf', 'ccrs_2_taiga_needleleaf', 'ccrs_5_broadleaf', 
            'ccrs_6_mixed_forest', 'ccrs_8_shrubland', 'ccrs_10_grassland', 
            'ccrs_11_polar_shrubland', 'ccrs_12_polar_grassland', 'ccrs_13_polar_barren', 
            'ccrs_14_wetland', 'ccrs_15_cropland', 'ccrs_16_barren', 'ccrs_17_urban', 
            'ccrs_18_water', 'ccrs_19_snow',

            # --- HUMAN INFLUENCE INDEX (1) ---
            'hii',

            # --- ANNUAL DISTURBANCE (6) ---
            'dist_1_wildfire', 'dist_2_harvesting', 'dist_3_other', 
            'dist_4_water', 'dist_5_defoliation_harvest', 'dist_6_defoliation_all',
            
            # --- FIRE MASKS (2) ---
            'accumulated_mask', 'scaled_accumulated_mask'
        ]
    

    # Initialize the Cellular Automata model
    ca_model = AlexandridisCA(
        feature_names=all_feature_names, 
        stats_json=stats_json
    ).to(device)
    ca_model.eval()
    
    _, _, _, _, test_spacetime_loader = get_advanced_dataloaders(config, is_sat_age=False)
    print(len(test_spacetime_loader))
    
    print("\nGrabbing one batch from the dataloader...")
    batch = next(iter(test_spacetime_loader))
    
    x_batch = batch["input_grids"].to(device)
    y_batch = batch["label"].to(device)

    print(f"Input X shape: {x_batch.shape}")
    print(f"Target Y shape: {y_batch.shape}")

    # Run the CA Model
    print("\nRunning Cellular Automata Physics Engine...")
    with torch.no_grad():
        ca_preds = ca_model(x_batch, steps=1)

    # Verify Outputs
    print("\n--- CA OUTPUT VERIFICATION ---")
    print(f"Prediction shape: {ca_preds.shape} (Should be Batch, H, W)")
    print(f"Unique values in prediction: {torch.unique(ca_preds).tolist()} (Should be just [0.0, 1.0])")
    
    # Check if the fire actually grew compared to the input mask
    input_fire_mask = x_batch[:, all_feature_names.index('accumulated_mask'), :, :]
    print(f"Total burning pixels (Input):  {input_fire_mask.sum().item():.0f}")
    print(f"Total burning pixels (Output): {ca_preds.sum().item():.0f}")

if __name__ == "__main__":
    run_ca_dummy_test()