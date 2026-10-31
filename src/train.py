from __future__ import annotations

import torch
import torch.nn as nn
import time

from src.config import Config

from configs import settings

# Dataloders
from src.dataloader import get_dataloaders
from src.dataloader_timeseries import get_timeseries_dataloaders
# TO BE CHANGED
from src.dataloader_advanced_dual import get_advanced_dual_dataloaders
from src.dataloader_advanced_branches import get_advanced_dataloaders_with_branches

# Models (Mono-Temporal)
from src.models import unet
from src.models import unet_time_gap
from src.models import unet_segformer
from src.models import unet_attention
from src.models.asufm import asufm, config_asufm
from src.models import RCDA_v2
from src.models.umamba import UMambaEnc_2d
from src.models import unet_olmo_offline
from src.models import dual_dinov3

# Models (Multi-Temporal)
from src.models import unet_convlstm
from src.models import utae
from src.models.simvp2 import simvp_wrapper
from src.models.video_swin_hybrid_unet import video_swin_hybrid_unet

# Models (Multi-Modal)
from src.models.frp.MultimodalEncoder import StateBlock, DynamicBlock, ConstantBlock, CombinedBlock
from src.models.frp.BackboneEncoder import Encoder
from src.models.frp.BackboneDecoder import Decoder
from src.models.frp.Model import FullNetwork

# Training
from src.trainer import Trainer
from src.utils import seed_everything
from src.focal_loss import FocalLoss 
from src.dice_loss import DiceLoss
from src.logger import CometLogger

# Full Features list
FULL_FEATURES = [
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

def train(config: Config) -> None:
    """Sets up all components and executes the model training loop.

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

    device = Trainer._resolve_device(config.training.device)

    ####################### MONO-TEMPORAL #######################

    if mc.architecture == 'unet':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False)
        model = unet.UNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )
        
    elif mc.architecture == 'unet_time_gap':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=True)
        model = unet_time_gap.UNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )

    elif mc.architecture == 'unet_attention':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False)
        model = unet_attention.AttentionUNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
            use_attention=True
        )
    
    elif mc.architecture == 'unet_segformer':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False)
        model = unet_segformer.UNetSegFormer(
            in_channels=mc.input_channels,
            out_classes=mc.num_classes
        )
    
    elif mc.architecture == 'unet_olmo_offline':
        
        train_loader, val_loader, test_loader = get_dataloaders(config, 
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
    
    elif mc.architecture == 'asufm':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False,
                                                            is_coords=False,
                                                            is_loc_emb=False)
        
        asufm_config = config_asufm.get_asufm_configs(mc.input_channels)

        model = asufm.ASUFM(
            config=asufm_config, 
            num_classes=mc.num_classes
        )

    elif mc.architecture == 'simvpv2_spatial':
        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False,
                                                            is_coords=False,
                                                            is_loc_emb=False)
        
        model = simvp_wrapper.WildfireSimVPWrapper(num_timesteps=1, 
                                     channels_per_step=mc.input_channels, 
                                     img_size=settings.GRID_SIZE,
                                     is_spatial_only=True)

    elif mc.architecture == 'rcda_v2':

        train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                            is_sat_age=False,
                                                            is_coords=False,
                                                            is_loc_emb=False,
                                                            is_olmo_emb=False)
        
        model = RCDA_v2.RCDA(
            in_ch=mc.input_channels,
            num_classes=mc.num_classes,
            depth=4,
        )

    elif mc.architecture == 'dual_dinov3':

        # The sub-lists
        RGB_FEATURES = ['s2_b02', 's2_b03', 's2_b04']
        
        ENV_FEATURES = [f for f in FULL_FEATURES if f not in RGB_FEATURES]
        # ENV_FEATURES = [f for f in FULL_FEATURES if f not in RGB_FEATURES and f not in ['hii']]
            
        train_loader, val_loader, test_loader = get_advanced_dual_dataloaders(config,
                                                            all_features=FULL_FEATURES,
                                                            rgb_features=RGB_FEATURES,
                                                            env_features=ENV_FEATURES,                                                                 
                                                            is_sat_age=False,
                                                            is_coords=False,
                                                            is_loc_emb=False,
                                                            is_olmo_emb=False,
                                                            is_alpha_emb=False)
        model = dual_dinov3.DualDinoV3(
            env_channels=len(ENV_FEATURES),
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling
        )

    elif mc.architecture == 'umamba':
            
            train_loader, val_loader, test_loader = get_dataloaders(config, 
                                                                is_sat_age=False,
                                                                is_coords=False,
                                                                is_loc_emb=False,
                                                                is_olmo_emb=False)

            model = UMambaEnc_2d.UMambaEnc(
                input_size=(settings.GRID_SIZE, settings.GRID_SIZE),
                input_channels=mc.input_channels,
                n_stages=6,
                features_per_stage=[32, 64, 128, 256, 512, 512],
                conv_op=nn.Conv2d,
                kernel_sizes=[[3, 3]] * 6,
                strides=[[1, 1]] + [[2, 2]] * 5,
                n_conv_per_stage=[2, 2, 2, 2, 2, 2],
                num_classes=mc.num_classes,
                n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
                conv_bias=True,
                norm_op=nn.InstanceNorm2d,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace': True},
                deep_supervision=False
            )

    ####################### MULTI-TEMPORAL #######################

    elif mc.architecture == 'unet_convlstm':
        train_loader, val_loader, test_loader = get_timeseries_dataloaders(config)
        model = unet_convlstm.SpatiotemporalUNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections
        )

    elif mc.architecture == 'utae':
        train_loader, val_loader, test_loader = get_timeseries_dataloaders(
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
    
    elif mc.architecture == 'simvpv2_spatiotemporal':

        train_loader, val_loader, test_loader = get_timeseries_dataloaders(
            config,
            return_positions=False,
            return_loc_emb=False,
            return_sat_age=False,
            return_coords=False
        )
        
        model = simvp_wrapper.WildfireSimVPWrapper(num_timesteps=3, 
                                     channels_per_step=mc.input_channels, 
                                     img_size=settings.GRID_SIZE,
                                     is_spatial_only=False)
    
    elif mc.architecture == 'video_swin_unet':

        train_loader, val_loader, test_loader = get_timeseries_dataloaders(
            config,
            return_positions=False,
            return_loc_emb=False,
            return_sat_age=False,
            return_coords=False
        )
        
        model = video_swin_hybrid_unet.VideoSwinHybridUNet(
            in_channels=mc.input_channels,
            time_steps=3,
            input_size=(settings.GRID_SIZE,settings.GRID_SIZE),
            num_classes=mc.num_classes
        )
    
    elif mc.architecture == 'frp':

        # The sub-lists
        STATE_FEATURES = ['accumulated_mask', 'scaled_accumulated_mask']
        
        DYNAMIC_FEATURES = [
            'tmax', 'rh', 'ws', 'prec', 'u10', 'v10', 'evi', 'ndvi', 
            's2_b02', 's2_b03', 's2_b04', 's2_b08', 's2_b11', 's2_b12'
        ]
        
        # Dynamically grab all remaining features for the Static block
        STATIC_FEATURES = [f for f in FULL_FEATURES if f not in STATE_FEATURES and f not in DYNAMIC_FEATURES]

        train_loader, val_loader, test_loader = get_advanced_dataloaders_with_branches(
            config,
            full_features=FULL_FEATURES,
            state_features=STATE_FEATURES,
            dynamic_features=DYNAMIC_FEATURES,
            static_features=STATIC_FEATURES,
            is_coords=False,
            is_sat_age=False,
            is_loc_emb=False
        )
        
        state_channels = len(STATE_FEATURES)
        dynamic_channels = len(DYNAMIC_FEATURES)
        static_channels = len(STATIC_FEATURES)

        # Initialize the Blocks
        block1 = StateBlock(input_dim=state_channels,
                            hidden_dims=[4, 8, 16, 16],
                            kernel_sizes=[(5, 5), (3, 3), (3, 3), (1, 1)],
                            num_layers=4,
                            num_conv_filters=[8, 16, 16],
                            device=device)

        block2 = DynamicBlock(input_channels=dynamic_channels,
                            convlstm_hidden_channels=16,
                            conv_hidden_channels=[8, 16, 16],
                            device=device)

        block3 = ConstantBlock(input_channels=static_channels).to(device)

        # Combine and build the full network
        combined_block = CombinedBlock(block1, block2, block3).to(device)
        encoder = Encoder().to(device)
        decoder = Decoder().to(device)
        model = FullNetwork(combined_block, encoder, decoder).to(device)
    
    else:
        raise ValueError(f"Unknown architecture specified in config: '{mc.architecture}'")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    warmup_epochs = min(1, config.training.num_epochs - 1)
    steps_per_epoch = len(train_loader)
    total_steps = config.training.num_epochs * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

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

    # Optional Comet logger
    logger: CometLogger | None = None
    if config.comet.enabled:
        cc = config.comet

        # Auto-generates names like: unet-baseline-1704124800
        run_name = f"{cc.experiment_name}-{int(time.time())}"

        logger = CometLogger(
            project_name=cc.project_name,
            workspace=cc.workspace,
            experiment_name=run_name,
            experiment_tags=cc.experiment_tags or None,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer, 
        scheduler=scheduler,
        loss_fn=loss_fn, 
        config=config.training, 
        logger=logger
    )

    print(f"Training on device: {trainer.device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer.fit(train_loader, val_loader)

    return trainer, test_loader