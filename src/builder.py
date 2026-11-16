"""Model and dataloader factory shared by training, evaluation and the analyses.

``config.model.architecture`` selects one of the entries of :data:`ARCHITECTURES`.
Each entry records whether the model reads mono- or multi-temporal samples, how
the input channels are presented (see :class:`src.data.WildfireDataset`) and which
extra sample keys it needs.

The released checkpoints on Hugging Face use different folder names; see
:data:`HF_CHECKPOINT_NAMES`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from src.config import Config
from src.data import SPLITS, build_datasets, make_loaders
from src.features import DYNAMIC_FEATURES, NUM_CHANNELS, RGB_FEATURES, STATE_FEATURES, STATIC_FEATURES


@dataclass(frozen=True)
class ArchSpec:
    temporal: str                     # "mono" or "multi"
    layout: str = "grids"             # see WildfireDataset
    extra_keys: tuple[str, ...] = ()  # optional .npz keys passed to the model


ARCHITECTURES: dict[str, ArchSpec] = {
    # Mono-temporal
    "unet": ArchSpec("mono"),
    "unet_time_gap": ArchSpec("mono", extra_keys=("delta_t",)),
    "unet_attention": ArchSpec("mono"),
    "unet_segformer": ArchSpec("mono"),
    "asufm": ArchSpec("mono"),
    "umamba": ArchSpec("mono"),
    "unet_olmo_offline": ArchSpec("mono", extra_keys=("olmo_emb",)),
    "dual_dinov3": ArchSpec("mono", layout="rgb_env"),
    # Multi-temporal
    "unet_convlstm": ArchSpec("multi"),
    "utae": ArchSpec("multi"),
    "simvpv2_spatiotemporal": ArchSpec("multi"),
    "video_swin_unet": ArchSpec("multi"),
    # Multi-modal (multi-temporal samples split into state/dynamic/static branches)
    "frp": ArchSpec("multi", layout="branches"),
}

# Folder names of the released checkpoints -> architecture names used in the code.
HF_CHECKPOINT_NAMES = {
    "unet": "unet",
    "unet_with_time_gap": "unet_time_gap",
    "attention_unet": "unet_attention",
    "segformer_unet": "unet_segformer",
    "swin_unet": "asufm",
    "olmoearth": "unet_olmo_offline",
    "dinov3": "dual_dinov3",
    "unet_with_convlstm": "unet_convlstm",
    "utae": "utae",
    "simvpv2": "simvpv2_spatiotemporal",
    "video_swin_unet": "video_swin_unet",
    "multi_mode_model": "frp",
}
ARCH_TO_HF_NAME = {arch: hf for hf, arch in HF_CHECKPOINT_NAMES.items()}

NUM_TIMESTEPS = 3


def canonical_architecture(name: str) -> str:
    """Accept either a code architecture name or a Hugging Face checkpoint folder name."""
    if name in ARCHITECTURES:
        return name
    if name in HF_CHECKPOINT_NAMES:
        return HF_CHECKPOINT_NAMES[name]
    raise ValueError(f"Unknown architecture '{name}'. Choose from {sorted(ARCHITECTURES)} "
                     f"or the checkpoint names {sorted(HF_CHECKPOINT_NAMES)}.")


def resolve_device(device: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA, then Apple MPS, then CPU."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def samples_folder_for(architecture: str) -> str:
    """Samples folder read by ``architecture`` (``SAMPLE_FOLDER`` or ``TIMESERIES_SAMPLE_FOLDER``)."""
    from configs import settings

    spec = ARCHITECTURES[canonical_architecture(architecture)]
    return settings.TIMESERIES_SAMPLE_FOLDER if spec.temporal == "multi" else settings.SAMPLE_FOLDER


def build_datasets_for(config: Config, splits: tuple[str, ...] = SPLITS, **kwargs) -> dict:
    """Datasets for ``config.model.architecture`` (keyed by split)."""
    arch = canonical_architecture(config.model.architecture)
    spec = ARCHITECTURES[arch]
    return build_datasets(
        samples_folder_for(arch), splits, layout=spec.layout, extra_keys=spec.extra_keys, **kwargs
    )


def build_dataloaders(config: Config, splits: tuple[str, ...] = SPLITS, **kwargs) -> dict:
    """DataLoaders for ``config.model.architecture`` (keyed by split, ``None`` for empty splits)."""
    datasets = build_datasets_for(config, splits, **kwargs)
    return make_loaders(datasets, config.training.batch_size, config.training.num_workers)


def build_model(config: Config, device: torch.device | str = "cpu") -> nn.Module:
    """Instantiate ``config.model.architecture`` with the hyper-parameters of ``config.model``."""
    mc = config.model
    arch = canonical_architecture(mc.architecture)
    grid = config.dataset.height
    unet_kwargs = dict(
        input_channels=mc.input_channels,
        num_classes=mc.num_classes,
        hidden_features=mc.hidden_features,
        use_skip_connections=mc.use_skip_connections,
        use_activation_after_upsampling=mc.use_activation_after_upsampling,
    )

    if arch == "unet":
        from src.models import unet
        model = unet.UNet(**unet_kwargs)
    elif arch == "unet_time_gap":
        from src.models import unet_time_gap
        model = unet_time_gap.UNet(**unet_kwargs)
    elif arch == "unet_attention":
        from src.models import unet_attention
        model = unet_attention.AttentionUNet(**unet_kwargs, use_attention=True)
    elif arch == "unet_segformer":
        from src.models import unet_segformer
        model = unet_segformer.UNetSegFormer(in_channels=mc.input_channels, out_classes=mc.num_classes)
    elif arch == "unet_olmo_offline":
        from src.models import unet_olmo_offline
        model = unet_olmo_offline.UNetOlmo(**unet_kwargs, use_olmo=True, olmo_channels=768)
    elif arch == "asufm":
        from src.models.asufm import asufm, config_asufm
        model = asufm.ASUFM(config=config_asufm.get_asufm_configs(mc.input_channels), num_classes=mc.num_classes)
    elif arch == "dual_dinov3":
        from src.models import dual_dinov3
        model = dual_dinov3.DualDinoV3(
            env_channels=NUM_CHANNELS - len(RGB_FEATURES),
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
            use_activation_after_upsampling=mc.use_activation_after_upsampling,
        )
    elif arch == "umamba":
        from src.models.umamba import UMambaEnc_2d  # requires mamba-ssm (CUDA)
        model = UMambaEnc_2d.UMambaEnc(
            input_size=(grid, grid),
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
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )
    elif arch == "unet_convlstm":
        from src.models import unet_convlstm
        model = unet_convlstm.SpatiotemporalUNet(
            input_channels=mc.input_channels,
            num_classes=mc.num_classes,
            hidden_features=mc.hidden_features,
            use_skip_connections=mc.use_skip_connections,
        )
    elif arch == "utae":
        from src.models import utae
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
    elif arch == "simvpv2_spatiotemporal":
        from src.models.simvp2 import simvp_wrapper
        model = simvp_wrapper.WildfireSimVPWrapper(
            num_timesteps=NUM_TIMESTEPS, channels_per_step=mc.input_channels, img_size=grid, is_spatial_only=False
        )
    elif arch == "video_swin_unet":
        from src.models.video_swin_hybrid_unet import video_swin_hybrid_unet
        model = video_swin_hybrid_unet.VideoSwinHybridUNet(
            in_channels=mc.input_channels, time_steps=NUM_TIMESTEPS, input_size=(grid, grid), num_classes=mc.num_classes
        )
    elif arch == "frp":
        from src.models.frp.BackboneDecoder import Decoder
        from src.models.frp.BackboneEncoder import Encoder
        from src.models.frp.Model import FullNetwork
        from src.models.frp.MultimodalEncoder import CombinedBlock, ConstantBlock, DynamicBlock, StateBlock

        device = torch.device(device)
        state = StateBlock(
            input_dim=len(STATE_FEATURES),
            hidden_dims=[4, 8, 16, 16],
            kernel_sizes=[(5, 5), (3, 3), (3, 3), (1, 1)],
            num_layers=4,
            num_conv_filters=[8, 16, 16],
            device=device,
        )
        dynamic = DynamicBlock(
            input_channels=len(DYNAMIC_FEATURES),
            convlstm_hidden_channels=16,
            conv_hidden_channels=[8, 16, 16],
            device=device,
        )
        constant = ConstantBlock(input_channels=len(STATIC_FEATURES))
        model = FullNetwork(CombinedBlock(state, dynamic, constant), Encoder(), Decoder())
    else:  # pragma: no cover - guarded by canonical_architecture
        raise ValueError(arch)

    return model.to(device)


def forward_batch(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    """Run ``model`` on a batch produced by :class:`src.data.WildfireDataset`; returns logits."""
    kwargs = {}
    if "satellite_age" in batch:
        kwargs["delta_t"] = batch["satellite_age"].to(device)
    if "olmo_emb" in batch:
        kwargs["olmo_emb"] = batch["olmo_emb"].to(device)

    if "input_grids" in batch:
        return model(batch["input_grids"].to(device), **kwargs)
    if "input_rgb" in batch:
        return model(batch["input_rgb"].to(device), batch["input_env"].to(device), **kwargs)
    if "state" in batch:
        return model(batch["state"].to(device), batch["dynamic"].to(device), batch["constant"].to(device))
    raise ValueError(f"Unrecognised batch keys: {sorted(batch)}")


def load_weights(model: nn.Module, checkpoint_path: str | Path, device: torch.device | str = "cpu") -> dict:
    """Load ``model_state_dict`` from a checkpoint written by :class:`src.trainer.Trainer`.

    Returns the checkpoint dict (with ``epoch`` and ``best_val_loss``).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def find_checkpoint(
    checkpoint_root: str | Path,
    architecture: str,
    run_id: int,
    filename: str = "best_checkpoint.pt",
) -> Path | None:
    """Locate the checkpoint of ``architecture`` / ``run_id`` under ``checkpoint_root``.

    Recognised layouts (``<name>`` is the code or the Hugging Face name)::

        <root>/checkpoints_<name>_run<N>/best_checkpoint.pt          (written by scripts/parallel_main.py)
        <root>/<hf_name>/checkpoints_<hf_name>_run_<N>/best_checkpoint.pt   (Hugging Face benchmark repo)
    """
    root = Path(os.path.expanduser(str(checkpoint_root)))
    arch = canonical_architecture(architecture)
    names = {arch, ARCH_TO_HF_NAME.get(arch, arch)}
    candidates = []
    for name in names:
        for run in (f"run{run_id}", f"run_{run_id}"):
            folder = f"checkpoints_{name}_{run}"
            candidates += [root / folder / filename, root / name / folder / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
