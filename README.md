# 🍁 Code for DailyWildfireCAN: Next-Day Wildfire Spread Prediction in Canada

[📄 Paper](https://arxiv.org/abs/2611.14207) · [🌐 Project page](https://dailywildfirecan.github.io) · [🗂️ Dataset](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) · [🏆 Benchmark checkpoints](https://huggingface.co/DailyWildfireCAN/DailyWildfireCAN-Benchmark)

This repository contains the code to **build the dataset** (`data_preparation/`, `samples_generation/`) and to **train and evaluate the benchmark models** (`src/`, `main.py`). The dataset content, splits, sample format, channel order and licenses are documented on the [dataset card](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN). Trained checkpoints and results are on the [benchmark card](https://huggingface.co/DailyWildfireCAN/DailyWildfireCAN-Benchmark).

![DailyWildfireCAN overview](figures/Fig1_DailyWildfireCAN.png)

## Citation

```bibtex
@article{dailywildfirecan,
  title={DailyWildfireCAN: A Dataset and Benchmark for Next-Day Wildfire Spread Prediction in Canada},
  author={Ettahri, Mahacine and Abdelwahed, Hager Radi and Olutayo, Toluwaleke and Aloise, Daniel and Caron, Ga{\"e}tan Marceau},
  journal={arXiv preprint arXiv:2611.14207},
  year={2026}
}
```

## Setup

Python 3.10. With [uv](https://docs.astral.sh/uv/) (versions pinned in `uv.lock`):

```bash
git clone https://github.com/hagerradi/DailyWildFireCAN && cd DailyWildFireCAN
uv sync && source .venv/bin/activate
```

or with pip: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

Optional, model-specific packages:

* U-Mamba: `pip install mamba-ssm` (CUDA only).
* OlmoEarth: `pip install olmoearth-pretrain` (only to compute the embeddings, see below).
* DINOv3: accept the license of [`facebook/dinov3-vitl16-pretrain-sat493m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) and run `hf auth login`.

All paths are defined in [`configs/settings.py`](configs/settings.py) and can be overridden with environment variables:

| Variable | Default | Content |
|---|---|---|
| `WILDFIRE_SAMPLE_FOLDER` | `<OUTPUT_FOLDER>/Samples` | Mono-temporal samples |
| `WILDFIRE_TIMESERIES_SAMPLE_FOLDER` | `<OUTPUT_FOLDER>/Timeseries_Samples` | Multi-temporal samples |
| `WILDFIRE_OUTPUT_FOLDER` | `output_folder` | Generated data (fire cubes, metadata, samples) |
| `WILDFIRE_DATA_FOLDER` | `data_folder` | Raw sources (only to rebuild the fire cubes) |

All commands below are run from the project root.

## 📂 Data

### Pre-built samples (recommended)

Download the mono-temporal (`mono_temporal_samples/`, ~150 GB) and/or multi-temporal (`multi_temporal_samples/`, ~360 GB) samples from the [dataset repository](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) and extract them:

```bash
hf download DailyWildfireCAN/DailyWildfireCAN --repo-type dataset \
    --include "mono_temporal_samples/*" --local-dir hf_data
mkdir -p Samples && for t in hf_data/mono_temporal_samples/*.tar; do tar -xf "$t" -C Samples; done
export WILDFIRE_SAMPLE_FOLDER=$PWD/Samples   # WILDFIRE_TIMESERIES_SAMPLE_FOLDER for multi-temporal samples
```

Use `--include "mono_temporal_samples/test_hard.tar"` to start with Test-Hard only (2.6 GB). The test archives extract to `test_time/` (Test-Easy), `test_space/` (Test-Medium) and `test_spacetime/` (Test-Hard); the code accepts both these names and `test_easy/`, `test_medium/`, `test_hard/`. The sample format and the 50 input channels are documented on the dataset card and in [`src/features.py`](src/features.py).

### Generating the samples from the fire cubes

Download `raw_data/` from the dataset repository and place `fire_data/*.h5` in `H5_OUTPUT_FOLDER`, `metadata_fires_per_year/*.json` in `METADATA_FOLDER` and `fire_growth_points_CFSDS/` in `BASE_FOLDER` (see `configs/settings.py`). Then:

```bash
python -m samples_generation.data_generator_main --type simple       # mono-temporal  -> SAMPLE_FOLDER
python -m samples_generation.data_generator_main --type timeseries   # multi-temporal -> TIMESERIES_SAMPLE_FOLDER
```

Features, years and sequence length are set in `samples_generation/samples_configs/samples_settings.py`. The training statistics (`training_stats_*.json`) are written next to the samples. To rebuild the fire cubes from the original sources (ERA5-Land, SCANFI, Sentinel-2, ...), see [`data_preparation/README.md`](data_preparation/README.md).

The analyses of Section 5 need extra files that are not part of the release; they are generated from the fire cubes:

```bash
python -m samples_generation.metadata_generator_main --type simple            # per-sample metadata (generalization analysis)
python -m samples_generation.metadata_generator_main --type simple --labels   # relaxed labels (label-variation analysis)
python -m samples_generation.data_appender_olmo --folder $WILDFIRE_SAMPLE_FOLDER   # OlmoEarth embeddings (olmo_emb)
```

## 🤖 Models

| Model (paper) | `--arch` | Checkpoint folder on Hugging Face | Samples |
|---|---|---|---|
| U-Net | `unet` | `unet` | mono |
| U-Net with time-gap encoding | `unet_time_gap` | `unet_with_time_gap` | mono |
| Attention U-Net | `unet_attention` | `attention_unet` | mono |
| SegFormer U-Net | `unet_segformer` | `segformer_unet` | mono |
| Swin-U-Net | `asufm` | `swin_unet` | mono |
| U-Mamba | `umamba` | – | mono |
| OlmoEarth | `unet_olmo_offline` | `olmoearth` | mono + `olmo_emb` |
| DINOv3 | `dual_dinov3` | `dinov3` | mono |
| U-Net with ConvLSTM | `unet_convlstm` | `unet_with_convlstm` | multi |
| U-TAE | `utae` | `utae` | multi |
| SimVPv2 | `simvpv2_spatiotemporal` | `simvpv2` | multi |
| Video Swin-U-Net | `video_swin_unet` | `video_swin_unet` | multi |
| Multi-modal (FRP) | `frp` | `multi_mode_model` | multi |

Every command that takes an architecture accepts both the `--arch` name and the Hugging Face folder name. The implementations are in `src/models/` and are instantiated by `src/builder.py`; the references are listed in the paper.

### Training

[`configs/default.yaml`](configs/default.yaml) holds the hyper-parameters of the paper (Appendix B): AdamW (lr 1e-3, weight decay 1e-4), batch size 16, 20 epochs, one warm-up epoch followed by a cosine schedule, and a 0.5/0.5 Focal (α = 0.85, γ = 2) + Dice loss.

```bash
python main.py --arch unet_time_gap
```

trains on `train`, keeps the checkpoint with the lowest validation loss (`checkpoints_<arch>/best_checkpoint.pt`) and evaluates it on the three test splits. Training resumes automatically from `latest_checkpoint.pt` if present. Set `comet.enabled: true` and `COMET_API_KEY` to log to [Comet](https://www.comet.com).

The paper reports the mean and standard deviation over three seeds. `scripts.parallel_main` trains one seed and writes `checkpoints_<arch>_run<N>/`:

```bash
for run in 1 2 3; do python -m scripts.parallel_main --arch utae --run_id $run; done
sbatch scripts/parallel_script.sh utae      # same on SLURM (array job)
```

### Evaluation

`src.evaluate` evaluates every run of every architecture on the test splits and writes one CSV row per architecture, run and split (AUC-PR, average precision, F1, IoU, precision, recall at 0.5). To reproduce the benchmark with the released checkpoints:

```bash
hf download DailyWildfireCAN/DailyWildfireCAN-Benchmark --local-dir .   # -> benchmark_models_checkpoints/
python -m src.evaluate --checkpoint_dir benchmark_models_checkpoints \
    --architectures unet unet_with_time_gap utae --splits test_hard --output results.csv
```

It finds both `checkpoints_<arch>_run<N>/` (written by `scripts.parallel_main`) and the released `<name>/checkpoints_<name>_run_<N>/`.

The heuristic baselines (persistence, dilation and cellular automata) are evaluated on the mono-temporal samples with:

```bash
python -m src.baselines.evaluate --stats hf_data/mono_temporal_samples/training_stats_mono_temporal_samples.json
```

### Analyses (Section 5)

```bash
python -m src.evaluation.run_analysis --analysis <analysis> --architecture <arch> --checkpoint_dir benchmark_models_checkpoints
# optional: --runs 1 2 3  --device cuda  --metadata_folder <dir>  --output <file>.csv
```

| `--analysis` | Paper | Test data | Extra files |
|---|---|---|---|
| `generalization` | Tables 7, 8 and C.3: fire size, fire age, month (Test-Hard), year (Test-Medium), province (Test-Easy + Test-Hard) | as listed | metadata JSONs |
| `spatial` | Figure 4, Appendix C.4: boundary F1 (90/180/270 m) and spread-distance F1 | combined test sets | – |
| `variation` | Table 9: original, 1-pixel dilated and 3-day merged labels | combined test sets | `<split>_temporal_labels/` |

Each run writes `<analysis>_<arch>.csv` (mean and std over runs) and `<analysis>_<arch>.runs.csv` (per run). See [Generating the samples](#generating-the-samples-from-the-fire-cubes) for the extra files.

## Repository structure

```text
├── main.py              # Train and test one model
├── configs/             # settings.py (paths) and default.yaml (hyper-parameters)
├── data_preparation/    # Build the per-fire .h5 cubes from the original sources
├── samples_generation/  # Build the .npz samples, metadata and relaxed labels from the cubes
├── src/
│   ├── features.py      # The 50 input channels
│   ├── data.py          # Dataset and splits
│   ├── builder.py       # Architecture registry, model construction, checkpoint lookup
│   ├── train.py, trainer.py, losses.py, metrics.py
│   ├── evaluate.py      # Benchmark evaluation
│   ├── baselines/       # Persistence, dilation and cellular automata
│   ├── evaluation/      # Section 5 analyses
│   └── models/
├── scripts/             # Multi-seed training and SLURM scripts
├── notebooks/           # Walkthrough of the representative sample
├── legacy/              # Earlier, unmaintained experiments (not used in the paper)
└── figures/
```

## License

The code in this repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license. The [dataset](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) is released under CC BY-NC-SA 4.0 and includes third-party data subject to their own licenses (see the dataset card).

## Contact

For inquiries, contact Mahacine Ettahri (mahacine.ettahri@etud.polymtl.ca) or Hager Radi (hager.radi@mila.quebec).
