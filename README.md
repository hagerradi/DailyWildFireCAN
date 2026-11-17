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

```bash
git clone https://github.com/hagerradi/DailyWildFireCAN && cd DailyWildFireCAN
python -m venv env_wildfires_spread
source env_wildfires_spread/bin/activate   # Windows: env_wildfires_spread\Scripts\activate
pip install -r requirements.txt
```

Edit `configs/settings.py` and set absolute paths for:

* `PROJECT_FOLDER`: this repository.
* `DATA_FOLDER`: raw downloaded sources (see [`data_preparation/README.md`](data_preparation/README.md)).
* `OUTPUT_FOLDER`: generated data. Its subfolders are created automatically the first time any script runs.

## 📂 Data

### Option 1 (recommended): pre-built samples

Download `mono_temporal_samples/` or `multi_temporal_samples/` from the [dataset repository](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) (the card lists sizes and download snippets) and extract each `.tar` into the sample folder from `configs/settings.py`: `SAMPLE_FOLDER` for mono-temporal, `TIMESERIES_SAMPLE_FOLDER` for multi-temporal:

```text
Samples/
├── train/
├── val/
├── test_easy/
├── test_medium/
└── test_hard/
```

The normalization statistics (`training_stats_*.json`) are in the same repository.

### Option 2: generate samples from the raw data

Download `raw_data/` from the dataset repository and place it as follows:

| Raw data | Destination (`configs/settings.py`) |
|---|---|
| `fire_data/*.h5` | `H5_OUTPUT_FOLDER` |
| `metadata_fires_per_year/*.json` | `METADATA_FOLDER` |
| `fire_growth_points_CFSDS/` (one `Firegrowth_pts_v1_1_<year>/` folder per year) | `BASE_FOLDER` |

Then run from the project root (`metadata_generator_main` builds the per-sample metadata used by the analysis scripts in `src/evaluation/` and is optional):

```bash
python -m samples_generation.metadata_generator_main --config configs/default.yaml --type simple   # optional
python -m samples_generation.data_generator_main --config configs/default.yaml --type simple      # mono-temporal
python -m samples_generation.data_generator_main --config configs/default.yaml --type timeseries  # multi-temporal
```

Feature selection, target years and sequence length (default 3) are set in `samples_generation/samples_configs/samples_settings.py`. The splitting logic, spatial-leakage guarantees and `.npz` keys are described on the dataset card. Normalization statistics are computed on the training split only and saved as a `.json` next to the samples.

To rebuild the raw `.h5` files from the original sources (ERA5-Land, SCANFI, ASTER, Sentinel-2, etc.), see [`data_preparation/README.md`](data_preparation/README.md).

## 🤖 Modeling


Pre-trained checkpoints and AUC-PR results for these models are on the [benchmark card](https://huggingface.co/DailyWildfireCAN/DailyWildfireCAN-Benchmark).

### Configuration (`configs/default.yaml`)
All training hyperparameters, hardware settings, and logging preferences are centralized in `configs/default.yaml`. Before training, you can adjust this file to suit your needs.

### Model Selection & Customization
Multiple model architectures are implemented in the `src/models/` directory to handle different temporal and spatial requirements. 

To switch between architectures (which will automatically configure the corresponding dataloaders, such as swapping from single-day static prediction to a 3-day sliding window), open your `configs/default.yaml` and update the `architecture` parameter under the `model` section to one of the following options:

#### Mono-Temporal Models:
1. **Standard UNet** (`architecture: 'unet'`):
   The baseline spatial U-Net model.
2. **U-Net with time-gap encoding** (`architecture: 'unet_time_gap'`): 
   A U-Net that explicitly encodes the satellite age (the time gap in days between the fire event and the satellite acquisition).
3. **Attention UNet** (`architecture: 'unet_attention'`): 
   A U-Net utilizing attention gates in the skip connections to help the model focus on the most critical spatial features and suppress irrelevant background noise.
4. **UNet-SegFormer** (`architecture: 'unet_segformer'`): 
   A hybrid vision-transformer architecture that replaces the standard CNN encoder with SegFormer's Mix Vision Transformer (MiT), paired with a standard U-Net decoder for heavy pixel-level accuracy. 
5. **Swin-U-Net** (`architecture: 'asufm'`): 
   A U-shaped encoder-decoder architecture with skip connection, where convolutional blocks are replaced with Swin Transformers blocks (Cao et al., 2022).
6. **U-Mamba** (`architecture: 'umamba'`): 
   An encoder-decoder architecture that incorporates Mamba-based state space blocks alongside traditional convolutions for feature extraction (Ma et al., 2024).
7. **Geospatial Foundation Models** (`architecture: 'unet_olmo_offline'` and `architecture: 'dual_dinov3'`): 
   Two recent foundation models, OlmoEarth (Herzog et al., 2026) and DINOv3 (Siméoni et al., 2025), using their pre-trained representations for the next-day wildfire spread prediction task.

#### Multi-Temporal Models:
1. **Spatiotemporal UNet** (`architecture: 'unet_convlstm'`): 
   A U-Net featuring a **ConvLSTM** bottleneck for recurrent time-series processing (e.g., 3-day sliding window forecasting).
2. **UTAE** (`architecture: 'utae'`):
   A temporal attention encoder-decoder baseline adapted from the ICCV 2021 U-TAE model for satellite image time series. It uses the multi-temporal samples and their sequence positions.
3. **Simpler yet better video prediction (SimVPv2)** (`architecture: 'simvpv2_spatiotemporal'`): 
   Originally designed for video prediction, SimVPv2 (Tan et al., 2025) is utilized here as a purely convolutional baseline for spatio-temporal spread forecasting.
4. **Video Swin-U-Net** (`architecture: 'video_swin_unet'`): 
   Extended Swin-U-Net by incorporating 3D Video Swin Transformers blocks (Srivastava et al., 2026) to capture joint spatial and temporal dynamics across 3-day sequences.
5. **A Multi-Modal Architecture** (`architecture: 'frp'`): 
   A CNN-based multibranch architecture inspired by (Dong et al., 2024), with separate branches for daily weather variables, fire progression sequences, and static features such topography and land cover, followed by late fusion.

Full references for the cited models are given in the [paper](https://arxiv.org/abs/2611.14207).

### Training the Model
The main entry point for the training pipeline is `main.py`, located at the root of the project. 

To run the training loop locally:
```bash
cd /path/to/your/PROJECT_FOLDER/

python -m main --config configs/default.yaml
```

**Key Training Features:**
* **Custom Loss:** Automatically initializes a custom `CombinedLoss` (Focal + Dice) to handle the extreme class imbalance of wildfire pixels.
* **Automatic Checkpointing:** The `Trainer` monitors the Validation IoU. Whenever the model improves, it automatically overwrites and saves `best_checkpoint.pt` to your configured `checkpoint_dir`.
* **Comet.ml Integration:** If `enabled: true` in your config, the pipeline will automatically log learning rates, loss curves, and epoch-by-epoch evaluation metrics directly to your Comet dashboard. It also uploads visual grid predictions at the end of epochs so you can watch the model learn.

### Automatic Evaluation (Testing)
At the end of the `train` loop, `main.py` automatically looks for the `best_checkpoint.pt` generated during training. 

If found, it runs the `test()` protocol on the three test splits (Test-Easy, Test-Medium and Test-Hard). This step computes the final metrics. Furthermore, it uploads high-resolution visual predictions (Previous Fire Mask, Ground Truth, Model Prediction) to CometML for your final visual analysis.

### Global Evaluation (Optional)
You can run a global evaluation to test all your models at once. This script evaluates every model across its multiple runs on the three test splits and writes `evaluation_results.csv` with one row per model, run and split.

Run the following command from your project root:
```bash
cd /path/to/your/PROJECT_FOLDER/
python -m src.evaluate --config configs/default.yaml

```

> **⚠️ Note:** the released [benchmark checkpoints](https://huggingface.co/DailyWildfireCAN/DailyWildfireCAN-Benchmark) are named `checkpoints_<model>_run_<N>` and are grouped in folders such as `unet_with_time_gap`. The script accepts both `run<N>` and `run_<N>`, but `<model>` must be the `architecture` value (e.g. `unet_time_gap`), so move the checkpoint folders next to each other in the project root.
>
> **Directory Structure:** For the script to locate your weights, the files must be named `best_checkpoint.pt` and organized in directories following the `checkpoints_<modelName>_run<N>` naming convention.
> For example, a UNet evaluated across three runs should look like this:
> ```text
> 📁 PROJECT_FOLDER/
>  ├── 📁 checkpoints_unet_run1/
>  │    └── 📄 best_checkpoint.pt
>  ├── 📁 checkpoints_unet_run2/
>  │    └── 📄 best_checkpoint.pt
>  └── 📁 checkpoints_unet_run3/
>       └── 📄 best_checkpoint.pt
> 
> ```

On a SLURM cluster, `scripts/parallel_script.sh` trains the three runs of one architecture as an array job (`python -m scripts.parallel_main --arch <architecture> --run_id <N>`) and saves them in this layout. `scripts/evaluate_script.sh` then runs the global evaluation.

## Repository structure

```text
├── main.py              # Train and test a model (python -m main --config configs/default.yaml)
├── configs/             # settings.py (paths) and YAML training configs
├── data_preparation/    # Build the raw per-fire .h5 files from the original sources
├── samples_generation/  # Build train/val/test .npz samples from the raw data
├── src/                 # Dataloaders, models, training, evaluation and baselines
├── scripts/             # Parallel-training and SLURM entry points
├── notebooks/           # Walkthrough of the representative sample (downloaded from Hugging Face)
├── legacy/              # Earlier experiments (single test split, cluster-based splits, logistic regression); not used for the paper
└── figures/
```

## License

The code in this repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license. The [dataset](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) is released under CC BY-NC-SA 4.0 and includes third-party data subject to their own licenses (see the dataset card).

## Contact

For inquiries, contact Mahacine Ettahri (mahacine.ettahri@etud.polymtl.ca) or Hager Radi (hager.radi@mila.quebec).
