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
uv sync                                     # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Some models need extra packages or access:

| Model | Requirement |
|---|---|
| U-Mamba | `pip install mamba-ssm` (CUDA only) |
| OlmoEarth | `pip install olmoearth-pretrain`, then precompute the embeddings with `python -m samples_generation.data_appender_olmo` |
| DINOv3 | Accept the license of [`facebook/dinov3-vitl16-pretrain-sat493m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m) and run `hf auth login` |

Edit `configs/settings.py` and set absolute paths for:

* `PROJECT_FOLDER`: this repository.
* `DATA_FOLDER`: raw downloaded sources (see [`data_preparation/README.md`](data_preparation/README.md)).
* `OUTPUT_FOLDER`: generated data. Its subfolders are created automatically the first time any script runs.

All commands below are run from the project root.

## 📂 Data

### Option 1 (recommended): pre-built samples

Download `mono_temporal_samples/` or `multi_temporal_samples/` from the [dataset repository](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) (the card lists sizes and download snippets) and extract each `.tar` into the sample folder from `configs/settings.py`: `SAMPLE_FOLDER` for mono-temporal, `TIMESERIES_SAMPLE_FOLDER` for multi-temporal:

```text
Samples/
├── train/
├── val/
├── test_easy/     # or test_time/
├── test_medium/   # or test_space/
└── test_hard/     # or test_spacetime/
```

The archives extract the test splits as `test_time`, `test_space` and `test_spacetime`; the code accepts both these names and the paper names. The normalization statistics (`training_stats_*.json`) are in the same repository.

### Option 2: generate samples from the raw data

Download `raw_data/` from the dataset repository and place it as follows:

| Raw data | Destination (`configs/settings.py`) |
|---|---|
| `fire_data/*.h5` | `H5_OUTPUT_FOLDER` |
| `metadata_fires_per_year/*.json` | `METADATA_FOLDER` |
| `fire_growth_points_CFSDS/` (one `Firegrowth_pts_v1_1_<year>/` folder per year) | `BASE_FOLDER` |

Then run:

```bash
python -m samples_generation.data_generator_main --config configs/default.yaml --type simple      # mono-temporal
python -m samples_generation.data_generator_main --config configs/default.yaml --type timeseries  # multi-temporal
python -m samples_generation.metadata_generator_main --config configs/default.yaml --type simple  # per-sample metadata, needed only for the analyses
```

Feature selection, target years and sequence length (default 3) are set in `samples_generation/samples_configs/samples_settings.py`. The splitting logic, spatial-leakage guarantees and `.npz` keys are described on the dataset card. Normalization statistics are computed on the training split only and saved as a `.json` next to the samples.

To rebuild the raw `.h5` files from the original sources (ERA5-Land, SCANFI, ASTER, Sentinel-2, etc.), see [`data_preparation/README.md`](data_preparation/README.md).

## 🤖 Models

Set `model.architecture` in [`configs/default.yaml`](configs/default.yaml) to choose a model; the matching data loader (mono- or multi-temporal) is selected automatically. Released checkpoints and results are on the [benchmark card](https://huggingface.co/DailyWildfireCAN/DailyWildfireCAN-Benchmark), whose folder names differ from the `architecture` values:

| Model (paper) | `architecture` | Benchmark folder |
|---|---|---|
| **Mono-temporal** | | |
| U-Net | `unet` | `unet` |
| U-Net + Time-Gap Encoding | `unet_time_gap` | `unet_with_time_gap` |
| Attention U-Net | `unet_attention` | `attention_unet` |
| SegFormer U-Net | `unet_segformer` | `segformer_unet` |
| Swin-U-Net | `asufm` | `swin_unet` |
| U-Mamba | `umamba` | – |
| OlmoEarth | `unet_olmo_offline` | `olmoearth` |
| DINOv3 | `dual_dinov3` | `dinov3` |
| **Multi-temporal** | | |
| U-Net ConvLSTM | `unet_convlstm` | `unet_with_convlstm` |
| U-TAE | `utae` | `utae` |
| SimVPv2 | `simvpv2_spatiotemporal` | `simvpv2` |
| Video Swin-U-Net | `video_swin_unet` | `video_swin_unet` |
| Multi-Model | `frp` | `multi_mode_model` |

Model descriptions and references are in Section 4 of the [paper](https://arxiv.org/abs/2611.14207).

## 🏋️ Training

```bash
python -m main --config configs/default.yaml
```

trains on `train`, keeps the checkpoint with the lowest validation loss as `best_checkpoint.pt` in `training.checkpoint_dir`, then evaluates it on Test-Easy, Test-Medium and Test-Hard (AUC-PR, precision, recall). `configs/utae.yaml` is the same configuration with `architecture: utae`.

`configs/default.yaml` holds the hyper-parameters of the paper (Appendix B): AdamW (learning rate 1e-3, weight decay 1e-4), batch size 16, 20 epochs, a one-epoch linear warm-up followed by cosine decay, and a 0.5 Focal (α = 0.85, γ = 2) + 0.5 Dice loss.

Logging to [Comet](https://www.comet.com) is on by default (`comet.enabled`): export `COMET_API_KEY` and set `comet.project_name` and `comet.workspace`, or set `comet.enabled: false`.

The paper reports the mean and standard deviation over three runs. `scripts.parallel_main` trains one run and saves it in `checkpoints_<architecture>_run<N>/`:

```bash
for run in 1 2 3; do python -m scripts.parallel_main --arch utae --run_id $run; done
sbatch scripts/parallel_script.sh   # the same three runs as a SLURM array job (set --arch in the script)
```

## 📊 Evaluation

```bash
python -m src.evaluate --config configs/default.yaml
```

evaluates the three runs of each architecture listed in `architectures` in [`src/evaluate.py`](src/evaluate.py) (by default `unet`, `unet_attention`, `unet_time_gap`, `unet_convlstm`, `unet_segformer` and `utae`) on the three test splits and writes `evaluation_results.csv` (one row per architecture, run and split). Checkpoints must be at `checkpoints_<architecture>_run<N>/best_checkpoint.pt` (`run_<N>` also works). To evaluate the released checkpoints, download them and rename each `<benchmark folder>/checkpoints_<benchmark folder>_run_<N>/` to `checkpoints_<architecture>_run_<N>/` using the table above:

```bash
hf download DailyWildfireCAN/DailyWildfireCAN-Benchmark --local-dir .
mv benchmark_models_checkpoints/unet_with_time_gap/checkpoints_unet_with_time_gap_run_1 checkpoints_unet_time_gap_run_1
```

`scripts/evaluate_script.sh` runs the same evaluation on SLURM.

The persistence, dilation and cellular-automata baselines are implemented in [`src/baselines/`](src/baselines).

## 🔬 Analyses (Section 5)

```bash
python -m src.evaluation.run_analysis --config configs/default.yaml
```

runs the generalization analysis (fire size, fire duration, month, year and province; Tables 7, 8 and C.3) and the spatial error analysis (boundary and spread-distance F1; Figure 4 and Appendix C.4) for `model.architecture` (`unet`, `unet_olmo_offline` or `utae`). It reads the three runs from `checkpoints_<architecture>/run_<N>/best_checkpoint.pt` and needs the metadata JSONs from `metadata_generator_main` (see [Option 2](#option-2-generate-samples-from-the-raw-data)). The label-variation analysis (Table 9) is in `src/evaluation/variation_analysis.py`.

## Repository structure

```text
├── main.py              # Train and test a model
├── configs/             # settings.py (paths) and YAML training configs
├── data_preparation/    # Build the raw per-fire .h5 files from the original sources
├── samples_generation/  # Build the .npz samples and metadata from the raw data
├── src/                 # Data loaders, models, training, evaluation, baselines and analyses
├── scripts/             # Multi-run training, evaluation and SLURM scripts
├── notebooks/           # Walkthrough of the representative sample (downloaded from Hugging Face)
├── legacy/              # Earlier experiments (single test split, cluster-based splits, logistic regression); not used in the paper
└── figures/
```

## License

The code in this repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license. The [dataset](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) is released under CC BY-NC-SA 4.0 and includes third-party data subject to their own licenses (see the dataset card).

## Contact

For inquiries, contact Mahacine Ettahri (mahacine.ettahri@etud.polymtl.ca) or Hager Radi (hager.radi@mila.quebec).
