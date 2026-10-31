# DailyWildfireCAN: A Large-Scale Dataset for Daily Wildfire Spread in Canada
[Currently under review]

This repository contains the code base for the dataset and the benchmark, DailyWildfireCAN, a dataset for daily wildfire spread prediction.

![alt text](figures/image.png)
---

## 🛠️ Part 1 : Path Configuration (settings.py)
Before running any scripts, open configs/settings.py. This file acts as the central nervous system for the pipeline. You must update the following three variables with their **absolute** paths on your machine:

* `PROJECT_FOLDER`: The absolute path to the root repository folder that contains all of the code (e.g., the src, configs, and data_preparation directories).
* `DATA_FOLDER`: The absolute path to the directory containing the raw downloaded datasets, structured exactly as shown in Section 1.2 above.
* `OUTPUT_FOLDER`: The absolute path to the directory where all generated data will be saved. This is where the scripts will output the final .h5 PyTorch datasets.

** Note on Folder Creation:** You only need to manually set up the directories containing the downloaded raw data (like `ERA5/`, `SCANFI/`, etc.). As soon as you run any script in this project, `settings.py` will automatically create the entire required `OUTPUT_FOLDER` structure (including the DEM, Satellite Status, and Metadata directories) if they do not already exist.

---

## ⚙️ Part 2: Environment Setup

### 2.1 Virtual Environment
Create and activate a fresh Python virtual environment:
```bash
# Create the environment
python -m venv env_wildfires_spread

# Activate (Linux/Mac)
source env_wildfires_spread/bin/activate

# Activate (Windows)
env_wildfires_spread\Scripts\activate
```

### 2.2 Install Python Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

## 📂 Part 3: Samples Generation

This is the final data preparation step before model training. At this stage, you have two options: you can either use our pre-compiled samples directly, or you can run this script to generate them from scratch using the raw data.

### Option 1: Use Pre-compiled Samples (Recommended)
If you want to skip the data generation process and jump straight to training, you can use our finalized samples.
Mono- and multi-temporal samples can be downloaded from [here](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN-v1/tree/main/data_samples)
Place them inside the `SAMPLE_FOLDER_ADVANCED` (or `TIMESERIES_SAMPLE_FOLDER_ADVANCED`) defined in your `settings.py`, following this exact structure:
```text
SAMPLES/
   ├── train/
   ├── val/
   ├── test_easy/
   ├── test_medium/
   └── test_hard/
```

### Option 2: Generate Samples from Scratch
If you prefer to run the sample generation process yourself, this script will read the raw daily `.h5` files and the JSON tile mappers to compile the finalized samples. 

> ⚠️ **Important Data Requirements:** > To run this process, you must have the `.h5` files, the JSON mappers (fires metadata), and the raw Fire Growth CSV files. All of these raw data files are provided in our Hugging Face repository in the `raw_data` folder: [DailyWildfireCAN-v1](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN-v1).

Make sure to organize the downloaded files according to the directories defined in your `settings.py`:

**1. Fire Growth CSVs:** Place these in your `BASE_FOLDER`. The structure must look exactly like this:
```text
└── BASE_FOLDER/
    ├── Firegrowth_pts_v1_1_2024/Firegrowth_pts_v1_1_2024.csv
    ├── Firegrowth_pts_v1_1_2023/Firegrowth_pts_v1_1_2023.csv
    ├── Firegrowth_pts_v1_1_2022/Firegrowth_pts_v1_1_2022.csv
    ├── Firegrowth_pts_v1_1_2021/Firegrowth_pts_v1_1_2021.csv
    ├── Firegrowth_pts_v1_1_2020/Firegrowth_pts_v1_1_2020.csv
    ├── Firegrowth_pts_v1_1_2019/Firegrowth_pts_v1_1_2019.csv
    ├── Firegrowth_pts_v1_1_2018/Firegrowth_pts_v1_1_2018.csv
    └── Firegrowth_pts_v1_1_2017/Firegrowth_pts_v1_1_2017.csv
```

**2. JSON Mappers:** Place these in your `METADATA_FOLDER`.
```text
└── METADATA_FOLDER/
    ├── tile_dob_mapper_2024.json
    ├── tile_dob_mapper_2023.json
    ├── tile_dob_mapper_2022.json
    ├── tile_dob_mapper_2021.json
    ├── tile_dob_mapper_2020.json
    ├── tile_dob_mapper_2019.json
    ├── tile_dob_mapper_2018.json
    └── tile_dob_mapper_2017.json
```

**3. H5 Files:** Place these in your `H5_OUTPUT_FOLDER`.

> **⚙️ Customizing Sample Generation:**
> The default variables for feature selection (dynamic/static variables), target years, and the time-series sequence length (default is 3) are pre-configured. If you need to modify any of these before generating your data, edit the settings file located at:
> `samples_generation/samples_configs/samples_settings.py`

This step handles the **Train/Validation/Test splitting** using the CSV fire growth data. It uses the tile mappers to guarantee that geographically overlapping fires are kept strictly within the same fold, ensuring zero spatial data leakage between your training and testing sets.

To generate the dataset, run the following command from the root of your project:

```bash
cd /path/to/your/PROJECT_FOLDER/

python -m samples_generation.data_generator_main --config configs/default.yaml --type simple
```

### Script Arguments:
* `--config`: The path to your configuration file (e.g., `configs/default.yaml`).
* `--type`: The formatting style of the generated samples. 
  * `choices=["simple", "timeseries"]`
  * **`simple`**: Generates standard single-step spatial inputs (Day $T \rightarrow$ Predict Day $T+1$). Best for standard U-Net architectures.
  * **`timeseries`**: Generates sequential temporal inputs (e.g., Days $T, T+1, T+2 \rightarrow$ Predict Day $T+3$). Best for spatio-temporal architectures like ConvLSTM.

**Key Execution Notes:**
* **Dataset Normalization:** During generation, the script automatically calculates the global mean and standard deviation for all features across the training split and saves a `.json` file. This ensures the validation and test sets are normalized exclusively using training statistics.
* **Output Location:** The script saves each generated sample as a numpy (`.npz`) file inside designated split subfolders (e.g., `train`, `val`, `test_easy`, `test_medium`, `test_hard`) within either your `SAMPLE_FOLDER` or `TIMESERIES_SAMPLE_FOLDER`. Each file contains a dictionary with the following:
  * `x`: The input features.
  * `y`: The ground truth target mask.
  * `delta_t`: The satellite image age in days (included for `simple` samples only).
  * `positions`: The sequence positions (included for `timeseries` samples only).
* **Optimization Note:** It is possible to build a PyTorch Dataset that reads directly from the raw daily `.h5` files during training using the code provided in `samples_generation/data_generator.py` and `samples_generation/data_generator_timeseries.py`. However, we pre-compute and save these ready-to-batch `.npz` arrays purely for optimization purposes to significantly accelerate the training loop and maximize GPU utilization.

## 🤖 Part 4: Modeling

With the samples available, the model is ready to train.

### 4.1 Configuration (`configs/default.yaml`)
All training hyperparameters, hardware settings, and logging preferences are centralized in `configs/default.yaml`. Before training, you can adjust this file to suit your needs.

### 4.2 Model Selection & Customization
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
   A U-shaped encoder-decoder architecture with skip connection, where convolutional blocks are replaced with Swin Transformers blocks (Li & Rad, 2024).
6. **U-Mamba** (`architecture: 'umamba'`): 
   An encoder-decoder architecture that incorporates Mamba-based state space blocks alongside traditional convolutions for feature extraction (Ma et al., 2024).
7. **Geospatial Foundation Models** (`architecture: 'unet_olmo_offline'` and `architecture: 'dual_dinov3'`): 
   Two recent foundation models, OlmoEarth (Herzog et al., 2026) and DINOv3 (Siméoni et al., 2025), using their pre-trained representations for the next-day wildfire spread prediction task.

#### Multi-Temporal Models:
1. **Spatiotemporal UNet** (`architecture: 'unet_convlstm'`): 
   A U-Net featuring a **ConvLSTM** bottleneck for recurrent time-series processing (e.g., 3-day sliding window forecasting).
2. **UTAE** (`architecture: 'utae'`):
   A temporal attention encoder-decoder baseline adapted from the ICCV 2021 U-TAE model for satellite image time series. This baseline uses the time-series offline samples from `Timeseries_Samples/`, and the generator now stores sequence positions for the temporal attention encoder when you regenerate those samples.
3. **Simpler yet better video prediction (SimVPv2)** (`architecture: 'simvpv2_spatiotemporal'`): 
   Originally designed for video prediction, SimVPv2 (Tan et al., 2025) is utilized here as a purely convolutional baseline for spatio-temporal spread forecasting.
4. **Video Swin-U-Net** (`architecture: 'video_swin_unet'`): 
   Extended Swin-U-Net by incorporating 3D Video Swin Transformers blocks (Srivastava et al., 2026) to capture joint spatial and temporal dynamics across 3-day sequences.
5. **A Multi-Modal Architecture** (`architecture: 'frp'`): 
   A CNN-based multibranch architecture inspired by (Dong et al., 2024), with separate branches for daily weather variables, fire progression sequences, and static features such topography and land cover, followed by late fusion.

#### References
   Li, B. S., & Rad, R. (2024, June). Wildfire spread prediction in North America using satellite imagery and vision transformer. In *2024 IEEE Conference on Artificial Intelligence (CAI)* (pp. 1536–1541). IEEE.

   Ma, J., Li, F., & Wang, B. (2024). U-mamba: Enhancing long-range dependency for biomedical image segmentation. arXiv preprint arXiv:2401.04722.

   Herzog, H., Bastani, F., Zhang, Y., Tseng, G., Redmon, J., Sablon, H., ... & Beukema, P. (2026). Olmoearth: Stable latent image modeling for multimodal earth observation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 34806-34817).

   Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., ... & Bojanowski, P. (2025). Dinov3. arXiv preprint arXiv:2508.10104.

   Tan, C., Gao, Z., Li, S., & Li, S. Z. (2025). SimVPv2: Towards simple yet powerful spatiotemporal predictive learning. IEEE Transactions on Multimedia, 27, 5170-5184.

   Srivastava, M., Saha, E., & Wang, H. (2026). Spatio-Temporal Wildfire Spread Prediction in Canada using a Video Swin-Hybrid-U-Net and Satellite Imagery. arXiv preprint arXiv:2606.20693.

   Dong, Z., Zhao, F., Wang, G., Tian, Y., & Li, H. (2024). A deep learning framework: Predicting fire radiative power from the combination of polar-orbiting and geostationary satellite data during wildfire spread. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17, 10827-10841.

### 4.3 Training the Model
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

### 4.4 Automatic Evaluation (Testing)
At the end of the `train` loop, `main.py` automatically looks for the `best_checkpoint.pt` generated during training. 

If found, it initiates the `test()` protocol on the holdout test split. This step computes the final metrics. Furthermore, it uploads high-resolution visual predictions (Previous Fire Mask, Ground Truth, Model Prediction) to CometML for your final visual analysis.

### 4.5 Global Evaluation (Optional)
You can run a global evaluation to test all your models at once. This script evaluates every model across its multiple runs and outputs a consolidated CSV file containing all the grouped metrics.

Run the following command from your project root:
```bash
cd /path/to/your/PROJECT_FOLDER/
python -m src.evaluate --config configs/default.yaml

```

> **⚠️ Important: Directory Structure** > For the script to locate your weights, the files must be named `best_checkpoint.pt` and organized in directories following the `checkpoints_<modelName>_run<N>` naming convention.
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

### 4.6 Logistic Regression Baseline

This repository also includes an additive classical logistic-regression baseline. It is separate from the PyTorch trainer, reads the precomputed simple samples from `SAMPLE_FOLDER/{train,val,test}`, ignores `delta_t`, extracts a configurable spatial patch around each pixel, and trains `sklearn.linear_model.SGDClassifier(loss="log_loss")`.

Following the approah in Next Day Wildfire Spread [Huot et al], the baseline performs search over the no-fire class-weight multiplier `W` configured in `training.w_values`. The number of SGD epochs is a fixed config input (`training.epochs`) and is not part of the hyperparameter search. The selected model is the one with the highest validation AUC-PR.

```bash
python -m logistic_regression_main --config configs/logistic_regression.yaml
```

For a quick wiring check:

```bash
python -m logistic_regression_main --config configs/logistic_regression_smoke.yaml
```

On a SLURM cluster:

```bash
sbatch slurm_logistic_regression.sbatch
```

Set `WILDFIRE_SAMPLE_FOLDER=/path/to/Samples` to avoid editing `configs/settings.py`, or set `data.sample_folder` directly in the logistic-regression config. Outputs include `w_sweep_metrics.csv/json`, `final_metrics.json`, `model.joblib`, `scaler.joblib`, and the feature manifest. Reported metrics include `auc_pr`, `average_precision` (AP), IoU, Dice/F1, precision, recall, and accuracy.
