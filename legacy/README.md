# Legacy code

Earlier experimental pipelines kept for reference. They are **not** used for the paper results or the released [dataset](https://huggingface.co/datasets/DailyWildfireCAN/DailyWildfireCAN) and [benchmark](https://huggingface.co/DailyWildfireCAN/DailyWildfireCAN-Benchmark), which use the main pipeline (`main.py`, `src/`, `samples_generation/`) with the Test-Easy, Test-Medium and Test-Hard splits.

| Pipeline | Files | Samples folder (`configs/settings.py`) |
|---|---|---|
| Single test split (stratified train/val/test) | `main.py`, `parallel_main.py`, `train.py`, `dataloader.py`, `dataloader_timeseries.py`, `samples_generation/data_generator*.py` | `LEGACY_SAMPLE_FOLDER`, `LEGACY_TIMESERIES_SAMPLE_FOLDER` |
| Cluster-based splits | `main_clusters.py`, `train_clusters.py`, `dataloader_clusters.py`, `samples_generation/clusters_generation/` | `SAMPLE_FOLDER_CLUSTERS`, `TIMESERIES_SAMPLE_FOLDER_CLUSTERS` |

Run them from the project root, for example:

```bash
python -m legacy.samples_generation.data_generator_main --config configs/default.yaml --type simple
python -m legacy.main --config configs/default.yaml
```

## Logistic regression baseline

A classical logistic-regression baseline. It is separate from the PyTorch trainer, reads the mono-temporal samples from `SAMPLE_FOLDER/{train,val,test_easy,test_medium,test_hard}`, ignores `delta_t`, extracts a configurable spatial patch around each pixel, and trains `sklearn.linear_model.SGDClassifier(loss="log_loss")`.

Following the approach in Next Day Wildfire Spread [Huot et al], the baseline performs search over the no-fire class-weight multiplier `W` configured in `training.w_values`. The number of SGD epochs is a fixed config input (`training.epochs`) and is not part of the hyperparameter search. The selected model is the one with the highest validation AUC-PR.

```bash
python -m legacy.logistic_regression_main --config legacy/configs/logistic_regression.yaml
```

For a quick wiring check:

```bash
python -m legacy.logistic_regression_main --config legacy/configs/logistic_regression_smoke.yaml
```

On a SLURM cluster:

```bash
sbatch legacy/slurm_logistic_regression.sbatch
```

Set `WILDFIRE_SAMPLE_FOLDER=/path/to/Samples` to avoid editing `configs/settings.py`, or set `data.sample_folder` directly in the logistic-regression config. Outputs include `w_sweep_metrics.csv/json`, `final_metrics.json`, `model.joblib`, `scaler.joblib`, and the feature manifest. Reported metrics include `auc_pr`, `average_precision` (AP), IoU, Dice/F1, precision, recall, and accuracy.
