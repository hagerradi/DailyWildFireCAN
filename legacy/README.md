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
