"""Generate the per-sample metadata and alternative labels used by the analyses (Section 5).

    # Metadata JSONs (fire size, date, fire age, location) for the robustness analysis
    python -m samples_generation.metadata_generator_main --type simple
    python -m samples_generation.metadata_generator_main --type timeseries

    # Dilated and multi-day labels (<split>_temporal_labels/) for the label-variation analysis
    python -m samples_generation.metadata_generator_main --type simple --labels

Both require the fire cubes produced by ``data_preparation``.
"""
import argparse

from samples_generation.samples_configs.samples_settings import SEQUENCE_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the metadata and alternative labels of the samples.")
    parser.add_argument("--type", required=True, choices=["simple", "timeseries"],
                        help="'simple' (mono-temporal) or 'timeseries' (multi-temporal) samples.")
    parser.add_argument("--labels", action="store_true",
                        help="Generate the label-variation targets instead of the metadata (simple only).")
    args = parser.parse_args()

    if args.labels:
        if args.type != "simple":
            parser.error("--labels is only available for --type simple")
        from samples_generation.data_generator import generate_labels_all_datasets
        generate_labels_all_datasets()
    elif args.type == "simple":
        from samples_generation.data_generator import generate_metadata_all_datasets
        generate_metadata_all_datasets()
    else:
        from samples_generation.data_generator_timeseries import generate_timeseries_metadata_all_datasets
        generate_timeseries_metadata_all_datasets(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
