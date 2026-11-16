"""Generate the DailyWildfireCAN samples from the fire cubes produced by ``data_preparation``.

    python -m samples_generation.data_generator_main --type simple       # mono-temporal
    python -m samples_generation.data_generator_main --type timeseries   # multi-temporal

Features, years and split settings are in ``samples_generation/samples_configs/samples_settings.py``;
input and output folders in ``configs/settings.py``.
"""
import argparse

from samples_generation.samples_configs.samples_settings import SEQUENCE_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DailyWildfireCAN samples.")
    parser.add_argument("--type", required=True, choices=["simple", "timeseries"],
                        help="'simple' (mono-temporal) or 'timeseries' (multi-temporal) samples.")
    args = parser.parse_args()

    if args.type == "simple":
        from samples_generation.data_generator import generate_simple_offline_data
        generate_simple_offline_data()
    else:
        from samples_generation.data_generator_timeseries import generate_timeseries_offline_data
        generate_timeseries_offline_data(seq_length=SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
