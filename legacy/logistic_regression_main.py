import argparse

from legacy.baselines.logistic_regression import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="DailyWildfireCAN logistic-regression baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="legacy/configs/logistic_regression.yaml",
        help="Path to the logistic-regression YAML config.",
    )
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
