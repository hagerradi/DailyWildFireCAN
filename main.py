"""Train one model and evaluate it on Test-Easy, Test-Medium and Test-Hard.

Example::

    python main.py --config configs/default.yaml --arch unet_time_gap
"""
import argparse

from src.builder import canonical_architecture
from src.config import Config
from src.train import test, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test a DailyWildfireCAN model.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file.")
    parser.add_argument("--arch", default=None, help="Override model.architecture.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.num_epochs.")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override training.checkpoint_dir (default with --arch: checkpoints_<arch>).")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.arch is not None:
        config.model.architecture = canonical_architecture(args.arch)
        config.training.checkpoint_dir = f"checkpoints_{config.model.architecture}"
        config.comet.experiment_name = config.model.architecture
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.epochs is not None:
        config.training.num_epochs = args.epochs

    trainer, loaders = train(config)
    test(trainer, loaders)


if __name__ == "__main__":
    main()
