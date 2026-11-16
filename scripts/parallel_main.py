"""Train and test one seed (``--run_id``) of an architecture; used by ``scripts/parallel_script.sh``.

The seed is derived from ``config.seed`` and the run id, and checkpoints are written
to ``checkpoints_<arch>_run<run_id>/``, the layout read by ``src.evaluate``.

Example::

    python -m scripts.parallel_main --config configs/default.yaml --arch unet --run_id 1
"""
import argparse
from pathlib import Path

from src.builder import canonical_architecture
from src.config import Config
from src.train import test, train


def run_seed(base_seed: int, run_id: int) -> int:
    """Seed used for run ``run_id`` (run 1 uses ``base_seed``)."""
    return (run_id * (base_seed + (run_id - 1))) % (2**31 - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and test one run of a DailyWildfireCAN model.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file.")
    parser.add_argument("--run_id", type=int, default=1, help="Run index (1, 2 or 3 in the paper).")
    parser.add_argument("--arch", default=None, help="Override model.architecture.")
    parser.add_argument("--checkpoint_root", default=".", help="Parent folder of the checkpoint folders.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.arch is not None:
        config.model.architecture = canonical_architecture(args.arch)
    base_seed = config.seed
    config.seed = run_seed(base_seed, args.run_id)
    config.training.checkpoint_dir = str(
        Path(args.checkpoint_root) / f"checkpoints_{config.model.architecture}_run{args.run_id}"
    )
    if config.comet.enabled:
        config.comet.experiment_name = f"{config.model.architecture}_run{args.run_id}"

    print(f"Architecture: {config.model.architecture} | run {args.run_id} | seed {config.seed} "
          f"(base {base_seed}) | checkpoints: {config.training.checkpoint_dir}")

    trainer, loaders = train(config)
    test(trainer, loaders)


if __name__ == "__main__":
    main()
