"""Command-line entry point for the analyses of Section 5 of the paper.

Examples::

    # Spatial error analysis of the released U-TAE checkpoints (combined test sets)
    python -m src.evaluation.run_analysis --analysis spatial --architecture utae \
        --checkpoint_dir benchmark_models_checkpoints

    # Robustness analysis (needs the metadata JSONs, see the README)
    python -m src.evaluation.run_analysis --analysis generalization --architecture unet_with_time_gap \
        --checkpoint_dir benchmark_models_checkpoints
"""
import argparse

from src.config import Config

ANALYSES = ("generalization", "spatial", "variation")


def main() -> None:
    parser = argparse.ArgumentParser(description="DailyWildfireCAN analyses (Section 5).")
    parser.add_argument("--analysis", required=True, choices=ANALYSES)
    parser.add_argument("--architecture", required=True, help="Code or Hugging Face architecture name.")
    parser.add_argument("--checkpoint_dir", default=".", help="Folder containing the checkpoint folders.")
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--metadata_folder", default=None, help="Override settings.METADATA_FOLDER.")
    parser.add_argument("--device", default=None, help="Override training.device (auto, cuda, mps, cpu).")
    parser.add_argument("--output", default=None, help="Output CSV (default: <analysis>_<architecture>.csv).")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.device:
        config.training.device = args.device
    output = args.output or f"{args.analysis}_{args.architecture}.csv"
    common = dict(config=config, architecture=args.architecture, checkpoint_dir=args.checkpoint_dir,
                  runs=args.runs, output=output)

    if args.analysis == "generalization":
        from src.evaluation.generalization_analysis import run_generalization_analysis
        run_generalization_analysis(**common, metadata_folder=args.metadata_folder)
    elif args.analysis == "spatial":
        from src.evaluation.spatial_analysis import run_spatial_analysis
        run_spatial_analysis(**common)
    else:
        from src.evaluation.variation_analysis import run_variation_analysis
        run_variation_analysis(**common)


if __name__ == "__main__":
    main()
