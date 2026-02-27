"""Script for preprocessing raw RDPS data into the format required for inference."""

import argparse
import logging

from resoterre.experiments.rdps_to_hrdps_workflow import preprocessing_raw_to_preprocessed


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing script to convert raw RDPS data into preprocessed format for inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Preprocess raw RDPS data using a config file
        python scripts/preprocessing/downscaling_preprocessing_rdps_to_hrdps.py \\
            configs/downscaling/downscaling_preprocessing_rdps_to_hrdps.yaml

        # Preprocess with verbose logging
        python scripts/preprocessing/downscaling_preprocessing_rdps_to_hrdps.py \\
            configs/downscaling/downscaling_preprocessing_rdps_to_hrdps.yaml --verbose
        """,
    )
    parser.add_argument("config", type=str, help="Path to the configuration file for preprocessing.")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output."
    )
    return parser.parse_args()


def main() -> None:
    """Main function to execute the preprocessing process."""
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    print("=" * 80)
    print("RDPS to HRDPS Preprocessing Pipeline")
    print("=" * 80)
    print(f"\nConfiguration file: {args.config}")
    print("\nStarting preprocessing of raw RDPS data...")

    # Run preprocessing
    output_files = preprocessing_raw_to_preprocessed(config=args.config)

    print("\n" + "=" * 80)
    print("Preprocessing Complete!")
    print("=" * 80)
    print(f"\nGenerated {len(output_files)} preprocessed batch file(s):")
    for file in output_files:
        print(f"  - {file}")
    print("\nYou can now use these files for inference with:")
    print("  python scripts/inference/downscaling_inference_rdps_to_hrdps.py <inference_config.yaml>")


if __name__ == "__main__":
    main()
