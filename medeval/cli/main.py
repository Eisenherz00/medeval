"""Main CLI entry point for medeval."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from medeval.cli.evaluate import evaluate_command

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML or JSON file.

    Parameters
    ----------
    config_path : Path
        Path to config file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def main(args: Optional[list] = None) -> int:
    """
    Main entry point for medeval CLI.

    Parameters
    ----------
    args : list, optional
        Command-line arguments (for testing). If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="MedEval: Medical Imaging Evaluation Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate metrics on dataset")
    eval_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to CSV/JSON manifest file with predictions and targets",
    )
    eval_parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML/JSON config file",
    )
    eval_parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results",
    )
    eval_parser.add_argument(
        "--task",
        type=str,
        choices=["segmentation", "classification", "detection", "registration"],
        help="Task type",
    )
    eval_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 1

    # Set logging level
    if parsed_args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config if provided
    config = {}
    if parsed_args.config:
        config = load_config(parsed_args.config)

    try:
        if parsed_args.command == "evaluate":
            return evaluate_command(parsed_args, config)
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=parsed_args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())

