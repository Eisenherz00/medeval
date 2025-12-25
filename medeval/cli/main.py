"""Main CLI entry point for medeval."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

from medeval.cli.evaluate import evaluate_command

# Configure logging - will be adjusted by verbosity flag
logger = logging.getLogger(__name__)


def _configure_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level.

    Parameters
    ----------
    verbosity : int
        0 = WARNING, 1 = INFO, 2+ = DEBUG
    """
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    # Configure root logger and all handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers and add a properly configured one
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        # Update existing handlers' levels
        for handler in root_logger.handlers:
            handler.setLevel(level)


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
        help="Output directory for results. Re-running with the same directory will "
             "overwrite existing files; use a new directory for each run to preserve results.",
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
        action="count",
        default=0,
        help="Verbose output (-v for INFO, -vv for DEBUG)",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 1

    # Configure logging based on verbosity
    _configure_logging(parsed_args.verbose)

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
        logger.error(f"Error: {e}", exc_info=parsed_args.verbose >= 1)
        return 1


if __name__ == "__main__":
    sys.exit(main())

