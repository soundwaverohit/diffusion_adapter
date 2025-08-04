"""
Shared utilities for inference: configuration loading and logging setup.
"""
import yaml
import logging


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Configure root logger to output to console and optionally to a file.
    Returns the logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

