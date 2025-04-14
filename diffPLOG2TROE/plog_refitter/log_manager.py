import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_name: Optional[str], console_output: bool = True) -> logging.Logger:
    """
    Setup logging for PLOG refitting.

    Args:
        log_name: Name of log file
        console_output: Whether to output to console

    Returns:
        Logger object
    """
    logger = logging.getLogger("PlogRefitter")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers

    formatter = logging.Formatter("%(message)s")

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_name:
        log_path = Path.cwd()
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / log_name, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_initialization(
    logger: logging.Logger,
    fitting_mode: str,
    primary_falloff_type: str,
    secondary_falloff_type: Optional[str],
    T_range: tuple,
    P_range: tuple,
    loss_name: str,
) -> None:
    """
    Log initial configuration settings.

    Args:
        logger: Logger object
        fitting_mode: Fitting mode ("single" or "duplicate")
        primary_falloff_type: Primary falloff type
        secondary_falloff_type: Secondary falloff type (for duplicate mode)
        T_range: Temperature range
        P_range: Pressure range
        loss_name: Loss function name
    """
    logger.info("=" * 89)

    if fitting_mode == "single":
        logger.info(f"Plog to {primary_falloff_type} refitter")
    else:
        logger.info(f"Plog to Duplicate Reactions ({primary_falloff_type} + {secondary_falloff_type}) refitter")

    logger.info(f" Temperature range [K]: {T_range}")
    logger.info(f" Pressure range [atm]: {P_range}")
    logger.info(f" Loss function: {loss_name}")
