import logging
import os
import warnings

import colorlog
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from transformers import logging as hf_logging

LOGGING_LEVELS_MAPPING = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


"""
Define the logger
"""


def get_logger(name: str, log_level: str = "info") -> logging.Logger:
    logger = logging.getLogger(name)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(levelname)s%(reset)s] - %(bold_yellow)s%(module)s:%(lineno)s%(reset)s$ %(message_log_color)s%(message)s",  # noqa: E501
        datefmt="%Y-%m-%dT%H:%M:%S",
        # reset=True,
        log_colors={
            "DEBUG": "bold_cyan",
            "INFO": "bold_green",
            "WARNING": "bold_yellow",
            "ERROR": "bold_red",
            "CRITICAL": "bold_red,bg_white",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "white",
                "ERROR": "white",
                "CRITICAL": "bg_red,white",
            }
        },
        style="%",
    )

    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(LOGGING_LEVELS_MAPPING.get(log_level))

    return logger


"""
Methods to interactively modify the logging level
"""


def get_cwd_info(logger=None) -> None:
    print_fn = logger.info if logger else print
    print_fn(f"Is Hydra changing the working directory: {HydraConfig().get().job.chdir}")
    print_fn(f"Original working directory: {get_original_cwd()}")
    print_fn(f"Current working directory: {os.getcwd()}")
    print_fn(f"Hydra run directory: {HydraConfig().get().runtime.output_dir}")


def set_ignore_warnings():
    warnings.simplefilter("ignore")
    # set os environ variable for multiprocesses
    os.environ["PYTHONWARNINGS"] = "ignore"
    hf_logging.set_verbosity_error()
