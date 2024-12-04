import os
import sys
import time
from loguru import logger
from typing import Iterable

from .parallel import MP_CONTEXT
from .io import StrPath
from .distributed import test_rank, get_rank_id


def get_loguru_format():
    loguru_format = os.environ.get("LOGURU_FORMAT")
    if loguru_format is None:
        if not test_rank():
            loguru_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<magenta>{process: <7}</magenta> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            )
        else:
            loguru_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<yellow>{extra[rank_id]: <6}</yellow> | "
                "<magenta>{process: <7}</magenta> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
            )
    return loguru_format


def init_loguru_logger(*paths: Iterable[StrPath], default_sink=None):
    """Initialize the logger for logruru, compatible with distributed processes."""

    loguru_format = get_loguru_format()

    # https://loguru.readthedocs.io/en/stable/resources/recipes.html#compatibility-with-multiprocessing-using-enqueue-argument
    shared_config = {"format": loguru_format, "enqueue": True, "context": MP_CONTEXT}

    # https://github.com/Delgan/loguru/issues/135#issuecomment-1374701618
    if default_sink is None:
        try:
            from tqdm import tqdm as _tqdm

        except ImportError:
            default_handler = {"sink": sys.stdout, **shared_config}
        else:
            default_handler = {
                "sink": lambda msg: _tqdm.write(msg, end=""),
                "colorize": True,
                **shared_config,
            }
    else:
        default_handler = {"sink": default_sink, **shared_config}

    extra = (
        {"rank_id": get_rank_id(), "log_id": f"d{get_rank_id()}"}
        if test_rank()
        else {"log_id": "c"}
    )

    timestamp = int(time.time() * 1000)
    config = {
        "handlers": [
            default_handler,
            *(
                {"sink": str(path).format(**extra, timestamp=timestamp), **shared_config}
                for path in paths
            ),
        ],
        "extra": extra,
    }
    logger.configure(**config)
    return logger
