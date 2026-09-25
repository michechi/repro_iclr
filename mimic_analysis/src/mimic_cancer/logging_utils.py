from __future__ import annotations

import logging
import os
import resource
import sys

import psutil


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def peak_rss_gib() -> float:
    """Peak resident set size in GiB, combining psutil (current) and
    resource.ru_maxrss (historical max). On Linux ru_maxrss is in kB.
    """
    proc = psutil.Process(os.getpid())
    rss_now = proc.memory_info().rss / (1024**3)
    ru_max_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_max_gib = ru_max_kb / (1024**2)  # kB -> GiB
    return max(rss_now, ru_max_gib)


def log_peak_memory(logger: logging.Logger, label: str = "") -> None:
    gib = peak_rss_gib()
    suffix = f" ({label})" if label else ""
    logger.info("peak RSS: %.2f GiB%s", gib, suffix)
