"""
统一 logging 配置（v0.2.1）

用法：
  from scripts._logging import get_logger, setup_logging
  logger = get_logger(__name__)
  setup_logging(level="DEBUG")
"""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: Optional[str] = None,
) -> None:
    """全局配置 logging

    Args:
        level: DEBUG/INFO/WARNING/ERROR
        log_file: 可选，写到文件
        fmt: 自定义格式
    """
    root = logging.getLogger()
    root.setLevel(_LEVELS.get(level.upper(), logging.INFO))

    # 清空旧 handler（避免重复）
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(fmt or _DEFAULT_FORMAT, datefmt=_DATE_FORMAT)

    # console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # 可选文件 handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """取一个标准命名的 logger"""
    return logging.getLogger(name)