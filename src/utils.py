"""
General helper functions, file I/O, device management, and reproducibility helpers.
"""

import json
import os
from pathlib import Path
import pickle
import random
from typing import Any, Dict, List, Optional, Union


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across random, numpy, and torch (if installed) for reproducibility.

    Args:
        seed (int): The integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_project_root() -> Path:
    """Returns the absolute Path to the root directory of the CLNLP project.

    Returns:
        Path: Project root directory path.
    """
    # Assuming this file is located at <project_root>/src/utils.py
    return Path(__file__).resolve().parent.parent


def get_device() -> str:
    """Detects and returns the best available computing device ('cuda', 'mps', or 'cpu').

    Returns:
        str: Device name string.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def read_text(filepath: Union[str, Path], encoding: str = "utf-8") -> str:
    """Reads a text file and returns its content.

    Args:
        filepath: Path to the text file.
        encoding: Character encoding (default 'utf-8').

    Returns:
        str: File content as string.
    """
    with open(filepath, mode="r", encoding=encoding, errors="replace") as f:
        return f.read()


def write_text(content: str, filepath: Union[str, Path], encoding: str = "utf-8") -> None:
    """Writes a string content to a file, creating parent directories if needed.

    Args:
        content: Text content to write.
        filepath: Target file path.
        encoding: Character encoding.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, mode="w", encoding=encoding) as f:
        f.write(content)


def load_json(filepath: Union[str, Path]) -> Any:
    """Loads data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Saves data to a JSON file."""
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """Saves a Python object using pickle."""
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Loads a Python object from a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
