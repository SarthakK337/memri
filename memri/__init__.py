"""memri — Observational Memory for coding agents."""

from .config import MemriConfig
from .core.memory import MemriMemory

__version__ = "0.1.0"
__all__ = ["MemriMemory", "MemriConfig"]
