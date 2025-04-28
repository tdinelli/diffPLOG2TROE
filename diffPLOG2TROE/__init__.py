"""diffPLOG2TROE - A differentiable PLOG to TROE refitter"""

from __future__ import annotations
from ._version import version as __version__
import jax


jax.config.update("jax_enable_x64", True)


__all__ = ["__version__"]
