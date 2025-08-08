from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

from ._version import version as __version__
# from .utilities.physical_constants import constants




__all__ = ["__version__"]
