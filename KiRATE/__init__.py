"""
Copyright (c) 2024-2026 Timoteo Dinelli
Licensed under the MIT License - see LICENSE file for details
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from ._version import version as __version__

__all__ = ["__version__"]
