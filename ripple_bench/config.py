"""Configuration for Ripple Bench.

This file contains default configuration values. To override these locally,
create a config_local.py file in the same directory with your custom values.
"""

# Default HuggingFace cache directory (Roy's path)
DEFAULT_CACHE_DIR = "/n/home04/rrinberg/data_dir/HF_cache"

# Try to import local overrides
try:
    from .config_local import *  # noqa: F401, F403
except ImportError:
    # No local config, use defaults
    pass
