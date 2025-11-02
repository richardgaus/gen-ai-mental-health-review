"""
Model definitions for data preprocessing.

This module loads model classifications from the YAML configuration file.
"""

import yaml
from pathlib import Path

# Load model definitions from YAML configuration
config_path = Path(__file__).parent.parent / "utils" / "data_cleaning" / "cleaning_config.yaml"

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    NON_GENERATIVE_MODELS = config.get('non_generative_models', [])
    GENERATIVE_MODELS = config.get('generative_models', [])
    
except Exception as e:
    print(f"Warning: Could not load model configuration from {config_path}: {e}")
    # Fallback to empty lists
    NON_GENERATIVE_MODELS = []
    GENERATIVE_MODELS = []

__all__ = ['NON_GENERATIVE_MODELS', 'GENERATIVE_MODELS']
