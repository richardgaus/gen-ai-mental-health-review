from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Standard data directories
DATA_UNPROCESSED_DIR = PROJECT_ROOT / "data" / "unprocessed"
DATA_INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Ensure directories exist
DATA_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
