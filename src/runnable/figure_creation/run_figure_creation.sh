#!/bin/bash

# Batch script to run all figure creation scripts in order with their specific arguments
# Usage: ./run_figure_creation.sh

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located (figure_creation directory)
FIGURE_CREATION_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (3 levels up from figure_creation)
PROJECT_ROOT="$( cd "$FIGURE_CREATION_DIR/../../.." && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Figure Creation Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Initialize mamba for bash (try multiple common locations)
if [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
    source "$HOME/mambaforge/etc/profile.d/conda.sh"
    source "$HOME/mambaforge/etc/profile.d/mamba.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    source "$HOME/miniforge3/etc/profile.d/mamba.sh"
elif [ -f "/opt/homebrew/Caskroom/mambaforge/base/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/Caskroom/mambaforge/base/etc/profile.d/conda.sh"
    source "/opt/homebrew/Caskroom/mambaforge/base/etc/profile.d/mamba.sh"
else
    echo -e "${YELLOW}Warning: Could not find mamba initialization files. Trying direct activation...${NC}"
fi

# Test environment activation and get python path
PYTHON_PATH=$(mamba run -n llm_psychotherapy_review which python)

if [ $? -eq 0 ] && [ -n "$PYTHON_PATH" ]; then
    echo -e "${GREEN}✓ Successfully found environment: llm_psychotherapy_review${NC}"
    echo -e "${GREEN}  Python path: $PYTHON_PATH${NC}"
    echo ""
else
    echo -e "${RED}✗ Error accessing environment: llm_psychotherapy_review${NC}"
    echo -e "${RED}Please ensure mamba is installed and the environment exists.${NC}"
    exit 1
fi

# Change to the project root directory for relative path consistency
cd "$PROJECT_ROOT"

# Script 1: Create panel overview
echo -e "${BLUE}1. Running: 1_create_panel_overview.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$FIGURE_CREATION_DIR/1_create_panel_overview.py" \
    --input "data/processed/final_data.csv" --exclude-population-surveys

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 1_create_panel_overview.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 1_create_panel_overview.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 2: Create panel clinical
echo -e "${BLUE}2. Running: 2_create_panel_clinical.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$FIGURE_CREATION_DIR/2_create_panel_clinical.py" \
    --input "data/processed/final_data.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 2_create_panel_clinical.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 2_create_panel_clinical.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 3: Find controlled clinical studies
echo -e "${BLUE}3. Running: 3_find_controlled_clinical_studies.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$FIGURE_CREATION_DIR/3_find_controlled_clinical_studies.py" \
    --input "data/processed/final_data.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 3_find_controlled_clinical_studies.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 3_find_controlled_clinical_studies.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 4: Create panel model types
echo -e "${BLUE}4. Running: 4_create_panel_model_types.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$FIGURE_CREATION_DIR/4_create_panel_model_types.py" \
    --input "data/processed/final_data.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 4_create_panel_model_types.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 4_create_panel_model_types.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 5-1: Create panel datasets
echo -e "${BLUE}5-1. Running: 5-1_create_panel_datasets.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$FIGURE_CREATION_DIR/5-1_create_panel_datasets.py" \
    --datasets_file "data/processed/datasets_file.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 5-1_create_panel_datasets.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 5-1_create_panel_datasets.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 5-2: Create panel measurements
echo -e "${BLUE}5-2. Running: 5-2_create_panel_measurements.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$FIGURE_CREATION_DIR/5-2_create_panel_measurements.py" \
    --input "data/processed/measurements_file.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 5-2_create_panel_measurements.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 5-2_create_panel_measurements.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 6: Create panel READI
echo -e "${BLUE}6. Running: 6_create_panel_readi.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$FIGURE_CREATION_DIR/6_create_panel_readi.py" \
    --input "data/processed/final_data.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 6_create_panel_readi.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 6_create_panel_readi.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Figure Creation Pipeline Completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Pipeline Summary:${NC}"
echo -e "  1. ✓ Created panel overview"
echo -e "  2. ✓ Created panel clinical"
echo -e "  3. ✓ Found controlled clinical studies"
echo -e "  4. ✓ Created panel model types"
echo -e "  5-1. ✓ Created panel datasets"
echo -e "  5-2. ✓ Created panel measurements"
echo -e "  6. ✓ Created panel READI"
echo ""
echo -e "${GREEN}All figures and reports saved to: results/${NC}"
echo -e "${GREEN}Individual panel outputs saved to their respective subdirectories${NC}"

