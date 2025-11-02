#!/bin/bash

# Batch script to run all data cleaning scripts in order with their specific arguments
# Usage: ./run_data_cleaning.sh

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located (data_cleaning directory)
DATA_CLEANING_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (3 levels up from data_cleaning)
PROJECT_ROOT="$( cd "$DATA_CLEANING_DIR/../../.." && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Data Cleaning Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Activate the mamba environment
echo -e "${BLUE}Activating mamba environment: llm_psychotherapy_review${NC}"

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

# Script 1: Transform 1st search data
echo -e "${BLUE}1. Running: 1_transform_1st_search_data.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/1_transform_1st_search_data.py" \
    --input "data/unprocessed/covidence_export_1st_search_20251030.csv" \
    --output "data/intermediate/1st_search_data_transformed_for_fusion.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 1_transform_1st_search_data.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 1_transform_1st_search_data.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 2: Preprocess 2nd search data
echo -e "${BLUE}2. Running: 2_preprocess_2nd_search_data.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/2_preprocess_2nd_search_data.py" \
    --input "data/unprocessed/covidence_export_2nd_search_20251030.csv" \
    --output "data/intermediate/2nd_search_data_preprocessed.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 2_preprocess_2nd_search_data.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 2_preprocess_2nd_search_data.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 3: Fuse datasets
echo -e "${BLUE}3. Running: 3_fuse_datasets.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/3_fuse_datasets.py" \
    --first_search "data/intermediate/1st_search_data_transformed_for_fusion.csv" \
    --second_search "data/intermediate/2nd_search_data_preprocessed.csv" \
    --output "data/intermediate/fused_data.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 3_fuse_datasets.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 3_fuse_datasets.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 4: Perform basic cleaning
echo -e "${BLUE}4. Running: 4_perform_basic_cleaning.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/4_perform_basic_cleaning.py" \
    --input "data/intermediate/fused_data.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 4_perform_basic_cleaning.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 4_perform_basic_cleaning.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 5: Remove non-generative studies
echo -e "${BLUE}5. Running: 5_remove_non_generative_studies.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/5_remove_non_generative_studies.py" \
    --input "data/intermediate/fused_data_cleaned.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 5_remove_non_generative_studies.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 5_remove_non_generative_studies.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 6: Calculate interrater agreement
echo -e "${BLUE}6. Running: 6_calculate_interrater_agreement.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/6_calculate_interrater_agreement.py" \
    --input "data/intermediate/fused_data_cleaned_generative_only.csv" \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 6_calculate_interrater_agreement.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 6_calculate_interrater_agreement.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 7: Perform final cleaning
echo -e "${BLUE}7. Running: 7_perform_final_cleaning.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/7_perform_final_cleaning.py" \
    --input "data/intermediate/fused_data_cleaned_generative_only.csv" \
    --output "data/processed" \
    --remove_population_surveys \
    --verbose

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 7_perform_final_cleaning.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 7_perform_final_cleaning.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 8: Extract dataset information
echo -e "${BLUE}8. Running: 8_extract_dataset_information.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/8_extract_dataset_information.py" \
    --input-large "data/processed/final_data.csv" \
    --input-small "data/unprocessed/llm_review_datasets_20251030.csv" \
    --output "data/processed" \
    --reference-column "title"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 8_extract_dataset_information.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 8_extract_dataset_information.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

# Script 9: Extract measurement information
echo -e "${BLUE}9. Running: 9_extract_measurement_information.py${NC}"
echo "----------------------------------------"
mamba run -n llm_psychotherapy_review python "$DATA_CLEANING_DIR/9_extract_measurement_information.py" \
    --input "data/processed/final_data.csv" \
    --output "data/processed"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Successfully completed: 9_extract_measurement_information.py${NC}"
    echo ""
else
    echo -e "${RED}✗ Error running: 9_extract_measurement_information.py${NC}"
    echo -e "${RED}Pipeline stopped.${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Data Cleaning Pipeline Completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Pipeline Summary:${NC}"
echo -e "  1. ✓ Transformed 1st search data"
echo -e "  2. ✓ Preprocessed 2nd search data"
echo -e "  3. ✓ Fused datasets"
echo -e "  4. ✓ Performed basic cleaning"
echo -e "  5. ✓ Removed non-generative studies"
echo -e "  6. ✓ Calculated interrater agreement"
echo -e "  7. ✓ Performed final cleaning"
echo -e "  8. ✓ Extracted dataset information"
echo -e "  9. ✓ Extracted measurement information"
echo ""
echo -e "${GREEN}All processing reports saved to: results/reports_data_processing/${NC}"
echo -e "${GREEN}Final outputs saved to: data/processed/${NC}"
