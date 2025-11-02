# Data Preprocessing Pipeline

This directory contains the modular data preprocessing pipeline for the LLMs in Psychotherapy research project.

## Structure

```
src/
├── data_preprocessing/          # Core preprocessing modules
│   ├── __init__.py             # Package initialization
│   ├── id_management.py        # ID assignment and management functions
│   └── consensus_filtering.py  # Consensus review filtering functions
└── runnable/                   # Lightweight runnable scripts
    └── preprocess_data.py      # Main preprocessing script
```

## Usage

### Prerequisites

Activate the conda/mamba environment:
```bash
mamba activate llm_psychotherapy_review
```

### Running the Preprocessing Pipeline

From the `src/runnable/` directory:

```bash
# Basic usage (uses default input file)
python preprocess_data.py

# With custom input file (output filename generated automatically)
python preprocess_data.py --input my_data.csv

# With verbose output
python preprocess_data.py --verbose
```

### Default Behavior

- **Input**: `../../data/unprocessed/covidence_export_2nd_search_20250921.csv`
- **Output**: `../../data/intermediate/{input_filename}_intermediate.csv`

The output filename is automatically generated based on the input filename with "_intermediate" suffix.

## Current Functionality

### Step 1: ID Management (`assign_unique_ids`)

The first preprocessing step handles study identification:

1. **Assigns unique study IDs**: Creates a sequential `study_id` for each row (1, 2, 3, ...)
2. **Removes Study ID column**: Deletes the original `Study ID` column from the dataset
3. **Renames Covidence column**: Changes `Covidence #` to `covidence_id`
4. **Reorders columns**: Places `study_id` and `covidence_id` at the beginning

### Step 2: Consensus Filtering (`filter_consensus_reviews`)

The second preprocessing step filters for consensus reviews:

1. **Identifies consensus availability**: Analyzes which studies have consensus reviews
2. **Provides detailed warnings**: Shows exactly which studies lack consensus reviews
3. **Interactive fallback option**: Prompts user whether to use "Richard Gaus" reviews as fallback
4. **Smart filtering**: 
   - Keeps only "Consensus" reviews when available
   - Uses "Richard Gaus" reviews for studies without consensus (if user agrees)
   - Excludes studies entirely if no consensus and user declines fallback

#### Consensus Analysis Results

From the current dataset:
- **104 total unique studies**
- **91 studies with consensus reviews** ✅
- **13 studies without consensus reviews** ⚠️
- **11 studies with Richard Gaus fallback available**
- **2 studies with no fallback available** (Covidence IDs: 118, 152)

#### Final Dataset Options

- **With fallback (y)**: 102 rows from 102 studies (91 Consensus + 11 Richard Gaus)
- **Without fallback (n)**: 91 rows from 91 studies (91 Consensus only)

### Data Summary

The initial dataset contains:
- **291 total rows** (representing individual reviewer assessments)
- **104 unique Covidence IDs** (representing unique studies)
- **110 columns** of research data

The final processed dataset varies based on consensus filtering choice.

## Module Design

The pipeline follows a strictly modular design:

- **`data_preprocessing/`**: Contains all core preprocessing logic
- **`runnable/`**: Contains lightweight scripts that orchestrate the preprocessing modules
- **Separation of concerns**: Each module handles a specific aspect of data cleaning
- **Reusable functions**: All functions are designed to be imported and reused

## Future Extensions

Additional preprocessing modules can be added to `data_preprocessing/` and integrated into the main runnable script following the same modular pattern.
