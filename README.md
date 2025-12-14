# A Scoping Review of Generative AI in Mental Health Support

## Overview

This repository contains the code and data for the paper _A Scoping Review of Generative AI in Mental Health Support_.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/richardgaus/gen-ai-mental-health-review.git
cd gen-ai-mental-health-review
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate gen_ai_mh_review
```

## Project Structure

```
├── src/
│   ├── runnable/                    # Executable scripts
│   │   ├── data_cleaning/           # Data preprocessing pipeline (steps 1-9)
│   │   └── figure_creation/         # Visualization scripts
│   ├── utils/                       # Helper modules
│   │   ├── data_cleaning/           # Data processing utilities
│   │   ├── figure_creation/         # Plotting utilities
│   │   └── interrater_agreement/    # Kappa calculation utilities
│   └── settings.py                  # Project configuration
├── data/
│   ├── unprocessed/                 # Raw input data
│   ├── intermediate/                # Data between processing steps
│   ├── processed/                   # Final cleaned datasets
│   └── auxiliary/                   # Reference data (common datasets)
├── results/                         # Output files (figures, reports)
└── environment.yml                  # Conda environment specification
```

The `src/` module is organized into **runnable scripts** and their supporting **utility functions**:
- `runnable/data_cleaning/` contains numbered scripts (1-9) that form the data preprocessing pipeline
- `runnable/figure_creation/` contains scripts for generating publication figures
- `utils/` provides helper functions used by the runnable scripts

## Data

### Input Data (`data/unprocessed/`)
- `covidence_export_1st_search_*.csv` - Extraction data from the first literature search
- `covidence_export_2nd_search_*.csv` - Extraction data from the second (updated) literature search

### Processed Data (`data/processed/`)
- `final_data.csv` - Main analysis dataset with all extracted variables per study
- `datasets_file.csv` - Extracted information about datasets used in the reviewed studies
- `measurements_file.csv` - Extracted information about measurement outcomes and evaluation metrics used

### Reference Data (`data/auxiliary/`)
- `common_datasets.csv` - Reference list of commonly used mental health datasets

## Usage

### Data Cleaning Pipeline

Run the full data preprocessing pipeline (steps 1-9):

```bash
cd src/runnable/data_cleaning
./run_data_cleaning.sh
```

This executes:
1. Transform 1st search data
2. Preprocess 2nd search data
3. Fuse datasets
4. Perform basic cleaning (column renaming, etc.)
5. Remove non-generative studies
6. Calculate interrater agreement
7. Perform final cleaning (keep only consensus data, derive new variables, map values to categories)
8. Extract dataset information
9. Extract measurement information

### Figure Creation Pipeline

Generate all publication figures:

```bash
cd src/runnable/figure_creation
./run_figure_creation.sh
```

This creates figures for:
1. Overview panel (publication trends)
2. Clinical panel (study characteristics)
3. Model types panel
4. Datasets panel
5. Measurements panel
6. READI framework panel

## Results

Generated outputs are saved to `results/`:

| Folder | Contents |
|--------|----------|
| `1_overview/` | Publication trends over time |
| `2_panel_clinical/` | Study design, sample sizes, intervention types |
| `3_controlled_clinical_studies/` | List of controlled clinical trials |
| `4_model_types/` | LLM families, development approaches, open vs closed weights |
| `5-1_datasets/` | Dataset characteristics (type, language, availability) |
| `5-2_measurements/` | Evaluation metrics and benchmark quality |
| `6_readi/` | READI framework compliance |
| `reports_data_processing/` | Logs from each data cleaning step |

Additionally, `full_literature_table.csv` contains all study characteristics for the supplementary materials.

## Citation

If you use this code or data, please cite:

> Gaus, R., Gross, F., Korman, M., Klaassen, F., Maspero, S., Martignoni, L., Urquijo, M., Boger, S., Jebrini, T., Wolf, J., Hager, P., Stade, E.C., Terhorst, Y., Volkert, J., Kambeitz, J., Stubbe, H.C., Padberg, F., Wiltsey Stirman, S., Koutsouleris, N.\*, & Eichstaedt, J.C.\* (2025). A Scoping Review of Generative AI in Mental Health Support. *[Journal TBD]*.

## License

- **Code**: MIT License
- **Data**: CC BY 4.0 (Creative Commons Attribution 4.0 International)

## Contact

**Corresponding authors:**
- Richard Gaus (richardgaus@outlook.de) - University Hospital LMU Munich / Stanford University
- Johannes C. Eichstaedt (eichstaedt@stanford.edu) - Stanford University
