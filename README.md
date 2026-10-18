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
│   │   ├── data_cleaning/           # Data preprocessing pipeline (steps 1-10)
│   │   ├── design_purpose/          # Companion search and agreement for the design purpose classification
│   │   ├── figure_creation/         # Visualization scripts
│   │   └── fulltext_exclusions/     # Reasons for the full-text exclusions (Figure 1)
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
- `runnable/data_cleaning/` contains numbered scripts (1-10) that form the data preprocessing pipeline
- `runnable/design_purpose/` contains the scripts for the design purpose classification added during peer review (see below)
- `runnable/figure_creation/` contains scripts for generating publication figures
- `runnable/fulltext_exclusions/` contains the scripts for the reasons for the full-text exclusions in Figure 1, added during peer review (see below)
- `utils/` provides helper functions used by the runnable scripts

## Data

### Input Data (`data/unprocessed/`)
- `covidence_export_1st_search_*.csv` - Extraction data from the first literature search
- `covidence_export_2nd_search_*.csv` - Extraction data from the second (updated) literature search
- `participation_mode_human_participants.xlsx` - Participation mode of the articles with human participants (added during peer review)
- `covidence_export_design_purpose_20261016.csv` - Dual coding of design purpose for 44 articles (added during peer review)
- `companion_screen_coding.xlsx` - Verification of the full-text search for companion applications (added during peer review)
- `fulltext_exclusion_reasons_assigned_20261018.xlsx` - Full-text exclusion reasons assigned during peer review for records without a recorded reason (added during peer review)

### Processed Data (`data/processed/`)
- `final_data.csv` - Main analysis dataset with all extracted variables per study
- `datasets_file.csv` - Extracted information about datasets used in the reviewed studies
- `measurements_file.csv` - Extracted information about measurement outcomes and evaluation metrics used

### Reference Data (`data/auxiliary/`)
- `common_datasets.csv` - Reference list of commonly used mental health datasets
- `companion_screen_terms.yaml` - Search terms for the full-text search for companion applications, with sources

## Usage

### Data Cleaning Pipeline

Run the full data preprocessing pipeline (steps 1-10):

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
10. Add classifications made during peer review (participation mode, design purpose)

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
7. Summary table of all included studies
8. Sensitivity analysis by participation mode (Supplementary Table S5)
9. READI-derived criteria by human participation (Results, Figure 4 paragraph)

## Classifications added during peer review

Two classifications were added after data charting, in response to peer review. Both are merged into `final_data.csv` in data cleaning step 10; all other columns are left unchanged.

**Participation mode** (Supplementary Notes S5.3; column `participation_mode`). For the 36 articles evaluating client-facing applications with human participants: whether participants used the application (`used`) or only rated LLM-generated outputs (`rated`). Classified by RG and verified by FG in `data/unprocessed/participation_mode_human_participants.xlsx`.

**Design purpose** (Supplementary Notes S5.2; columns `design_purpose` and `design_purpose_basis`). For the 114 articles evaluating client-facing applications, one of three mutually exclusive groups: `General-purpose only`, `Purpose-built` or `Companion`. Empty for the other application types. The classification combines three steps:

1. Articles whose charted development approach (`llm_development_approach`) was "Only fine-tuning", "Fine-tuning + other modules" or "Prompting + other modules" (n = 70) are purpose-built (`design_purpose_basis` = `development approach`).
2. The remaining articles ("Only prompting" or "Other", n = 44) were coded independently by two reviewers (RG, SM) in Covidence on three yes/no items (`design_purpose_basis` = `dual coding`). Input: `data/unprocessed/covidence_export_design_purpose_20261016.csv` (reviewers anonymized as Reviewer 1 and Reviewer 2, with consensus).
3. The full texts of all 114 articles were searched for companion applications (terms: `data/auxiliary/companion_screen_terms.yaml`; hits: `data/intermediate/companion_screen_hits.csv`), and every article with a hit was verified by two reviewers (`data/unprocessed/companion_screen_coding.xlsx`).

Articles evaluating a companion application are assigned to `Companion`, other articles evaluating a purpose-built application to `Purpose-built`, and the remaining articles to `General-purpose only`.

Scripts in `src/runnable/design_purpose/` (run from the repository root):

```bash
# Full-text search (requires a local copy of the full-text PDFs, which are not distributed for copyright reasons)
python src/runnable/design_purpose/1_search_companion_terms.py --pdf-dir PDF_FOLDER --pdf-index PDF_INDEX.csv

# Percentage agreement and Cohen's kappa for Steps 2 and 3
python src/runnable/design_purpose/2_calculate_design_purpose_agreement.py
```

The agreement report is saved to `results/reports_data_processing/design_purpose_agreement.txt`; the counts per group (overall and by human participants) are in `results/reports_data_processing/10_add_peer_review_classifications.txt`.

## Full-text exclusion reasons (Figure 1)

Figure 1 reports 258 full texts screened and 172 records retained, so 86 records were excluded at the full-text stage. The reasons shown in Figure 1 were added during peer review and are derived as follows:

1. Full-text exclusions recorded in Covidence (n = 83). The excluded records were later moved back to screening in Covidence, which removed the recorded reasons. Before that, the Covidence page "Excluded - Full text review" was saved as HTML (30 September 2025; 81 records); two further excluded records, not on the saved page, were exported separately (Covidence CSV export, 18 October 2026). Both files contain personal account data and are not distributed; script 1 extracts the records and their reasons into `data/intermediate/covidence_fulltext_excluded.csv`. SHA-256 of the inputs:
   - `Excluded - Full text review _ LLMs in Psychotherapy 2nd database search _ Covidence.html`: `640d54adffb9b1c2d0bcbb6ee0315b16de0a10b255803462baf8e7e51230cc4c`
   - `review_628555_excluded_csv_20261019033848.csv`: `746e1ea3c36139ac33fede9f4156301a7bdd56e7cd5cab2982a7df40bc39fad7`
2. Reasons assigned from the full texts on 18 October 2026 (RG; `data/unprocessed/fulltext_exclusion_reasons_assigned_20261018.xlsx`, n = 7): five Covidence exclusions without a recorded reason, one excluded record missing from both Covidence files, and one record without completed data charting (removed in data cleaning step 4).
3. One record charted in both Covidence reviews; the 2nd search copy was removed as a duplicate in data cleaning step 3.

Script 2 combines these sources, checks that every record has exactly one reason and that the total equals 258 - 172 = 86, and groups the Covidence reasons into the reasons shown in Figure 1.

Scripts in `src/runnable/fulltext_exclusions/` (run from the repository root):

```bash
# Extraction from Covidence (requires local copies of the saved page and the export, which are not distributed)
python src/runnable/fulltext_exclusions/1_extract_fulltext_exclusions.py --html SAVED_PAGE.html --export EXPORT.csv

# Exclusion reasons per record and the counts shown in Figure 1
python src/runnable/fulltext_exclusions/2_summarize_fulltext_exclusions.py
```

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
| `8_sensitivity_participation_mode/` | Sensitivity analysis by participation mode (Supplementary Table S5) |
| `9_readi_by_human_participation/` | READI-derived criteria for articles with and without human participants (Results, Figure 4 paragraph) |
| `fulltext_exclusions/` | Full-text exclusion reasons per record and per Figure 1 reason |
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
- Richard Gaus (richardgaus at outlook.de) - University Hospital LMU Munich / Stanford University
- Johannes C. Eichstaedt (eichstaedt at stanford.edu) - Stanford University
