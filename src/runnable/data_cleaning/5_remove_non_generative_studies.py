#!/usr/bin/env python3
"""
Remove non-generative studies script for LLMs in Psychotherapy research.

This script processes CSV files to:
- Identify consensus rows (where reviewer_name == "Consensus") for each study
- Parse models from the models_employed column using transform_multi_choice_column_other
- Count generative and non-generative models ONLY from consensus rows using configuration dictionaries
- Add num_generative and num_non_generative columns to ALL rows based on consensus counts
- Remove ALL rows for studies where consensus shows num_generative == 0

Usage:
    python 5_remove_non_generative_studies.py --input PATH [--verbose]

Example:
    python 5_remove_non_generative_studies.py --input data/processed/final_processed_data.csv --verbose
"""

import argparse
import sys
import time
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import yaml
from utils.utils import transform_multi_choice_column_other
from utils.data_cleaning.report_generator import generate_processing_report, save_report_to_file, get_dataframe_summary, compare_dataframes


def load_model_dictionaries(config_path: str = None) -> tuple[list, list]:
    """
    Load generative and non-generative model lists from configuration file.
    
    Args:
        config_path (str): Path to configuration file. If None, uses default path.
        
    Returns:
        tuple[list, list]: (generative_models, non_generative_models)
    """
    if config_path is None:
        # Default path relative to src directory
        config_path = Path(__file__).parent.parent.parent / "utils" / "data_cleaning" / "final_cleaning_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        generative_models = config.get('generative_models', [])
        non_generative_models = config.get('non_generative_models', [])
        
        return generative_models, non_generative_models
        
    except Exception as e:
        print(f"Error: Could not load model configuration from {config_path}: {e}")
        sys.exit(1)


def parse_models_from_string(models_string: str) -> list[str]:
    """
    Parse models from a string using transform_multi_choice_column_other and split by semicolon.
    
    Args:
        models_string (str): String containing models information
        
    Returns:
        list[str]: List of individual model names
    """
    if pd.isna(models_string) or not models_string:
        return []
    
    # Transform the string to handle "Other:" format
    transformed = transform_multi_choice_column_other(str(models_string))
    
    # Split by semicolon and clean up
    models = [model.strip() for model in transformed.split(';') if model.strip()]
    
    return models


def count_model_types(models_list: list[str], generative_models: list, non_generative_models: list) -> tuple[int, int]:
    """
    Count the number of generative and non-generative models in a list.
    
    Args:
        models_list (list[str]): List of model names
        generative_models (list): List of generative model names from config
        non_generative_models (list): List of non-generative model names from config
        
    Returns:
        tuple[int, int]: (num_generative, num_non_generative)
    """
    num_generative = 0
    num_non_generative = 0
    
    for model in models_list:
        if model in generative_models:
            num_generative += 1
        elif model in non_generative_models:
            num_non_generative += 1
        # Models not in either list are ignored (could be unknown/unclassified)
    
    return num_generative, num_non_generative


def process_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Process the dataframe to add model counts and remove non-generative studies.
    Only counts models from consensus rows and removes all rows for studies 
    where consensus shows num_generative == 0.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        pd.DataFrame: Processed dataframe with generative studies only
    """
    required_columns = ['models_employed', 'reviewer_name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Required columns not found in dataframe: {missing_columns}")
    
    # Load model dictionaries
    if verbose:
        print("Loading model dictionaries from configuration...")
    
    generative_models, non_generative_models = load_model_dictionaries()
    
    if verbose:
        print(f"✓ Loaded {len(generative_models)} generative models and {len(non_generative_models)} non-generative models")
    
    # Process consensus rows to determine model counts per study
    if verbose:
        print("Processing consensus rows to determine model counts per study...")
    
    df_processed = df.copy()
    
    # Initialize new columns
    df_processed['num_generative'] = 0
    df_processed['num_non_generative'] = 0
    
    # Get consensus rows only for model counting
    consensus_rows = df_processed[df_processed['reviewer_name'] == 'Consensus'].copy()
    
    if verbose:
        print(f"✓ Found {len(consensus_rows)} consensus rows out of {len(df_processed)} total rows")
    
    # Create a dictionary to store model counts per study (covidence_id)
    study_model_counts = {}
    
    for idx, row in consensus_rows.iterrows():
        covidence_id = idx  # Assuming covidence_id is the index
        models_string = row['models_employed']
        
        # Parse models from the string
        models_list = parse_models_from_string(models_string)
        
        # Count generative and non-generative models
        num_gen, num_non_gen = count_model_types(models_list, generative_models, non_generative_models)
        
        # Store counts for this study
        study_model_counts[covidence_id] = {
            'num_generative': num_gen,
            'num_non_generative': num_non_gen
        }
    
    # Apply model counts to all rows based on their covidence_id
    for idx, row in df_processed.iterrows():
        covidence_id = idx  # Assuming covidence_id is the index
        
        if covidence_id in study_model_counts:
            df_processed.loc[idx, 'num_generative'] = study_model_counts[covidence_id]['num_generative']
            df_processed.loc[idx, 'num_non_generative'] = study_model_counts[covidence_id]['num_non_generative']
        else:
            # If no consensus row found for this study, set counts to 0
            df_processed.loc[idx, 'num_generative'] = 0
            df_processed.loc[idx, 'num_non_generative'] = 0
    
    if verbose:
        print(f"✓ Processed model counts for {len(study_model_counts)} studies based on consensus")
        print(f"  - Total generative model instances (from consensus): {sum(counts['num_generative'] for counts in study_model_counts.values())}")
        print(f"  - Total non-generative model instances (from consensus): {sum(counts['num_non_generative'] for counts in study_model_counts.values())}")
    
    # Identify studies to remove (where consensus shows num_generative == 0)
    studies_to_remove = [study_id for study_id, counts in study_model_counts.items() 
                        if counts['num_generative'] == 0]
    
    # Also remove studies that have no consensus row
    studies_without_consensus = set(df_processed.index) - set(study_model_counts.keys())
    studies_to_remove.extend(list(studies_without_consensus))
    
    if verbose and studies_without_consensus:
        print(f"⚠ Found {len(studies_without_consensus)} studies without consensus rows - these will be removed")
    
    # Remove all rows for studies that should be filtered out
    initial_count = len(df_processed)
    initial_studies = df_processed.index.nunique()
    
    df_filtered = df_processed[~df_processed.index.isin(studies_to_remove)].copy()
    
    removed_rows = initial_count - len(df_filtered)
    removed_studies = len(studies_to_remove)
    remaining_studies = df_filtered.index.nunique()
    
    if verbose:
        print(f"\nFiltering out studies with no generative models (based on consensus)...")
        print(f"✓ Removed {removed_studies} studies ({removed_rows} total rows)")
        print(f"  - Studies with num_generative == 0 (consensus): {len([s for s in studies_to_remove if s in study_model_counts and study_model_counts[s]['num_generative'] == 0])}")
        print(f"  - Studies without consensus rows: {len(studies_without_consensus)}")
        print(f"✓ Remaining studies: {remaining_studies} ({len(df_filtered)} total rows)")
    
    return df_filtered


def main():
    """Main function to run the non-generative studies removal pipeline."""
    parser = argparse.ArgumentParser(
        description="Remove studies with no generative models from CSV data"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file (relative to current working directory)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle input path (resolve relative to current working directory)
    input_path = Path(args.input).resolve()
    
    # Generate output filename
    input_stem = input_path.stem
    output_filename = f"{input_stem}_generative_only.csv"
    output_path = input_path.parent / output_filename
    
    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load the data
        if args.verbose:
            print("Loading data...")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path, index_col=0)  # Assuming first column is index (e.g., covidence_id)
        df_original = df.copy()
        
        if args.verbose:
            print(f"Loaded {len(df):,} rows with {len(df.columns):,} columns")
        
        # Process the dataframe
        df_final = process_dataframe(df, verbose=args.verbose)
        
        # Save the filtered data
        if args.verbose:
            print(f"\nSaving filtered data to {output_path}...")
        
        df_final.to_csv(output_path)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate processing report
        if args.verbose:
            print(f"\nGenerating processing report...")
        
        # Calculate study counts
        original_studies = df_original.index.nunique() if hasattr(df_original.index, 'nunique') else len(df_original)
        final_studies = df_final.index.nunique() if hasattr(df_final.index, 'nunique') else len(df_final)
        studies_removed = original_studies - final_studies
        
        # Get model configuration details
        generative_models, non_generative_models = load_model_dictionaries()
        
        # Get consensus-specific statistics for reporting
        consensus_rows = df_original[df_original['reviewer_name'] == 'Consensus']
        consensus_studies = consensus_rows.index.nunique() if hasattr(consensus_rows.index, 'nunique') else len(consensus_rows)
        studies_without_consensus = original_studies - consensus_studies
        
        processing_steps = {
            "step_1_data_loading": f"Loaded cleaned dataset with {len(df_original)} rows and {len(df_original.columns)} columns representing {original_studies} unique studies",
            "step_2_consensus_identification": f"Identified {len(consensus_rows)} consensus rows representing {consensus_studies} studies with consensus reviews. Found {studies_without_consensus} studies without consensus rows",
            "step_3_model_dictionary_loading": f"Loaded model configuration with {len(generative_models)} generative models and {len(non_generative_models)} non-generative models from final_cleaning_config.yaml",
            "step_4_consensus_model_parsing": "Parsed models from 'models_employed' column ONLY from consensus rows using transform_multi_choice_column_other function to handle 'Other:' format and semicolon separation",
            "step_5_consensus_model_counting": "Counted generative and non-generative models for each study based ONLY on consensus reviewer assessments, matching parsed models against configuration dictionaries",
            "step_6_column_addition": "Added 'num_generative' and 'num_non_generative' columns to ALL rows based on consensus-derived counts for each study (covidence_id)",
            "step_7_study_filtering": f"Removed {studies_removed} complete studies (all rows per covidence_id) where consensus showed num_generative == 0 or no consensus row existed. This removed studies that employ only non-generative models or have no consensus assessment. Final dataset: {final_studies} studies with {len(df_final)} rows"
        }
        
        # Create simplified summary statistics
        summary_stats = {
            "input_studies": original_studies,
            "input_rows": len(df_original),
            "input_columns": len(df_original.columns),
            "consensus_studies": consensus_studies,
            "consensus_rows": len(consensus_rows),
            "studies_without_consensus": studies_without_consensus,
            "output_studies": final_studies,
            "output_rows": len(df_final),
            "output_columns": len(df_final.columns),
            "studies_removed": studies_removed,
            "rows_removed": len(df_original) - len(df_final),
            "columns_added": 2,  # num_generative and num_non_generative
            "generative_models_in_config": len(generative_models),
            "non_generative_models_in_config": len(non_generative_models),
            "total_generative_model_instances_consensus": df_final['num_generative'].sum() if 'num_generative' in df_final.columns else 0,
            "total_non_generative_model_instances_consensus": df_final['num_non_generative'].sum() if 'num_non_generative' in df_final.columns else 0,
            "filtering_method": "consensus_based",
            "execution_time_seconds": execution_time
        }
        
        report = generate_processing_report(
            script_name="5_remove_non_generative_studies.py",
            input_file=input_path,
            output_file=output_path,
            processing_steps=processing_steps,
            summary_stats=summary_stats,
            execution_time=execution_time
        )
        
        # Save report
        reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
        report_filename = "5_remove_non_generative_studies"
        report_path = save_report_to_file(report, str(reports_dir), report_filename, save_json=False)
        
        if args.verbose:
            print(f"✓ Successfully saved filtered data to {output_path}")
            print(f"✓ Processing report saved to {report_path}")
            print(f"✓ Non-generative studies removal complete!")
            print(f"\nFinal dataset: {len(df_final):,} rows with {len(df_final.columns):,} columns")
            print(f"Execution time: {execution_time:.2f} seconds")
        else:
            print(f"Non-generative studies removal complete. Output saved to: {output_path}")
            print(f"Processing report saved to: {report_path}")
            
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
