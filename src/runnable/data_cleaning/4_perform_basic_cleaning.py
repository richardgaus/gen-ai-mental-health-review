#!/usr/bin/env python3
"""
Basic data cleaning script for LLMs in Psychotherapy research.

This script performs basic cleaning operations on CSV files including:
- Removing studies without consensus reviewer (all rows for studies with no consensus reviewer)
- Unicode normalization of column names
- Column renaming according to configuration
- Column reordering according to configuration
- Column filtering (dropping columns not in configuration)
- Cleaning 'Other: ' fields
- Setting covidence_id as index
- Filling specified fields with "n" if empty
- Lowercasing specified fields

Usage:
    python 4_perform_basic_cleaning.py --input PATH [--verbose]

Example:
    python 4_perform_basic_cleaning.py --input data/intermediate/fused_data.csv --verbose
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from utils.data_cleaning.basic_cleaning_utils import perform_basic_cleaning_pipeline
from utils.data_cleaning.report_generator import generate_processing_report, save_report_to_file, get_dataframe_summary, compare_dataframes


def remove_studies_without_consensus(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove all rows for studies that do not have a consensus reviewer.
    
    A study is identified by its covidence_id. If a study has no row where
    Reviewer Name == 'Consensus', then ALL rows for that study are removed.
    
    Args:
        df (pd.DataFrame): Input dataframe with covidence_id and Reviewer Name columns
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Filtered dataframe and processing details dictionary
    """
    # Check for required columns
    if 'covidence_id' not in df.columns:
        raise KeyError("Column 'covidence_id' not found in dataframe")
    
    # Handle both possible column names (before and after renaming)
    reviewer_col = None
    if 'Reviewer Name' in df.columns:
        reviewer_col = 'Reviewer Name'
    elif 'reviewer_name' in df.columns:
        reviewer_col = 'reviewer_name'
    else:
        raise KeyError("Neither 'Reviewer Name' nor 'reviewer_name' column found in dataframe")
    
    # Store original counts
    original_rows = len(df)
    original_studies = df['covidence_id'].nunique()
    
    # Find studies that have at least one consensus reviewer row
    studies_with_consensus = df[df[reviewer_col] == 'Consensus']['covidence_id'].unique()
    
    # Filter to keep only rows from studies with consensus
    df_filtered = df[df['covidence_id'].isin(studies_with_consensus)].copy()
    
    # Calculate what was removed
    final_rows = len(df_filtered)
    final_studies = df_filtered['covidence_id'].nunique()
    rows_removed = original_rows - final_rows
    studies_removed = original_studies - final_studies
    
    if verbose:
        print(f"✓ Removed studies without consensus reviewer:")
        print(f"  - Original studies: {original_studies:,}")
        print(f"  - Studies with consensus: {final_studies:,}")
        print(f"  - Studies removed: {studies_removed:,}")
        print(f"  - Original rows: {original_rows:,}")
        print(f"  - Final rows: {final_rows:,}")
        print(f"  - Rows removed: {rows_removed:,}")
    
    processing_details = {
        'original_rows': original_rows,
        'original_studies': original_studies,
        'final_rows': final_rows,
        'final_studies': final_studies,
        'rows_removed': rows_removed,
        'studies_removed': studies_removed
    }
    
    return df_filtered, processing_details


def main():
    """Main function to run the basic cleaning pipeline."""
    parser = argparse.ArgumentParser(
        description="Perform basic cleaning operations on CSV data"
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
    output_filename = f"{input_stem}_cleaned.csv"
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
        
        df = pd.read_csv(input_path)
        df_original = df.copy()
        
        if args.verbose:
            print(f"Loaded {len(df):,} rows with {len(df.columns):,} columns")
        
        # Remove studies without consensus reviewer
        if args.verbose:
            print("Removing studies without consensus reviewer...")
        
        df_before_consensus_filter = df.copy()
        df, studies_removed_no_consensus = remove_studies_without_consensus(df, verbose=args.verbose)
        
        if args.verbose:
            print(f"Removed {studies_removed_no_consensus['studies_removed']} studies without consensus reviewer")
            print(f"Rows removed: {studies_removed_no_consensus['rows_removed']}")
        
        # Perform the complete cleaning pipeline and capture processing details
        df_final, processing_details = perform_basic_cleaning_pipeline(df, verbose=args.verbose, return_details=True)
        
        # Save the cleaned data
        if args.verbose:
            print(f"\nSaving cleaned data to {output_path}...")
        
        df_final.to_csv(output_path)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate processing report
        if args.verbose:
            print(f"\nGenerating processing report...")
        
        # Calculate study counts
        original_studies = df_original['covidence_id'].nunique() if 'covidence_id' in df_original.columns else len(df_original)
        final_studies = df_final.index.nunique() if hasattr(df_final.index, 'nunique') else len(df_final)
        
        # Create detailed processing steps with specific information
        consensus_filter_text = f"Removed {studies_removed_no_consensus['studies_removed']} studies that have no consensus reviewer (removed all {studies_removed_no_consensus['rows_removed']} rows for these studies). Studies with consensus: {studies_removed_no_consensus['final_studies']} (down from {studies_removed_no_consensus['original_studies']})"
        
        columns_dropped_text = f"Dropped {len(processing_details['columns_dropped'])} columns not specified in configuration: {', '.join(processing_details['columns_dropped'])}" if processing_details['columns_dropped'] else "No columns were dropped (all columns were in configuration)"
        
        columns_filled_text = f"Filled specified fields with 'n' if empty (reviewers usually mean 'no' if they don't fill a field). Columns processed: {', '.join(processing_details['columns_filled_with_n'])}" if processing_details['columns_filled_with_n'] else "No columns needed to be filled with 'n'"
        
        columns_lowercased_text = f"Converted specified fields to lowercase for consistency. Columns processed: {', '.join(processing_details['columns_lowercased'])}" if processing_details['columns_lowercased'] else "No columns needed case normalization"
        
        processing_steps = {
            "step_1_data_loading": f"Loaded fused dataset with {len(df_original)} rows and {len(df_original.columns)} columns",
            "step_2_consensus_filtering": consensus_filter_text,
            "step_3_unicode_normalization": "Normalized Unicode characters in column names to ensure consistent encoding",
            "step_4_column_renaming": "Renamed columns according to configuration mapping (e.g., standardized field names)",
            "step_5_column_reordering": "Reordered columns according to configuration to match expected schema",
            "step_6_column_filtering": columns_dropped_text,
            "step_7_other_field_cleaning": "Cleaned 'Other: ' fields by setting them to empty strings for consistency",
            "step_8_index_setting": "Set covidence_id as index column for efficient data access",
            "step_9_empty_value_filling": columns_filled_text,
            "step_10_case_normalization": f"{columns_lowercased_text}. Final dataset: {final_studies} studies with {len(df_final)} rows and {len(df_final.columns)} columns"
        }
        
        # Create simplified summary statistics
        summary_stats = {
            "input_studies": original_studies,
            "input_rows": len(df_original),
            "input_columns": len(df_original.columns),
            "output_studies": final_studies,
            "output_rows": len(df_final),
            "output_columns": len(df_final.columns),
            "studies_removed": studies_removed_no_consensus['studies_removed'],
            "rows_removed": studies_removed_no_consensus['rows_removed'],
            "columns_removed": len(df_original.columns) - len(df_final.columns),
            "index_column_set": df_final.index.name,
            "execution_time_seconds": execution_time
        }
        
        report = generate_processing_report(
            script_name="4_perform_basic_cleaning.py",
            input_file=input_path,
            output_file=output_path,
            processing_steps=processing_steps,
            summary_stats=summary_stats,
            execution_time=execution_time
        )
        
        # Save report
        reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
        report_filename = "4_perform_basic_cleaning"
        report_path = save_report_to_file(report, str(reports_dir), report_filename, save_json=False)
        
        if args.verbose:
            print(f"✓ Successfully saved cleaned data to {output_path}")
            print(f"✓ Processing report saved to {report_path}")
            print(f"✓ Basic cleaning complete!")
            print(f"\nFinal dataset: {len(df_final):,} rows with {len(df_final.columns):,} columns")
            print(f"Index: {df_final.index.name} ({len(df_final.index):,} entries)")
            print(f"Execution time: {execution_time:.2f} seconds")
        else:
            print(f"Basic cleaning complete. Output saved to: {output_path}")
            print(f"Processing report saved to: {report_path}")
            
    except Exception as e:
        print(f"Error during cleaning: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()