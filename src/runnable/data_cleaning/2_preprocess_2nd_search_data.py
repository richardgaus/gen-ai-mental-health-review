#!/usr/bin/env python3
"""
2nd search data preprocessing script for LLMs in Psychotherapy research.

This script processes the raw Covidence export data from the 2nd search and creates cleaned datasets
ready for analysis. It assigns unique IDs and performs basic preprocessing.

Usage:
    python 2_preprocess_2nd_search_data.py [--input FILENAME] [--output PATH] [--verbose]

Example:
    python 2_preprocess_2nd_search_data.py --input data/unprocessed/covidence_export_2nd_search_20250921.csv --verbose
    python 2_preprocess_2nd_search_data.py --input data/unprocessed/covidence_export_2nd_search_20250921.csv --output data/intermediate/
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from utils.data_cleaning.id_management import assign_unique_ids, get_unique_studies_summary
from settings import DATA_UNPROCESSED_DIR, DATA_INTERMEDIATE_DIR
from utils.data_cleaning.report_generator import generate_processing_report, save_report_to_file, get_dataframe_summary, compare_dataframes


def main():
    """Main function to run the data preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess Covidence export data for LLMs in Psychotherapy research"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="covidence_export_2nd_search_20250921.csv",
        help="Path to the input CSV file (relative to current working directory, default: covidence_export_2nd_search_20250921.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path for the output processed CSV file (relative to current working directory, optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    
    args = parser.parse_args()
    
    # Handle input path (resolve relative to current working directory)
    input_path = Path(args.input).resolve()
    
    # Handle output path
    if args.output:
        output_path = Path(args.output).resolve()
        # If output path is a directory, generate filename within that directory
        if output_path.is_dir() or str(output_path).endswith('/'):
            input_stem = Path(args.input).stem
            output_filename = f"{input_stem}_preprocessed.csv"
            output_path = output_path / output_filename
    else:
        # Generate output filename based on input filename
        input_stem = Path(args.input).stem  # Get filename without extension
        output_filename = f"{input_stem}_preprocessed.csv"
        output_path = DATA_INTERMEDIATE_DIR / output_filename
    
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
            print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Get initial summary
        if args.verbose:
            try:
                total_rows, unique_studies = get_unique_studies_summary(df)
                print(f"Initial data: {total_rows} total rows, {unique_studies} unique Covidence IDs")
            except KeyError as e:
                print(f"Warning: Could not generate initial summary: {e}")
        
        # Step 1: Assign unique IDs and clean ID columns
        if args.verbose:
            print("Assigning unique study IDs and cleaning ID columns...")
        
        df_processed = assign_unique_ids(df)
        
        if args.verbose:
            print(f"✓ Assigned unique study IDs to {len(df_processed)} rows")
            print(f"✓ Renamed 'Covidence #' to 'covidence_id'")
            print(f"✓ Removed 'Study ID' column")
        
        # No filtering - keep all data after ID management
        df_filtered = df_processed
        
        if args.verbose:
            print(f"\n✓ ID management complete - no filtering applied")
            print(f"  - Final rows: {len(df_filtered)}")
            print(f"  - Unique studies: {df_filtered['covidence_id'].nunique() if 'covidence_id' in df_filtered.columns else 'N/A'}")
        
        # Save the processed data
        if args.verbose:
            print(f"Saving processed data to {output_path}...")
        
        df_filtered.to_csv(output_path, index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate processing report
        if args.verbose:
            print(f"\nGenerating processing report...")
        
        # Calculate study counts (unique covidence_id)
        original_studies = df_original['covidence_id'].nunique() if 'covidence_id' in df_original.columns else len(df_original)
        processed_studies = df_filtered['covidence_id'].nunique() if 'covidence_id' in df_filtered.columns else len(df_filtered)
        
        processing_steps = {
            "step_1_data_loading": f"Loaded raw Covidence export data from 2nd search with {len(df_original)} rows and {len(df_original.columns)} columns",
            "step_2_id_assignment": "Assigned unique study IDs and cleaned ID columns (renamed 'Covidence #' to 'covidence_id', removed 'Study ID' column)",
            "step_3_data_retention": f"Retained all data after ID management - no filtering applied. Kept {processed_studies} studies with {len(df_filtered)} total rows (including all reviewer entries)"
        }
        
        # Create simplified summary statistics
        summary_stats = {
            "input_studies": original_studies,
            "input_rows": len(df_original),
            "input_columns": len(df_original.columns),
            "output_studies": processed_studies,
            "output_rows": len(df_filtered),
            "output_columns": len(df_filtered.columns),
            "studies_removed": 0,
            "rows_removed": 0,
            "columns_added": len(df_filtered.columns) - len(df_original.columns),
            "execution_time_seconds": execution_time
        }
        
        report = generate_processing_report(
            script_name="2_preprocess_2nd_search_data.py",
            input_file=input_path,
            output_file=output_path,
            processing_steps=processing_steps,
            summary_stats=summary_stats,
            execution_time=execution_time
        )
        
        # Save report
        reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
        report_filename = "2_preprocess_2nd_search_data"
        report_path = save_report_to_file(report, str(reports_dir), report_filename, save_json=False)
        
        if args.verbose:
            print(f"✓ Successfully saved processed data to {output_path}")
            print(f"✓ Processing report saved to {report_path}")
            print(f"✓ Processing complete!")
            print(f"Execution time: {execution_time:.2f} seconds")
        else:
            print(f"Data preprocessing complete. Output saved to: {output_path}")
            print(f"Processing report saved to: {report_path}")
            
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
