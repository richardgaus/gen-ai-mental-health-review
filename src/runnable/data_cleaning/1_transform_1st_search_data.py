#!/usr/bin/env python3
"""
1st search data transformation script.

This script transforms the 1st search data to match the structure of the 2nd search data.
This is a standalone transformation step, separate from any data fusion operations.

Usage:
    python transform_1st_search_data.py --input PATH [--output PATH] [--verbose]

Example:
    python transform_1st_search_data.py --input data/unprocessed/covidence_export_1st_search_20250921.csv --verbose
    python transform_1st_search_data.py --input data/my_data.csv --output data/transformed_data.csv
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from utils.data_cleaning.transform_1st_search_data import transform_first_search_data, get_transformation_summary, validate_transformation
from utils.data_cleaning.id_management import assign_unique_ids, get_unique_studies_summary
from utils.data_cleaning.report_generator import generate_processing_report, save_report_to_file, get_dataframe_summary, compare_dataframes


def main():
    """Main function to run the 1st search data transformation pipeline."""
    parser = argparse.ArgumentParser(
        description="Transform 1st search data to match 2nd search structure"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the 1st search CSV file to transform (relative to current working directory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path for the output transformed CSV file (relative to current working directory, optional)"
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
            input_stem = input_path.stem
            output_filename = f"{input_stem}_transformed_for_fusion.csv"
            output_path = output_path / output_filename
    else:
        # Generate output filename based on input file
        input_stem = input_path.stem
        output_filename = f"{input_stem}_transformed_for_fusion.csv"
        output_path = input_path.parent / output_filename
    
    if args.verbose:
        print(f"Input file (1st search): {input_path}")
        print(f"Output file (transformed): {output_path}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load the 1st search data
        if args.verbose:
            print("Loading 1st search data...")
        
        if not input_path.exists():
            raise FileNotFoundError(f"1st search file not found: {input_path}")
        
        df_first_search = pd.read_csv(input_path)
        
        if args.verbose:
            print(f"Loaded {len(df_first_search):,} rows with {len(df_first_search.columns):,} columns")
        
        # Store original for comparison
        df_original = df_first_search.copy()
        
        # Get initial summary
        if args.verbose:
            try:
                total_rows, unique_studies = get_unique_studies_summary(df_first_search)
                print(f"Initial data: {total_rows} total rows, {unique_studies} unique Covidence IDs")
            except KeyError as e:
                print(f"Warning: Could not generate initial summary: {e}")
        
        # Step 1: Assign unique IDs and clean ID columns
        if args.verbose:
            print("\nAssigning unique study IDs and cleaning ID columns...")
        
        df_with_ids = assign_unique_ids(df_first_search)
        
        if args.verbose:
            print(f"✓ Assigned unique study IDs to {len(df_with_ids)} rows")
            print(f"✓ Renamed 'Covidence #' to 'covidence_id'")
            print(f"✓ Removed 'Study ID' column")
        
        # Step 2: Transform 1st search data to match 2nd search structure
        if args.verbose:
            print("\nTransforming 1st search data to match 2nd search structure...")
        
        df_transformed, dropped_rows_info = transform_first_search_data(df_with_ids)
        
        if args.verbose:
            # Get transformation summary
            summary = get_transformation_summary(df_original, df_transformed)
            print(f"✓ Transformation completed:")
            print(f"  - Original rows: {summary['original_rows']:,}")
            print(f"  - Final rows: {summary['final_rows']:,}")
            print(f"  - Rows dropped: {summary['rows_dropped']:,}")
            print(f"  - Original columns: {summary['original_columns']:,}")
            print(f"  - Final columns: {summary['final_columns']:,}")
            print(f"  - Columns added: {summary['columns_added']:,}")
            print(f"  - Columns renamed: {summary['columns_renamed']:,}")
            print(f"  - Values transformed: {summary['values_transformed']:,}")
            
            if summary['transformation_details']:
                print(f"  - Value transformation details:")
                for transformation, count in summary['transformation_details'].items():
                    print(f"    • {transformation}: {count:,} cases")
        
        # Validate transformation
        if args.verbose:
            print("\nValidating transformation...")
            
        validation = validate_transformation(df_transformed)
        
        if args.verbose:
            print(f"✓ Validation results:")
            print(f"  - New column created: {validation['has_new_column']}")
            if validation['has_new_column']:
                stats = validation['column_stats']
                print(f"  - Column statistics:")
                print(f"    • Total values: {stats['total_values']:,}")
                print(f"    • Non-null values: {stats['non_null_values']:,}")
                print(f"    • Null values: {stats['null_values']:,}")
                print(f"    • Unique values: {stats['unique_values']:,}")
                
                if validation['sample_values']:
                    print(f"  - Sample values:")
                    for value in validation['sample_values']:
                        print(f"    • {value}")
        
        # Save the transformed data
        if args.verbose:
            print(f"\nSaving transformed data to {output_path}...")
        
        df_transformed.to_csv(output_path, index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate processing report
        if args.verbose:
            print(f"\nGenerating processing report...")
        
        # Calculate study counts (unique covidence_id)
        # df_original doesn't have covidence_id yet, so we need to use df_with_ids for accurate study counts
        original_studies = df_with_ids['covidence_id'].nunique()
        transformed_studies = df_transformed['covidence_id'].nunique()
        
        # Calculate how many studies were actually filtered out
        # Get the unique studies that were in df_with_ids but not in df_transformed
        original_study_ids = set(df_with_ids['covidence_id'].unique())
        transformed_study_ids = set(df_transformed['covidence_id'].unique())
        filtered_study_ids = original_study_ids - transformed_study_ids
        studies_filtered = len(filtered_study_ids)
        
        # Create detailed description of dropped rows
        dropped_rows_description = ""
        if dropped_rows_info["dropped_studies"]:
            dropped_rows_description = f" DROPPED STUDIES: "
            for i, study in enumerate(dropped_rows_info["dropped_studies"]):
                if i > 0:
                    dropped_rows_description += "; "
                dropped_rows_description += f"Study {study['covidence_id']} ('{study['title'][:50]}{'...' if len(study['title']) > 50 else ''}') - Consensus marked as '{study['consensus_study_type']}' - {study['total_rows_dropped']} rows removed"
        else:
            dropped_rows_description = " No studies were dropped."
        
        processing_steps = {
            "step_1_id_assignment": "Assigned unique study IDs and cleaned ID columns (renamed 'Covidence #' to 'covidence_id', removed 'Study ID' column)",
            "step_2_drop_study_type_ii": "Dropped the 'Study Type II' column as it's not needed in the final dataset",
            "step_3_transform_study_type": f"Transformed Study Type values: 'Tool Development and Evaluation' and 'Direct LLM performance evaluation' → 'Empirical research involving an LLM'. Filtered out {studies_filtered} studies where the Consensus reviewer marked them as non-empirical types: 'Opinion, commentary, perspective, correspondence', 'Conceptual or theoretical work (e.g. on ethics or safety)', or 'Review (systematic or other)' from {original_studies} total studies, keeping {transformed_studies} studies. Individual reviewer assessments (Richard Gaus, Reviewer Two) do not trigger study removal - only Consensus decisions. This removed {len(df_original) - len(df_transformed)} rows (including all reviewer entries for filtered studies).{dropped_rows_description}",
            "step_4_transform_development_approach": "Renamed Development Approach column from 'If Study Type == Tool Development and Evaluation: Development Approach' to 'If Study Type == Empirical research involving an LLM: Development Approach' to match 2nd search format",
            "step_5_rename_application_type": "Renamed 'If Experimental Research or Population Survey: Application Type' to 'Application Type' for consistency",
            "step_6_drop_unwanted_columns": "Removed 4 columns: 'Main Safety-Related Discussion Contents', 'P-1 Protection of user information Considered in tool design? (Y/N)', 'P-1 Protection of user information if YES: Notes (paste text passage)', 'Safety-Related Discussion'",
            "step_7_add_missing_columns": "Added 8 new empty columns that exist in 2nd search data: 'Perplexity Used (Y/N)', 'Perplexity How it compares against benchmark (B/S/W)', 'Perplexity Benchmark quality (H/L)', 'Perplexity Notes on benchmark quality', 'Lexical diversity Used (Y/N)', 'Lexical diversity How it compares against benchmark (B/S/W)', 'Lexical diversity Benchmark quality (H/L)', 'Lexical diversity Notes on benchmark quality'",
            "step_8_validation": "Validated transformation results to ensure data integrity and proper column structure"
        }
        
        # Create simplified summary statistics
        summary_stats = {
            "input_studies": original_studies,
            "input_rows": len(df_original),
            "input_columns": len(df_original.columns),
            "output_studies": transformed_studies,
            "output_rows": len(df_transformed),
            "output_columns": len(df_transformed.columns),
            "studies_removed": studies_filtered,
            "rows_removed": len(df_original) - len(df_transformed),
            "columns_added": len(df_transformed.columns) - len(df_original.columns),
            "execution_time_seconds": execution_time
        }
        
        report = generate_processing_report(
            script_name="1_transform_1st_search_data.py",
            input_file=input_path,
            output_file=output_path,
            processing_steps=processing_steps,
            summary_stats=summary_stats,
            execution_time=execution_time
        )
        
        # Save report
        reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
        report_filename = "1_transform_1st_search_data"
        report_path = save_report_to_file(report, str(reports_dir), report_filename, save_json=False)
        
        if args.verbose:
            print(f"✓ Successfully saved transformed data to {output_path}")
            print(f"✓ Processing report saved to {report_path}")
            print(f"✓ Transformation complete!")
            print(f"\nTransformed dataset: {len(df_transformed):,} rows with {len(df_transformed.columns):,} columns")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"\nNote: This transformed file is ready for data fusion or further processing.")
        else:
            print(f"1st search data transformation complete. Output saved to: {output_path}")
            print(f"Processing report saved to: {report_path}")
            
    except Exception as e:
        print(f"Error during transformation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
