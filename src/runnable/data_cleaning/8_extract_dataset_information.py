#!/usr/bin/env python3
"""
Extract dataset information from large CSV and merge with small dataset file.

This script extracts dataset-related columns from a large CSV file, adds a reference column,
and merges the result with a provided smaller dataset file.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data_cleaning.dataset_extraction_utils import extract_and_merge_datasets, generate_report


def main():
    """Main function to extract and merge dataset information."""
    parser = argparse.ArgumentParser(
        description="Extract dataset information and merge with another dataset"
    )
    parser.add_argument(
        "--input-large",
        required=True,
        help="Relative path to the large CSV file containing dataset information"
    )
    parser.add_argument(
        "--input-small", 
        required=True,
        help="Relative path to the small CSV file with existing dataset information"
    )
    parser.add_argument(
        "--output",
        help="Output folder path where the merged dataset will be saved (optional, defaults to data/processed/)"
    )
    parser.add_argument(
        "--reference-column",
        default="title",
        help="Column name to use as reference from large file (default: title)"
    )
    
    args = parser.parse_args()
    
    try:
        # Convert relative paths to absolute paths
        project_root = Path(__file__).parent.parent.parent.parent
        input_large_path = project_root / args.input_large
        input_small_path = project_root / args.input_small
        
        # Generate output path
        if args.output:
            output_folder = project_root / args.output
        else:
            # Default to data/processed/ folder
            output_folder = project_root / "data" / "processed"
        
        # Create output directory if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_path = output_folder / "datasets_file.csv"
        
        print(f"Input large file: {input_large_path}")
        print(f"Input small file: {input_small_path}")
        print(f"Output file: {output_path}")
        print(f"Reference column: {args.reference_column}")
        
        # Perform the extraction and merging
        result_path = extract_and_merge_datasets(
            str(input_large_path),
            str(input_small_path), 
            str(output_path),
            args.reference_column
        )
        
        # Generate and save report
        final_df = pd.read_csv(result_path, index_col=0)  # Read with dataset_id as index
        report = generate_report(final_df, result_path)
        
        # Save report in reports_data_processing folder
        reports_folder = project_root / "results" / "reports_data_processing"
        reports_folder.mkdir(parents=True, exist_ok=True)
        report_path = reports_folder / "8_extract_dataset_information.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        print(f"Successfully extracted and merged dataset information to: {result_path}")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during dataset extraction and merging: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
