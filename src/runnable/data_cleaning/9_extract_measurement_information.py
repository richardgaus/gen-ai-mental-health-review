#!/usr/bin/env python3
"""
Extract measurement information from CSV file.

This script extracts measurement/metric-related information from a CSV file,
creating a new measurements_file.csv with standardized measurement records.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.data_cleaning.measurement_extraction_utils import extract_measurements, generate_report


def main():
    """Main function to extract measurement information."""
    parser = argparse.ArgumentParser(
        description="Extract measurement information from a CSV file. "
                    "By default, excludes studies with human participants that are not empirical research."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Relative path to the input CSV file containing measurement information"
    )
    parser.add_argument(
        "--output",
        help="Output folder path where measurements_file.csv will be saved (optional, defaults to data/processed/)"
    )
    parser.add_argument(
        "--include-human-participant-studies",
        action="store_true",
        help="Include all studies with human participants (default: exclude non-empirical studies with participants)"
    )
    parser.add_argument(
        "--include-non-empirical-studies",
        action="store_true",
        help="Include all non-empirical studies (default: exclude them if they have participants)"
    )
    
    args = parser.parse_args()
    
    try:
        # Convert relative paths to absolute paths
        project_root = Path(__file__).parent.parent.parent.parent
        input_path = project_root / args.input
        
        # Generate output path
        if args.output:
            output_folder = project_root / args.output
        else:
            # Default to data/processed/ folder
            output_folder = project_root / "data" / "processed"
        
        # Create output directory if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        output_path = output_folder / "measurements_file.csv"
        
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        
        # Perform the extraction
        result_path = extract_measurements(
            str(input_path),
            str(output_path),
            include_human_participant_studies=args.include_human_participant_studies,
            include_non_empirical_studies=args.include_non_empirical_studies
        )
        
        # Generate and save report
        measurements_df = pd.read_csv(result_path)
        report = generate_report(measurements_df, result_path)
        
        # Save report in reports_data_processing folder
        reports_folder = project_root / "results" / "reports_data_processing"
        reports_folder.mkdir(parents=True, exist_ok=True)
        report_path = reports_folder / "9_extract_measurement_information.txt"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nSuccessfully extracted measurement information to: {result_path}")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error during measurement extraction: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

