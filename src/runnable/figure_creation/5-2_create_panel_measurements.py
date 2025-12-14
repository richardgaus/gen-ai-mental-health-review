#!/usr/bin/env python3
"""
Create Panel: Measurements Metrics

This script creates visualizations and metrics for measurement characteristics
in the LLMs in psychotherapy study. It generates a horizontal bar chart
showing the distribution of metric supercategories by article count and produces a comprehensive report.

Usage:
    python 5-2_create_panel_measurements.py --input path/to/measurements.csv --final-data path/to/final_data.csv

Arguments:
    --input: Path to CSV file containing measurements information
    --final-data: Path to final_data.csv file (to calculate total studies without participants)

Outputs:
    - Metric supercategories distribution chart (SVG)
    - Comprehensive measurements metrics report (TXT)
    - All files saved to results/5-2_measurements/
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.figure_creation.panel_measurements import create_measurements_metrics_panel


def main():
    """Main function to create measurements metrics panel."""
    parser = argparse.ArgumentParser(
        description="Create measurements metrics panel with charts and reports"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV file containing measurements information"
    )
    parser.add_argument(
        "--final-data",
        required=True,
        help="Path to final_data.csv file (to calculate total studies without participants)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Measurements file '{args.input}' not found.")
        sys.exit(1)
    
    # Validate final_data file exists
    if not os.path.exists(args.final_data):
        print(f"Error: Final data file '{args.final_data}' not found.")
        sys.exit(1)
    
    # Set output directory - go up to project root from src_dir
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(
        project_root,
        "results",
        "5-2_measurements"
    )
    
    print(f"Creating measurements metrics panel...")
    print(f"Input file: {args.input}")
    print(f"Final data file: {args.final_data}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load final_data.csv to calculate total studies without participants
        print(f"\nCalculating total studies without human participants...")
        final_df = pd.read_csv(args.final_data)
        
        # Count studies where client_type is "No clients/patients involved" or empty/null
        total_studies_without_participants = len(final_df[
            (final_df['client_type'] == 'No clients/patients involved') | 
            (final_df['client_type'].isna()) | 
            (final_df['client_type'].astype(str).str.strip() == '')
        ])
        
        print(f"Total studies without human participants: {total_studies_without_participants}")
        
        # Create the panel
        results = create_measurements_metrics_panel(
            args.input, 
            output_dir,
            total_studies_without_participants
        )
        
        # Print summary
        print("\nMeasurements Metrics Panel Created Successfully!")
        print("=" * 50)
        print(f"Total studies without human participants: {results['total_studies_without_participants']}")
        print(f"Articles with measurements: {results['unique_articles']}")
        print(f"Articles without measurements: {results['articles_without_measurements']}")
        print(f"Total measurements analyzed: {results['total_measurements']}")
        print(f"Unique metric supercategories: {results['unique_supercategories']}")
        if results['most_common_supercategory']:
            print(f"Most common supercategory: {results['most_common_supercategory']}")
        
        print(f"\nFiles created:")
        print(f"  Metric Supercategories Chart: {results['chart_path']}")
        print(f"  Benchmark Quality Chart: {results['benchmark_chart_path']}")
        print(f"  Benchmark Quality Stacked Chart: {results['benchmark_stacked_chart_path']}")
        print(f"  Performance vs High Quality Benchmarks Chart: {results['performance_high_chart_path']}")
        print(f"  Performance vs Low Quality Benchmarks Chart: {results['performance_low_chart_path']}")
        print(f"  Performance Combined Chart: {results['performance_combined_chart_path']}")
        print(f"  Report: {results['report_path']}")
        
    except Exception as e:
        print(f"Error creating measurements metrics panel: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
