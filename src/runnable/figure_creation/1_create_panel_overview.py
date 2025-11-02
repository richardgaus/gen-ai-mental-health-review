#!/usr/bin/env python3
"""
Script to create overview panel and analysis of LLM applications in psychotherapy research.

This script generates comprehensive reports and visualizations showing the distribution
of application types and their subtypes, along with publication trends by outlet field
categories over time.

Usage:
    python 1_create_panel_overview.py --input data/processed/final_data.csv
    python 1_create_panel_overview.py --input data/processed/final_data.csv --exclude-population-surveys

Options:
    --exclude-population-surveys    Exclude studies with study_type == "Population survey"
                                   from the analysis (focuses on empirical LLM research)

Requirements:
    - Input CSV must contain application_type, subtype, outlet_field, and year columns
    - Output reports and figures will be saved to the results/1_overview/ directory

Outputs:
    - 1_overview_report.txt: Comprehensive text report with application type analysis
    - outlet_field_time_series.svg: Time series chart of publications by outlet field
"""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path for imports
script_dir = Path(__file__).parent.absolute()
src_dir = script_dir.parent.parent
sys.path.insert(0, str(src_dir))

import pandas as pd
from utils.figure_creation.panel_overview import generate_overview_report, create_outlet_field_time_series, get_outlet_field_statistics


def main():
    """Main function to parse arguments and create the overview analysis."""
    
    parser = argparse.ArgumentParser(
        description="Create overview analysis of LLM applications in psychotherapy research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Relative path to input CSV file containing the data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/1_overview",
        help="Directory to save output reports (default: results/1_overview)"
    )
    
    parser.add_argument(
        "--exclude-population-surveys",
        action="store_true",
        help="Exclude studies with study_type == 'Population survey'"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Convert input path to absolute path relative to project root
    project_root = script_dir.parent.parent.parent
    input_path = project_root / args.input
    
    # Validate input file exists
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Reading data from: {input_path}")
        print(f"Output directory: {output_dir}")
    
    try:
        # Read the data
        df = pd.read_csv(input_path)
        initial_count = len(df)
        if args.verbose:
            print(f"Loaded {initial_count} rows from input file")
        
        # Apply population survey filter if requested
        if args.exclude_population_surveys:
            df = df[df['study_type'] != 'Population survey'].copy()
            filtered_count = len(df)
            if args.verbose:
                print(f"Excluded population surveys: {filtered_count} studies remaining (from {initial_count} total)")
            else:
                print(f"Excluded population surveys: analyzing {filtered_count} studies (from {initial_count} total)")
        else:
            filtered_count = initial_count
        
        # Define output paths
        report_path = output_dir / "1_overview_report.txt"
        time_series_path = output_dir / "outlet_field_time_series.svg"
        
        # Generate and save the report
        if args.verbose:
            print(f"\nGenerating overview report...")
        
        report_content = generate_overview_report(
            df, 
            excluded_population_surveys=args.exclude_population_surveys,
            original_count=initial_count if args.exclude_population_surveys else None
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if args.verbose:
            print(f"  - Report saved: {report_path}")
        
        # Create the outlet field time series chart
        if args.verbose:
            print(f"\nCreating outlet field time series chart...")
        
        fig = create_outlet_field_time_series(
            df=df,
            title="Publication numbers by field of publication venue over time",
            xlabel="Year",
            ylabel="Number of articles",
            figsize=(8, 4),
            save_path=str(time_series_path)
        )
        
        if args.verbose:
            print(f"  - Time series chart saved: {time_series_path}")
        
        print(f"\n✓ Successfully created overview analysis")
        if args.exclude_population_surveys:
            print(f"✓ Analyzed {len(df)} studies (excluded {initial_count - len(df)} population surveys)")
        else:
            print(f"✓ Analyzed {len(df)} total studies")
        print(f"✓ Report and figure saved to: {output_dir}")
        
        # Print summary statistics
        app_type_counts = df['application_type'].value_counts()
        print(f"\nApplication Type Summary:")
        for app_type, count in app_type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {app_type}: {count} studies ({percentage:.1f}%)")
        
        # Print outlet field category statistics
        outlet_stats = get_outlet_field_statistics(df)
        print(f"\nOutlet Field Category Summary:")
        for category, count in outlet_stats['category_distribution'].items():
            percentage = (count / len(df)) * 100
            print(f"  - {category}: {count} studies ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error processing data: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
