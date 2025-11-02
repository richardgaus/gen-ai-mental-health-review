#!/usr/bin/env python3
"""
Create Panel: Measurements Metrics

This script creates visualizations and metrics for measurement characteristics
in the LLMs in psychotherapy study. It generates a horizontal bar chart
showing the distribution of metric supercategories by article count and produces a comprehensive report.

Usage:
    python 5-2_create_panel_measurements.py --input path/to/measurements.csv

Arguments:
    --input: Path to CSV file containing measurements information

Outputs:
    - Metric supercategories distribution chart (SVG)
    - Comprehensive measurements metrics report (TXT)
    - All files saved to results/5-2_measurements/
"""

import argparse
import sys
import os
from pathlib import Path

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
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Measurements file '{args.input}' not found.")
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
    print(f"Output directory: {output_dir}")
    
    try:
        # Create the panel
        results = create_measurements_metrics_panel(args.input, output_dir)
        
        # Print summary
        print("\nMeasurements Metrics Panel Created Successfully!")
        print("=" * 50)
        print(f"Total measurements analyzed: {results['total_measurements']}")
        print(f"Unique articles: {results['unique_articles']}")
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
