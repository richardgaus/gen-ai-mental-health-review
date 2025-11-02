#!/usr/bin/env python3
"""
Create Panel: Datasets Metrics

This script creates visualizations and metrics for dataset characteristics
in the LLMs in psychotherapy study. It generates a horizontal bar chart
showing the distribution of dataset types and produces a comprehensive report.

Usage:
    python 5-1_create_panel_datasets.py --datasets_file path/to/datasets.csv

Arguments:
    --datasets_file: Path to CSV file containing dataset information

Outputs:
    - Dataset type distribution chart (SVG)
    - Comprehensive datasets metrics report (TXT)
    - All files saved to results/5-1_datasets/
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.figure_creation.panel_datasets import create_datasets_metrics_panel


def main():
    """Main function to create datasets metrics panel."""
    parser = argparse.ArgumentParser(
        description="Create datasets metrics panel with charts and reports"
    )
    parser.add_argument(
        "--datasets_file",
        required=True,
        help="Path to CSV file containing dataset information"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.datasets_file):
        print(f"Error: Dataset file '{args.datasets_file}' not found.")
        sys.exit(1)
    
    # Set output directory - go up to project root from src_dir
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(
        project_root,
        "results",
        "5-1_datasets"
    )
    
    print(f"Creating datasets metrics panel...")
    print(f"Input file: {args.datasets_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Create the panel
        results = create_datasets_metrics_panel(args.datasets_file, output_dir)
        
        # Print summary
        print("\nDatasets Metrics Panel Created Successfully!")
        print("=" * 50)
        print(f"Total datasets analyzed: {results['total_datasets']}")
        print(f"Datasets with high-level type info: {results['datasets_with_high_level_type']}")
        print(f"Unique high-level dataset types: {results['unique_high_level_types']}")
        if results['most_common_high_level_type']:
            print(f"Most common high-level type: {results['most_common_high_level_type']}")
        
        print(f"\nFiles created:")
        print(f"  Dataset Types Chart: {results['chart_path']}")
        print(f"  Psychopathology Status Chart: {results['psychopathology_chart_path']}")
        print(f"  Responder Type Chart: {results['responder_chart_path']}")
        print(f"  Combined Psychopathology & Responder Chart: {results['combined_psycho_responder_chart_path']}")
        print(f"  Dataset Reuse Chart: {results['reuse_chart_path']}")
        print(f"  Reused Datasets Chart: {results['reused_datasets_chart_path']}")
        print(f"  Dataset Language Chart: {results['language_chart_path']}")
        print(f"  Dataset Public Chart: {results['public_chart_path']}")
        print(f"  Report: {results['report_path']}")
        
    except Exception as e:
        print(f"Error creating datasets metrics panel: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
