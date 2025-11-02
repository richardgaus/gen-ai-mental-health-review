#!/usr/bin/env python3
"""
Script to create READI Framework adoption visualization.

This script creates a comprehensive visualization showing the adoption rates of
READI (Responsible, Equitable, Accessible, Deployable, Interpretable) framework
components in client-facing LLM applications for psychotherapy research.

Usage:
    python 6_create_panel_readi.py --input data/processed/final_data.csv

Requirements:
    - Input CSV must contain READI-related columns and client_type information
    - Output figures will be saved to the results/6_readi/ directory
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
from utils.figure_creation.panel_readi import create_readi_panel, generate_readi_report


def main():
    """Main function to parse arguments and create the READI panel."""
    
    parser = argparse.ArgumentParser(
        description="Create READI Framework adoption visualization",
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
        default="results/6_readi",
        help="Directory to save output figures (default: results/6_readi)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="readi_framework",
        help="Prefix for output files (default: readi_framework)"
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
        if args.verbose:
            print(f"Loaded {len(df)} rows from input file")
        
        # Filter for client-facing applications AND empirical LLM studies
        initial_count = len(df)
        filtered_df = df[
            (df['application_type'] == 'Client-facing application') &
            (df['study_type'] == 'Empirical research involving an LLM')
        ].copy()
        filtered_count = len(filtered_df)
        
        if args.verbose:
            print(f"Filtered to client-facing empirical LLM studies: {filtered_count} studies (from {initial_count} total)")
        
        if filtered_count == 0:
            print("Warning: No studies found with both application_type == 'Client-facing application' and study_type == 'Empirical research involving an LLM'")
            sys.exit(1)
        
        # Define output paths
        figure_path = output_dir / f"{args.output_prefix}.svg"
        report_path = output_dir / f"6_readi_report.txt"
        
        # Generate and save the report
        if args.verbose:
            print(f"\nGenerating READI report...")
        
        report_content = generate_readi_report(filtered_df)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if args.verbose:
            print(f"  - Report saved: {report_path}")
        
        # Create the READI framework figure
        if args.verbose:
            print(f"\nCreating READI framework SVG figure...")
        
        fig = create_readi_panel(
            df=filtered_df,
            title="Adoption of responsible development practices\n(READI framework) in client-facing applications",
            figsize=(12, 10),
            save_path=str(figure_path)
        )
        
        if args.verbose:
            print(f"  - Figure saved: {figure_path}")
        
        print(f"\n✓ Successfully created READI framework panel")
        print(f"✓ Analyzed {filtered_count} client-facing empirical LLM studies")
        print(f"✓ Report and figure saved to: {output_dir}")
        
        # Print summary statistics
        readi_columns = [
            's1_risk_detection_considered',
            's2_content_safety_considered', 
            'p1_on_premise_model_considered',
            'p2_privacy_awareness_considered',
            'e1_demographics_reporting_considered',
            'e2_outcomes_by_demographics_considered',
            'g1_early_discontinuation_considered',
            'g2_overuse_considered',
            'f1_validated_outcomes_considered',
            'f2_control_condition_considered',
            'i1_multilevel_feasibility_considered',
            'i2_healthcare_integration_considered'
        ]
        
        yes_counts = []
        for col in readi_columns:
            if col in filtered_df.columns:
                yes_count = filtered_df[col].astype(str).str.lower().isin(['y', 'yes']).sum()
                yes_counts.append(yes_count)
            else:
                yes_counts.append(0)
        
        overall_avg = sum(yes_counts) / (len(yes_counts) * filtered_count) if filtered_count > 0 else 0
        
        print(f"\nREADI Framework Summary:")
        print(f"  - Overall adoption rate: {overall_avg:.1%}")
        print(f"  - Highest adoption: {max(yes_counts)}/{filtered_count} studies ({max(yes_counts)/filtered_count:.1%})")
        print(f"  - Lowest adoption: {min(yes_counts)}/{filtered_count} studies ({min(yes_counts)/filtered_count:.1%})")
            
    except Exception as e:
        print(f"Error processing data: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
