#!/usr/bin/env python3
"""
Script to create a horizontal bar chart showing the distribution of language models employed in studies.

This script reads a CSV file containing study data and creates a visualization showing
the top 6 most frequently used language models (excluding "Other" category).

Usage:
    python 4_create_panel_model_types.py --input data/processed/final_data.csv

Requirements:
    - Input CSV must contain a 'models_employed' column with semicolon-separated model names
    - Output figures will be saved to the results/figures/ directory
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
from utils.figure_creation.panel_model_types import create_model_types_panel, create_llm_approach_panel, create_open_vs_closed_weight_time_series, get_model_statistics, generate_model_types_report


def main():
    """Main function to parse arguments and create the model types panel."""
    
    parser = argparse.ArgumentParser(
        description="Create a horizontal bar chart showing language model distribution",
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
        default="results/4_model_types",
        help="Directory to save output figures (default: results/4_model_types)"
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="model_types",
        help="Prefix for output files (default: model_types)"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of top models to show (default: 8)"
    )
    
    parser.add_argument(
        "--include-other",
        action="store_true",
        help="Include 'Other' category in results (default: exclude)"
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
    
    print(f"Reading data from: {input_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Read the data
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from input file")
        
        # Validate required columns exist
        required_columns = ['models_employed', 'study_type', 'llm_development_approach', 'p1_on_premise_model_considered', 'year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Input file must contain the following columns: {', '.join(missing_columns)}")
            sys.exit(1)
        
        # Filter for empirical research involving LLMs
        initial_count = len(df)
        df = df[df['study_type'] == 'Empirical research involving an LLM'].copy()
        filtered_count = len(df)
        
        print(f"Filtered to empirical LLM research: {filtered_count} studies (from {initial_count} total)")
        
        if filtered_count == 0:
            print("Warning: No studies found with study_type == 'Empirical research involving an LLM'")
            sys.exit(1)
        
        # Get statistics
        stats = get_model_statistics(
            df, 
            exclude_other=not args.include_other
        )
        
        print(f"\nDataset statistics:")
        print(f"  - Studies with model information: {stats['studies_with_models']}")
        print(f"  - Total model mentions: {stats['total_model_mentions']}")
        print(f"  - Unique models: {stats['unique_models']}")
        print(f"  - Average models per study: {stats['average_models_per_study']:.2f}")
        
        if stats['most_common_model']:
            model_name, count = stats['most_common_model']
            print(f"  - Most common model: {model_name} ({count} studies)")
        
        # Define output paths
        models_figure_path = output_dir / f"{args.output_prefix}.svg"
        approaches_figure_path = output_dir / f"llm_development_approaches.svg"
        time_series_figure_path = output_dir / f"open_vs_closed_weight_time_series.svg"
        report_path = output_dir / f"4_model_types.txt"
        
        # Generate and save the report
        print(f"\nGenerating report...")
        
        report_content = generate_model_types_report(
            df=df,
            top_n=args.top_n,
            exclude_other=not args.include_other
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"  - Report saved: {report_path}")
        
        # Create the models figure
        print(f"\nCreating models SVG figure...")
        
        models_fig = create_model_types_panel(
            df=df,
            title="LLM types used in studies",
            xlabel="Number of articles",
            ylabel=None,
            figsize=(5, 4),
            top_n=args.top_n,
            exclude_other=not args.include_other,
            save_path=str(models_figure_path)
        )
        
        print(f"  - Models figure saved: {models_figure_path}")
        
        # Create the LLM development approaches figure
        print(f"\nCreating LLM development approaches SVG figure...")
        
        approaches_fig = create_llm_approach_panel(
            df=df,
            title="Development approaches of\nLLM applications in studies",
            xlabel="Number of articles",
            ylabel=None,
            figsize=(5, 3.2),
            top_n=8,
            save_path=str(approaches_figure_path)
        )
        
        print(f"  - Approaches figure saved: {approaches_figure_path}")
        
        # Create the open vs closed weight time series figure
        print(f"\nCreating open vs closed weight time series SVG figure...")
        
        time_series_fig = create_open_vs_closed_weight_time_series(
            df=df,
            title="Time trend of open vs. closed weight model usage",
            xlabel="Publication year",
            ylabel="Number of articles",
            figsize=(7.5, 4),
            save_path=str(time_series_figure_path)
        )
        
        print(f"  - Time series figure saved: {time_series_figure_path}")
        
        print(f"\n✓ Successfully created model types panel with top {args.top_n} models")
        print(f"✓ Successfully created LLM development approaches panel")
        print(f"✓ Successfully created open vs closed weight time series chart")
        print(f"✓ Report and figures saved to: {output_dir}")
        
        # Print top models for reference
        print(f"\nTop {args.top_n} models shown:")
        top_models = list(stats['model_distribution'].items())[:args.top_n]
        for i, (model, count) in enumerate(top_models, 1):
            percentage = (count / stats['total_model_mentions']) * 100
            print(f"  {i}. {model}: {count} studies ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"Error processing data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
