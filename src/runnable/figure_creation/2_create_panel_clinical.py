#!/usr/bin/env python3
"""
Clinical panel analysis script.

This script generates a report on human participation in empirical LLM studies
with client-facing applications. It analyzes which studies involve human 
participants and provides a breakdown of client types.

Usage:
    python 2_create_panel_clinical.py [--input FILENAME] [--output FILENAME] [--verbose]

Example:
    python 2_create_panel_clinical.py --input final_data.csv --verbose
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from settings import DATA_PROCESSED_DIR
from utils.figure_creation.panel_clinical import generate_clinical_report, create_sample_size_figure, create_inclusion_criteria_figure, create_combined_sample_size_inclusion_criteria_figure, create_combined_intervention_control_outcome_figure, create_intervention_type_figure, create_primary_clinical_outcome_chart, create_control_group_chart, create_time_series_human_participation_chart


def filter_clinical_empirical_studies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to only include client-facing empirical LLM studies with clients involved.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Apply filtering criteria:
    # - application_type is "Client-facing application" 
    # - study_type is "Empirical research involving an LLM"
    # - client_type is other than "No clients/patients involved"
    
    required_columns = ['application_type', 'study_type', 'client_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns for filtering: {missing_columns}")
        return df  # Return original data if filtering columns are missing
    
    client_facing_mask = df['application_type'] == 'Client-facing application'
    empirical_llm_mask = df['study_type'] == 'Empirical research involving an LLM'
    clients_involved_mask = df['client_type'] != 'No clients/patients involved'
    
    filtered_df = df[client_facing_mask & empirical_llm_mask & clients_involved_mask].copy()
    
    return filtered_df


def main():
    """Main function to run the clinical panel analysis."""
    parser = argparse.ArgumentParser(
        description="Generate clinical panel report on human participation in LLM studies"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/final_data.csv",
        help="Path to input CSV file (default: data/processed/final_data.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/2_panel_clinical/2_panel_clinical_report.txt",
        help="Path for output report (default: results/2_panel_clinical/2_panel_clinical_report.txt)"
    )
    
    parser.add_argument(
        "--figure",
        type=str,
        default="results/2_panel_clinical/sample_size_distribution.svg",
        help="Path for output figure (default: results/2_panel_clinical/sample_size_distribution.svg)"
    )
    
    parser.add_argument(
        "--inclusion-criteria-chart",
        type=str,
        default="results/2_panel_clinical/inclusion_criteria_distribution.svg",
        help="Path for inclusion criteria chart (default: results/2_panel_clinical/inclusion_criteria_distribution.svg)"
    )
    
    parser.add_argument(
        "--combined-chart",
        type=str,
        default="results/2_panel_clinical/combined_sample_size_inclusion_criteria.svg",
        help="Path for combined sample size and inclusion criteria chart (default: results/2_panel_clinical/combined_sample_size_inclusion_criteria.svg)"
    )
    
    parser.add_argument(
        "--combined-intervention-chart",
        type=str,
        default="results/2_panel_clinical/combined_intervention_control_outcome.svg",
        help="Path for combined intervention, control, and outcome chart (default: results/2_panel_clinical/combined_intervention_control_outcome.svg)"
    )
    
    parser.add_argument(
        "--intervention-chart",
        type=str,
        default="results/2_panel_clinical/intended_intervention_type_distribution.svg",
        help="Path for intervention type chart (default: results/2_panel_clinical/intended_intervention_type_distribution.svg)"
    )
    
    parser.add_argument(
        "--outcome-chart",
        type=str,
        default="results/2_panel_clinical/primary_clinical_outcome_distribution.svg",
        help="Path for primary clinical outcome chart (default: results/2_panel_clinical/primary_clinical_outcome_distribution.svg)"
    )
    
    parser.add_argument(
        "--control-chart",
        type=str,
        default="results/2_panel_clinical/control_group_presence.svg",
        help="Path for control group presence chart (default: results/2_panel_clinical/control_group_presence.svg)"
    )
    
    parser.add_argument(
        "--time-series-chart",
        type=str,
        default="results/2_panel_clinical/time_series_human_participation.svg",
        help="Path for time series human participation chart (default: results/2_panel_clinical/time_series_human_participation.svg)"
    )
    
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Build full paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    figure_path = Path(args.figure)
    inclusion_criteria_chart_path = Path(getattr(args, 'inclusion_criteria_chart'))
    combined_chart_path = Path(getattr(args, 'combined_chart'))
    combined_intervention_chart_path = Path(getattr(args, 'combined_intervention_chart'))
    intervention_chart_path = Path(getattr(args, 'intervention_chart'))
    outcome_chart_path = Path(getattr(args, 'outcome_chart'))
    control_chart_path = Path(getattr(args, 'control_chart'))
    time_series_chart_path = Path(getattr(args, 'time_series_chart'))
    
    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    inclusion_criteria_chart_path.parent.mkdir(parents=True, exist_ok=True)
    combined_chart_path.parent.mkdir(parents=True, exist_ok=True)
    combined_intervention_chart_path.parent.mkdir(parents=True, exist_ok=True)
    intervention_chart_path.parent.mkdir(parents=True, exist_ok=True)
    outcome_chart_path.parent.mkdir(parents=True, exist_ok=True)
    control_chart_path.parent.mkdir(parents=True, exist_ok=True)
    time_series_chart_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output report: {output_path}")
        print(f"Output figure: {figure_path}")
        print(f"Inclusion criteria chart: {inclusion_criteria_chart_path}")
        print(f"Combined chart: {combined_chart_path}")
        print(f"Combined intervention chart: {combined_intervention_chart_path}")
        print(f"Intervention chart: {intervention_chart_path}")
        print(f"Outcome chart: {outcome_chart_path}")
        print(f"Control chart: {control_chart_path}")
        print(f"Time series chart: {time_series_chart_path}")
    
    try:
        # Load the data
        if args.verbose:
            print("Loading data...")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        
        if args.verbose:
            print(f"Loaded {len(df):,} rows with {len(df.columns):,} columns")
        
        # Generate the report
        if args.verbose:
            print("Generating clinical panel report...")
        
        report_content = generate_clinical_report(df)
        
        # Save the report
        if args.verbose:
            print(f"Saving report to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Filter data for all clinical charts
        if args.verbose:
            print("Filtering data for clinical charts...")
        
        clinical_df = filter_clinical_empirical_studies(df)
        
        if args.verbose:
            print(f"Filtered to {len(clinical_df):,} client-facing empirical LLM studies with clients involved (from {len(df):,} total)")
        
        # Generate and save the figure
        if args.verbose:
            print("Generating sample size distribution figure...")
        
        fig = create_sample_size_figure(clinical_df, save_path=str(figure_path))
        
        # Generate inclusion criteria chart
        if args.verbose:
            print("Generating inclusion criteria chart...")
        
        try:
            inclusion_criteria_fig = create_inclusion_criteria_figure(clinical_df, save_path=str(inclusion_criteria_chart_path))
        except (KeyError, ValueError) as e:
            if args.verbose:
                print(f"Warning: Could not generate inclusion criteria chart: {e}")
            inclusion_criteria_fig = None
        
        # Generate combined sample size and inclusion criteria chart
        if args.verbose:
            print("Generating combined sample size and inclusion criteria chart...")
        
        try:
            combined_fig = create_combined_sample_size_inclusion_criteria_figure(clinical_df, save_path=str(combined_chart_path))
        except (KeyError, ValueError) as e:
            if args.verbose:
                print(f"Warning: Could not generate combined chart: {e}")
            combined_fig = None
        
        # Generate combined intervention, control, and outcome chart
        if args.verbose:
            print("Generating combined intervention, control, and outcome chart...")
        
        try:
            combined_intervention_fig = create_combined_intervention_control_outcome_figure(clinical_df, save_path=str(combined_intervention_chart_path))
        except (KeyError, ValueError) as e:
            if args.verbose:
                print(f"Warning: Could not generate combined intervention chart: {e}")
            combined_intervention_fig = None
        
        # Generate intervention type chart
        if args.verbose:
            print("Generating intended intervention type chart...")
        
        intervention_fig = create_intervention_type_figure(clinical_df, save_path=str(intervention_chart_path))
        
        # Generate primary clinical outcome chart
        if args.verbose:
            print("Generating primary clinical outcome chart...")
        
        try:
            outcome_fig = create_primary_clinical_outcome_chart(clinical_df, save_path=str(outcome_chart_path))
        except (KeyError, ValueError) as e:
            if args.verbose:
                print(f"Warning: Could not generate primary clinical outcome chart: {e}")
            outcome_fig = None
        
        # Generate control group presence chart
        if args.verbose:
            print("Generating control group presence chart...")
        
        try:
            control_fig = create_control_group_chart(clinical_df, save_path=str(control_chart_path))
        except (KeyError, ValueError) as e:
            if args.verbose:
                print(f"Warning: Could not generate control group chart: {e}")
            control_fig = None
        
        # Generate time series human participation chart (uses full dataset with its own filtering)
        if args.verbose:
            print("Generating time series human participation chart...")
        
        try:
            time_series_fig = create_time_series_human_participation_chart(
                df,
                save_path=str(time_series_chart_path),
                figsize=(6, 4)
            )
        except (KeyError, ValueError) as e:
            if args.verbose:
                print(f"Warning: Could not generate time series chart: {e}")
            time_series_fig = None
        
        
        if args.verbose:
            print(f"✓ Successfully generated clinical panel report: {output_path}")
            print(f"✓ Successfully generated sample size figure: {figure_path}")
            if inclusion_criteria_fig is not None:
                print(f"✓ Successfully generated inclusion criteria chart: {inclusion_criteria_chart_path}")
            if combined_fig is not None:
                print(f"✓ Successfully generated combined chart: {combined_chart_path}")
            if combined_intervention_fig is not None:
                print(f"✓ Successfully generated combined intervention chart: {combined_intervention_chart_path}")
            print(f"✓ Successfully generated intervention type chart: {intervention_chart_path}")
            if outcome_fig is not None:
                print(f"✓ Successfully generated primary clinical outcome chart: {outcome_chart_path}")
            if control_fig is not None:
                print(f"✓ Successfully generated control group presence chart: {control_chart_path}")
            if time_series_fig is not None:
                print(f"✓ Successfully generated time series human participation chart: {time_series_chart_path}")
            print(f"✓ Analysis complete!")
        else:
            print(f"Clinical panel report generated: {output_path}")
            print(f"Sample size figure generated: {figure_path}")
            if inclusion_criteria_fig is not None:
                print(f"Inclusion criteria chart generated: {inclusion_criteria_chart_path}")
            if combined_fig is not None:
                print(f"Combined chart generated: {combined_chart_path}")
            if combined_intervention_fig is not None:
                print(f"Combined intervention chart generated: {combined_intervention_chart_path}")
            print(f"Intervention type chart generated: {intervention_chart_path}")
            if outcome_fig is not None:
                print(f"Primary clinical outcome chart generated: {outcome_chart_path}")
            if control_fig is not None:
                print(f"Control group presence chart generated: {control_chart_path}")
            if time_series_fig is not None:
                print(f"Time series human participation chart generated: {time_series_chart_path}")
            
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
