#!/usr/bin/env python3
"""
Controlled Clinical Studies Finder.

This script identifies and reports on empirical LLM studies with client-facing
applications that use both validated outcomes and control conditions.

Usage:
    python 3_find_controlled_clinical_studies.py [--input FILENAME] [--output FILENAME] [--verbose]

Example:
    python 3_find_controlled_clinical_studies.py --input final_data.csv --verbose
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from settings import DATA_PROCESSED_DIR


def filter_controlled_clinical_studies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to controlled clinical studies with validated outcomes.
    
    Criteria:
    - study_type == 'Empirical research involving an LLM'
    - application_type == 'Client-facing application'
    - f1_validated_outcomes_considered == 'y'
    - f2_control_condition_considered == 'y'
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    filtered = df[
        (df['study_type'] == 'Empirical research involving an LLM') &
        (df['application_type'] == 'Client-facing application') &
        (df['f1_validated_outcomes_considered'] == 'y') &
        (df['f2_control_condition_considered'] == 'y')
    ].copy()
    
    return filtered


def generate_controlled_studies_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive report on controlled clinical studies.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        str: Formatted text report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("CONTROLLED CLINICAL STUDIES REPORT")
    report_lines.append("Empirical LLM Studies with Validated Outcomes & Control Conditions")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Filter studies
    filtered_df = filter_controlled_clinical_studies(df)
    
    if filtered_df.empty:
        report_lines.append("No studies found matching all criteria:")
        report_lines.append("  - study_type == 'Empirical research involving an LLM'")
        report_lines.append("  - application_type == 'Client-facing application'")
        report_lines.append("  - f1_validated_outcomes_considered == 'y'")
        report_lines.append("  - f2_control_condition_considered == 'y'")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        return "\n".join(report_lines)
    
    # Overview
    report_lines.append("OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total studies found: {len(filtered_df)}")
    report_lines.append("")
    report_lines.append("Selection criteria:")
    report_lines.append("  ✓ Empirical research involving an LLM")
    report_lines.append("  ✓ Client-facing application")
    report_lines.append("  ✓ Validated outcomes considered")
    report_lines.append("  ✓ Control condition considered")
    report_lines.append("")
    
    # Detailed information for each study
    report_lines.append("=" * 80)
    report_lines.append("DETAILED STUDY INFORMATION")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for idx, (_, row) in enumerate(filtered_df.iterrows(), 1):
        report_lines.append(f"STUDY {idx}")
        report_lines.append("-" * 80)
        
        # Basic information
        report_lines.append(f"Study ID: {row.get('study_id', 'N/A')}")
        report_lines.append(f"Covidence ID: {row.get('covidence_id', 'N/A')}")
        report_lines.append("")
        
        report_lines.append(f"Title: {row.get('title', 'N/A')}")
        report_lines.append("")
        
        # Publication information
        year = row.get('year', 'N/A')
        month = row.get('month', 'N/A')
        day = row.get('day', 'N/A')
        report_lines.append(f"Publication Date: {year}-{month}-{day}")
        report_lines.append(f"Outlet Type: {row.get('outlet_type', 'N/A')}")
        report_lines.append(f"Outlet Field: {row.get('outlet_field', 'N/A')}")
        report_lines.append(f"Author Country: {row.get('author_country', 'N/A')}")
        report_lines.append("")
        
        # Participant information
        report_lines.append("PARTICIPANT INFORMATION:")
        client_type = row.get('client_type', 'N/A')
        client_count = row.get('client_count', 'N/A')
        report_lines.append(f"  Client Type: {client_type}")
        report_lines.append(f"  Sample Size: {client_count}")
        report_lines.append("")
        
        # Application details
        report_lines.append("APPLICATION DETAILS:")
        report_lines.append(f"  Application Subtype (Client-Facing): {row.get('application_subtype_client_facing', 'N/A')}")
        report_lines.append(f"  Intervention Type: {row.get('intervention_type', 'N/A')}")
        report_lines.append(f"  Intended Intervention Type: {row.get('intended_intervention_type', 'N/A')}")
        report_lines.append(f"  Models Employed: {row.get('models_employed', 'N/A')}")
        report_lines.append("")
        
        # Dataset information
        report_lines.append("DATASET INFORMATION:")
        report_lines.append(f"  Dataset Source: {row.get('dataset_source', 'N/A')}")
        report_lines.append(f"  Dataset Type: {row.get('dataset_type', 'N/A')}")
        report_lines.append(f"  Dataset Language: {row.get('dataset_language', 'N/A')}")
        report_lines.append(f"  Dataset Is Public: {row.get('dataset_is_public', 'N/A')}")
        report_lines.append(f"  Dataset User Psychopathology Status: {row.get('dataset_user_psychopathology_status', 'N/A')}")
        report_lines.append("")
        
        # Validation details
        report_lines.append("VALIDATION & CONTROL:")
        report_lines.append(f"  Validated Outcomes: {row.get('f1_validated_outcomes_considered', 'N/A')}")
        report_lines.append(f"  Validated Outcomes Notes: {row.get('f1_validated_outcomes_notes', 'N/A')}")
        report_lines.append(f"  Control Condition: {row.get('f2_control_condition_considered', 'N/A')}")
        report_lines.append(f"  Control Condition Notes: {row.get('f2_control_condition_notes', 'N/A')}")
        report_lines.append("")
        
        # UX Assessment
        report_lines.append("UX ASSESSMENT:")
        ux_present = row.get('ux_assessment_is_present', 'N/A')
        report_lines.append(f"  UX Assessment Present: {ux_present}")
        if ux_present == 'Yes':
            report_lines.append(f"  Standard Instrument Used: {row.get('ux_uses_standard_instrument', 'N/A')}")
            report_lines.append(f"  Qualitative Assessment: {row.get('ux_uses_qualitative_assessment', 'N/A')}")
            report_lines.append(f"  Quantitative Assessment: {row.get('ux_uses_quantitative_assessment', 'N/A')}")
            report_lines.append(f"  Results Reported: {row.get('ux_results_reported', 'N/A')}")
            report_lines.append(f"  Assessment Instrument: {row.get('ux_assessment_instrument', 'N/A')}")
        report_lines.append("")
        
        # Evaluation metrics
        report_lines.append("EVALUATION METRICS:")
        
        metrics_used = []
        if row.get('lexical_overlap_used') == 'y':
            metrics_used.append(f"Lexical Overlap (vs benchmark: {row.get('lexical_overlap_vs_benchmark', 'N/A')})")
        if row.get('embedding_similarity_used') == 'y':
            metrics_used.append(f"Embedding Similarity (vs benchmark: {row.get('embedding_similarity_vs_benchmark', 'N/A')})")
        if row.get('classification_used') == 'y':
            metrics_used.append(f"Classification (vs benchmark: {row.get('classification_vs_benchmark', 'N/A')})")
        if row.get('continuous_metrics_used') == 'y':
            metrics_used.append(f"Continuous Metrics (vs benchmark: {row.get('continuous_metrics_vs_benchmark', 'N/A')})")
        if row.get('expert_rating_used') == 'y':
            metrics_used.append(f"Expert Rating (vs benchmark: {row.get('expert_rating_vs_benchmark', 'N/A')})")
        if row.get('llm_judge_used') == 'y':
            metrics_used.append(f"LLM Judge (vs benchmark: {row.get('llm_judge_vs_benchmark', 'N/A')})")
        if row.get('perplexity_used') == 'y':
            metrics_used.append(f"Perplexity (vs benchmark: {row.get('perplexity_vs_benchmark', 'N/A')})")
        if row.get('lexical_diversity_used') == 'y':
            metrics_used.append(f"Lexical Diversity (vs benchmark: {row.get('lexical_diversity_vs_benchmark', 'N/A')})")
        
        if metrics_used:
            for metric in metrics_used:
                report_lines.append(f"  • {metric}")
        else:
            report_lines.append("  No standard metrics reported")
        report_lines.append("")
        
        # Custom metrics
        custom_metrics = []
        metric1 = row.get('metric1_name', '')
        metric2 = row.get('metric2_name', '')
        metric3 = row.get('metric3_name', '')
        
        if pd.notna(metric1) and str(metric1).strip():
            custom_metrics.append(f"Metric 1: {metric1} (vs benchmark: {row.get('metric1_vs_benchmark', 'N/A')})")
        if pd.notna(metric2) and str(metric2).strip():
            custom_metrics.append(f"Metric 2: {metric2} (vs benchmark: {row.get('metric2_vs_benchmark', 'N/A')})")
        if pd.notna(metric3) and str(metric3).strip():
            custom_metrics.append(f"Metric 3: {metric3} (vs benchmark: {row.get('metric3_vs_benchmark', 'N/A')})")
        
        if custom_metrics:
            report_lines.append("CUSTOM METRICS:")
            for metric in custom_metrics:
                report_lines.append(f"  • {metric}")
            report_lines.append("")
        
        # Safety and ethical considerations
        report_lines.append("SAFETY & ETHICAL CONSIDERATIONS:")
        considerations = []
        
        if row.get('s1_risk_detection_considered') == 'y':
            considerations.append(f"Risk Detection: Yes - {row.get('s1_risk_detection_notes', 'N/A')}")
        if row.get('s2_content_safety_considered') == 'y':
            considerations.append(f"Content Safety: Yes - {row.get('s2_content_safety_notes', 'N/A')}")
        if row.get('p1_on_premise_model_considered') == 'y':
            considerations.append(f"On-Premise Model: Yes - {row.get('p1_on_premise_model_notes', 'N/A')}")
        if row.get('p2_privacy_awareness_considered') == 'y':
            considerations.append(f"Privacy Awareness: Yes - {row.get('p2_privacy_awareness_notes', 'N/A')}")
        if row.get('e1_demographics_reporting_considered') == 'y':
            considerations.append(f"Demographics Reporting: Yes - {row.get('e1_demographics_reporting_notes', 'N/A')}")
        if row.get('e2_outcomes_by_demographics_considered') == 'y':
            considerations.append(f"Outcomes by Demographics: Yes - {row.get('e2_outcomes_by_demographics_considered', 'N/A')}")
        
        if considerations:
            for consideration in considerations:
                report_lines.append(f"  • {consideration}")
        else:
            report_lines.append("  No specific safety/ethical considerations documented")
        report_lines.append("")
        
        # General notes
        general_notes = row.get('general_notes', '')
        if pd.notna(general_notes) and str(general_notes).strip():
            report_lines.append(f"GENERAL NOTES:")
            report_lines.append(f"  {general_notes}")
            report_lines.append("")
        
        report_lines.append("")
    
    # Summary statistics
    report_lines.append("=" * 80)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Sample size statistics
    numeric_sizes = []
    for val in filtered_df['client_count']:
        if pd.notna(val):
            try:
                numeric_sizes.append(int(val))
            except (ValueError, TypeError):
                pass
    
    if numeric_sizes:
        report_lines.append("Sample Size Statistics:")
        report_lines.append(f"  Studies with numeric sample sizes: {len(numeric_sizes)}")
        report_lines.append(f"  Total participants: {sum(numeric_sizes):,}")
        report_lines.append(f"  Range: {min(numeric_sizes)} - {max(numeric_sizes):,}")
        report_lines.append(f"  Mean: {sum(numeric_sizes)/len(numeric_sizes):.1f}")
        report_lines.append(f"  Median: {sorted(numeric_sizes)[len(numeric_sizes)//2]}")
        report_lines.append("")
    
    # Intervention types
    intervention_types = filtered_df['intended_intervention_type'].value_counts()
    if not intervention_types.empty:
        report_lines.append("Intervention Types:")
        for intervention_type, count in intervention_types.items():
            report_lines.append(f"  {intervention_type}: {count}")
        report_lines.append("")
    
    # Models used
    models = filtered_df['models_employed'].value_counts()
    if not models.empty:
        report_lines.append("Models Employed (top 10):")
        for model, count in models.head(10).items():
            model_str = str(model)[:70] + "..." if len(str(model)) > 70 else str(model)
            report_lines.append(f"  {model_str}: {count}")
        report_lines.append("")
    
    # Client types
    client_types = filtered_df['client_type'].value_counts()
    if not client_types.empty:
        report_lines.append("Client Types (top 10):")
        for client_type, count in client_types.head(10).items():
            client_str = str(client_type)[:70] + "..." if len(str(client_type)) > 70 else str(client_type)
            report_lines.append(f"  {client_str}: {count}")
        report_lines.append("")
    
    # Publication years
    years = filtered_df['year'].value_counts().sort_index()
    if not years.empty:
        report_lines.append("Publication Years:")
        for year, count in years.items():
            report_lines.append(f"  {year}: {count}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def main():
    """Main function to find and report on controlled clinical studies."""
    parser = argparse.ArgumentParser(
        description="Find and report on controlled clinical studies with validated outcomes"
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
        default="results/3_controlled_clinical_studies/3_controlled_clinical_studies.txt",
        help="Path for output report (default: results/3_controlled_clinical_studies/3_controlled_clinical_studies.txt)"
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
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output report: {output_path}")
    
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
            print("Generating controlled clinical studies report...")
        
        report_content = generate_controlled_studies_report(df)
        
        # Save the report
        if args.verbose:
            print(f"Saving report to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if args.verbose:
            print(f"✓ Successfully generated controlled clinical studies report: {output_path}")
            print(f"✓ Analysis complete!")
        else:
            print(f"Controlled clinical studies report generated: {output_path}")
            
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


