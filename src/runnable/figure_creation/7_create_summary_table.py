#!/usr/bin/env python3
"""
Study Characteristics Summary Table Generator.

This script creates a comprehensive summary table of all studies in the final dataset,
highlighting key characteristics relevant for high-impact scientific publications.

The table includes:
- Study identification (first author + year)
- Study title
- Application type
- Intervention type (intended)
- Models employed
- Population characteristics
- Sample size
- Control group presence
- Clinical outcome assessment
- Risk detection consideration

Usage:
    python 7_create_summary_table.py --input data/processed/final_data.csv \
        --input_covidence_export data/unprocessed/covidence_export_1st_search_20251030.csv \
        --output results/summary_table.csv

Example:
    python 7_create_summary_table.py \
        --input data/processed/final_data.csv \
        --input_covidence_export data/unprocessed/covidence_export_1st_search_20251030.csv,data/unprocessed/covidence_export_2nd_search_20251030.csv \
        --output results/summary_table.csv \
        --verbose
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def format_study_id(study_id: str, year: float) -> str:
    """
    Format study identification as 'First Author et al. (Year)'.
    
    Args:
        study_id (str): Study ID from covidence (e.g., 'Smith 2023')
        year (float): Publication year
        
    Returns:
        str: Formatted study identification
    """
    if pd.isna(study_id):
        if pd.notna(year):
            return f"Unknown et al. ({int(year)})"
        return "Unknown et al."
    
    # Extract author name from study_id (format is typically "Author YYYY")
    parts = str(study_id).strip().split()
    if len(parts) >= 1:
        author = parts[0]
        # Add year if available
        if pd.notna(year):
            return f"{author} et al. ({int(year)})"
        return f"{author} et al."
    
    return str(study_id)


def format_application_type(app_type: str) -> str:
    """Format application type for table display."""
    if pd.isna(app_type):
        return "Not specified"
    return str(app_type)


def format_intervention_type(intervention: str) -> str:
    """Format intervention type for table display."""
    if pd.isna(intervention):
        return "Unspecified"
    return str(intervention)


def format_models(models: str) -> str:
    """
    Format model names for concise display.
    
    Args:
        models (str): Model names/descriptions
        
    Returns:
        str: Formatted model string
    """
    if pd.isna(models):
        return "Not specified"
    
    model_str = str(models).strip()
    
    # Truncate very long model descriptions
    if len(model_str) > 100:
        return model_str[:97] + "..."
    
    return model_str


def format_sample_size(count, population: str = None) -> str:
    """
    Format sample size for display.
    
    Args:
        count: Sample size value
        population: Population description (if "No clients/patients involved", return "-")
    """
    # If no clients/patients involved, return "-"
    if population and "No clients/patients involved" in str(population):
        return "-"
    
    if pd.isna(count):
        return "Not reported"
    
    try:
        count_int = int(float(count))
        return f"{count_int:,}"
    except (ValueError, TypeError):
        return str(count)


def format_client_type(client_type: str) -> str:
    """
    Format client/population type for concise display.
    
    Args:
        client_type (str): Client type description
        
    Returns:
        str: Formatted client type
    """
    if pd.isna(client_type):
        return "Not specified"
    
    client_str = str(client_type).strip()
    
    # Shorten common long descriptions
    replacements = {
        "General population (e.g., college student, adult, adolescent)": "General population",
        "Patients recruited in hospital or outpatient treatment facility": "Clinical patients (hospital/outpatient)",
        "Patients with disorder explicitly based on ICD or DSM": "Clinical patients (ICD/DSM diagnosis)",
    }
    
    for long_form, short_form in replacements.items():
        if long_form in client_str:
            client_str = client_str.replace(long_form, short_form)
    
    # Truncate if still too long
    if len(client_str) > 80:
        return client_str[:77] + "..."
    
    return client_str


def format_yes_no(value: str) -> str:
    """
    Convert y/n values to Yes/No for table display.
    
    Args:
        value (str): Input value (y/n)
        
    Returns:
        str: Yes/No/Not reported
    """
    if pd.isna(value):
        return "Not reported"
    
    value_str = str(value).strip().lower()
    
    if value_str == 'y' or value_str == 'yes':
        return "Yes"
    elif value_str == 'n' or value_str == 'no':
        return "No"
    else:
        return "Not reported"


def format_clinical_outcome(outcome: str) -> str:
    """Format clinical outcome for display."""
    if pd.isna(outcome):
        return "Not specified"
    
    outcome_str = str(outcome).strip()
    
    # Truncate if too long
    if len(outcome_str) > 80:
        return outcome_str[:77] + "..."
    
    return outcome_str


def create_summary_table(df: pd.DataFrame, covidence_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create a summary table of all studies with key characteristics.
    
    Args:
        df (pd.DataFrame): Main dataset (final_data.csv)
        covidence_df (pd.DataFrame): Optional covidence export data for author names
        
    Returns:
        pd.DataFrame: Summary table
    """
    # Merge with covidence data if available to get study_id (author year format)
    if covidence_df is not None:
        # Standardize column names
        covidence_df = covidence_df.rename(columns={
            'Covidence #': 'covidence_id_raw',
            'Study ID': 'author_year_id'
        })
        
        # Convert covidence_id to string
        covidence_df['covidence_id_raw'] = covidence_df['covidence_id_raw'].astype(str)
        
        # Create mappings for both 1st and 2nd search
        # In final_data.csv, covidence_id is like "1st_search_214" or "2nd_search_214"
        # In covidence export, it's just "214"
        covidence_df['covidence_id_1st'] = '1st_search_' + covidence_df['covidence_id_raw']
        covidence_df['covidence_id_2nd'] = '2nd_search_' + covidence_df['covidence_id_raw']
        
        # Convert df covidence_id to string
        df['covidence_id'] = df['covidence_id'].astype(str)
        
        # Try to merge - first create a combined lookup
        lookup = pd.concat([
            covidence_df[['covidence_id_1st', 'author_year_id']].rename(columns={'covidence_id_1st': 'covidence_id'}),
            covidence_df[['covidence_id_2nd', 'author_year_id']].rename(columns={'covidence_id_2nd': 'covidence_id'})
        ]).drop_duplicates(subset='covidence_id')
        
        # Merge to get author_year_id
        df = df.merge(
            lookup, 
            on='covidence_id', 
            how='left'
        )
    else:
        # Use study_id as fallback
        df['author_year_id'] = df['study_id']
    
    # Format client type first (needed for sample size formatting)
    formatted_population = df['client_type'].apply(format_client_type)
    
    # Create the summary table
    summary = pd.DataFrame({
        'Study': df.apply(lambda row: format_study_id(row['author_year_id'], row['year']), axis=1),
        'Title': df['title'],
        'Application': df['application_type'].apply(format_application_type),
        'Intervention': df['intended_intervention_type'].apply(format_intervention_type),
        'Model': df['models_employed'].apply(format_models),
        'Population': formatted_population,
        'Sample (N)': df.apply(lambda row: format_sample_size(row['client_count'], formatted_population[row.name]), axis=1),
        'Control Group': df['f2_control_condition_considered'].apply(format_yes_no),
        'Clinical Outcome': df['primary_clinical_outcome'].apply(format_clinical_outcome),
        'Risk Detection': df['s1_risk_detection_considered'].apply(format_yes_no),
    })
    
    return summary


def main():
    """Main function to generate the summary table."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive summary table of study characteristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file (final_data.csv)"
    )
    
    parser.add_argument(
        "--input_covidence_export",
        type=str,
        required=True,
        help="Path(s) to covidence export CSV file(s), comma-separated for multiple files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output summary table CSV file"
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
        print(f"Covidence export file(s): {args.input_covidence_export}")
        print(f"Output file: {output_path}")
        print()
    
    try:
        # Load the main data
        if args.verbose:
            print("Loading main dataset...")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        df = pd.read_csv(input_path)
        
        if args.verbose:
            print(f"  ✓ Loaded {len(df):,} studies with {len(df.columns):,} columns")
        
        # Load covidence export file(s)
        covidence_df = None
        if args.input_covidence_export:
            if args.verbose:
                print("Loading covidence export data...")
            
            # Handle multiple files (comma-separated)
            covidence_paths = [Path(p.strip()) for p in args.input_covidence_export.split(',')]
            covidence_dfs = []
            
            for cov_path in covidence_paths:
                if not cov_path.exists():
                    print(f"  Warning: Covidence export file not found: {cov_path}")
                    continue
                
                cov_df = pd.read_csv(cov_path)
                covidence_dfs.append(cov_df)
                
                if args.verbose:
                    print(f"  ✓ Loaded {len(cov_df):,} entries from {cov_path.name}")
            
            # Combine all covidence exports
            if covidence_dfs:
                covidence_df = pd.concat(covidence_dfs, ignore_index=True)
                if args.verbose:
                    print(f"  ✓ Combined total: {len(covidence_df):,} entries")
        
        # Generate the summary table
        if args.verbose:
            print("\nGenerating summary table...")
        
        summary_table = create_summary_table(df, covidence_df)
        
        if args.verbose:
            print(f"  ✓ Created table with {len(summary_table):,} rows and {len(summary_table.columns):,} columns")
        
        # Save the table
        if args.verbose:
            print(f"\nSaving summary table to {output_path}...")
        
        summary_table.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n{'='*80}")
        print(f"✓ Successfully generated summary table!")
        print(f"{'='*80}")
        print(f"Studies summarized: {len(summary_table):,}")
        print(f"Output saved to: {output_path}")
        print()
        
        # Print some summary statistics
        if args.verbose:
            print("Column Summary:")
            print(f"  Application types: {summary_table['Application'].nunique()} unique")
            print(f"  Intervention types: {summary_table['Intervention'].nunique()} unique")
            print(f"  Models: {summary_table['Model'].nunique()} unique")
            print()
            
            # Control group statistics
            control_counts = summary_table['Control Group'].value_counts()
            print("Control Group Presence:")
            for status, count in control_counts.items():
                pct = (count / len(summary_table)) * 100
                print(f"  {status}: {count} ({pct:.1f}%)")
            print()
            
            # Risk detection statistics
            risk_counts = summary_table['Risk Detection'].value_counts()
            print("Risk Detection Consideration:")
            for status, count in risk_counts.items():
                pct = (count / len(summary_table)) * 100
                print(f"  {status}: {count} ({pct:.1f}%)")
            print()
            
            # Show first few rows as preview
            print("Preview (first 5 rows):")
            print(summary_table.head().to_string(index=False))
            print()
            
    except Exception as e:
        print(f"\nError during table generation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

