#!/usr/bin/env python3
"""
Data analysis script for intermediate datasets.

This script analyzes CSV files in the data/intermediate directory and generates
comprehensive column-by-column reports showing categorical distributions,
including detailed breakdowns of "Other:" categories.

Usage:
    python analyze_data.py [--input FILENAME] [--verbose] [--filter_consensus_only]

Example:
    python analyze_data.py --input intermediate_data.csv --verbose
    python analyze_data.py --input intermediate_data.csv --filter_consensus_only --verbose
"""

import argparse
import sys
import re
from pathlib import Path
import pandas as pd
from collections import Counter

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import DATA_INTERMEDIATE_DIR

# Define multiple-choice columns that need to be split on semicolons
MULTIPLE_CHOICE_COLUMNS = [
    "intervention_type",
    "models_employed", 
    "client_type"
]


def analyze_multiple_choice_column(series: pd.Series, column_name: str) -> dict:
    """
    Analyze a multiple-choice column where entries are separated by semicolons.
    
    Args:
        series (pd.Series): The column data to analyze
        column_name (str): Name of the column
        
    Returns:
        dict: Analysis results for multiple-choice data
    """
    # Collect all individual choices
    all_choices = []
    other_categories = {}
    other_count = 0
    
    other_pattern = re.compile(r'^Other:\s*(.+)', re.IGNORECASE)
    
    for value in series.dropna():
        if pd.isna(value):
            continue
            
        value_str = str(value).strip()
        if not value_str:
            continue
            
        # Split by semicolon and process each choice
        choices = [choice.strip() for choice in value_str.split(';') if choice.strip()]
        
        for choice in choices:
            all_choices.append(choice)
            
            # Check for "Other:" categories
            match = other_pattern.match(choice)
            if match:
                other_text = match.group(1).strip()
                if other_text:
                    other_categories[other_text] = other_categories.get(other_text, 0) + 1
                else:
                    other_categories['[blank]'] = other_categories.get('[blank]', 0) + 1
                other_count += 1
    
    # Count all choices
    choice_counts = Counter(all_choices)
    
    # Basic statistics
    total_responses = series.notna().sum()
    null_count = len(series) - total_responses
    total_choices = len(all_choices)
    unique_choices = len(choice_counts)
    
    return {
        'column_name': column_name,
        'is_multiple_choice': True,
        'total_responses': total_responses,
        'null_count': null_count,
        'total_choices': total_choices,
        'unique_choices': unique_choices,
        'choice_counts': dict(choice_counts),
        'other_categories': {
            'total_other_count': other_count,
            'other_categories': other_categories
        },
        'avg_choices_per_response': total_choices / total_responses if total_responses > 0 else 0
    }


def analyze_column(series: pd.Series, column_name: str) -> dict:
    """
    Analyze a single column and return analysis results.
    
    Args:
        series (pd.Series): The column data to analyze
        column_name (str): Name of the column
        
    Returns:
        dict: Analysis results including counts, other categories, etc.
    """
    # Basic statistics
    total_count = len(series)
    non_null_count = series.notna().sum()
    null_count = total_count - non_null_count
    
    # Check if column appears to be categorical (non-numeric or limited unique values)
    unique_count = series.nunique()
    is_likely_categorical = (
        series.dtype == 'object' or 
        unique_count <= 50 or  # Arbitrary threshold for categorical
        unique_count / non_null_count <= 0.1  # Less than 10% unique values
    )
    
    result = {
        'column_name': column_name,
        'total_count': total_count,
        'non_null_count': non_null_count,
        'null_count': null_count,
        'unique_count': unique_count,
        'is_likely_categorical': is_likely_categorical,
        'dtype': str(series.dtype)
    }
    
    if is_likely_categorical and non_null_count > 0:
        # Get value counts for categorical data
        value_counts = series.value_counts(dropna=False)
        result['value_counts'] = value_counts.to_dict()
        
        # Analyze "Other:" categories
        other_categories = analyze_other_categories(series)
        result['other_categories'] = other_categories
        
        # Get sample values if too many unique values
        if unique_count > 20:
            sample_values = series.dropna().unique()[:20]
            result['sample_values'] = list(sample_values)
    else:
        # For non-categorical data, provide basic statistics
        if series.dtype in ['int64', 'float64']:
            try:
                result['min'] = series.min()
                result['max'] = series.max()
                result['mean'] = series.mean()
                result['median'] = series.median()
            except:
                pass
        
        # Show sample values for non-categorical data
        sample_values = series.dropna().unique()[:10]
        result['sample_values'] = list(sample_values)
    
    return result


def analyze_other_categories(series: pd.Series) -> dict:
    """
    Analyze "Other:" categories in a series.
    
    Args:
        series (pd.Series): The series to analyze
        
    Returns:
        dict: Dictionary with other category analysis
    """
    other_pattern = re.compile(r'^Other:\s*(.+)', re.IGNORECASE)
    other_categories = {}
    other_count = 0
    
    for value in series.dropna():
        if pd.isna(value):
            continue
            
        value_str = str(value).strip()
        match = other_pattern.match(value_str)
        
        if match:
            other_text = match.group(1).strip()
            if other_text:
                other_categories[other_text] = other_categories.get(other_text, 0) + 1
            else:
                other_categories['[blank]'] = other_categories.get('[blank]', 0) + 1
            other_count += 1
    
    return {
        'total_other_count': other_count,
        'other_categories': other_categories
    }


def generate_report(df: pd.DataFrame, input_filename: str, consensus_filtered: bool = False) -> str:
    """
    Generate a comprehensive text report for the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
        input_filename (str): Name of the input file
        consensus_filtered (bool): Whether the data was filtered for consensus only
        
    Returns:
        str: The complete report as a string
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append(f"DATA ANALYSIS REPORT")
    report_lines.append(f"Input File: {input_filename}")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if consensus_filtered:
        report_lines.append(f"Filter: Consensus reviewer only")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Dataset overview
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total rows: {len(df):,}")
    report_lines.append(f"Total columns: {len(df.columns):,}")
    report_lines.append("")
    
    # Column-by-column analysis
    report_lines.append("COLUMN-BY-COLUMN ANALYSIS")
    report_lines.append("-" * 40)
    report_lines.append("")
    
    for i, column in enumerate(df.columns, 1):
        # Check if this is a multiple-choice column
        if column in MULTIPLE_CHOICE_COLUMNS:
            analysis = analyze_multiple_choice_column(df[column], column)
            
            report_lines.append(f"{i}. {column}")
            report_lines.append("   " + "=" * (len(str(i)) + len(column) + 2))
            report_lines.append(f"   Category: MULTIPLE CHOICE (semicolon-separated)")
            report_lines.append(f"   Total responses: {analysis['total_responses']:,}")
            report_lines.append(f"   Null responses: {analysis['null_count']:,}")
            report_lines.append(f"   Total individual choices: {analysis['total_choices']:,}")
            report_lines.append(f"   Unique choice options: {analysis['unique_choices']:,}")
            report_lines.append(f"   Average choices per response: {analysis['avg_choices_per_response']:.1f}")
            report_lines.append("")
            
            # Show individual choice counts
            report_lines.append("   Individual Choice Counts:")
            sorted_choices = sorted(analysis['choice_counts'].items(), key=lambda x: x[1], reverse=True)
            for choice, count in sorted_choices:
                percentage = (count / analysis['total_choices'] * 100) if analysis['total_choices'] > 0 else 0
                report_lines.append(f"     {choice}: {count:,} ({percentage:.1f}%)")
            report_lines.append("")
            
            # Show "Other:" categories if any
            other_info = analysis['other_categories']
            if other_info.get('total_other_count', 0) > 0:
                report_lines.append("   'Other:' Categories Breakdown:")
                report_lines.append(f"     Total 'Other:' entries: {other_info['total_other_count']:,}")
                report_lines.append("")
                # Sort by count (descending) for better readability
                sorted_other = sorted(other_info['other_categories'].items(), key=lambda x: x[1], reverse=True)
                for other_text, count in sorted_other:
                    report_lines.append(f"     • Other: {other_text} — {count:,} occurrence(s)")
                report_lines.append("")
            
        else:
            # Regular single-value column analysis
            analysis = analyze_column(df[column], column)
            
            report_lines.append(f"{i}. {column}")
            report_lines.append("   " + "=" * (len(str(i)) + len(column) + 2))
            report_lines.append(f"   Data type: {analysis['dtype']}")
            report_lines.append(f"   Total values: {analysis['total_count']:,}")
            report_lines.append(f"   Non-null values: {analysis['non_null_count']:,}")
            report_lines.append(f"   Null values: {analysis['null_count']:,}")
            report_lines.append(f"   Unique values: {analysis['unique_count']:,}")
            
            if analysis['is_likely_categorical']:
                report_lines.append(f"   Category: CATEGORICAL")
                report_lines.append("")
                
                # Show value counts
                if 'value_counts' in analysis:
                    report_lines.append("   Value Counts:")
                    for value, count in analysis['value_counts'].items():
                        value_display = str(value) if pd.notna(value) else "[NULL]"
                        percentage = (count / analysis['non_null_count'] * 100) if analysis['non_null_count'] > 0 else 0
                        report_lines.append(f"     {value_display}: {count:,} ({percentage:.1f}%)")
                    report_lines.append("")
                
                # Show "Other:" categories if any
                other_info = analysis.get('other_categories', {})
                if other_info.get('total_other_count', 0) > 0:
                    report_lines.append("   'Other:' Categories Breakdown:")
                    report_lines.append(f"     Total 'Other:' entries: {other_info['total_other_count']:,}")
                    report_lines.append("")
                    # Sort by count (descending) for better readability
                    sorted_other = sorted(other_info['other_categories'].items(), key=lambda x: x[1], reverse=True)
                    for other_text, count in sorted_other:
                        report_lines.append(f"     • Other: {other_text} — {count:,} occurrence(s)")
                    report_lines.append("")
                
                # Show sample values if too many categories
                if 'sample_values' in analysis:
                    report_lines.append("   Sample values (first 20):")
                    for value in analysis['sample_values']:
                        value_display = str(value) if pd.notna(value) else "[NULL]"
                        report_lines.append(f"     {value_display}")
                    report_lines.append("")
            
            else:
                report_lines.append(f"   Category: NUMERICAL/CONTINUOUS")
                report_lines.append("")
                
                # Show numerical statistics
                if 'min' in analysis:
                    report_lines.append("   Statistical Summary:")
                    report_lines.append(f"     Min: {analysis['min']}")
                    report_lines.append(f"     Max: {analysis['max']}")
                    report_lines.append(f"     Mean: {analysis['mean']:.2f}")
                    report_lines.append(f"     Median: {analysis['median']:.2f}")
                    report_lines.append("")
                
                # Show sample values
                if 'sample_values' in analysis:
                    report_lines.append("   Sample values (first 10):")
                    for value in analysis['sample_values']:
                        value_display = str(value) if pd.notna(value) else "[NULL]"
                        report_lines.append(f"     {value_display}")
                    report_lines.append("")
        
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def main():
    """Main function to run the data analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze intermediate CSV data files and generate column reports"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="intermediate/intermediate_data.csv",
        help="Filename of input CSV file in data/intermediate/ (default: intermediate_data.csv)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--filter_consensus_only",
        action="store_true",
        help="Filter to keep only rows where reviewer_name/Reviewer Name is 'Consensus' before analysis"
    )
    
    args = parser.parse_args()
    
    # Build full paths using settings
    input_path = Path(args.input)
    
    # Generate output filename (replace .csv with _report.txt)
    output_filename = args.input.replace('.csv', '_report.txt')
    output_path = Path(output_filename)
    
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
        
        # Filter for consensus only if requested
        if args.filter_consensus_only:
            original_rows = len(df)
            
            # Determine which reviewer column name is present
            reviewer_col = None
            if 'Reviewer Name' in df.columns:
                reviewer_col = 'Reviewer Name'
            elif 'reviewer_name' in df.columns:
                reviewer_col = 'reviewer_name'
            else:
                raise KeyError("Neither 'Reviewer Name' nor 'reviewer_name' column found in dataframe. Cannot filter for consensus.")
            
            # Filter for consensus rows only
            df = df[df[reviewer_col] == 'Consensus'].copy()
            
            if args.verbose:
                print(f"Filtered for consensus reviewer only:")
                print(f"  - Original rows: {original_rows:,}")
                print(f"  - Consensus rows: {len(df):,}")
                print(f"  - Rows removed: {original_rows - len(df):,}")
        
        # Generate the report
        if args.verbose:
            print("Analyzing columns and generating report...")
        
        report_content = generate_report(df, args.input, consensus_filtered=args.filter_consensus_only)
        
        # Save the report
        if args.verbose:
            print(f"Saving report to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        if args.verbose:
            print(f"✓ Successfully generated analysis report: {output_path}")
            print(f"✓ Analysis complete!")
        else:
            print(f"Analysis report generated: {output_path}")
            
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
