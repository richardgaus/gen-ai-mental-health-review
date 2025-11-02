"""
Measurement extraction utility functions for LLMs in Psychotherapy research.

This module contains utility functions for extracting measurement/metric information from CSV files.
"""

import sys
from pathlib import Path
from typing import List
import pandas as pd
import yaml


def load_metrics_config() -> dict:
    """
    Load the metrics configuration from the YAML file.
    
    Returns:
        dict: Configuration dictionary with metric_columns list and metric_categories mapping
    """
    config_path = Path(__file__).parent / "measurements_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def categorize_metric(metric_name: str, metric_categories: dict) -> str:
    """
    Categorize a metric name based on the metric_categories mapping.
    
    Args:
        metric_name (str): The metric name to categorize
        metric_categories (dict): Dictionary mapping categories to lists of metric names
        
    Returns:
        str: The category name, or the original metric_name if no match found
    """
    if pd.isna(metric_name) or not metric_name:
        return str(metric_name) if metric_name else 'Unknown'
    
    metric_name_str = str(metric_name).strip()
    
    # Search through categories for exact matches
    for category, metric_list in metric_categories.items():
        if metric_name_str in metric_list:
            return category
    
    # If no exact match found, return the original metric name
    return metric_name_str


def categorize_metric_supercategory(metric_category: str, metric_supercategories: dict) -> str:
    """
    Categorize a metric category into a supercategory based on the metric_supercategories mapping.
    
    Args:
        metric_category (str): The metric category to supercategorize
        metric_supercategories (dict): Dictionary mapping supercategories to lists of categories
        
    Returns:
        str: The supercategory name, or the original metric_category if no match found
    """
    if pd.isna(metric_category) or not metric_category:
        return str(metric_category) if metric_category else 'Unknown'
    
    metric_category_str = str(metric_category).strip()
    
    # Search through supercategories for exact matches
    for supercategory, category_list in metric_supercategories.items():
        if metric_category_str in category_list:
            return supercategory
    
    # If no exact match found, return the original metric category
    return metric_category_str


def add_metric_categories(measurements_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add metric_category and metric_supercategory columns to the measurements DataFrame.
    
    Args:
        measurements_df (pd.DataFrame): DataFrame with measurement information
        config (dict): Configuration dictionary with metric_categories and metric_supercategories mappings
        
    Returns:
        pd.DataFrame: DataFrame with added metric_category and metric_supercategory columns
    """
    if 'metric_categories' not in config:
        print("Warning: metric_categories not found in config. Using metric names as categories.")
        measurements_df['metric_category'] = measurements_df['metric_name']
    else:
        metric_categories = config['metric_categories']
        print(f"Applying metric categorization using {len(metric_categories)} categories...")
        
        # Apply categorization
        measurements_df['metric_category'] = measurements_df['metric_name'].apply(
            lambda x: categorize_metric(x, metric_categories)
        )
        
        # Report categorization results
        category_counts = measurements_df['metric_category'].value_counts()
        print("Metric categorization results:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} measurements")
    
    # Add supercategories
    if 'metric_supercategories' not in config:
        print("Warning: metric_supercategories not found in config. Using metric categories as supercategories.")
        measurements_df['metric_supercategory'] = measurements_df['metric_category']
    else:
        metric_supercategories = config['metric_supercategories']
        print(f"Applying metric supercategorization using {len(metric_supercategories)} supercategories...")
        
        # Apply supercategorization
        measurements_df['metric_supercategory'] = measurements_df['metric_category'].apply(
            lambda x: categorize_metric_supercategory(x, metric_supercategories)
        )
        
        # Report supercategorization results
        supercategory_counts = measurements_df['metric_supercategory'].value_counts()
        print("Metric supercategorization results:")
        for supercategory, count in supercategory_counts.items():
            print(f"  {supercategory}: {count} measurements")
    
    return measurements_df


def extract_measurements_from_row(row: pd.Series, metric_columns: List[str]) -> List[dict]:
    """
    Extract measurement information from a single row.
    
    Args:
        row (pd.Series): A row from the input DataFrame
        metric_columns (List[str]): List of metric column names from config
        
    Returns:
        List[dict]: List of measurement dictionaries extracted from this row
    """
    measurements = []
    reference_title = row.get('title', '')
    
    # Process columns ending with _used
    used_columns = [col for col in metric_columns if col.endswith('_used')]
    for used_col in used_columns:
        if used_col in row and str(row[used_col]).strip().lower() == 'y':
            # Extract the metric prefix (e.g., 'lexical_overlap' from 'lexical_overlap_used')
            metric_prefix = used_col[:-5]  # Remove '_used' suffix
            
            # Build the measurement record
            measurement = {
                'reference_title': reference_title,
                'metric_name': metric_prefix,
                'benchmark_quality': row.get(f'{metric_prefix}_benchmark_quality', ''),
                'performance_vs_benchmark': row.get(f'{metric_prefix}_vs_benchmark', ''),
                'benchmark_notes': row.get(f'{metric_prefix}_notes', '')
            }
            measurements.append(measurement)
    
    # Process columns ending with _name
    name_columns = [col for col in metric_columns if col.endswith('_name')]
    for name_col in name_columns:
        metric_value = row.get(name_col, '')
        # Only process if the field is not empty and not "n"
        if pd.notna(metric_value) and str(metric_value).strip() != '' and str(metric_value).strip().lower() != 'n':
            # Extract the metric prefix (e.g., 'metric1' from 'metric1_name')
            metric_prefix = name_col[:-5]  # Remove '_name' suffix
            
            # Build the measurement record
            measurement = {
                'reference_title': reference_title,
                'metric_name': str(metric_value).strip(),
                'benchmark_quality': row.get(f'{metric_prefix}_benchmark_quality', ''),
                'performance_vs_benchmark': row.get(f'{metric_prefix}_vs_benchmark', ''),
                'benchmark_notes': row.get(f'{metric_prefix}_notes', '')
            }
            measurements.append(measurement)
    
    return measurements


def extract_measurements(input_path: str, output_path: str,
                        include_human_participant_studies: bool = False,
                        include_non_empirical_studies: bool = False) -> str:
    """
    Extract measurement information from input CSV and save to output CSV.
    
    By default, excludes studies with human participants that are not empirical research.
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str): Path for the output measurements CSV file
        include_human_participant_studies (bool): If True, include all studies with human participants
                                                   (default False excludes non-empirical studies with participants)
        include_non_empirical_studies (bool): If True, include all non-empirical studies
                                              (default False excludes them if they have participants)
        
    Returns:
        str: Path to the output file
    """
    print("Starting measurement extraction workflow")
    
    # Load the metrics configuration
    print("Loading metrics configuration...")
    config = load_metrics_config()
    metric_columns = config.get('metric_columns', [])
    print(f"Found {len(metric_columns)} metric columns in configuration")
    
    # Load the input CSV
    print(f"Loading input data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")
    
    # Apply filtering based on flags
    initial_count = len(df)
    if not include_human_participant_studies or not include_non_empirical_studies:
        print("\nApplying exclusion criteria...")
        
        # By default, we exclude studies that have human participants OR are not empirical research
        # We keep studies that:
        # 1. Have no clients/patients involved (client_type is "No clients/patients involved" OR empty/null) AND
        # 2. Are empirical research (study_type is "Empirical research involving an LLM")
        
        if not include_human_participant_studies and not include_non_empirical_studies:
            print("  Excluding studies with human participants OR non-empirical studies")
            print("  Keeping only: Studies without clients/patients AND empirical research")
            
            df = df[
                ((df['client_type'] == 'No clients/patients involved') | 
                 (df['client_type'].isna()) | 
                 (df['client_type'].astype(str).str.strip() == '')) &
                (df['study_type'] == 'Empirical research involving an LLM')
            ]
        elif not include_human_participant_studies:
            print("  Excluding all studies with human participants (regardless of empirical status)")
            
            df = df[
                (df['client_type'] == 'No clients/patients involved') | 
                (df['client_type'].isna()) | 
                (df['client_type'].astype(str).str.strip() == '')
            ]
        elif not include_non_empirical_studies:
            print("  Excluding all non-empirical studies (regardless of participant status)")
            
            df = df[
                df['study_type'] == 'Empirical research involving an LLM'
            ]
        
        excluded_count = initial_count - len(df)
        print(f"  Excluded {excluded_count} rows (from {initial_count} to {len(df)})")
    else:
        print("\nIncluding all studies (no exclusion criteria applied)")
    
    print(f"Processing {len(df)} rows for measurement extraction")
    
    # Extract measurements from each row
    print("Extracting measurements from each row...")
    all_measurements = []
    for idx, row in df.iterrows():
        measurements = extract_measurements_from_row(row, metric_columns)
        all_measurements.extend(measurements)
    
    print(f"Extracted {len(all_measurements)} measurements from {len(df)} rows")
    
    # Create the output DataFrame
    measurements_df = pd.DataFrame(all_measurements, columns=[
        'reference_title',
        'metric_name',
        'benchmark_quality',
        'performance_vs_benchmark',
        'benchmark_notes'
    ])
    
    # Filter out llm_judge measurements
    initial_measurement_count = len(measurements_df)
    measurements_df = measurements_df[measurements_df['metric_name'] != 'llm_judge']
    filtered_measurement_count = initial_measurement_count - len(measurements_df)
    
    if filtered_measurement_count > 0:
        print(f"Filtered out {filtered_measurement_count} 'llm_judge' measurements")
    print(f"Remaining measurements after filtering: {len(measurements_df)}")
    
    # Add metric categories
    print("Adding metric categories...")
    measurements_df = add_metric_categories(measurements_df, config)
    
    # Reorder columns to put metric_category and metric_supercategory after metric_name
    column_order = [
        'reference_title',
        'metric_name',
        'metric_category',
        'metric_supercategory',
        'benchmark_quality',
        'performance_vs_benchmark',
        'benchmark_notes'
    ]
    measurements_df = measurements_df[column_order]
    
    # Save to CSV
    print(f"Saving measurements to: {output_path}")
    measurements_df.to_csv(output_path, index=False)
    
    print("Measurement extraction completed successfully")
    
    return output_path


def generate_report(measurements_df: pd.DataFrame, output_path: str) -> str:
    """
    Generate a report about the measurement extraction process.
    
    Args:
        measurements_df (pd.DataFrame): The measurements DataFrame
        output_path (str): Path where the CSV was saved
        
    Returns:
        str: Report text
    """
    report_lines = [
        "Measurement Extraction Report",
        "=" * 50,
        "",
        f"Output file: {output_path}",
        f"Total measurements extracted: {len(measurements_df)}",
        "",
        "Column summary:",
    ]
    
    for col in measurements_df.columns:
        non_null_count = measurements_df[col].notna().sum()
        null_count = measurements_df[col].isna().sum()
        non_empty_count = (measurements_df[col].astype(str).str.strip() != '').sum()
        report_lines.append(f"  {col}: {non_null_count} non-null, {null_count} null, {non_empty_count} non-empty")
    
    report_lines.extend([
        "",
        "Metric supercategory distribution:",
    ])
    
    if 'metric_supercategory' in measurements_df.columns:
        supercategory_counts = measurements_df['metric_supercategory'].value_counts()
        for supercategory, count in supercategory_counts.items():
            report_lines.append(f"  {supercategory}: {count}")
    else:
        report_lines.append("  metric_supercategory column not available")
    
    report_lines.extend([
        "",
        "Metric category distribution:",
    ])
    
    if 'metric_category' in measurements_df.columns:
        category_counts = measurements_df['metric_category'].value_counts()
        for category, count in category_counts.items():
            report_lines.append(f"  {category}: {count}")
    else:
        report_lines.append("  metric_category column not available")
    
    report_lines.extend([
        "",
        "Metric name distribution:",
    ])
    
    if 'metric_name' in measurements_df.columns:
        metric_counts = measurements_df['metric_name'].value_counts()
        for metric, count in metric_counts.items():
            report_lines.append(f"  {metric}: {count}")
    else:
        report_lines.append("  metric_name column not available")
    
    report_lines.extend([
        "",
        "Benchmark quality distribution:",
    ])
    
    if 'benchmark_quality' in measurements_df.columns:
        quality_counts = measurements_df['benchmark_quality'].value_counts()
        for quality, count in quality_counts.items():
            if pd.notna(quality) and str(quality).strip() != '':
                report_lines.append(f"  {quality}: {count}")
    else:
        report_lines.append("  benchmark_quality column not available")
    
    report_lines.extend([
        "",
        "Performance vs benchmark distribution:",
    ])
    
    if 'performance_vs_benchmark' in measurements_df.columns:
        perf_counts = measurements_df['performance_vs_benchmark'].value_counts()
        for perf, count in perf_counts.items():
            if pd.notna(perf) and str(perf).strip() != '':
                report_lines.append(f"  {perf}: {count}")
    else:
        report_lines.append("  performance_vs_benchmark column not available")
    
    report_lines.extend([
        "",
        "Unique references:",
    ])
    
    if 'reference_title' in measurements_df.columns:
        unique_refs = measurements_df['reference_title'].nunique()
        report_lines.append(f"  Total unique reference titles: {unique_refs}")
    else:
        report_lines.append("  reference_title column not available")
    
    return "\n".join(report_lines)

