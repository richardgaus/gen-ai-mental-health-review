"""
Dataset extraction utility functions for LLMs in Psychotherapy research.

This module contains utility functions for extracting dataset information from large CSV files
and merging with smaller dataset files.
"""

import sys
from pathlib import Path
from typing import Union, Tuple, List
import pandas as pd
import yaml

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_datasets_config() -> dict:
    """
    Load the datasets configuration from the YAML file.
    
    Returns:
        dict: Configuration dictionary with dataset_ids mapping
    """
    config_path = Path(__file__).parent / "datasets_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def generate_dataset_id(dataset_notes: str, config: dict) -> str:
    """
    Generate a dataset ID based on dataset_notes matching supercategories in config.
    
    Args:
        dataset_notes (str): The dataset notes to match against
        config (dict): Configuration dictionary with dataset_ids mapping
        
    Returns:
        str: Dataset ID (supercategory name or generic dataset_i)
    """
    if pd.isna(dataset_notes) or not dataset_notes:
        return None
    
    dataset_notes_str = str(dataset_notes).strip()
    
    # Check if dataset_notes matches any supercategory patterns
    if 'dataset_ids' in config:
        for supercategory, patterns in config['dataset_ids'].items():
            for pattern in patterns:
                pattern_str = str(pattern).strip()
                if pattern_str and dataset_notes_str == pattern_str:
                    return supercategory
    
    return None  # Will be assigned generic ID later


def add_dataset_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dataset_id column to DataFrame based on dataset_notes matching.
    
    Args:
        df (pd.DataFrame): DataFrame with dataset_notes column
        
    Returns:
        pd.DataFrame: DataFrame with added dataset_id column
    """
    if 'dataset_notes' not in df.columns:
        print("Warning: dataset_notes column not found. Adding empty dataset_id column.")
        df['dataset_id'] = None
        return df
    
    # Load configuration
    config = load_datasets_config()
    
    # Generate dataset IDs based on matching
    df['dataset_id'] = df['dataset_notes'].apply(lambda x: generate_dataset_id(x, config))
    
    # Assign generic IDs for unmatched entries
    unmatched_mask = df['dataset_id'].isna()
    unmatched_count = unmatched_mask.sum()
    
    if unmatched_count > 0:
        # Generate generic dataset IDs
        generic_ids = [f"dataset_{i+1}" for i in range(unmatched_count)]
        df.loc[unmatched_mask, 'dataset_id'] = generic_ids
        print(f"Assigned {unmatched_count} generic dataset IDs (dataset_1 to dataset_{unmatched_count})")
    
    # Count matched entries
    matched_count = len(df) - unmatched_count
    if matched_count > 0:
        print(f"Matched {matched_count} entries to supercategories")
    
    return df


def reduce_dataset_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce dataset by collapsing rows with the same dataset_id.
    
    Args:
        df (pd.DataFrame): DataFrame with dataset_id column
        
    Returns:
        pd.DataFrame: Reduced DataFrame with dataset_id as index and references column
    """
    if 'dataset_id' not in df.columns:
        raise ValueError("DataFrame must have a 'dataset_id' column")
    
    if 'reference' not in df.columns:
        raise ValueError("DataFrame must have a 'reference' column")
    
    # Group by dataset_id and aggregate
    print("Collapsing rows by dataset_id...")
    
    # Create references column by combining all reference values for each dataset_id
    grouped = df.groupby('dataset_id').agg({
        'reference': lambda x: list(x.dropna().unique()),  # List of unique references
        'row_id': lambda x: list(x),  # Keep all row IDs for this dataset
        **{col: 'first' for col in df.columns if col not in ['dataset_id', 'reference', 'row_id']}
    }).reset_index()
    
    # Convert reference list to a string representation and count references
    grouped['references'] = grouped['reference'].apply(lambda x: '; '.join(x) if x else '')
    grouped['reference_count'] = grouped['reference'].apply(lambda x: len(x) if x else 0)
    grouped['row_ids'] = grouped['row_id'].apply(lambda x: '; '.join(map(str, x)) if x else '')
    
    # Print reference counts during collapsing
    print("Reference counts per dataset:")
    for _, row in grouped.iterrows():
        dataset_id = row['dataset_id']
        ref_count = row['reference_count']
        if ref_count > 1:
            print(f"  {dataset_id}: {ref_count} references")
    
    # Drop the original reference and row_id columns (keeping the aggregated versions)
    grouped = grouped.drop(['reference', 'row_id'], axis=1)
    
    # Set dataset_id as index
    grouped = grouped.set_index('dataset_id')
    
    print(f"Reduced from {len(df)} rows to {len(grouped)} unique dataset IDs")
    
    return grouped


def merge_with_common_datasets(df: pd.DataFrame, common_datasets_path: str) -> pd.DataFrame:
    """
    Merge with common_datasets.csv and overwrite columns where available.
    
    Args:
        df (pd.DataFrame): DataFrame with dataset_id as index
        common_datasets_path (str): Path to common_datasets.csv file
        
    Returns:
        pd.DataFrame: DataFrame with overwritten columns from common_datasets
    """
    print(f"Loading common datasets from: {common_datasets_path}")
    
    # Load common datasets (using semicolon delimiter)
    common_df = pd.read_csv(common_datasets_path, delimiter=';')
    
    # Set dataset_id as index for merging
    if 'dataset_id' in common_df.columns:
        common_df = common_df.set_index('dataset_id')
    else:
        raise ValueError("common_datasets.csv must have a 'dataset_id' column")
    
    print(f"Found {len(common_df)} entries in common datasets")
    
    # Find overlapping columns (excluding the index)
    overlapping_columns = list(set(df.columns) & set(common_df.columns))
    print(f"Overlapping columns to overwrite: {overlapping_columns}")
    
    # Create a copy of the original dataframe
    result_df = df.copy()
    
    # Overwrite columns where common_datasets has data
    for col in overlapping_columns:
        # Only overwrite where common_df has non-null values
        mask = common_df[col].notna()
        matching_indices = result_df.index.intersection(common_df.index[mask])
        
        if len(matching_indices) > 0:
            result_df.loc[matching_indices, col] = common_df.loc[matching_indices, col]
            print(f"  Overwritten {col} for {len(matching_indices)} entries")
    
    return result_df


def rename_dataset_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename dataset_type values according to dataset_types_mapping in config.
    
    Args:
        df (pd.DataFrame): DataFrame with dataset_type column
        
    Returns:
        pd.DataFrame: DataFrame with renamed dataset_type values
    """
    if 'dataset_type' not in df.columns:
        print("Warning: dataset_type column not found. Skipping dataset type renaming.")
        return df
    
    # Load configuration
    config = load_datasets_config()
    
    if 'dataset_types_mapping' not in config:
        print("Warning: dataset_types_mapping not found in config. Skipping dataset type renaming.")
        return df
    
    # Create reverse mapping (from old values to new keys)
    type_mapping = {}
    for new_type, old_types in config['dataset_types_mapping'].items():
        for old_type in old_types:
            type_mapping[old_type] = new_type
    
    print(f"Dataset type mapping loaded with {len(type_mapping)} mappings")
    
    # Fill empty values with "Other: unknown"
    original_values = df['dataset_type'].value_counts()
    empty_mask = df['dataset_type'].isna() | (df['dataset_type'] == '') | (df['dataset_type'] == 'nan')
    empty_count = empty_mask.sum()
    if empty_count > 0:
        df.loc[empty_mask, 'dataset_type'] = 'Other: unknown'
        print(f"  Filled {empty_count} empty dataset_type values with 'Other: unknown'")
    
    # Apply the mapping with smart "Other: " prefix handling
    def smart_map_dataset_type(value):
        if pd.isna(value):
            return value
        
        value_str = str(value)
        
        # First try direct mapping
        if value_str in type_mapping:
            return type_mapping[value_str]
        
        # If it starts with "Other: ", try mapping the cleaned version
        if value_str.startswith("Other: "):
            cleaned_val = value_str[7:]  # Remove "Other: " prefix
            if cleaned_val in type_mapping:
                return type_mapping[cleaned_val]
        
        # No mapping found, return original value
        return value_str
    
    # Apply the smart mapping
    df['dataset_type'] = df['dataset_type'].apply(smart_map_dataset_type)
    
    # Report changes
    final_values = df['dataset_type'].value_counts()
    mapping_changed_count = 0
    other_removed_count = 0
    
    for orig_val, count in original_values.items():
        orig_str = str(orig_val)
        
        # Check if this value was mapped
        if orig_str in type_mapping:
            new_val = type_mapping[orig_str]
            if new_val != orig_str:
                print(f"  Mapped '{orig_str}' -> '{new_val}' ({count} entries)")
                mapping_changed_count += count
        elif orig_str.startswith("Other: "):
            cleaned_val = orig_str[7:]
            if cleaned_val in type_mapping:
                new_val = type_mapping[cleaned_val]
                print(f"  Removed 'Other: ' prefix and mapped '{orig_str}' -> '{new_val}' ({count} entries)")
                mapping_changed_count += count
                other_removed_count += count
    
    if mapping_changed_count > 0:
        print(f"Total dataset type entries mapped: {mapping_changed_count}")
    else:
        print("No dataset type values were mapped")
    
    return df


def add_high_level_dataset_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add dataset_type_high_level column based on dataset_types_mapping_high_level in config.
    
    Args:
        df (pd.DataFrame): DataFrame with dataset_type column
        
    Returns:
        pd.DataFrame: DataFrame with added dataset_type_high_level column
    """
    if 'dataset_type' not in df.columns:
        print("Warning: dataset_type column not found. Adding empty dataset_type_high_level column.")
        df['dataset_type_high_level'] = 'Other'
        return df
    
    # Load configuration
    config = load_datasets_config()
    
    if 'dataset_types_mapping_high_level' not in config:
        print("Warning: dataset_types_mapping_high_level not found in config. Filling with 'Other'.")
        df['dataset_type_high_level'] = 'Other'
        return df
    
    # Create mapping from detailed types to high-level types
    high_level_mapping = {}
    for high_level_type, detailed_types in config['dataset_types_mapping_high_level'].items():
        for detailed_type in detailed_types:
            high_level_mapping[detailed_type] = high_level_type
    
    print(f"High-level dataset type mapping loaded with {len(high_level_mapping)} mappings")
    
    # Apply the mapping
    df['dataset_type_high_level'] = df['dataset_type'].map(high_level_mapping).fillna('Other')
    
    # Report the mapping results
    high_level_counts = df['dataset_type_high_level'].value_counts()
    mapped_count = len(df) - (df['dataset_type_high_level'] == 'Other').sum()
    unspecified_count = (df['dataset_type_high_level'] == 'Other').sum()
    
    print(f"High-level dataset type distribution:")
    for high_level_type, count in high_level_counts.items():
        print(f"  {high_level_type}: {count} entries")
    
    print(f"Total entries mapped to high-level types: {mapped_count}")
    print(f"Total entries marked as 'Other': {unspecified_count}")
    
    return df


def extract_dataset_columns(df: pd.DataFrame, reference_column: str = 'title') -> pd.DataFrame:
    """
    Extract dataset-related columns from a DataFrame and create a reference column.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing dataset information
        reference_column (str): Column name to use as reference (default: 'title')
        
    Returns:
        pd.DataFrame: DataFrame with extracted dataset columns and reference
    """
    # Define the dataset columns to extract
    dataset_columns = [
        'dataset_source',
        'dataset_notes', 
        'dataset_type',
        'dataset_language',
        'dataset_contains_synthetic_data',
        'dataset_is_public',
        'dataset_user_psychopathology_status',
        'dataset_responder_type'
    ]
    
    # Check if reference column exists
    if reference_column not in df.columns:
        raise ValueError(f"Reference column '{reference_column}' not found in DataFrame")
    
    # Check which dataset columns exist in the DataFrame
    available_columns = [col for col in dataset_columns if col in df.columns]
    missing_columns = [col for col in dataset_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in input data: {missing_columns}")
    
    if not available_columns:
        raise ValueError("No dataset columns found in input DataFrame")
    
    # Extract the columns
    extracted_df = df[[reference_column] + available_columns].copy()
    
    # Rename the reference column to 'reference'
    extracted_df = extracted_df.rename(columns={reference_column: 'reference'})
    
    # Filter out rows where dataset_source is "No dataset used for development or evaluation"
    initial_count = len(extracted_df)
    if 'dataset_source' in extracted_df.columns:
        extracted_df = extracted_df[
            extracted_df['dataset_source'] != "No dataset used for development or evaluation"
        ]
        filtered_count = initial_count - len(extracted_df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} rows with 'No dataset used for development or evaluation'")
    
    print(f"Extracted {len(available_columns)} dataset columns from {len(extracted_df)} rows (after filtering)")
    
    return extracted_df


def merge_dataset_files(large_df: pd.DataFrame, small_df: pd.DataFrame, 
                       reference_column: str = 'title') -> pd.DataFrame:
    """
    Merge dataset information from a large DataFrame with a smaller dataset file.
    
    Args:
        large_df (pd.DataFrame): Large DataFrame to extract dataset info from
        small_df (pd.DataFrame): Small DataFrame with existing dataset info
        reference_column (str): Column name to use as reference from large_df
        
    Returns:
        pd.DataFrame: Merged DataFrame containing both datasets
    """
    # Extract dataset columns from large DataFrame
    extracted_df = extract_dataset_columns(large_df, reference_column)
    
    # Check if small_df has 'reference' column
    if 'reference' not in small_df.columns:
        raise ValueError("Small dataset must have a 'reference' column")
    
    # Get common columns between the two DataFrames
    common_columns = set(extracted_df.columns) & set(small_df.columns)
    print(f"Common columns for merging: {list(common_columns)}")
    
    # Concatenate the DataFrames
    merged_df = pd.concat([extracted_df, small_df], ignore_index=True)
    
    # Add a unique row ID column since one article can have multiple datasets
    merged_df['row_id'] = range(1, len(merged_df) + 1)
    
    print(f"Merged datasets: {len(extracted_df)} + {len(small_df)} = {len(merged_df)} total entries (keeping all - one article can have multiple datasets)")
    
    return merged_df


def extract_and_merge_datasets(input_large_path: str, input_small_path: str, 
                              output_path: str, reference_column: str = 'title') -> str:
    """
    Complete workflow to extract dataset information and merge with another dataset.
    
    Args:
        input_large_path (str): Path to the large CSV file
        input_small_path (str): Path to the small dataset CSV file  
        output_path (str): Path for the output merged CSV file
        reference_column (str): Column name to use as reference from large file
        
    Returns:
        str: Path to the output file
    """
    print("Starting dataset extraction and merging workflow")
    
    # Load the input files
    print(f"Loading large dataset from: {input_large_path}")
    large_df = pd.read_csv(input_large_path)
    
    print(f"Loading small dataset from: {input_small_path}")
    small_df = pd.read_csv(input_small_path)
    
    # Perform the merge
    merged_df = merge_dataset_files(large_df, small_df, reference_column)
    
    # Add dataset IDs
    print("Adding dataset IDs based on dataset_notes matching...")
    merged_df = add_dataset_ids(merged_df)
    
    # Reduce dataset by collapsing rows with same dataset_id
    print("Reducing dataset by collapsing rows with same dataset_id...")
    reduced_df = reduce_dataset_by_id(merged_df)
    
    # Merge with common datasets to overwrite columns
    data_root = Path(output_path).parent.parent  # This gives us the data/ directory
    common_datasets_path = data_root / "auxiliary" / "common_datasets.csv"
    
    if common_datasets_path.exists():
        print("Merging with common datasets...")
        final_df = merge_with_common_datasets(reduced_df, str(common_datasets_path))
    else:
        print(f"Warning: Common datasets file not found at {common_datasets_path}")
        final_df = reduced_df
    
    # Rename dataset types according to mapping
    print("Renaming dataset types according to mapping...")
    final_df = rename_dataset_types(final_df)
    
    # Add high-level dataset types
    print("Adding high-level dataset types...")
    final_df = add_high_level_dataset_types(final_df)
    
    # Save the result
    print(f"Saving reduced dataset to: {output_path}")
    final_df.to_csv(output_path, index=True)
    
    print("Dataset extraction and merging completed successfully")
    
    return output_path


def generate_report(final_df: pd.DataFrame, output_path: str) -> str:
    """
    Generate a report about the dataset extraction, merging, and reduction process.
    
    Args:
        final_df (pd.DataFrame): The final reduced DataFrame with dataset_id as index
        output_path (str): Path where the CSV was saved
        
    Returns:
        str: Report text
    """
    report_lines = [
        "Dataset Extraction, Merging, and Reduction Report",
        "=" * 50,
        "",
        f"Output file: {output_path}",
        f"Total unique datasets: {len(final_df)}",
        "",
        "Column summary:",
    ]
    
    # Include index in column summary
    all_columns = ['dataset_id (index)'] + list(final_df.columns)
    for col in all_columns:
        if col == 'dataset_id (index)':
            non_null_count = len(final_df)
            null_count = 0
        else:
            non_null_count = final_df[col].notna().sum()
            null_count = final_df[col].isna().sum()
        report_lines.append(f"  {col}: {non_null_count} non-null, {null_count} null")
    
    report_lines.extend([
        "",
        "Dataset source distribution:",
    ])
    
    if 'dataset_source' in final_df.columns:
        source_counts = final_df['dataset_source'].value_counts()
        for source, count in source_counts.items():
            report_lines.append(f"  {source}: {count}")
    else:
        report_lines.append("  dataset_source column not available")
    
    report_lines.extend([
        "",
        "Dataset ID distribution (all unique):",
    ])
    
    # Since dataset_id is now the index, list all unique IDs
    dataset_ids = final_df.index.tolist()
    for dataset_id in sorted(dataset_ids):
        report_lines.append(f"  {dataset_id}: 1")
    
    report_lines.extend([
        "",
        "References summary:",
    ])
    
    if 'reference_count' in final_df.columns:
        # Use the reference_count column for more accurate counting
        total_refs = final_df['reference_count'].sum()
        report_lines.append(f"  Total studies referenced: {total_refs}")
        
        # Show references per dataset using reference_count
        for dataset_id in sorted(dataset_ids):
            ref_count = final_df.loc[dataset_id, 'reference_count']
            if ref_count > 0:
                report_lines.append(f"  {dataset_id}: {ref_count} studies")
    elif 'references' in final_df.columns:
        # Fallback to old method if reference_count not available
        total_refs = 0
        for refs in final_df['references'].dropna():
            if refs:
                total_refs += len(refs.split('; '))
        report_lines.append(f"  Total studies referenced: {total_refs}")
        
        # Show references per dataset
        for dataset_id in sorted(dataset_ids):
            refs = final_df.loc[dataset_id, 'references']
            if pd.notna(refs) and refs:
                ref_count = len(refs.split('; '))
                report_lines.append(f"  {dataset_id}: {ref_count} studies")
    else:
        report_lines.append("  references and reference_count columns not available")
    
    return "\n".join(report_lines)
