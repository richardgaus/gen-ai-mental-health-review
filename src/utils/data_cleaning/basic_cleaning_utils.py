"""
Basic data cleaning utility functions for LLMs in Psychotherapy research.

This module contains utility functions for performing basic cleaning operations on CSV files including:
- Loading cleaning configuration
- Filling empty values with "n"
- Converting columns to lowercase
- Dropping columns not in configuration
- Cleaning "Other: " fields
- Setting covidence_id as index
"""

import sys
from pathlib import Path
from typing import Union, Tuple, Dict, List
import pandas as pd
import yaml

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_cleaning_config(config_path: str = None) -> dict:
    """
    Load cleaning configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file. If None, uses default path.
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Default path relative to src directory
        config_path = Path(__file__).parent / "basic_cleaning_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error: Could not load cleaning configuration from {config_path}: {e}")
        sys.exit(1)


def load_column_rename_config(config_path: str = None) -> Dict[str, str]:
    """
    Load column rename configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file. If None, uses default path.
        
    Returns:
        Dict[str, str]: Dictionary mapping old column names to new column names
    """
    config = load_cleaning_config(config_path)
    return config.get('rename_columns', {})


def load_column_order_config(config_path: str = None) -> List[str]:
    """
    Load column order configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file. If None, uses default path.
        
    Returns:
        List[str]: List of column names in desired order
    """
    config = load_cleaning_config(config_path)
    return config.get('column_order', [])


def normalize_unicode_characters(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalize Unicode characters in column names to regular ASCII equivalents.
    
    This function targets READI columns that may have:
    - Unicode hyphens (‑) → regular dash (-)
    - Non-breaking spaces (U+00A0) → regular space
    - Other Unicode dash-like and space-like characters
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Dataframe with normalized column names and list of changed columns
    """
    df_processed = df.copy()
    
    # Define various Unicode characters to normalize
    unicode_replacements = [
        # Dash-like characters → regular ASCII dash
        ('‑', '-'),  # Unicode hyphen (U+2011)
        ('–', '-'),  # En dash (U+2013) 
        ('—', '-'),  # Em dash (U+2014)
        ('−', '-'),  # Minus sign (U+2212)
        ('⁻', '-'),  # Superscript minus (U+207B)
        
        # Space-like characters → regular ASCII space
        ('\u00A0', ' '),  # Non-breaking space (U+00A0)
        ('\u2000', ' '),  # En quad (U+2000)
        ('\u2001', ' '),  # Em quad (U+2001)
        ('\u2002', ' '),  # En space (U+2002)
        ('\u2003', ' '),  # Em space (U+2003)
        ('\u2004', ' '),  # Three-per-em space (U+2004)
        ('\u2005', ' '),  # Four-per-em space (U+2005)
        ('\u2006', ' '),  # Six-per-em space (U+2006)
        ('\u2007', ' '),  # Figure space (U+2007)
        ('\u2008', ' '),  # Punctuation space (U+2008)
        ('\u2009', ' '),  # Thin space (U+2009)
        ('\u200A', ' '),  # Hair space (U+200A)
        ('\u202F', ' '),  # Narrow no-break space (U+202F)
        ('\u205F', ' '),  # Medium mathematical space (U+205F)
        ('\u3000', ' '),  # Ideographic space (U+3000)
    ]
    
    changed_columns = []
    column_mapping = {}
    
    for old_col in df.columns:
        new_col = old_col
        
        # Apply all Unicode character replacements
        for unicode_char, replacement in unicode_replacements:
            if unicode_char in new_col:
                new_col = new_col.replace(unicode_char, replacement)
        
        # If column name changed, track it
        if new_col != old_col:
            changed_columns.append(f"{old_col} → {new_col}")
            column_mapping[old_col] = new_col
    
    # Apply column renaming
    if column_mapping:
        df_processed = df_processed.rename(columns=column_mapping)
        
        if verbose:
            print(f"✓ Normalized Unicode characters in {len(column_mapping)} column names:")
            for old_col, new_col in column_mapping.items():
                print(f"  - {old_col}")
                print(f"    → {new_col}")
    else:
        if verbose:
            print("✓ No Unicode character normalization needed")
    
    return df_processed, changed_columns


def rename_columns_from_config(df: pd.DataFrame, config_path: str = None, verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Rename columns according to the specification in basic_cleaning_config.yaml.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config_path (str): Path to configuration file
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Dataframe with renamed columns and renaming info
    """
    # Load rename mapping
    rename_mapping = load_column_rename_config(config_path)
    
    if not rename_mapping:
        if verbose:
            print("✓ No column rename configuration found, keeping current names")
        return df.copy(), {'renamed': False, 'reason': 'No config found'}
    
    # Get current columns
    current_columns = list(df.columns)
    
    # Create the rename mapping for columns that exist in the dataframe
    actual_rename_mapping = {}
    for old_name, new_name in rename_mapping.items():
        if old_name in current_columns:
            actual_rename_mapping[old_name] = new_name
    
    # Apply the renaming
    df_renamed = df.rename(columns=actual_rename_mapping)
    
    renaming_info = {
        'renamed': True,
        'total_columns': len(current_columns),
        'columns_renamed': len(actual_rename_mapping),
        'columns_not_found': len(rename_mapping) - len(actual_rename_mapping),
        'rename_mapping': actual_rename_mapping
    }
    
    if verbose:
        print(f"✓ Renamed columns according to configuration:")
        print(f"  - Total columns: {renaming_info['total_columns']}")
        print(f"  - Columns renamed: {renaming_info['columns_renamed']}")
        print(f"  - Columns in config but not found: {renaming_info['columns_not_found']}")
        
        if actual_rename_mapping:
            print(f"  - Sample renames:")
            sample_renames = list(actual_rename_mapping.items())[:5]
            for old_name, new_name in sample_renames:
                print(f"    • {old_name} → {new_name}")
            if len(actual_rename_mapping) > 5:
                print(f"    ... and {len(actual_rename_mapping) - 5} more")
    
    return df_renamed, renaming_info


def reorder_columns(df: pd.DataFrame, config_path: str = None, verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Reorder columns according to the specification in basic_cleaning_config.yaml.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config_path (str): Path to configuration file
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Dataframe with reordered columns and reordering info
    """
    # Load desired column order
    desired_order = load_column_order_config(config_path)
    
    if not desired_order:
        if verbose:
            print("✓ No column ordering configuration found, keeping current order")
        return df.copy(), {'reordered': False, 'reason': 'No config found'}
    
    # Get current columns
    current_columns = list(df.columns)
    
    # Create the new column order
    # Start with columns that are in the desired order AND present in the dataframe
    ordered_columns = []
    for col in desired_order:
        if col in current_columns:
            ordered_columns.append(col)
    
    # Add any remaining columns that weren't in the desired order at the end
    remaining_columns = [col for col in current_columns if col not in ordered_columns]
    final_order = ordered_columns + remaining_columns
    
    # Reorder the dataframe
    df_reordered = df[final_order]
    
    reordering_info = {
        'reordered': True,
        'total_columns': len(current_columns),
        'ordered_by_config': len(ordered_columns),
        'remaining_columns': len(remaining_columns),
        'columns_not_in_config': remaining_columns
    }
    
    if verbose:
        print(f"✓ Reordered columns according to configuration:")
        print(f"  - Total columns: {reordering_info['total_columns']}")
        print(f"  - Ordered by config: {reordering_info['ordered_by_config']}")
        print(f"  - Remaining columns: {reordering_info['remaining_columns']}")
        
        if remaining_columns:
            print(f"  - Columns not in config (added at end):")
            for col in remaining_columns[:5]:  # Show first 5
                print(f"    • {col}")
            if len(remaining_columns) > 5:
                print(f"    ... and {len(remaining_columns) - 5} more")
    
    return df_reordered, reordering_info


def fill_empty_with_n(df: pd.DataFrame, columns_to_fill: list, verbose: bool = False, return_filled: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """
    Fill empty values in specified columns with "n".
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_fill (list): List of column names to fill
        verbose (bool): Enable verbose logging
        
    Returns:
        pd.DataFrame: Dataframe with filled values
    """
    df_processed = df.copy()
    filled_count = 0
    columns_filled = []
    
    for column in columns_to_fill:
        if column in df_processed.columns:
            # Count empty values before filling
            empty_mask = df_processed[column].isna() | (df_processed[column].astype(str).str.strip() == '')
            empty_count = empty_mask.sum()
            
            if empty_count > 0:
                df_processed.loc[empty_mask, column] = 'n'
                filled_count += empty_count
                columns_filled.append(column)
                
                if verbose:
                    print(f"  - {column}: filled {empty_count} empty values with 'n'")
        elif verbose:
            print(f"  - Warning: Column '{column}' not found in dataframe")
    
    if verbose:
        print(f"✓ Filled {filled_count} empty values with 'n' across {len(columns_to_fill)} specified columns")
    
    if return_filled:
        return df_processed, columns_filled
    else:
        return df_processed


def lowercase_columns(df: pd.DataFrame, columns_to_lowercase: list, verbose: bool = False, return_processed: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """
    Convert specified columns to lowercase.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns_to_lowercase (list): List of column names to lowercase
        verbose (bool): Enable verbose logging
        
    Returns:
        pd.DataFrame: Dataframe with lowercased values
    """
    df_processed = df.copy()
    processed_count = 0
    columns_processed = []
    
    for column in columns_to_lowercase:
        if column in df_processed.columns:
            # Convert to string and lowercase, preserving NaN values
            original_values = df_processed[column].copy()
            df_processed[column] = df_processed[column].astype(str).str.lower()
            
            # Restore NaN values (they become 'nan' strings after str conversion)
            nan_mask = original_values.isna()
            df_processed.loc[nan_mask, column] = pd.NA
            
            processed_count += 1
            columns_processed.append(column)
            
            if verbose:
                print(f"  - {column}: converted to lowercase")
        elif verbose:
            print(f"  - Warning: Column '{column}' not found in dataframe")
    
    if verbose:
        print(f"✓ Converted {processed_count} columns to lowercase")
    
    if return_processed:
        return df_processed, columns_processed
    else:
        return df_processed


def drop_remaining_columns(df: pd.DataFrame, config_path: str = None, verbose: bool = False, return_dropped: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """
    Drop columns that are not in the reordering configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config_path (str): Path to configuration file
        verbose (bool): Enable verbose logging
        
    Returns:
        pd.DataFrame: Dataframe with only configured columns
    """
    if config_path is None:
        # Default path relative to src directory
        config_path = Path(__file__).parent / "basic_cleaning_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        desired_order = config.get('column_order', [])
        if not desired_order:
            if verbose:
                print("✓ No column ordering configuration found, keeping all columns")
            return df.copy()
        
        # Get current columns
        current_columns = list(df.columns)
        
        # Find columns to keep (those in the configuration)
        columns_to_keep = [col for col in desired_order if col in current_columns]
        
        # Find columns to drop (those not in the configuration)
        columns_to_drop = [col for col in current_columns if col not in desired_order]
        
        # Drop the unwanted columns
        df_processed = df[columns_to_keep].copy()
        
        if verbose:
            print(f"✓ Dropped columns not in configuration:")
            print(f"  - Total original columns: {len(current_columns)}")
            print(f"  - Columns kept: {len(columns_to_keep)}")
            print(f"  - Columns dropped: {len(columns_to_drop)}")
            if columns_to_drop:
                print(f"  - Dropped columns:")
                for col in columns_to_drop:
                    print(f"    • {col}")
        
        if return_dropped:
            return df_processed, columns_to_drop
        else:
            return df_processed
        
    except Exception as e:
        print(f"Warning: Could not load column order configuration: {e}")
        if return_dropped:
            return df.copy(), []
        else:
            return df.copy()


def clean_other_fields(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Clean fields that contain only "Other: " string by setting them to empty.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        pd.DataFrame: Dataframe with "Other: " fields cleaned
    """
    df_processed = df.copy()
    cleaned_count = 0
    
    # Iterate through all columns and rows to find "Other: " values
    for column in df_processed.columns:
        # Find cells that contain exactly "Other: " (exact string match)
        other_mask = df_processed[column].astype(str) == 'Other: '
        other_count = other_mask.sum()
        
        if other_count > 0:
            # Set these cells to NaN (empty)
            df_processed.loc[other_mask, column] = pd.NA
            cleaned_count += other_count
            
            if verbose:
                print(f"  - {column}: cleaned {other_count} 'Other: ' entries")
    
    if verbose:
        print(f"✓ Cleaned {cleaned_count} 'Other: ' entries across the dataset")
    
    return df_processed


def set_covidence_id_as_index(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Set covidence_id as the index column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        pd.DataFrame: Dataframe with covidence_id as index
    """
    if 'covidence_id' not in df.columns:
        raise KeyError("Column 'covidence_id' not found in dataframe")
    
    df_processed = df.set_index('covidence_id')
    
    if verbose:
        print(f"✓ Set 'covidence_id' as index column")
        print(f"  - Index name: {df_processed.index.name}")
        print(f"  - Index size: {len(df_processed.index)}")
    
    return df_processed


def perform_basic_cleaning_pipeline(df: pd.DataFrame, config_path: str = None, verbose: bool = False, return_details: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Perform the complete basic cleaning pipeline on a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config_path (str): Path to configuration file
        verbose (bool): Enable verbose logging
        return_details (bool): If True, return processing details along with dataframe
        
    Returns:
        pd.DataFrame or Tuple[pd.DataFrame, Dict]: Fully cleaned dataframe, optionally with processing details
    """
    # Load cleaning configuration
    if verbose:
        print("Loading cleaning configuration...")
    
    config = load_cleaning_config(config_path)
    fill_columns = config.get('fill_with_n_if_empty', [])
    lowercase_columns_list = config.get('disregard_case', [])
    
    if verbose:
        print(f"✓ Configuration loaded:")
        print(f"  - Columns to fill with 'n': {len(fill_columns)}")
        print(f"  - Columns to lowercase: {len(lowercase_columns_list)}")
    
    # Step 1: Normalize Unicode characters in column names
    if verbose:
        print("\nStep 1: Normalizing Unicode characters in column names...")
    
    df_normalized, normalized_columns = normalize_unicode_characters(df, verbose=verbose)
    
    # Step 2: Rename columns according to configuration
    if verbose:
        print("\nStep 2: Renaming columns according to configuration...")
    
    df_renamed, rename_info = rename_columns_from_config(df_normalized, verbose=verbose)
    
    # Step 3: Reorder columns according to configuration
    if verbose:
        print("\nStep 3: Reordering columns according to configuration...")
    
    df_reordered, reorder_info = reorder_columns(df_renamed, verbose=verbose)
    
    # Step 4: Drop columns not in configuration
    if verbose:
        print("\nStep 4: Dropping columns not in configuration...")
    
    df_filtered, dropped_columns = drop_remaining_columns(df_reordered, config_path, verbose=verbose, return_dropped=True)
    
    # Step 5: Clean 'Other: ' fields
    if verbose:
        print("\nStep 5: Cleaning 'Other: ' fields...")
    
    df_cleaned_other = clean_other_fields(df_filtered, verbose=verbose)
    
    # Step 6: Set covidence_id as index
    if verbose:
        print("\nStep 6: Setting covidence_id as index...")
    
    df_indexed = set_covidence_id_as_index(df_cleaned_other, verbose=verbose)
    
    # Step 7: Fill specified fields with "n" if empty
    if verbose:
        print(f"\nStep 7: Filling {len(fill_columns)} specified fields with 'n' if empty...")
    
    df_filled, columns_filled = fill_empty_with_n(df_indexed, fill_columns, verbose=verbose, return_filled=True)
    
    # Step 8: Lowercase specified fields
    if verbose:
        print(f"\nStep 8: Converting {len(lowercase_columns_list)} specified fields to lowercase...")
    
    df_final, columns_lowercased = lowercase_columns(df_filled, lowercase_columns_list, verbose=verbose, return_processed=True)
    
    # Collect processing details if requested
    if return_details:
        processing_details = {
            'columns_dropped': dropped_columns,
            'columns_filled_with_n': columns_filled,
            'columns_lowercased': columns_lowercased
        }
        return df_final, processing_details
    else:
        return df_final
