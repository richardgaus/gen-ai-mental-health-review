"""
ID management functions for data preprocessing.

This module handles the assignment of unique identifiers and management
of ID columns in the research dataset.
"""

import pandas as pd
from typing import Tuple


def assign_unique_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign unique study IDs and clean up ID columns.
    
    This function:
    1. Creates a unique 'study_id' for each study in the dataset
    2. Removes the original 'Study ID' column
    3. Renames 'Covidence #' to 'covidence_id'
    
    Args:
        df (pd.DataFrame): Input dataframe with 'Covidence #' and 'Study ID' columns
        
    Returns:
        pd.DataFrame: Processed dataframe with unique study IDs and cleaned columns
        
    Raises:
        KeyError: If required columns ('Covidence #', 'Study ID') are not found
        ValueError: If dataframe is empty
    """
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Check for required columns
    required_columns = ['Covidence #', 'Study ID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Create unique study IDs based on row index
    # This ensures each study gets a unique identifier regardless of duplicates
    df_processed['study_id'] = range(1, len(df_processed) + 1)
    
    # Rename 'Covidence #' to 'covidence_id'
    df_processed = df_processed.rename(columns={'Covidence #': 'covidence_id'})
    
    # Remove the original 'Study ID' column
    df_processed = df_processed.drop(columns=['Study ID'])
    
    # Reorder columns to put study_id first, then covidence_id
    cols = ['study_id', 'covidence_id'] + [col for col in df_processed.columns 
                                          if col not in ['study_id', 'covidence_id']]
    df_processed = df_processed[cols]
    
    return df_processed


def get_unique_studies_summary(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Get summary statistics about unique studies in the dataset.
    
    Args:
        df (pd.DataFrame): Dataframe with 'covidence_id' column
        
    Returns:
        Tuple[int, int]: (total_rows, unique_covidence_ids)
    """
    if 'covidence_id' not in df.columns:
        raise KeyError("Column 'covidence_id' not found in dataframe")
    
    total_rows = len(df)
    unique_covidence_ids = df['covidence_id'].nunique()
    
    return total_rows, unique_covidence_ids
