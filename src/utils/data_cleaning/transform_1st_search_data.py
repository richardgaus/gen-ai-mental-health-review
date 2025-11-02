"""
1st search data transformation functions.

This module handles the transformation of 1st search datasets to match
the structure of 2nd search data, including column renaming and value standardization.
"""

import pandas as pd
from typing import Tuple


def transform_first_search_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Transform 1st search data to match the structure of 2nd search data.
    
    This function handles column renaming, value transformations, row filtering,
    and column additions to standardize the datasets.
    
    Args:
        df (pd.DataFrame): Input dataframe from 1st search
        
    Returns:
        tuple: (transformed_dataframe, dropped_rows_info)
        
    Raises:
        KeyError: If required columns are not found
    """
    # Create a copy to avoid modifying the original dataframe
    df_transformed = df.copy()
    
    # Step 1: Drop "Study Type II" column
    df_transformed = drop_study_type_ii_column(df_transformed)
    
    # Step 2: Handle the Development Approach column transformation
    df_transformed = transform_development_approach_column(df_transformed)

    # Step 3: Transform Study Type column and filter rows
    df_transformed, dropped_rows_info = transform_study_type_column(df_transformed)
    
    # Step 4: Rename Application Type column
    df_transformed = rename_application_type_column(df_transformed)
    
    # Step 5: Drop unwanted columns
    df_transformed = drop_unwanted_columns(df_transformed)
    
    # Step 6: Add missing columns from 2nd search structure
    df_transformed = add_missing_columns(df_transformed)
    
    return df_transformed, dropped_rows_info


def transform_development_approach_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the Development Approach column from 1st search format to 2nd search format.
    
    Column transformation:
    - Rename: "If Study Type == Tool Development and Evaluation: Development Approach" 
      to: "If Study Type == Empirical research involving an LLM: Development Approach"
    
    Value transformations:
    1. If Study Type is "Direct LLM performance evaluation (only via prompting)" → change to "Only prompting"
    2. If value is "Only fine-tuning" → keep as "Only fine-tuning"
    3. If value is "Custom pipeline with integrated LLM which is only prompted" → change to "Prompting + other modules"
    4. Otherwise → keep existing value
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Transformed dataframe
    """
    df_transformed = df.copy()
    
    # Define column names
    old_dev_approach_col = "If Study Type == Tool Development and Evaluation: Development Approach"
    new_dev_approach_col = "If Study Type == Empirical research involving an LLM: Development Approach"
    study_type_col = "Study Type"
    
    # Check if required columns exist
    if old_dev_approach_col not in df.columns:
        print(f"Warning: Column '{old_dev_approach_col}' not found. Skipping transformation.")
        return df_transformed
    
    if study_type_col not in df.columns:
        print(f"Warning: Column '{study_type_col}' not found. Using only direct value transformations.")
        # Still proceed with basic transformations if Study Type column is missing
    
    # Step 1: Rename the column
    df_transformed = df_transformed.rename(columns={old_dev_approach_col: new_dev_approach_col})
    
    # Step 2: Transform values
    def transform_development_approach_value(row):
        """Transform individual development approach values."""
        dev_approach = row[new_dev_approach_col] if pd.notna(row[new_dev_approach_col]) else ""
        study_type = row.get(study_type_col, "") if study_type_col in df.columns else ""
        
        # Convert to strings for comparison
        dev_approach_str = str(dev_approach).strip()
        study_type_str = str(study_type).strip()
        
        # Transformation logic
        if study_type_str == "Direct LLM performance evaluation (only via prompting)":
            return "Only prompting"
        elif dev_approach_str == "Only fine-tuning":
            return "Only fine-tuning"
        elif dev_approach_str == "Custom pipeline with integrated LLM which is only prompted":
            return "Prompting + other modules"
        else:
            # Keep existing value (including NaN)
            return dev_approach
    
    # Apply transformation
    df_transformed[new_dev_approach_col] = df_transformed.apply(transform_development_approach_value, axis=1)
    
    return df_transformed


def drop_study_type_ii_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the "Study Type II" column if it exists.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with "Study Type II" column removed
    """
    df_processed = df.copy()
    
    if "Study Type II" in df_processed.columns:
        df_processed = df_processed.drop(columns=["Study Type II"])
        print(f"✓ Dropped 'Study Type II' column")
    else:
        print(f"Warning: 'Study Type II' column not found")
    
    return df_processed


def transform_study_type_column(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Transform Study Type values and filter rows based on study type.
    
    Transformations:
    - "Tool Development and Evaluation" or "Direct LLM performance evaluation (only via prompting)" 
      → "Empirical research involving an LLM"
    - "Population survey" → keep as is
    - Drop ENTIRE STUDIES (all rows for a study) ONLY if the Consensus reviewer marks them as:
      • "Opinion, commentary, perspective, correspondence"
      • "Conceptual or theoretical work (e.g. on ethics or safety)"
      • "Review (systematic or other)"
    - Individual reviewer assessments (Richard Gaus, Reviewer Two) do not trigger study removal
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (transformed_dataframe, dropped_rows_info)
    """
    if "Study Type" not in df.columns:
        print(f"Warning: 'Study Type' column not found. Skipping transformation.")
        return df.copy(), {"dropped_rows": [], "dropped_studies": []}
    
    if "Reviewer Name" not in df.columns:
        print(f"Warning: 'Reviewer Name' column not found. Cannot filter by consensus reviewer.")
        return df.copy(), {"dropped_rows": [], "dropped_studies": []}
    
    df_processed = df.copy()
    original_count = len(df_processed)
    
    # Define study types to transform to "Empirical research involving an LLM"
    empirical_types = [
        "Tool Development and Evaluation",
        "Direct LLM performance evaluation (only via prompting)"
    ]
    
    # Transform study types (but don't filter yet)
    def transform_study_type_value(study_type):
        if pd.isna(study_type):
            return study_type
        
        study_type_str = str(study_type).strip()
        
        if study_type_str in empirical_types:
            return "Empirical research involving an LLM"
        elif study_type_str == "Population survey":
            return "Population survey"
        else:
            return study_type_str
    
    # Apply transformations
    df_processed["Study Type"] = df_processed["Study Type"].apply(transform_study_type_value)
    
    # Define study types that should be dropped when marked by Consensus reviewer
    drop_types = [
        "Opinion, commentary, perspective, correspondence",
        "Conceptual or theoretical work (e.g. on ethics or safety)",
        "Review (systematic or other)"
    ]
    
    # Find studies where Consensus reviewer marks them as any non-empirical type
    consensus_drop_mask = (
        (df_processed["Reviewer Name"] == "Consensus") & 
        (df_processed["Study Type"].isin(drop_types))
    )
    
    # Get the covidence_ids of studies that should be dropped
    studies_to_drop = df_processed[consensus_drop_mask]["covidence_id"].unique()
    
    # Collect information about dropped rows before removing them
    dropped_rows_info = {"dropped_rows": [], "dropped_studies": []}
    
    if len(studies_to_drop) > 0:
        # Get all rows that will be dropped (all reviewers for the dropped studies)
        rows_to_drop_mask = df_processed["covidence_id"].isin(studies_to_drop)
        dropped_rows_df = df_processed[rows_to_drop_mask].copy()
        
        # Collect detailed information about dropped rows
        for _, row in dropped_rows_df.iterrows():
            dropped_row_info = {
                "covidence_id": row["covidence_id"],
                "title": row.get("Title", "N/A"),
                "reviewer_name": row["Reviewer Name"],
                "study_type": row["Study Type"]
            }
            dropped_rows_info["dropped_rows"].append(dropped_row_info)
        
        # Collect study-level information
        for study_id in studies_to_drop:
            study_rows = dropped_rows_df[dropped_rows_df["covidence_id"] == study_id]
            consensus_row = study_rows[study_rows["Reviewer Name"] == "Consensus"]
            if not consensus_row.empty:
                study_info = {
                    "covidence_id": study_id,
                    "title": consensus_row.iloc[0].get("Title", "N/A"),
                    "consensus_study_type": consensus_row.iloc[0]["Study Type"],
                    "total_rows_dropped": len(study_rows)
                }
                dropped_rows_info["dropped_studies"].append(study_info)
        
        # Remove ALL rows (all reviewers) for these studies
        df_processed = df_processed[~df_processed["covidence_id"].isin(studies_to_drop)].reset_index(drop=True)
    
    final_count = len(df_processed)
    dropped_count = original_count - final_count
    studies_dropped = len(studies_to_drop)
    
    print(f"✓ Study Type column transformed")
    print(f"  - Studies dropped (where Consensus marked as non-empirical): {studies_dropped}")
    print(f"  - Rows dropped: {dropped_count}")
    print(f"  - Remaining rows: {final_count}")
    
    if studies_dropped > 0:
        print(f"  - Dropped study IDs: {list(studies_to_drop)}")
        # Show which study types caused the drops
        dropped_study_types = [study["consensus_study_type"] for study in dropped_rows_info["dropped_studies"]]
        if dropped_study_types:
            print(f"  - Consensus study types that caused drops: {list(set(dropped_study_types))}")
    
    return df_processed, dropped_rows_info


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add missing columns that exist in 2nd search data but not in 1st search data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with additional empty columns
    """
    df_processed = df.copy()
    
    # Define new columns to add (all empty/NaN)
    new_columns = [
        "Perplexity Used (Y/N)",
        "Perplexity How it compares against benchmark (B/S/W)",
        "Perplexity Benchmark quality (H/L)",
        "Perplexity Notes on benchmark quality",
        "Lexical diversity Used (Y/N)",
        "Lexical diversity How it compares against benchmark (B/S/W)",
        "Lexical diversity Benchmark quality (H/L)",
        "Lexical diversity Notes on benchmark quality"
    ]
    
    # Add columns that don't already exist
    added_columns = []
    for col in new_columns:
        if col not in df_processed.columns:
            df_processed[col] = pd.NA  # Add as pandas NA (empty)
            added_columns.append(col)
    
    if added_columns:
        print(f"✓ Added {len(added_columns)} new columns:")
        for col in added_columns:
            print(f"  - {col}")
    else:
        print(f"All required columns already exist")
    
    return df_processed


def get_transformation_summary(df_original: pd.DataFrame, df_transformed: pd.DataFrame) -> dict:
    """
    Generate a summary of the transformation process.
    
    Args:
        df_original (pd.DataFrame): Original dataframe
        df_transformed (pd.DataFrame): Transformed dataframe
        
    Returns:
        dict: Summary statistics
    """
    old_col = "If Study Type == Tool Development and Evaluation: Development Approach"
    new_col = "If Study Type == Empirical research involving an LLM: Development Approach"
    
    summary = {
        'original_rows': len(df_original),
        'final_rows': len(df_transformed),
        'rows_dropped': len(df_original) - len(df_transformed),
        'original_columns': len(df_original.columns),
        'final_columns': len(df_transformed.columns),
        'columns_added': len(df_transformed.columns) - len(df_original.columns),
        'columns_renamed': 0,
        'values_transformed': 0,
        'transformation_details': {}
    }
    
    # Check if column was renamed
    if old_col in df_original.columns and new_col in df_transformed.columns:
        summary['columns_renamed'] = 1
        
        # Count value transformations (need to handle different row counts due to filtering)
        if new_col in df_transformed.columns and old_col in df_original.columns:
            # Create a mapping based on study_id or row index for comparison
            if 'study_id' in df_original.columns and 'study_id' in df_transformed.columns:
                # Use study_id for matching
                orig_mapping = df_original.set_index('study_id')[old_col].fillna('').astype(str)
                trans_mapping = df_transformed.set_index('study_id')[new_col].fillna('').astype(str)
                
                # Find common study_ids
                common_ids = orig_mapping.index.intersection(trans_mapping.index)
                
                if len(common_ids) > 0:
                    orig_subset = orig_mapping.loc[common_ids]
                    trans_subset = trans_mapping.loc[common_ids]
                    
                    # Count changes
                    changes = (orig_subset != trans_subset).sum()
                    summary['values_transformed'] = changes
                    
                    # Detail the transformations
                    transformation_counts = {}
                    for study_id in common_ids:
                        orig_val = orig_subset.loc[study_id]
                        trans_val = trans_subset.loc[study_id]
                        if orig_val != trans_val and orig_val.strip() != '' and trans_val.strip() != '':
                            key = f"{orig_val} → {trans_val}"
                            transformation_counts[key] = transformation_counts.get(key, 0) + 1
                    
                    summary['transformation_details'] = transformation_counts
    
    return summary


def validate_transformation(df_transformed: pd.DataFrame) -> dict:
    """
    Validate the transformation results.
    
    Args:
        df_transformed (pd.DataFrame): Transformed dataframe
        
    Returns:
        dict: Validation results
    """
    new_col = "If Study Type == Empirical research involving an LLM: Development Approach"
    
    validation = {
        'has_new_column': new_col in df_transformed.columns,
        'column_stats': {},
        'sample_values': []
    }
    
    if validation['has_new_column']:
        col_data = df_transformed[new_col]
        validation['column_stats'] = {
            'total_values': len(col_data),
            'non_null_values': col_data.notna().sum(),
            'null_values': col_data.isna().sum(),
            'unique_values': col_data.nunique()
        }
        
        # Get sample values
        sample_values = col_data.dropna().unique()[:10]
        validation['sample_values'] = list(sample_values)
    
    return validation


def rename_application_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the Application Type column to match the structure of 2nd search data.
    
    Column transformation:
    - Rename: "If Experimental Research or Population Survey: Application Type" 
      to: "Application Type"
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Transformed dataframe
    """
    df_transformed = df.copy()
    
    # Define column names
    old_col = "If Experimental Research or Population Survey: Application Type"
    new_col = "Application Type"
    
    # Check if the old column exists
    if old_col in df.columns:
        df_transformed = df_transformed.rename(columns={old_col: new_col})
        print(f"✓ Renamed '{old_col}' to '{new_col}'")
    else:
        print(f"Warning: Column '{old_col}' not found. Skipping rename.")
    
    return df_transformed


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unwanted columns that should not appear in the final dataset.
    
    Columns to drop:
    - Main Safety-Related Discussion Contents
    - P-1 Protection of user information Considered in tool design? (Y/N)
    - P-1 Protection of user information if YES: Notes (paste text passage)
    - READI P-2:
    - Safety-Related Discussion
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with unwanted columns removed
    """
    df_transformed = df.copy()
    
    # Define columns to drop
    columns_to_drop = [
        "Main Safety-Related Discussion Contents",
        "P-1 Protection of user information Considered in tool design? (Y/N)",
        "P-1 Protection of user information if YES: Notes (paste text passage)",
        "READI P-2:",
        "Safety-Related Discussion"
    ]
    
    # Drop columns that exist
    dropped_columns = []
    for col in columns_to_drop:
        if col in df_transformed.columns:
            df_transformed = df_transformed.drop(columns=[col])
            dropped_columns.append(col)
    
    if dropped_columns:
        print(f"✓ Dropped {len(dropped_columns)} unwanted columns:")
        for col in dropped_columns:
            print(f"  - {col}")
    else:
        print(f"No unwanted columns found to drop")
    
    return df_transformed
