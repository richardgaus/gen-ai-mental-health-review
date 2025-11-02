"""
Model cleaning and classification functions for data preprocessing.

This module handles cleaning and standardizing the "Models Employed" column,
including extracting models from "Other:" entries and classifying them as
generative or non-generative.
"""

import pandas as pd
import re
from typing import List, Tuple
from .model_definitions import NON_GENERATIVE_MODELS


def clean_models_employed_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the "Models Employed" column.
    
    This function:
    1. Splits semicolon-separated entries
    2. Extracts model names from "Other: model a, model b" format
    3. Rejoins all models with semicolons
    4. Adds a binary "non_generative_models_only" column
    
    Args:
        df (pd.DataFrame): Input dataframe with "Models Employed" column
        
    Returns:
        pd.DataFrame: Processed dataframe with cleaned models column and new binary column
        
    Raises:
        KeyError: If "Models Employed" column is not found
    """
    if 'Models Employed' not in df.columns:
        raise KeyError("Column 'Models Employed' not found in dataframe")
    
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Clean the Models Employed column
    df_processed['Models Employed'] = df_processed['Models Employed'].apply(clean_model_entry)
    
    # Add binary column for non-generative models only
    df_processed['non_generative_models_only'] = df_processed['Models Employed'].apply(
        lambda x: is_non_generative_only(x) if pd.notna(x) else False
    )
    
    return df_processed


def clean_model_entry(entry: str) -> str:
    """
    Clean a single Models Employed entry.
    
    Args:
        entry (str): The original model entry (may be semicolon-separated)
        
    Returns:
        str: Cleaned model entry with standardized format
    """
    if pd.isna(entry) or not entry.strip():
        return entry
    
    # Split by semicolons first
    model_parts = [part.strip() for part in str(entry).split(';') if part.strip()]
    
    cleaned_models = []
    
    for part in model_parts:
        # Extract models from this part
        extracted_models = extract_models_from_part(part)
        cleaned_models.extend(extracted_models)
    
    # Remove duplicates while preserving order
    unique_models = []
    seen = set()
    for model in cleaned_models:
        if model.lower() not in seen:
            unique_models.append(model)
            seen.add(model.lower())
    
    return '; '.join(unique_models) if unique_models else ''


def extract_models_from_part(part: str) -> List[str]:
    """
    Extract model names from a single part of the Models Employed entry.
    
    Handles formats like:
    - "GPT-4 family"
    - "Other: model a, model b"
    - "Other: model a; model b"
    
    Args:
        part (str): A single part of the model entry
        
    Returns:
        List[str]: List of extracted model names
    """
    part = part.strip()
    if not part:
        return []
    
    # Check if this is an "Other:" entry
    other_pattern = re.compile(r'^Other:\s*(.+)', re.IGNORECASE)
    match = other_pattern.match(part)
    
    if match:
        # Extract the content after "Other:"
        other_content = match.group(1).strip()
        
        # Split by common delimiters (comma, semicolon, "and")
        models = re.split(r'[,;]|\sand\s', other_content)
        
        # Clean each model name
        cleaned_models = []
        for model in models:
            model = model.strip()
            if model:
                # Remove common prefixes/suffixes that aren't part of model names
                model = re.sub(r'\s*\(.*?\)\s*$', '', model)  # Remove parenthetical notes at end
                model = model.strip()
                if model and model.lower() not in ['etc', 'etc.', 'among others']:
                    cleaned_models.append(model)
        
        return cleaned_models
    else:
        # Regular model entry (not "Other:")
        return [part]


def is_non_generative_only(models_entry: str) -> bool:
    """
    Check if all models in the entry are non-generative.
    
    Args:
        models_entry (str): Semicolon-separated model names
        
    Returns:
        bool: True if all models are non-generative, False otherwise
    """
    if pd.isna(models_entry) or not models_entry.strip():
        return False
    
    # Split by semicolons
    models = [model.strip() for model in models_entry.split(';') if model.strip()]
    
    if not models:
        return False
    
    # Check if all models are non-generative
    for model in models:
        if not is_model_non_generative(model):
            return False
    
    return True


def is_model_non_generative(model_name: str) -> bool:
    """
    Check if a single model is non-generative.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if the model is non-generative, False otherwise
    """
    if not model_name or pd.isna(model_name):
        return False
    
    model_lower = model_name.lower().strip()
    
    # Check against known non-generative models (case-insensitive)
    for non_gen_model in NON_GENERATIVE_MODELS:
        if non_gen_model.lower() in model_lower or model_lower in non_gen_model.lower():
            return True
    
    # Additional patterns for non-generative models
    non_generative_patterns = [
        r'\bbert\b',
        r'\broberta\b', 
        r'\bdistilbert\b',
        r'\bvader\b',
        r'\bsentiment\s*analysis\b',
        r'\bclassifier\b',
        r'\bembedding\b',
        r'\bsvm\b',
        r'\brandom\s*forest\b',
        r'\blogistic\s*regression\b',
        r'\bnaive\s*bayes\b',
        r'\bdecision\s*tree\b',
        r'\bxgboost\b',
        r'\blightgbm\b',
        r'\bcnn\b',
        r'\blstm\b',
        r'\brnn\b',
        r'\bmlp\b',
        r'\bknn\b',
        r'\bk-nearest\b'
    ]
    
    for pattern in non_generative_patterns:
        if re.search(pattern, model_lower):
            return True
    
    return False


def get_model_cleaning_summary(df_original: pd.DataFrame, df_processed: pd.DataFrame) -> dict:
    """
    Generate a summary of the model cleaning process.
    
    Args:
        df_original (pd.DataFrame): Original dataframe
        df_processed (pd.DataFrame): Processed dataframe
        
    Returns:
        dict: Summary statistics
    """
    # Count non-null entries
    original_non_null = df_original['Models Employed'].notna().sum()
    processed_non_null = df_processed['Models Employed'].notna().sum()
    
    # Count non-generative only studies
    non_gen_only_count = df_processed['non_generative_models_only'].sum()
    
    # Count entries that were modified (simplified check)
    modified_count = 0
    for i, (orig, proc) in enumerate(zip(df_original['Models Employed'], df_processed['Models Employed'])):
        if pd.notna(orig) and pd.notna(proc) and str(orig).strip() != str(proc).strip():
            modified_count += 1
    
    return {
        'total_rows': len(df_processed),
        'original_non_null_models': original_non_null,
        'processed_non_null_models': processed_non_null,
        'entries_modified': modified_count,
        'non_generative_only_studies': non_gen_only_count,
        'generative_studies': processed_non_null - non_gen_only_count
    }
