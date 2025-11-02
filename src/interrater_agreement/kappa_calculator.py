"""
Cohen's Kappa calculation functions for interrater agreement.

This module provides functions to calculate Cohen's kappa coefficient
for measuring agreement between two reviewers.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import cohen_kappa_score
import yaml
from pathlib import Path
from collections import Counter


def load_kappa_config(config_path: str = None) -> Tuple[List[str], Dict[str, List[str]], List[str]]:
    """
    Load Cohen's kappa configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file. If None, uses default path.
        
    Returns:
        Tuple[List[str], Dict[str, List[str]], List[str]]: Single-choice columns, multi-choice columns with options, and numerical columns
    """
    if config_path is None:
        # Default path relative to this file
        config_path = Path(__file__).parent / "metrics.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load single-choice categorical columns (renamed from cohens_kappa)
        single_choice_columns = config.get('categorical_single_choice', [])
        multi_choice_columns = config.get('categorical_multi_choice', {})
        numerical_columns = config.get('numerical', [])
        return single_choice_columns, multi_choice_columns, numerical_columns
    except Exception as e:
        print(f"Warning: Could not load Cohen's kappa configuration from {config_path}: {e}")
        return [], {}, []


def get_conditional_dependencies() -> Dict[str, str]:
    """
    Get mapping of conditional columns to their parent columns.
    
    Returns:
        Dict[str, str]: Mapping of dependent column to parent column
    """
    dependencies = {
        # Application subtype columns depend on application_type agreement
        'application_subtype_client_facing': 'application_type',
        'application_subtype_therapist_facing': 'application_type',
        
        # Dataset columns depend on dataset_source agreement
        'dataset_type': 'dataset_source',
        'dataset_language': 'dataset_source',
        'dataset_contains_synthetic_data': 'dataset_source',
        'dataset_is_public': 'dataset_source',
        'dataset_user_psychopathology_status': 'dataset_source',
        'dataset_responder_type': 'dataset_source',
        
        # UX columns depend on ux_assessment_is_present agreement
        'ux_uses_standard_instrument': 'ux_assessment_is_present',
        'ux_uses_qualitative_assessment': 'ux_assessment_is_present',
        'ux_uses_quantitative_assessment': 'ux_assessment_is_present',
        'ux_results_reported': 'ux_assessment_is_present',
    }
    
    # Add metric dependencies dynamically
    metric_bases = [
        'lexical_overlap', 'embedding_similarity', 'classification', 'continuous_metrics',
        'expert_rating', 'llm_judge', 'perplexity', 'lexical_diversity',
        'metric1', 'metric2', 'metric3'
    ]
    
    for base in metric_bases:
        used_col = f'{base}_used'
        dependencies[f'{base}_vs_benchmark'] = used_col
        dependencies[f'{base}_benchmark_quality'] = used_col
    
    return dependencies


def prepare_reviewer_data(df: pd.DataFrame, column: str, verbose: bool = False) -> Tuple[pd.Series, pd.Series, int]:
    """
    Prepare data for two reviewers for a specific column, applying conditional logic if needed.
    
    Args:
        df (pd.DataFrame): Input dataframe with reviewer data
        column (str): Column name to analyze
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.Series, pd.Series, int]: Reviewer 1 data, Reviewer 2 data, number of valid pairs
    """
    if 'reviewer_name' not in df.columns:
        raise KeyError("Column 'reviewer_name' not found in dataframe")
    
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")
    
    # Get unique reviewers (excluding 'Consensus')
    reviewers = df[df['reviewer_name'] != 'Consensus']['reviewer_name'].unique()
    
    if len(reviewers) < 2:
        if verbose:
            print(f"  - Warning: Only {len(reviewers)} non-consensus reviewers found for {column}")
        return pd.Series(dtype=object), pd.Series(dtype=object), 0
    
    # Take the first two reviewers
    reviewer1_name = reviewers[0]
    reviewer2_name = reviewers[1]
    
    if verbose:
        print(f"  - Comparing {reviewer1_name} vs {reviewer2_name}")
    
    # Get data for each reviewer
    reviewer1_data = df[df['reviewer_name'] == reviewer1_name]
    reviewer2_data = df[df['reviewer_name'] == reviewer2_name]
    
    # Find common studies (by index if covidence_id is index, or by covidence_id column)
    if df.index.name == 'covidence_id':
        common_studies = reviewer1_data.index.intersection(reviewer2_data.index)
        r1_values = reviewer1_data.loc[common_studies, column]
        r2_values = reviewer2_data.loc[common_studies, column]
    else:
        # Merge on covidence_id if it's a column
        if 'covidence_id' in df.columns:
            merged = pd.merge(
                reviewer1_data[['covidence_id', column]], 
                reviewer2_data[['covidence_id', column]], 
                on='covidence_id', 
                suffixes=('_r1', '_r2')
            )
            r1_values = merged[f'{column}_r1']
            r2_values = merged[f'{column}_r2']
        else:
            raise KeyError("Neither index 'covidence_id' nor column 'covidence_id' found")
    
    # Remove pairs where either reviewer has missing data
    valid_mask = r1_values.notna() & r2_values.notna()
    
    # Apply conditional logic if this column depends on another
    dependencies = get_conditional_dependencies()
    if column in dependencies:
        parent_column = dependencies[column]
        
        if verbose:
            print(f"    - Applying conditional logic: {column} depends on {parent_column}")
        
        if parent_column in df.columns:
            # Get parent column data for both reviewers
            if df.index.name == 'covidence_id':
                common_studies = r1_values.index.intersection(r2_values.index)
                parent_r1 = reviewer1_data.loc[common_studies, parent_column]
                parent_r2 = reviewer2_data.loc[common_studies, parent_column]
            else:
                if 'covidence_id' in df.columns:
                    merged_parent = pd.merge(
                        reviewer1_data[['covidence_id', parent_column]], 
                        reviewer2_data[['covidence_id', parent_column]], 
                        on='covidence_id', 
                        suffixes=('_r1', '_r2')
                    )
                    parent_r1 = merged_parent[f'{parent_column}_r1']
                    parent_r2 = merged_parent[f'{parent_column}_r2']
                else:
                    raise KeyError("Cannot apply conditional logic without covidence_id")
            
            # Only include pairs where both reviewers agreed on the parent column
            parent_agreement_mask = (
                parent_r1.notna() & parent_r2.notna() & 
                (parent_r1.astype(str) == parent_r2.astype(str))
            )
            
            # Combine with existing valid mask
            conditional_mask = valid_mask & parent_agreement_mask
            
            if verbose:
                original_pairs = valid_mask.sum()
                conditional_pairs = conditional_mask.sum()
                print(f"    - Original valid pairs: {original_pairs}")
                print(f"    - After conditional filtering: {conditional_pairs}")
            
            valid_mask = conditional_mask
        else:
            if verbose:
                print(f"    - Warning: Parent column '{parent_column}' not found, skipping conditional logic")
    
    r1_clean = r1_values[valid_mask]
    r2_clean = r2_values[valid_mask]
    
    return r1_clean, r2_clean, len(r1_clean)


def parse_multi_choice_response(response: str, valid_options: List[str]) -> List[str]:
    """
    Parse a multi-choice response string into individual selections.
    Disregards any "Other: xxx" responses.
    
    Args:
        response (str): The response string (e.g., "Option A; Option B")
        valid_options (List[str]): List of valid option strings
        
    Returns:
        List[str]: List of selected options (excluding "Other:" responses)
    """
    if pd.isna(response) or response == '':
        return []
    
    # Split by semicolon and clean up
    selections = [s.strip() for s in str(response).split(';')]
    
    # Filter out "Other:" responses
    selections = [s for s in selections if not is_other_response(s)]
    
    # Filter to only valid options (case-insensitive matching)
    valid_selections = []
    for selection in selections:
        for option in valid_options:
            if selection.lower().strip() == option.lower().strip():
                valid_selections.append(option)
                break
    
    return valid_selections


def create_onehot_encoding(reviewer1_data: pd.Series, reviewer2_data: pd.Series, 
                          valid_options: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one-hot encodings for multi-choice categorical data.
    
    Args:
        reviewer1_data (pd.Series): Data from reviewer 1
        reviewer2_data (pd.Series): Data from reviewer 2
        valid_options (List[str]): List of valid options for this column
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: One-hot encoded data for both reviewers
    """
    if len(reviewer1_data) != len(reviewer2_data):
        raise ValueError("Reviewer data series must have the same length")
    
    n_rows = len(reviewer1_data)
    
    # Initialize one-hot matrices
    r1_onehot = pd.DataFrame(0, index=reviewer1_data.index, columns=valid_options)
    r2_onehot = pd.DataFrame(0, index=reviewer2_data.index, columns=valid_options)
    
    # Fill one-hot encodings
    for idx in reviewer1_data.index:
        r1_selections = parse_multi_choice_response(reviewer1_data.loc[idx], valid_options)
        r2_selections = parse_multi_choice_response(reviewer2_data.loc[idx], valid_options)
        
        # Set 1s for selected options
        for option in r1_selections:
            if option in valid_options:
                r1_onehot.loc[idx, option] = 1
        
        for option in r2_selections:
            if option in valid_options:
                r2_onehot.loc[idx, option] = 1
    
    return r1_onehot, r2_onehot


def calculate_multi_choice_kappa(reviewer1_data: pd.Series, reviewer2_data: pd.Series,
                                valid_options: List[str], column_name: str = "", 
                                verbose: bool = False) -> Dict:
    """
    Calculate Cohen's kappa for multi-choice categorical data using pooled one-hot encoding.
    Instead of calculating kappa for each option separately, this pools all response options
    and computes a single kappa on the pooled data.
    
    Args:
        reviewer1_data (pd.Series): Data from reviewer 1
        reviewer2_data (pd.Series): Data from reviewer 2
        valid_options (List[str]): List of valid options for this column
        column_name (str): Name of the column being analyzed
        verbose (bool): Enable verbose logging
        
    Returns:
        Dict: Dictionary containing pooled kappa and statistics
    """
    if len(reviewer1_data) == 0 or len(reviewer2_data) == 0:
        return {
            'column': column_name,
            'overall_kappa': np.nan,
            'n_pairs': 0,
            'n_options': len(valid_options),
            'status': 'no_data'
        }
    
    # Apply "Other:" exclusion logic - but this is now handled in parse_multi_choice_response
    # We still exclude pairs where BOTH reviewers have only "Other:" responses
    both_other_mask = (
        reviewer1_data.astype(str).apply(is_other_response) & 
        reviewer2_data.astype(str).apply(is_other_response)
    )
    excluded_other_pairs = both_other_mask.sum()
    
    # Filter out the "both Other:" pairs
    valid_mask = ~both_other_mask
    r1_filtered = reviewer1_data[valid_mask]
    r2_filtered = reviewer2_data[valid_mask]
    
    if len(r1_filtered) == 0:
        return {
            'column': column_name,
            'overall_kappa': np.nan,
            'n_pairs': 0,
            'n_options': len(valid_options),
            'excluded_other_pairs': excluded_other_pairs,
            'status': 'no_valid_pairs_after_other_exclusion'
        }
    
    try:
        # Create one-hot encodings (now "Other:" responses are filtered out in parsing)
        r1_onehot, r2_onehot = create_onehot_encoding(r1_filtered, r2_filtered, valid_options)
        
        # Pool all response options into single arrays
        # Each study contributes multiple binary decisions (one per option)
        pooled_r1 = []
        pooled_r2 = []
        
        for option in valid_options:
            pooled_r1.extend(r1_onehot[option].tolist())
            pooled_r2.extend(r2_onehot[option].tolist())
        
        # Calculate single kappa on pooled data
        overall_kappa = cohen_kappa_score(pooled_r1, pooled_r2)
        
        result = {
            'column': column_name,
            'overall_kappa': overall_kappa,
            'n_pairs': len(r1_filtered),
            'n_options': len(valid_options),
            'n_pooled_decisions': len(pooled_r1),
            'excluded_other_pairs': excluded_other_pairs,
            'status': 'success'
        }
        
        if verbose:
            print(f"  - Multi-choice kappa for {column_name}:")
            print(f"    - Valid study pairs: {len(r1_filtered)}")
            print(f"    - Options analyzed: {len(valid_options)}")
            print(f"    - Pooled binary decisions: {len(pooled_r1)}")
            print(f"    - Pooled kappa: {overall_kappa:.3f}")
            if excluded_other_pairs > 0:
                print(f"    - Excluded 'Other:' pairs: {excluded_other_pairs}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  - Error calculating multi-choice kappa for {column_name}: {e}")
        return {
            'column': column_name,
            'overall_kappa': np.nan,
            'n_pairs': len(r1_filtered),
            'n_options': len(valid_options),
            'excluded_other_pairs': excluded_other_pairs,
            'status': f'calculation_error: {str(e)}'
        }


def calculate_gwets_ac1(reviewer1_data: pd.Series, reviewer2_data: pd.Series) -> float:
    """
    Calculate Gwet's AC1 coefficient for agreement between two reviewers.
    
    Gwet's AC1 is less sensitive to trait prevalence than Cohen's kappa and
    provides a more stable measure of interrater agreement.
    
    Args:
        reviewer1_data (pd.Series): Data from reviewer 1
        reviewer2_data (pd.Series): Data from reviewer 2
        
    Returns:
        float: Gwet's AC1 coefficient
    """
    if len(reviewer1_data) != len(reviewer2_data):
        raise ValueError("Reviewer data series must have the same length")
    
    if len(reviewer1_data) == 0:
        return np.nan
    
    # Convert to string for consistent handling
    r1_str = reviewer1_data.astype(str)
    r2_str = reviewer2_data.astype(str)
    
    # Get all unique categories
    all_categories = set(r1_str) | set(r2_str)
    n_categories = len(all_categories)
    n_pairs = len(r1_str)
    
    if n_categories <= 1:
        return 1.0  # Perfect agreement if only one category
    
    # Calculate observed agreement
    observed_agreement = (r1_str == r2_str).sum() / n_pairs
    
    # Calculate expected agreement by chance (Gwet's approach)
    # For AC1, the chance agreement is based on uniform distribution
    # across categories, not marginal probabilities
    chance_agreement = 1.0 / n_categories
    
    # Calculate AC1
    if chance_agreement == 1.0:
        return 1.0 if observed_agreement == 1.0 else 0.0
    
    ac1 = (observed_agreement - chance_agreement) / (1.0 - chance_agreement)
    
    return ac1


def is_other_response(value: str) -> bool:
    """
    Check if a value is an "Other:" response.
    
    Args:
        value (str): The value to check
        
    Returns:
        bool: True if the value starts with "Other:"
    """
    if pd.isna(value):
        return False
    return str(value).strip().startswith('Other:')


def calculate_cohens_kappa(reviewer1_data: pd.Series, reviewer2_data: pd.Series, 
                          column_name: str = "", verbose: bool = False) -> Dict:
    """
    Calculate Cohen's kappa and Gwet's AC1 coefficients between two reviewers for a specific column.
    Excludes pairs where both reviewers selected "Other:" responses.
    
    Args:
        reviewer1_data (pd.Series): Data from reviewer 1
        reviewer2_data (pd.Series): Data from reviewer 2
        column_name (str): Name of the column being analyzed
        verbose (bool): Enable verbose logging
        
    Returns:
        Dict: Dictionary containing kappa, AC1, and related statistics
    """
    if len(reviewer1_data) == 0 or len(reviewer2_data) == 0:
        return {
            'column': column_name,
            'kappa': np.nan,
            'ac1': np.nan,
            'n_pairs': 0,
            'agreement_rate': np.nan,
            'status': 'no_data',
            'excluded_other_pairs': 0
        }
    
    if len(reviewer1_data) != len(reviewer2_data):
        raise ValueError("Reviewer data series must have the same length")
    
    # Convert to string for exact matching (handles mixed types)
    r1_str = reviewer1_data.astype(str)
    r2_str = reviewer2_data.astype(str)
    
    # Exclude pairs where both reviewers selected "Other:" responses
    both_other_mask = (
        r1_str.apply(is_other_response) & 
        r2_str.apply(is_other_response)
    )
    excluded_other_pairs = both_other_mask.sum()
    
    # Filter out the "both Other:" pairs
    valid_mask = ~both_other_mask
    r1_filtered = r1_str[valid_mask]
    r2_filtered = r2_str[valid_mask]
    
    if verbose and excluded_other_pairs > 0:
        print(f"    - Excluded {excluded_other_pairs} pairs where both selected 'Other:' responses")
    
    if len(r1_filtered) == 0:
        return {
            'column': column_name,
            'kappa': np.nan,
            'ac1': np.nan,
            'n_pairs': 0,
            'agreement_rate': np.nan,
            'status': 'no_valid_pairs_after_other_exclusion',
            'excluded_other_pairs': excluded_other_pairs
        }
    
    # Calculate simple agreement rate (on filtered data)
    agreements = (r1_filtered == r2_filtered).sum()
    total_pairs = len(r1_filtered)
    agreement_rate = agreements / total_pairs if total_pairs > 0 else 0
    
    try:
        # Calculate Cohen's kappa (on filtered data)
        kappa = cohen_kappa_score(r1_filtered, r2_filtered)
        kappa_status = 'success'
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not calculate kappa for {column_name}: {e}")
        kappa = np.nan
        kappa_status = 'calculation_error'
    
    try:
        # Calculate Gwet's AC1 (on filtered data)
        ac1 = calculate_gwets_ac1(r1_filtered, r2_filtered)
        ac1_status = 'success'
    except Exception as e:
        if verbose:
            print(f"  - Warning: Could not calculate AC1 for {column_name}: {e}")
        ac1 = np.nan
        ac1_status = 'calculation_error'
    
    # Overall status
    if kappa_status == 'success' or ac1_status == 'success':
        status = 'success'
    else:
        status = 'calculation_error'
    
    result = {
        'column': column_name,
        'kappa': kappa,
        'ac1': ac1,
        'n_pairs': total_pairs,
        'agreement_rate': agreement_rate,
        'agreements': agreements,
        'status': status,
        'is_conditional': False,
        'parent_column': None,
        'excluded_other_pairs': excluded_other_pairs
    }
    
    if verbose:
        original_pairs = len(reviewer1_data)
        ac1_str = f", AC1: {ac1:.3f}" if not pd.isna(ac1) else ", AC1: N/A"
        print(f"  - Original pairs: {original_pairs}, After Other: exclusion: {total_pairs}, Agreements: {agreements}, Rate: {agreement_rate:.3f}, Kappa: {kappa:.3f}{ac1_str}")
    
    return result


def calculate_overall_agreement(df: pd.DataFrame, columns: List[str], 
                               verbose: bool = False) -> Dict:
    """
    Calculate overall percent agreement across all specified columns.
    
    Args:
        df (pd.DataFrame): Input dataframe with reviewer data
        columns (List[str]): List of column names to analyze
        verbose (bool): Enable verbose logging
        
    Returns:
        Dict: Overall agreement statistics
    """
    if 'reviewer_name' not in df.columns:
        raise KeyError("Column 'reviewer_name' not found in dataframe")
    
    # Get unique reviewers (excluding 'Consensus')
    reviewers = df[df['reviewer_name'] != 'Consensus']['reviewer_name'].unique()
    
    if len(reviewers) < 2:
        return {
            'total_comparisons': 0,
            'total_agreements': 0,
            'overall_agreement_rate': 0.0,
            'columns_analyzed': 0,
            'status': 'insufficient_reviewers'
        }
    
    # Take the first two reviewers
    reviewer1_name = reviewers[0]
    reviewer2_name = reviewers[1]
    
    if verbose:
        print(f"\n🔍 Calculating overall agreement between {reviewer1_name} and {reviewer2_name}")
    
    total_comparisons = 0
    total_agreements = 0
    columns_analyzed = 0
    total_excluded_other = 0
    
    # For overall kappa and AC1 calculation
    all_r1_values = []
    all_r2_values = []
    
    for column in columns:
        if column not in df.columns:
            if verbose:
                print(f"  - Skipping {column}: not found in dataframe")
            continue
        
        try:
            # Get data for both reviewers for this column
            r1_data, r2_data, n_pairs = prepare_reviewer_data(df, column, verbose=False)
            
            if n_pairs == 0:
                if verbose:
                    print(f"  - Skipping {column}: no valid pairs")
                continue
            
            # Count agreements for this column, excluding "Other:" pairs
            r1_str = r1_data.astype(str)
            r2_str = r2_data.astype(str)
            
            # Exclude pairs where both reviewers selected "Other:" responses
            both_other_mask = (
                r1_str.apply(is_other_response) & 
                r2_str.apply(is_other_response)
            )
            excluded_other_pairs = both_other_mask.sum()
            
            # Filter out the "both Other:" pairs
            valid_mask = ~both_other_mask
            r1_filtered = r1_str[valid_mask]
            r2_filtered = r2_str[valid_mask]
            
            if len(r1_filtered) == 0:
                if verbose:
                    print(f"  - Skipping {column}: no valid pairs after Other: exclusion")
                continue
            
            agreements = (r1_filtered == r2_filtered).sum()
            valid_pairs = len(r1_filtered)
            
            total_comparisons += valid_pairs
            total_agreements += agreements
            total_excluded_other += excluded_other_pairs
            columns_analyzed += 1
            
            # Collect data for overall kappa and AC1
            all_r1_values.extend(r1_filtered.tolist())
            all_r2_values.extend(r2_filtered.tolist())
            
            if verbose:
                agreement_rate = agreements / valid_pairs if valid_pairs > 0 else 0
                other_info = f" (excluded {excluded_other_pairs} 'Other:' pairs)" if excluded_other_pairs > 0 else ""
                print(f"  - {column}: {agreements}/{valid_pairs} ({agreement_rate:.1%}){other_info}")
                
        except Exception as e:
            if verbose:
                print(f"  - Error processing {column}: {e}")
            continue
    
    overall_agreement_rate = total_agreements / total_comparisons if total_comparisons > 0 else 0.0
    
    # Calculate overall Cohen's kappa and AC1
    overall_kappa = np.nan
    overall_ac1 = np.nan
    
    if len(all_r1_values) > 0 and len(all_r2_values) > 0:
        try:
            # Convert to pandas Series for consistency with other functions
            r1_series = pd.Series(all_r1_values)
            r2_series = pd.Series(all_r2_values)
            
            # Calculate overall Cohen's kappa
            overall_kappa = cohen_kappa_score(r1_series, r2_series)
            
            # Calculate overall Gwet's AC1
            overall_ac1 = calculate_gwets_ac1(r1_series, r2_series)
            
        except Exception as e:
            if verbose:
                print(f"  - Warning: Could not calculate overall kappa/AC1: {e}")
    
    if verbose:
        print(f"\n📊 Overall Agreement Summary:")
        print(f"  - Columns analyzed: {columns_analyzed}/{len(columns)}")
        print(f"  - Total comparisons: {total_comparisons:,}")
        print(f"  - Total agreements: {total_agreements:,}")
        print(f"  - Total excluded 'Other:' pairs: {total_excluded_other:,}")
        print(f"  - Overall agreement rate: {overall_agreement_rate:.1%}")
        
        if not pd.isna(overall_ac1):
            print(f"  - Overall AC1: {overall_ac1:.3f}")
        if not pd.isna(overall_kappa):
            print(f"  - Overall Cohen's kappa: {overall_kappa:.3f}")
    
    return {
        'total_comparisons': total_comparisons,
        'total_agreements': total_agreements,
        'overall_agreement_rate': overall_agreement_rate,
        'overall_kappa': overall_kappa,
        'overall_ac1': overall_ac1,
        'columns_analyzed': columns_analyzed,
        'total_columns': len(columns),
        'reviewer1': reviewer1_name,
        'reviewer2': reviewer2_name,
        'total_excluded_other': total_excluded_other,
        'status': 'success'
    }


def calculate_kappa_for_columns(df: pd.DataFrame, columns: List[str], 
                               multi_choice_columns: Dict[str, List[str]] = None,
                               verbose: bool = False) -> Tuple[Dict[str, Dict], Dict, Dict[str, Dict]]:
    """
    Calculate Cohen's kappa for multiple columns, overall agreement, and multi-choice columns.
    
    Args:
        df (pd.DataFrame): Input dataframe with reviewer data
        columns (List[str]): List of regular column names to analyze
        multi_choice_columns (Dict[str, List[str]]): Multi-choice columns with their valid options
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[Dict[str, Dict], Dict, Dict[str, Dict]]: Regular results, overall stats, multi-choice results
    """
    results = {}
    multi_choice_results = {}
    
    if multi_choice_columns is None:
        multi_choice_columns = {}
    
    # Calculate overall agreement first (only for regular columns)
    overall_stats = calculate_overall_agreement(df, columns, verbose=verbose)
    
    for column in columns:
        if verbose:
            print(f"\nCalculating Cohen's kappa for: {column}")
        
        try:
            # Prepare data for the two reviewers
            r1_data, r2_data, n_pairs = prepare_reviewer_data(df, column, verbose=verbose)
            
            if n_pairs == 0:
                results[column] = {
                    'column': column,
                    'kappa': np.nan,
                    'n_pairs': 0,
                    'agreement_rate': np.nan,
                    'status': 'no_valid_pairs'
                }
                if verbose:
                    print(f"  - No valid pairs found for {column}")
                continue
            
            # Calculate kappa
            kappa_result = calculate_cohens_kappa(r1_data, r2_data, column, verbose=verbose)
            
            # Add conditional information
            dependencies = get_conditional_dependencies()
            if column in dependencies:
                kappa_result['is_conditional'] = True
                kappa_result['parent_column'] = dependencies[column]
            
            results[column] = kappa_result
            
        except Exception as e:
            if verbose:
                print(f"  - Error processing {column}: {e}")
            results[column] = {
                'column': column,
                'kappa': np.nan,
                'n_pairs': 0,
                'agreement_rate': np.nan,
                'status': f'error: {str(e)}'
            }
    
    # Calculate multi-choice kappa for specified columns
    if multi_choice_columns and verbose:
        print(f"\n🔀 Calculating multi-choice kappa for {len(multi_choice_columns)} columns...")
    
    for column, valid_options in (multi_choice_columns or {}).items():
        if verbose:
            print(f"\nCalculating multi-choice kappa for: {column}")
        
        try:
            # Prepare data for the two reviewers
            r1_data, r2_data, n_pairs = prepare_reviewer_data(df, column, verbose=verbose)
            
            if n_pairs == 0:
                multi_choice_results[column] = {
                    'column': column,
                    'overall_kappa': np.nan,
                    'n_pairs': 0,
                    'n_options': len(valid_options),
                    'status': 'no_valid_pairs'
                }
                if verbose:
                    print(f"  - No valid pairs found for {column}")
                continue
            
            # Calculate multi-choice kappa
            mc_result = calculate_multi_choice_kappa(r1_data, r2_data, valid_options, column, verbose=verbose)
            
            # Add conditional information
            dependencies = get_conditional_dependencies()
            if column in dependencies:
                mc_result['is_conditional'] = True
                mc_result['parent_column'] = dependencies[column]
            else:
                mc_result['is_conditional'] = False
                mc_result['parent_column'] = None
            
            multi_choice_results[column] = mc_result
            
        except Exception as e:
            if verbose:
                print(f"  - Error processing multi-choice column {column}: {e}")
            multi_choice_results[column] = {
                'column': column,
                'overall_kappa': np.nan,
                'n_pairs': 0,
                'n_options': len(valid_options) if valid_options else 0,
                'status': f'error: {str(e)}'
            }
    
    return results, overall_stats, multi_choice_results


def generate_kappa_report(results: Dict[str, Dict], overall_stats: Dict, 
                         multi_choice_results: Dict[str, Dict] = None, output_path: str = None,
                         single_choice_count: int = 0, numerical_count: int = 0) -> str:
    """
    Generate a text report of Cohen's kappa results.
    
    Args:
        results (Dict[str, Dict]): Results from calculate_kappa_for_columns
        output_path (str): Optional path to save the report
        
    Returns:
        str: The report content as a string
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("COHEN'S KAPPA INTERRATER AGREEMENT REPORT")
    report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Methodology section
    report_lines.append("METHODOLOGY AND CALCULATIONS")
    report_lines.append("=" * 40)
    report_lines.append("")
    
    report_lines.append("COLUMN TYPES ANALYZED:")
    report_lines.append(f"  • Single-choice categorical columns: {single_choice_count}")
    report_lines.append(f"  • Multi-choice categorical columns: {len(multi_choice_results)}")
    report_lines.append(f"  • Numerical columns: {numerical_count} (identified but not analyzed)")
    report_lines.append("")
    
    report_lines.append("STEP-BY-STEP ANALYSIS PROCESS:")
    report_lines.append("")
    
    report_lines.append("1. DATA PREPARATION:")
    report_lines.append("   • Loaded data with covidence_id as index")
    report_lines.append("   • Identified non-consensus reviewers for comparison")
    report_lines.append("   • Applied conditional logic for dependent fields")
    report_lines.append("")
    
    report_lines.append("2. SINGLE-CHOICE CATEGORICAL ANALYSIS:")
    report_lines.append("   • Excluded pairs where both reviewers selected 'Other:' responses")
    report_lines.append("   • Applied conditional filtering (e.g., subtype fields only when main type agreed)")
    report_lines.append("   • Calculated three agreement metrics for each field:")
    report_lines.append("     - Percent Agreement: Simple observed agreement rate")
    report_lines.append("     - Cohen's Kappa: Chance-corrected agreement coefficient")
    report_lines.append("     - Gwet's AC1: Robust agreement coefficient less affected by prevalence")
    report_lines.append("")
    
    report_lines.append("3. MULTI-CHOICE CATEGORICAL ANALYSIS:")
    report_lines.append("   • Parsed semicolon-separated responses (e.g., 'Option A; Option B')")
    report_lines.append("   • Filtered out all 'Other: xxx' responses during parsing")
    report_lines.append("   • Created one-hot encoding matrices (1 = selected, 0 = not selected)")
    report_lines.append("   • Pooled all binary decisions across options and studies")
    report_lines.append("   • Calculated single Cohen's kappa on pooled data")
    report_lines.append("   • Example: 100 studies × 5 options = 500 pooled binary decisions")
    report_lines.append("")
    
    report_lines.append("4. CONDITIONAL LOGIC APPLIED:")
    report_lines.append("   • application_subtype_* fields: Only when application_type agreed")
    report_lines.append("   • dataset_* fields: Only when dataset_source agreed")
    report_lines.append("   • ux_* fields: Only when ux_assessment_is_present agreed")
    report_lines.append("   • *_vs_benchmark & *_benchmark_quality: Only when *_used agreed")
    report_lines.append("")
    
    report_lines.append("5. EXCLUSION CRITERIA:")
    report_lines.append("   • Pairs where both reviewers selected any 'Other:' response")
    report_lines.append("   • Studies not meeting conditional requirements")
    report_lines.append("   • Missing or invalid data entries")
    report_lines.append("")
    
    report_lines.append("6. OVERALL METRICS CALCULATION:")
    report_lines.append("   • Pooled all valid field comparisons across studies")
    report_lines.append("   • Calculated overall agreement rate, AC1, and Cohen's kappa")
    report_lines.append("   • Applied same exclusion rules as individual fields")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    valid_kappa_results = [r for r in results.values() if not pd.isna(r['kappa'])]
    valid_ac1_results = [r for r in results.values() if not pd.isna(r['ac1'])]
    valid_agreement_results = [r for r in results.values() if not pd.isna(r['agreement_rate'])]
    total_columns = len(results)
    valid_kappa_columns = len(valid_kappa_results)
    valid_ac1_columns = len(valid_ac1_results)
    
    # Multi-choice statistics
    if multi_choice_results is None:
        multi_choice_results = {}
    valid_mc_results = [r for r in multi_choice_results.values() if not pd.isna(r['overall_kappa'])]
    total_mc_columns = len(multi_choice_results)
    valid_mc_columns = len(valid_mc_results)
    
    report_lines.append("ANALYSIS SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Single-choice categorical columns analyzed: {total_columns}")
    report_lines.append(f"Columns with valid kappa: {valid_kappa_columns}")
    report_lines.append(f"Columns with valid AC1: {valid_ac1_columns}")
    report_lines.append(f"Columns with valid agreement rates: {len(valid_agreement_results)}")
    if total_mc_columns > 0:
        report_lines.append(f"Multi-choice categorical columns analyzed: {total_mc_columns}")
        report_lines.append(f"Multi-choice columns with valid kappa: {valid_mc_columns}")
    report_lines.append("")
    
    # Overall agreement statistics
    if overall_stats.get('status') == 'success':
        report_lines.append("OVERALL AGREEMENT ACROSS ALL FIELDS:")
        report_lines.append(f"  Reviewers compared: {overall_stats['reviewer1']} vs {overall_stats['reviewer2']}")
        report_lines.append(f"  Total field comparisons: {overall_stats['total_comparisons']:,}")
        report_lines.append(f"  Total agreements: {overall_stats['total_agreements']:,}")
        report_lines.append(f"  Overall agreement rate: {overall_stats['overall_agreement_rate']:.1%}")
        
        # Overall AC1 and Kappa
        overall_ac1 = overall_stats.get('overall_ac1', np.nan)
        overall_kappa = overall_stats.get('overall_kappa', np.nan)
        if not pd.isna(overall_ac1):
            report_lines.append(f"  Overall Gwet's AC1: {overall_ac1:.3f}")
        if not pd.isna(overall_kappa):
            report_lines.append(f"  Overall Cohen's kappa: {overall_kappa:.3f}")
        
        report_lines.append(f"  Fields contributing: {overall_stats['columns_analyzed']}/{overall_stats['total_columns']}")
        if overall_stats.get('total_excluded_other', 0) > 0:
            report_lines.append(f"  Excluded 'Other:' pairs: {overall_stats['total_excluded_other']:,}")
        report_lines.append("")
    
    # Cohen's Kappa Statistics (field-by-field)
    if valid_kappa_results:
        kappa_values = [r['kappa'] for r in valid_kappa_results]
        report_lines.append("FIELD-BY-FIELD KAPPA STATISTICS:")
        report_lines.append(f"  Mean kappa per field: {np.mean(kappa_values):.3f}")
        report_lines.append(f"  Median kappa per field: {np.median(kappa_values):.3f}")
        report_lines.append(f"  Min kappa per field: {np.min(kappa_values):.3f}")
        report_lines.append(f"  Max kappa per field: {np.max(kappa_values):.3f}")
        report_lines.append("")
    
    # Gwet's AC1 Statistics (field-by-field)
    if valid_ac1_results:
        ac1_values = [r['ac1'] for r in valid_ac1_results]
        report_lines.append("FIELD-BY-FIELD AC1 STATISTICS:")
        report_lines.append(f"  Mean AC1 per field: {np.mean(ac1_values):.3f}")
        report_lines.append(f"  Median AC1 per field: {np.median(ac1_values):.3f}")
        report_lines.append(f"  Min AC1 per field: {np.min(ac1_values):.3f}")
        report_lines.append(f"  Max AC1 per field: {np.max(ac1_values):.3f}")
        report_lines.append("")
    
    # Percent Agreement Statistics (by field)
    if valid_agreement_results:
        agreement_values = [r['agreement_rate'] for r in valid_agreement_results]
        report_lines.append("FIELD-BY-FIELD AGREEMENT STATISTICS:")
        report_lines.append(f"  Mean agreement per field: {np.mean(agreement_values):.1%}")
        report_lines.append(f"  Median agreement per field: {np.median(agreement_values):.1%}")
        report_lines.append(f"  Min agreement per field: {np.min(agreement_values):.1%}")
        report_lines.append(f"  Max agreement per field: {np.max(agreement_values):.1%}")
        report_lines.append("")
    
        report_lines.append("")
    
    # Multi-choice results section
    if multi_choice_results:
        report_lines.append("MULTI-CHOICE CATEGORICAL RESULTS")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Column':<40} {'Overall':<8} {'Options':<8} {'Pairs':<6} {'Excl':<5} {'Conditional':<12} {'Status'}")
        report_lines.append("-" * 97)
        
        # Sort by overall kappa (descending, NaN last)
        sorted_mc_results = sorted(multi_choice_results.items(), 
                                 key=lambda x: (pd.isna(x[1]['overall_kappa']), -x[1]['overall_kappa'] if not pd.isna(x[1]['overall_kappa']) else 0))
        
        for column, result in sorted_mc_results:
            overall_kappa = result.get('overall_kappa', np.nan)
            n_pairs = result.get('n_pairs', 0)
            n_options = result.get('n_options', 0)
            status = result.get('status', 'unknown')
            
            kappa_str = f"{overall_kappa:.3f}" if not pd.isna(overall_kappa) else "N/A"
            conditional_str = f"→{result.get('parent_column', '')[:10]}" if result.get('is_conditional', False) else ""
            excluded_other = result.get('excluded_other_pairs', 0)
            excl_str = str(excluded_other) if excluded_other > 0 else ""
            
            report_lines.append(f"{column:<40} {kappa_str:<8} {n_options:<8} {n_pairs:<6} {excl_str:<5} {conditional_str:<12} {status}")
        
        report_lines.append("")
        
        # Multi-choice statistics
        if valid_mc_results:
            mc_kappa_values = [r['overall_kappa'] for r in valid_mc_results]
            report_lines.append("MULTI-CHOICE KAPPA STATISTICS:")
            report_lines.append(f"  Mean overall kappa: {np.mean(mc_kappa_values):.3f}")
            report_lines.append(f"  Median overall kappa: {np.median(mc_kappa_values):.3f}")
            report_lines.append(f"  Min overall kappa: {np.min(mc_kappa_values):.3f}")
            report_lines.append(f"  Max overall kappa: {np.max(mc_kappa_values):.3f}")
            report_lines.append("")
    
    # Detailed results
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Column':<40} {'Agree%':<8} {'AC1':<8} {'Kappa':<8} {'Pairs':<6} {'Excl':<5} {'Conditional':<12} {'Status'}")
    report_lines.append("-" * 105)
    
    # Sort by agreement rate (descending, NaN last)
    sorted_results = sorted(results.items(), 
                          key=lambda x: (pd.isna(x[1]['agreement_rate']), -x[1]['agreement_rate'] if not pd.isna(x[1]['agreement_rate']) else 0))
    
    for column, result in sorted_results:
        kappa = result['kappa']
        n_pairs = result['n_pairs']
        agreement_rate = result['agreement_rate']
        status = result['status']
        
        kappa_str = f"{kappa:.3f}" if not pd.isna(kappa) else "N/A"
        ac1 = result.get('ac1', np.nan)
        ac1_str = f"{ac1:.3f}" if not pd.isna(ac1) else "N/A"
        agree_str = f"{agreement_rate:.1%}" if not pd.isna(agreement_rate) else "N/A"
        conditional_str = f"→{result.get('parent_column', '')[:10]}" if result.get('is_conditional', False) else ""
        excluded_other = result.get('excluded_other_pairs', 0)
        excl_str = str(excluded_other) if excluded_other > 0 else ""
        
        report_lines.append(f"{column:<40} {agree_str:<8} {ac1_str:<8} {kappa_str:<8} {n_pairs:<6} {excl_str:<5} {conditional_str:<12} {status}")
    
    report_lines.append("")
    
    # Interpretation guide
    # Conditional logic explanation
    conditional_results = [r for r in results.values() if r.get('is_conditional', False)]
    if conditional_results:
        report_lines.append("CONDITIONAL LOGIC APPLIED")
        report_lines.append("-" * 40)
        report_lines.append("Some columns were analyzed with conditional logic:")
        report_lines.append("Only pairs where reviewers agreed on the parent column were included.")
        report_lines.append("")
        
        # Group by parent column
        parent_groups = {}
        for result in conditional_results:
            parent = result.get('parent_column')
            if parent not in parent_groups:
                parent_groups[parent] = []
            parent_groups[parent].append(result['column'])
        
        for parent, children in parent_groups.items():
            report_lines.append(f"  {parent} → {len(children)} dependent columns:")
            for child in children:
                report_lines.append(f"    • {child}")
        report_lines.append("")
    
    report_lines.append("INTERPRETATION GUIDE")
    report_lines.append("-" * 40)
    report_lines.append("Percent Agreement Interpretation:")
    report_lines.append("  < 70%: Poor agreement")
    report_lines.append("  70-79%: Fair agreement")
    report_lines.append("  80-89%: Good agreement")
    report_lines.append("  90-95%: Very good agreement")
    report_lines.append("  > 95%: Excellent agreement")
    report_lines.append("")
    report_lines.append("Gwet's AC1 Interpretation:")
    report_lines.append("  < 0.00: Poor agreement")
    report_lines.append("  0.00-0.20: Slight agreement")
    report_lines.append("  0.21-0.40: Fair agreement")
    report_lines.append("  0.41-0.60: Moderate agreement")
    report_lines.append("  0.61-0.80: Substantial agreement")
    report_lines.append("  0.81-1.00: Almost perfect agreement")
    report_lines.append("")
    report_lines.append("Cohen's Kappa Interpretation:")
    report_lines.append("  < 0.00: Poor agreement")
    report_lines.append("  0.00-0.20: Slight agreement")
    report_lines.append("  0.21-0.40: Fair agreement")
    report_lines.append("  0.41-0.60: Moderate agreement")
    report_lines.append("  0.61-0.80: Substantial agreement")
    report_lines.append("  0.81-1.00: Almost perfect agreement")
    report_lines.append("")
    report_lines.append("Notes:")
    report_lines.append("- Percent agreement is often more intuitive and represents raw agreement.")
    report_lines.append("- Gwet's AC1 is generally preferred over Cohen's kappa as it's less affected")
    report_lines.append("  by trait prevalence and marginal probability imbalances.")
    report_lines.append("- Cohen's kappa adjusts for chance agreement based on marginal probabilities.")
    report_lines.append("- Gwet's AC1 adjusts for chance agreement assuming uniform distribution.")
    report_lines.append("- Pairs where both reviewers selected 'Other:' responses are excluded")
    report_lines.append("  from agreement calculations (shown in 'Excl' column).")
    report_lines.append("- Conditional columns only include pairs where reviewers agreed on parent column.")
    
    report_content = "\n".join(report_lines)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    return report_content
