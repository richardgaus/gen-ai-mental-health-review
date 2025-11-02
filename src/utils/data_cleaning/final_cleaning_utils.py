"""
Final data cleaning utility functions for LLMs in Psychotherapy research.

This module contains utility functions for performing final cleaning operations on CSV files including:
- Filtering for consensus reviewer data
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import Tuple, Dict, List


def filter_reviewer_data(df: pd.DataFrame, use_preliminary: bool = False, verbose: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Filter dataframe to keep only rows from the specified reviewer.
    
    Args:
        df (pd.DataFrame): Input dataframe
        use_preliminary (bool): If True, use 'Richard Gaus' reviewer for specific columns, 
                               otherwise use 'Consensus' reviewer
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, dict]: Filtered dataframe and processing details
    """
    if 'reviewer_name' not in df.columns:
        raise KeyError("Column 'reviewer_name' not found in dataframe")
    
    # Store original counts
    original_rows = len(df)
    original_studies = df.index.nunique() if hasattr(df.index, 'nunique') else len(df)
    
    # Get reviewer counts before filtering
    reviewer_counts = df['reviewer_name'].value_counts().to_dict()
    
    if use_preliminary:
        # Define columns that should use Richard Gaus data
        preliminary_columns = [
            'p1_on_premise_model_considered',
            'p2_privacy_awareness_considered', 
            'author_country'
        ]
        # Add all columns starting with 'dataset'
        dataset_columns = [col for col in df.columns if col.startswith('dataset')]
        preliminary_columns.extend(dataset_columns)
        
        # Start with consensus data
        df_consensus = df[df['reviewer_name'] == 'Consensus'].copy()
        
        # Get Richard Gaus data for specific columns
        df_richard = df[df['reviewer_name'] == 'Richard Gaus'].copy()
        
        if len(df_richard) == 0:
            if verbose:
                print("Warning: No 'Richard Gaus' reviewer data found. Using 'Consensus' data only.")
            df_filtered = df_consensus
            reviewer_used = 'Consensus'
        else:
            # Replace specific columns with Richard Gaus data
            for col in preliminary_columns:
                if col in df_richard.columns and col in df_consensus.columns:
                    # Match by index and replace values
                    common_indices = df_consensus.index.intersection(df_richard.index)
                    df_consensus.loc[common_indices, col] = df_richard.loc[common_indices, col]
            
            df_filtered = df_consensus
            reviewer_used = 'Consensus (with Richard Gaus preliminary decisions)'
            
            if verbose:
                print(f"✓ Using preliminary decisions from 'Richard Gaus' for columns: {', '.join(preliminary_columns)}")
    else:
        # Filter for consensus only
        df_filtered = df[df['reviewer_name'] == 'Consensus'].copy()
        reviewer_used = 'Consensus'
    
    # Get final counts
    final_rows = len(df_filtered)
    final_studies = df_filtered.index.nunique() if hasattr(df_filtered.index, 'nunique') else len(df_filtered)
    
    # Calculate removed counts
    rows_removed = original_rows - final_rows
    studies_removed = original_studies - final_studies
    
    if verbose:
        print(f"✓ Filtered for '{reviewer_used}' reviewer:")
        print(f"  - Original rows: {original_rows:,}")
        print(f"  - Final rows: {final_rows:,}")
        print(f"  - Rows removed: {rows_removed:,}")
        print(f"  - Original studies: {original_studies:,}")
        print(f"  - Final studies: {final_studies:,}")
        print(f"  - Studies removed: {studies_removed:,}")
        print(f"\n  Reviewer breakdown (before filtering):")
        for reviewer, count in reviewer_counts.items():
            print(f"    • {reviewer}: {count:,} rows")
    
    # Prepare processing details
    processing_details = {
        'original_rows': original_rows,
        'final_rows': final_rows,
        'rows_removed': rows_removed,
        'original_studies': original_studies,
        'final_studies': final_studies,
        'studies_removed': studies_removed,
        'reviewer_counts': reviewer_counts,
        'reviewer_used': reviewer_used,
        'use_preliminary': use_preliminary
    }
    
    return df_filtered, processing_details


def filter_consensus_only(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Backward compatibility function for filter_consensus_only.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, dict]: Filtered dataframe and processing details
    """
    return filter_reviewer_data(df, use_preliminary=False, verbose=verbose)


def create_intended_intervention_type(intervention_type_value: str) -> str:
    """
    Transform intervention_type value to intended_intervention_type value.
    
    Args:
        intervention_type_value (str): Original intervention_type value (semicolon-separated)
        
    Returns:
        str: Transformed intended_intervention_type value (semicolon-separated)
    """
    # Handle missing values
    if pd.isna(intervention_type_value) or intervention_type_value == '':
        return ''
    
    # Split by semicolon and strip whitespace
    techniques = [t.strip() for t in str(intervention_type_value).split(';') if t.strip()]
    
    # Step 1: Replace unspecified/informal counseling with "Non-specific support"
    informal_variations = [
        "Unspecified, might include formal therapy methods",
        "Informal counseling (e.g., emotional support conversation)"
    ]
    has_informal = any(tech in informal_variations for tech in techniques)
    if has_informal:
        techniques = [t for t in techniques if t not in informal_variations]
        techniques.append("Non-specific support")
    
    # Step 2: Rename "Other CBT techniques" to "CBT: Other techniques"
    techniques = ["CBT: Other techniques" if t == "Other CBT techniques" else t for t in techniques]
    
    # Step 3: Rename "CBT: Motivational Interviewing" to "Motivational interviewing"
    techniques = ["Motivational interviewing" if t == "CBT: Motivational interviewing" else t for t in techniques]
    
    # Step 4: Rename "Mix of formal therapy methods" to "Multiple therapeutic orientations"
    techniques = ["Multiple therapeutic orientations" if t == "Mix of formal therapy methods" else t for t in techniques]
    
    # Step 5: Rename any "Other: xxx" to just "Other"
    techniques = ["Other" if t.startswith("Other: ") else t for t in techniques]
    
    # Step 6: Remove "Non-specific support" if other techniques are present
    if len(techniques) > 1 and "Non-specific support" in techniques:
        techniques = [t for t in techniques if t != "Non-specific support"]
    
    # Step 7: Remove "Other" if other techniques are present
    if len(techniques) > 1 and "Other" in techniques:
        techniques = [t for t in techniques if t != "Other"]
    
    # Step 8: Remove "Peer support conversation" if other techniques are present
    if len(techniques) > 1 and "Peer support conversation" in techniques:
        techniques = [t for t in techniques if t != "Peer support conversation"]
    
    # Step 9: Remove "CBT: Other techniques" if "CBT: Cognitive restructuring" is present
    if "CBT: Cognitive restructuring" in techniques and "CBT: Other techniques" in techniques:
        techniques = [t for t in techniques if t != "CBT: Other techniques"]
    
    # Step 10: If multiple categories remain, replace all with "Multiple therapeutic orientations"
    if len(techniques) > 1:
        techniques = ["Multiple therapeutic orientations"]
    
    # Join back with semicolon and space
    return techniques[0]


def add_intended_intervention_type_column(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Add intended_intervention_type column based on intervention_type column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        tuple[pd.DataFrame, dict]: Dataframe with new intended_intervention_type column and processing details
    """
    if 'intervention_type' not in df.columns:
        if verbose:
            print("  Warning: 'intervention_type' column not found, skipping intended_intervention_type creation")
        return df, {'column_created': False, 'reason': 'intervention_type column not found'}
    
    if verbose:
        print("  Creating intended_intervention_type column...")
    
    # Apply transformation to each row
    df['intended_intervention_type'] = df['intervention_type'].apply(create_intended_intervention_type)
    
    # Get statistics about the transformations
    unique_mappings = df[['intervention_type', 'intended_intervention_type']].drop_duplicates()
    unique_original = len(unique_mappings)
    unique_transformed = df['intended_intervention_type'].nunique()
    
    # Count transformation types
    transformation_counts = {}
    for _, row in unique_mappings.iterrows():
        orig = row['intervention_type']
        new = row['intended_intervention_type']
        if orig != new:
            transformation_counts[f"'{orig}' → '{new}'"] = transformation_counts.get(f"'{orig}' → '{new}'", 0) + 1
    
    processing_details = {
        'column_created': True,
        'unique_original_values': unique_original,
        'unique_transformed_values': unique_transformed,
        'transformations_applied': len(transformation_counts),
        'example_transformations': list(transformation_counts.keys())[:5]  # First 5 examples
    }
    
    if verbose:
        print(f"  ✓ Created intended_intervention_type column")
        print(f"  - Found {unique_original:,} unique intervention_type values")
        print(f"  - Resulted in {unique_transformed:,} unique intended_intervention_type values")
        print(f"  - Applied {len(transformation_counts):,} different transformations")
        print(f"  - Example transformations (first 5):")
        for idx, example in enumerate(processing_details['example_transformations'], 1):
            print(f"    {idx}. {example}")
    
    return df, processing_details


def create_inclusion_criteria(client_type_value: str) -> str:
    """
    Transform client_type value to inclusion_criteria value.
    
    Args:
        client_type_value (str): Original client_type value (semicolon-separated)
        
    Returns:
        str: Transformed inclusion_criteria value
    """
    # Handle missing values
    if pd.isna(client_type_value) or client_type_value == '':
        return 'Unspecified'
    
    # Convert to string and check for specific patterns
    client_type_str = str(client_type_value).lower()
    
    # Check for "General population" (case-insensitive)
    if "general population" in client_type_str:
        return "No specific criteria"
    
    # Check for "People with some symptoms but not disorder"
    elif "people with some symptoms but not disorder" in client_type_str:
        return "Elevated symptoms"
    
    # Check for "Patients with disorder explicitly based on ICD or DSM"
    elif "patients with disorder explicitly based on icd or dsm" in client_type_str:
        return "Diagnosed mental disorder (ICD/DSM)"
    
    # Check for "Patients recruited in hospital or outpatient treatment facility"
    elif "patients recruited in hospital or outpatient treatment facility" in client_type_str:
        return "Recruited in in- or outpatient healthcare setting with no specific criteria"
    
    # Default case
    else:
        return "Unspecified"


def add_inclusion_criteria_column(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Add inclusion_criteria column based on client_type column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        tuple[pd.DataFrame, dict]: Dataframe with new inclusion_criteria column and processing details
    """
    if 'client_type' not in df.columns:
        if verbose:
            print("  Warning: 'client_type' column not found, skipping inclusion_criteria creation")
        return df, {'column_created': False, 'reason': 'client_type column not found'}
    
    if verbose:
        print("  Creating inclusion_criteria column...")
    
    # Apply transformation to each row
    df['inclusion_criteria'] = df['client_type'].apply(create_inclusion_criteria)
    
    # Get statistics about the transformations
    unique_mappings = df[['client_type', 'inclusion_criteria']].drop_duplicates()
    unique_original = len(unique_mappings)
    unique_transformed = df['inclusion_criteria'].nunique()
    
    # Get distribution of inclusion criteria
    criteria_counts = df['inclusion_criteria'].value_counts().to_dict()
    
    # Count transformation types
    transformation_counts = {}
    for _, row in unique_mappings.iterrows():
        orig = row['client_type']
        new = row['inclusion_criteria']
        if pd.notna(orig) and str(orig).strip():
            transformation_counts[f"'{orig}' → '{new}'"] = transformation_counts.get(f"'{orig}' → '{new}'", 0) + 1
    
    processing_details = {
        'column_created': True,
        'unique_original_values': unique_original,
        'unique_transformed_values': unique_transformed,
        'transformations_applied': len(transformation_counts),
        'example_transformations': list(transformation_counts.keys())[:5],  # First 5 examples
        'criteria_distribution': criteria_counts
    }
    
    if verbose:
        print(f"  ✓ Created inclusion_criteria column")
        print(f"  - Found {unique_original:,} unique client_type values")
        print(f"  - Resulted in {unique_transformed:,} unique inclusion_criteria values")
        print(f"  - Applied {len(transformation_counts):,} different transformations")
        print(f"  - Inclusion criteria distribution:")
        for criteria, count in criteria_counts.items():
            print(f"    • {criteria}: {count:,} rows")
        print(f"  - Example transformations (first 5):")
        for idx, example in enumerate(processing_details['example_transformations'], 1):
            print(f"    {idx}. {example}")
    
    return df, processing_details


def add_primary_clinical_outcome_column(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Add primary_clinical_outcome column based on application_type, research_type, client_type, and outcome measures.
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        tuple[pd.DataFrame, dict]: Dataframe with new primary_clinical_outcome column and processing details
    """
    required_columns = ['application_type', 'study_type', 'client_type', 'f1_validated_outcomes_considered', 'ux_assessment_is_present']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        if verbose:
            print(f"  Warning: Missing required columns {missing_columns}, skipping primary_clinical_outcome creation")
        return df, {'column_created': False, 'reason': f'Missing columns: {missing_columns}'}
    
    if verbose:
        print("  Creating primary_clinical_outcome column...")
    
    # Initialize the column with empty values
    df['primary_clinical_outcome'] = ''
    
    # Define the conditions for filling the column
    # Only fill for rows where:
    # - application_type is "Client-facing application" 
    # - it's empirical research involving an LLM
    # - client_type is other than "No clients/patients involved"
    
    # First, identify eligible rows
    client_facing_mask = df['application_type'] == 'Client-facing application'
    empirical_llm_mask = df['study_type'] == 'Empirical research involving an LLM'
    clients_involved_mask = df['client_type'] != 'No clients/patients involved'
    
    # If study_type column doesn't exist, raise an error since it's required
    if 'study_type' not in df.columns:
        if verbose:
            print("  Warning: 'study_type' column not found, skipping primary_clinical_outcome creation")
        return df, {'column_created': False, 'reason': 'study_type column not found'}
    
    eligible_mask = client_facing_mask & empirical_llm_mask & clients_involved_mask
    
    # Apply the logic for eligible rows
    eligible_rows = df[eligible_mask].copy()
    
    # Priority 1: Validated symptom/function scale
    validated_mask = eligible_rows['f1_validated_outcomes_considered'].str.lower() == 'y'
    df.loc[eligible_mask & validated_mask, 'primary_clinical_outcome'] = 'Validated symptom/function scale'
    
    # Priority 2: User experience assessment (for remaining rows)
    remaining_mask = eligible_mask & (df['primary_clinical_outcome'] == '')
    ux_mask = df.loc[remaining_mask, 'ux_assessment_is_present'].str.lower() == 'yes'
    df.loc[remaining_mask & ux_mask, 'primary_clinical_outcome'] = 'User experience assessment'
    
    # Priority 3: Other (for remaining eligible rows)
    still_remaining_mask = eligible_mask & (df['primary_clinical_outcome'] == '')
    df.loc[still_remaining_mask, 'primary_clinical_outcome'] = 'Other'
    
    # Get statistics
    outcome_counts = df['primary_clinical_outcome'].value_counts().to_dict()
    total_eligible = eligible_mask.sum()
    total_filled = (df['primary_clinical_outcome'] != '').sum()
    
    # Get study names for "Other" category
    other_studies = []
    if 'Other' in outcome_counts:
        other_mask = df['primary_clinical_outcome'] == 'Other'
        if 'study_name' in df.columns:
            other_studies = df.loc[other_mask, 'study_name'].dropna().unique().tolist()
        elif 'title' in df.columns:
            other_studies = df.loc[other_mask, 'title'].dropna().unique().tolist()
        else:
            # Use index if no study name column
            other_studies = df.loc[other_mask].index.tolist()
    
    processing_details = {
        'column_created': True,
        'total_eligible_rows': total_eligible,
        'total_filled_rows': total_filled,
        'outcome_counts': outcome_counts,
        'other_studies': other_studies,
        'conditions_applied': {
            'client_facing_applications': client_facing_mask.sum(),
            'empirical_llm_studies': empirical_llm_mask.sum(),
            'studies_with_clients_involved': clients_involved_mask.sum(),
            'all_conditions_met': total_eligible
        }
    }
    
    if verbose:
        print(f"  ✓ Created primary_clinical_outcome column")
        print(f"  - Condition breakdown:")
        print(f"    • Client-facing applications: {client_facing_mask.sum():,}")
        print(f"    • Empirical LLM studies: {empirical_llm_mask.sum():,}")
        print(f"    • Studies with clients involved: {clients_involved_mask.sum():,}")
        print(f"    • All conditions met: {total_eligible:,}")
        print(f"  - Total rows filled: {total_filled:,}")
        print(f"  - Outcome distribution:")
        for outcome, count in outcome_counts.items():
            if outcome != '':  # Skip empty values
                print(f"    • {outcome}: {count:,} rows")
        if other_studies:
            # Convert to strings for display
            study_names_str = [str(s) for s in other_studies[:5]]
            print(f"  - Studies with 'Other' outcome ({len(other_studies)}): {', '.join(study_names_str)}")
            if len(other_studies) > 5:
                print(f"    ... and {len(other_studies) - 5} more")
    
    return df, processing_details


def load_models_cleaning_mapping(config_path: str = None) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Load models cleaning mapping and non-generative models list from configuration file.
    
    Args:
        config_path (str): Path to configuration file. If None, uses default path.
        
    Returns:
        Tuple[Dict[str, List[str]], List[str]]: Dictionary mapping target model names to source model names, and list of non-generative models
    """
    if config_path is None:
        # Default path relative to this file
        config_path = Path(__file__).parent / "final_cleaning_config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        models_mapping = config.get('column_cleaning_mapping', {}).get('models_employed', {})
        non_generative_models = config.get('non_generative_models', [])
        return models_mapping, non_generative_models
    except Exception as e:
        print(f"Warning: Could not load models cleaning mapping from {config_path}: {e}")
        return {}, []


def clean_models_employed_column(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the models_employed column by removing non-generative models and mapping "Other: " entries.
    
    This function:
    1. Removes all non-generative models from semicolon-separated and comma-separated (in "Other:") lists
    2. Removes empty "Other: " entries that result from non-generative model removal
    3. Extracts remaining models from "Other: model1, model2" format
    4. Maps each extracted model using the models_employed cleaning configuration
    5. Replaces unmapped models with "Other"
    6. Combines all models back with semicolon separation
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, dict]: Dataframe with cleaned models_employed column and processing details
    """
    if 'models_employed' not in df.columns:
        if verbose:
            print("  Warning: 'models_employed' column not found, skipping models cleaning")
        return df, {'column_cleaned': False, 'reason': 'models_employed column not found'}
    
    if verbose:
        print("  Cleaning models_employed column...")
    
    # Load the cleaning mapping and non-generative models list
    models_mapping, non_generative_models = load_models_cleaning_mapping()
    
    if not models_mapping and not non_generative_models:
        if verbose:
            print("  Warning: No models cleaning mapping or non-generative models list found")
        return df, {'column_cleaned': False, 'reason': 'No models mapping or non-generative models found'}
    
    df_processed = df.copy()
    
    # Track processing statistics
    total_processed = 0
    total_mapped = 0
    total_unmapped = 0
    total_non_gen_removed = 0
    unmapped_models = set()  # For unique unmapped models
    unmapped_models_with_duplicates = []  # For all instances including duplicates
    mapping_examples = []
    non_gen_removed_examples = []
    
    # Process each row
    for idx, models_entry in df_processed['models_employed'].items():
        if pd.isna(models_entry) or not str(models_entry).strip():
            continue
        
        total_processed += 1
        original_entry = str(models_entry)
        
        # STEP 1: Remove non-generative models first
        # Split by semicolons and process each part
        model_parts = [part.strip() for part in original_entry.split(';') if part.strip()]
        after_non_gen_removal = []
        
        for part in model_parts:
            # Check if this part starts with "Other: "
            if part.lower().startswith('other: '):
                # Extract the content after "Other: "
                other_content = part[7:].strip()  # Remove "Other: " prefix
                
                # Split by commas to get individual models
                other_models = [model.strip() for model in other_content.split(',') if model.strip()]
                
                # Filter out non-generative models
                filtered_other_models = []
                for model in other_models:
                    if model in non_generative_models:
                        total_non_gen_removed += 1
                        non_gen_removed_examples.append(model)
                    else:
                        filtered_other_models.append(model)
                
                # If any models remain after filtering, reconstruct "Other: " part
                if filtered_other_models:
                    after_non_gen_removal.append(f"Other: {', '.join(filtered_other_models)}")
                # If no models remain, don't add anything (removes empty "Other: ")
            else:
                # Regular model entry (not "Other: "), check if it's non-generative
                if part in non_generative_models:
                    total_non_gen_removed += 1
                    non_gen_removed_examples.append(part)
                    # Don't add to after_non_gen_removal (removes it)
                else:
                    # Keep generative models
                    after_non_gen_removal.append(part)
        
        # STEP 2: Now process the remaining parts for mapping
        cleaned_parts = []
        
        for part in after_non_gen_removal:
            # Check if this part starts with "Other: "
            if part.lower().startswith('other: '):
                # Extract the content after "Other: "
                other_content = part[7:].strip()  # Remove "Other: " prefix
                
                # Split by commas to get individual models
                other_models = [model.strip() for model in other_content.split(',') if model.strip()]
                
                # Map each model using the configuration
                for model in other_models:
                    mapped = False
                    
                    # Check against all target -> source mappings
                    for target_model, source_models in models_mapping.items():
                        if model in source_models:
                            cleaned_parts.append(target_model)
                            total_mapped += 1
                            mapping_examples.append(f"'{model}' → '{target_model}'")
                            mapped = True
                            break
                    
                    # If not mapped, replace with "Other"
                    if not mapped:
                        cleaned_parts.append("Other")
                        unmapped_models.add(model)  # For unique list
                        unmapped_models_with_duplicates.append(model)  # For all instances
                        total_unmapped += 1
            else:
                # Regular model entry (not "Other: "), keep as is
                cleaned_parts.append(part)
        
        # Join back with semicolons
        cleaned_entry = '; '.join(cleaned_parts) if cleaned_parts else ''
        df_processed.loc[idx, 'models_employed'] = cleaned_entry
    
    processing_details = {
        'column_cleaned': True,
        'total_entries_processed': total_processed,
        'total_non_gen_models_removed': total_non_gen_removed,
        'non_gen_removed_examples': non_gen_removed_examples[:10],  # First 10 examples
        'total_models_mapped': total_mapped,
        'total_models_unmapped': total_unmapped,
        'unmapped_models': sorted(list(unmapped_models)),  # Unique unmapped models
        'unmapped_models_with_duplicates': unmapped_models_with_duplicates,  # All instances including duplicates
        'mapping_examples': mapping_examples[:10],  # First 10 examples
        'mapping_rules_available': len(models_mapping),
        'non_generative_models_available': len(non_generative_models)
    }
    
    if verbose:
        print(f"  ✓ Cleaned models_employed column")
        print(f"  - Entries processed: {total_processed:,}")
        print(f"  - Non-generative models removed: {total_non_gen_removed:,}")
        print(f"  - Models mapped: {total_mapped:,}")
        print(f"  - Models unmapped (set to 'Other'): {total_unmapped:,}")
        print(f"  - Mapping rules available: {len(models_mapping):,}")
        print(f"  - Non-generative models list size: {len(non_generative_models):,}")
        
        if non_gen_removed_examples:
            print(f"  - Example non-generative models removed:")
            unique_removed = sorted(list(set(non_gen_removed_examples)))
            for model in unique_removed[:5]:  # Show first 5 unique
                print(f"    • '{model}'")
            if len(unique_removed) > 5:
                print(f"    ... and {len(unique_removed) - 5} more")
        
        if mapping_examples:
            print(f"  - Example mappings:")
            for example in mapping_examples[:5]:  # Show first 5
                print(f"    • {example}")
            if len(mapping_examples) > 5:
                print(f"    ... and {len(mapping_examples) - 5} more")
        
        if unmapped_models:
            print(f"  - Unmapped models (set to 'Other'):")
            print(f"    Unique models: {len(unmapped_models)}, Total instances: {len(unmapped_models_with_duplicates)}")
            unmapped_list = sorted(list(unmapped_models))
            for model in unmapped_list[:5]:  # Show first 5 unique
                count_instances = unmapped_models_with_duplicates.count(model)
                print(f"    • '{model}' ({count_instances} instances)")
            if len(unmapped_list) > 5:
                print(f"    ... and {len(unmapped_list) - 5} more unique models")
    
    return df_processed, processing_details


def clean_llm_development_approach_column(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the llm_development_approach column by mapping values and collapsing multi-label entries.
    
    This function:
    1. Maps individual values using the llm_development_approach cleaning configuration
    2. Collapses multi-label entries using priority rules:
       - "Any RL" takes precedence over everything
       - Fine-tuning approaches take precedence over prompting approaches
       - Combined approaches take precedence over single approaches
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, dict]: Dataframe with cleaned llm_development_approach column and processing details
    """
    if 'llm_development_approach' not in df.columns:
        if verbose:
            print("  Warning: 'llm_development_approach' column not found, skipping approach cleaning")
        return df, {'column_cleaned': False, 'reason': 'llm_development_approach column not found'}
    
    if verbose:
        print("  Cleaning llm_development_approach column...")
    
    # Load the cleaning mapping
    models_mapping, _ = load_models_cleaning_mapping()  # We only need the config, not non-gen models
    
    try:
        with open(Path(__file__).parent / "final_cleaning_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        approach_mapping = config.get('column_cleaning_mapping', {}).get('llm_development_approach', {})
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load llm_development_approach mapping: {e}")
        return df, {'column_cleaned': False, 'reason': 'Could not load approach mapping'}
    
    if not approach_mapping:
        if verbose:
            print("  Warning: No llm_development_approach mapping found")
        return df, {'column_cleaned': False, 'reason': 'No approach mapping found'}
    
    df_processed = df.copy()
    
    # Track processing statistics
    total_processed = 0
    total_mapped = 0
    total_collapsed = 0
    total_final_categorized = 0
    mapping_examples = []
    collapse_examples = []
    final_categorization_examples = []
    
    # Process each row
    for idx, approach_entry in df_processed['llm_development_approach'].items():
        if pd.isna(approach_entry) or not str(approach_entry).strip():
            continue
        
        total_processed += 1
        original_entry = str(approach_entry)
        
        # Step 1: Split by semicolons and map individual values
        approach_parts = [part.strip() for part in original_entry.split(';') if part.strip()]
        mapped_parts = []
        
        for part in approach_parts:
            mapped = False
            
            # Handle "Other: " entries by extracting the content
            if part.lower().startswith('other: '):
                other_content = part[7:].strip()  # Remove "Other: " prefix
                
                # Check if the content matches any mapping
                for target_approach, source_approaches in approach_mapping.items():
                    if other_content in source_approaches:
                        mapped_parts.append(target_approach)
                        total_mapped += 1
                        mapping_examples.append(f"'Other: {other_content}' → '{target_approach}'")
                        mapped = True
                        break
                
                # If not mapped, keep as "Other"
                if not mapped:
                    mapped_parts.append("Other")
            else:
                # Check direct mappings for non-"Other:" entries
                for target_approach, source_approaches in approach_mapping.items():
                    if part in source_approaches:
                        mapped_parts.append(target_approach)
                        total_mapped += 1
                        mapping_examples.append(f"'{part}' → '{target_approach}'")
                        mapped = True
                        break
                
                # If not mapped, keep original
                if not mapped:
                    mapped_parts.append(part)
        
        # Step 2: Apply collapse rules to mapped parts
        original_mapped = mapped_parts.copy()
        
        # Remove duplicates while preserving order
        unique_parts = []
        seen = set()
        for part in mapped_parts:
            if part not in seen:
                unique_parts.append(part)
                seen.add(part)
        mapped_parts = unique_parts
        
        # Apply collapse rules
        if len(mapped_parts) > 1:
            # Rule: if "Any RL" -> drop all but "Any RL" (but this will be handled in final categorization)
            if "Any RL" in mapped_parts:
                # Keep main categories alongside "Any RL" for now, final categorization will handle it
                main_categories_present = {"Only prompting", "Only fine-tuning", "Prompting + other modules", "Fine-tuning + other modules"}
                main_in_mapped = [part for part in mapped_parts if part in main_categories_present]
                
                if main_in_mapped:
                    # Keep main categories and "Any RL" - final categorization will convert "Any RL" to "Other"
                    collapsed_parts = main_in_mapped + ["Any RL"]
                else:
                    # No main categories, just keep "Any RL"
                    collapsed_parts = ["Any RL"]
                
                if len(mapped_parts) > len(collapsed_parts):
                    total_collapsed += 1
                    collapse_examples.append(f"{'; '.join(original_mapped)} → {'; '.join(collapsed_parts)}")
            else:
                # Apply other collapse rules
                collapsed_parts = mapped_parts.copy()
                
                # Rule: "Only prompting" AND "Only fine-tuning" -> "Only fine-tuning"
                if "Only prompting" in collapsed_parts and "Only fine-tuning" in collapsed_parts:
                    collapsed_parts = [p for p in collapsed_parts if p != "Only prompting"]
                    total_collapsed += 1
                    collapse_examples.append(f"{'; '.join(original_mapped)} → {'; '.join(collapsed_parts)}")
                
                # Rule: "Prompting + other modules" AND "Fine-tuning + other modules" -> "Fine-tuning + other modules"
                elif "Prompting + other modules" in collapsed_parts and "Fine-tuning + other modules" in collapsed_parts:
                    collapsed_parts = [p for p in collapsed_parts if p != "Prompting + other modules"]
                    total_collapsed += 1
                    collapse_examples.append(f"{'; '.join(original_mapped)} → {'; '.join(collapsed_parts)}")
                
                # Rule: "Only prompting" AND "Prompting + other modules" -> "Prompting + other modules"
                elif "Only prompting" in collapsed_parts and "Prompting + other modules" in collapsed_parts:
                    collapsed_parts = [p for p in collapsed_parts if p != "Only prompting"]
                    total_collapsed += 1
                    collapse_examples.append(f"{'; '.join(original_mapped)} → {'; '.join(collapsed_parts)}")
                
                # Rule: "Only fine-tuning" AND "Fine-tuning + other modules" -> "Fine-tuning + other modules"
                elif "Only fine-tuning" in collapsed_parts and "Fine-tuning + other modules" in collapsed_parts:
                    collapsed_parts = [p for p in collapsed_parts if p != "Only fine-tuning"]
                    total_collapsed += 1
                    collapse_examples.append(f"{'; '.join(original_mapped)} → {'; '.join(collapsed_parts)}")
                
                # Rule: "Only fine-tuning" AND "Prompting + other modules" -> "Fine-tuning + other modules"
                elif "Only fine-tuning" in collapsed_parts and "Prompting + other modules" in collapsed_parts:
                    collapsed_parts = ["Fine-tuning + other modules"]
                    total_collapsed += 1
                    collapse_examples.append(f"{'; '.join(original_mapped)} → Fine-tuning + other modules")
                
                # Rule: "Fine-tuning + other modules" AND "Only prompting" -> "Fine-tuning + other modules"
                elif "Fine-tuning + other modules" in collapsed_parts and "Only prompting" in collapsed_parts:
                    collapsed_parts = [p for p in collapsed_parts if p != "Only prompting"]
                    total_collapsed += 1
                    collapse_examples.append(f"{'; '.join(original_mapped)} → {'; '.join(collapsed_parts)}")
        else:
            collapsed_parts = mapped_parts
        
        # Step 3: Final categorization - collapse non-main categories to "Other"
        # Special rule: Main category + non-main category → "Other"
        # But main category + main category should use existing collapse rules
        main_categories = {"Only prompting", "Only fine-tuning", "Prompting + other modules", "Fine-tuning + other modules"}
        original_collapsed = collapsed_parts.copy()
        
        # Check if we have main categories mixed with non-main categories (use original_mapped to check before collapse rules)
        main_categories_in_original = [part for part in original_mapped if part in main_categories]
        non_main_categories_in_original = [part for part in original_mapped if part not in main_categories]
        
        if len(main_categories_in_original) > 0 and len(non_main_categories_in_original) > 0:
            # Main category + non-main category → "Other"
            final_parts = ["Other"]
            total_final_categorized += 1
            final_categorization_examples.append(f"{'; '.join(original_mapped)} → Other")
        elif len(collapsed_parts) == 1 and collapsed_parts[0] in main_categories:
            # Single main category stays as-is
            final_parts = collapsed_parts
        elif all(part in main_categories for part in collapsed_parts):
            # All main categories - keep the result of collapse rules
            final_parts = collapsed_parts
        else:
            # Normal final categorization for non-main categories only
            final_parts = []
            for part in collapsed_parts:
                if part in main_categories:
                    final_parts.append(part)
                else:
                    final_parts.append("Other")
            
            # Deduplicate "Other" entries while preserving order
            deduplicated_parts = []
            seen = set()
            for part in final_parts:
                if part not in seen:
                    deduplicated_parts.append(part)
                    seen.add(part)
            final_parts = deduplicated_parts
            
            # Track if any final categorization happened
            if final_parts != original_collapsed:
                total_final_categorized += 1
                final_categorization_examples.append(f"{'; '.join(original_collapsed)} → {'; '.join(final_parts)}")
        
        collapsed_parts = final_parts
        
        # Join back with semicolons
        cleaned_entry = '; '.join(collapsed_parts) if collapsed_parts else ''
        df_processed.loc[idx, 'llm_development_approach'] = cleaned_entry
    
    processing_details = {
        'column_cleaned': True,
        'total_entries_processed': total_processed,
        'total_values_mapped': total_mapped,
        'total_entries_collapsed': total_collapsed,
        'total_entries_final_categorized': total_final_categorized,
        'mapping_examples': mapping_examples[:10],  # First 10 examples
        'collapse_examples': collapse_examples[:10],  # First 10 examples
        'final_categorization_examples': final_categorization_examples[:10],  # First 10 examples
        'mapping_rules_available': len(approach_mapping)
    }
    
    if verbose:
        print(f"  ✓ Cleaned llm_development_approach column")
        print(f"  - Entries processed: {total_processed:,}")
        print(f"  - Values mapped: {total_mapped:,}")
        print(f"  - Entries collapsed: {total_collapsed:,}")
        print(f"  - Entries final categorized: {total_final_categorized:,}")
        print(f"  - Mapping rules available: {len(approach_mapping):,}")
        
        if mapping_examples:
            print(f"  - Example mappings:")
            for example in mapping_examples[:5]:  # Show first 5
                print(f"    • {example}")
            if len(mapping_examples) > 5:
                print(f"    ... and {len(mapping_examples) - 5} more")
        
        if collapse_examples:
            print(f"  - Example collapses:")
            for example in collapse_examples[:5]:  # Show first 5
                print(f"    • {example}")
            if len(collapse_examples) > 5:
                print(f"    ... and {len(collapse_examples) - 5} more")
        
        if final_categorization_examples:
            print(f"  - Example final categorizations:")
            for example in final_categorization_examples[:5]:  # Show first 5
                print(f"    • {example}")
            if len(final_categorization_examples) > 5:
                print(f"    ... and {len(final_categorization_examples) - 5} more")
    
    return df_processed, processing_details


def clean_outlet_and_application_columns(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Clean outlet_field and application type columns by mapping values using configuration.
    
    This function maps values in:
    - outlet_field
    - application_type  
    - application_subtype_client_facing
    - application_subtype_therapist_facing
    
    Args:
        df (pd.DataFrame): Input dataframe
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, dict]: Dataframe with cleaned columns and processing details
    """
    columns_to_clean = [
        'outlet_field', 
        'application_type', 
        'application_subtype_client_facing', 
        'application_subtype_therapist_facing'
    ]
    
    # Check which columns exist
    existing_columns = [col for col in columns_to_clean if col in df.columns]
    
    if not existing_columns:
        if verbose:
            print("  Warning: None of the target columns found, skipping outlet/application cleaning")
        return df, {'columns_cleaned': False, 'reason': 'No target columns found'}
    
    if verbose:
        print(f"  Cleaning outlet and application columns: {', '.join(existing_columns)}")
    
    # Load the cleaning mapping
    try:
        with open(Path(__file__).parent / "final_cleaning_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        column_mappings = config.get('column_cleaning_mapping', {})
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load column mappings: {e}")
        return df, {'columns_cleaned': False, 'reason': 'Could not load mappings'}
    
    df_processed = df.copy()
    
    # Track processing statistics
    total_columns_processed = 0
    total_values_mapped = 0
    total_other_unmapped = 0
    mapping_examples = []
    other_unmapped_examples = []
    column_details = {}
    
    # Process each column
    for column in existing_columns:
        if column not in column_mappings:
            if verbose:
                print(f"    Warning: No mapping found for column '{column}'")
            continue
        
        column_mapping = column_mappings[column]
        column_mapped_count = 0
        column_other_unmapped_count = 0
        column_examples = []
        column_other_unmapped_examples = []
        
        total_columns_processed += 1
        
        # Apply mappings for this column
        for idx, value in df_processed[column].items():
            if pd.isna(value) or not str(value).strip():
                continue
            
            original_value = str(value)
            mapped = False
            
            # Handle "Other: " entries by extracting the content
            if original_value.lower().startswith('other: '):
                other_content = original_value[7:].strip()  # Remove "Other: " prefix
                
                # Check if the content matches any mapping
                for target_value, source_values in column_mapping.items():
                    if other_content in source_values:
                        df_processed.loc[idx, column] = target_value
                        column_mapped_count += 1
                        total_values_mapped += 1
                        mapping_example = f"'Other: {other_content}' → '{target_value}'"
                        column_examples.append(mapping_example)
                        mapping_examples.append(f"{column}: {mapping_example}")
                        mapped = True
                        break
                
                # If not mapped, keep as "Other" and track it
                if not mapped:
                    df_processed.loc[idx, column] = "Other"
                    column_other_unmapped_count += 1
                    total_other_unmapped += 1
                    column_other_unmapped_examples.append(other_content)
                    other_unmapped_examples.append(f"{column}: {other_content}")
            else:
                # Check direct mappings for non-"Other:" entries
                for target_value, source_values in column_mapping.items():
                    if original_value in source_values:
                        df_processed.loc[idx, column] = target_value
                        column_mapped_count += 1
                        total_values_mapped += 1
                        mapping_example = f"'{original_value}' → '{target_value}'"
                        column_examples.append(mapping_example)
                        mapping_examples.append(f"{column}: {mapping_example}")
                        mapped = True
                        break
        
        column_details[column] = {
            'values_mapped': column_mapped_count,
            'other_unmapped_count': column_other_unmapped_count,
            'mapping_examples': column_examples[:5],  # First 5 examples
            'other_unmapped_examples': column_other_unmapped_examples[:5],  # First 5 examples
            'mapping_rules_available': len(column_mapping)
        }
        
        if verbose:
            print(f"    ✓ {column}: {column_mapped_count:,} values mapped, {column_other_unmapped_count:,} 'Other:' unmapped using {len(column_mapping):,} rules")
            if column_examples:
                for example in column_examples[:3]:  # Show first 3
                    print(f"      • {example}")
                if len(column_examples) > 3:
                    print(f"      ... and {len(column_examples) - 3} more")
            if column_other_unmapped_examples:
                print(f"      Unmapped 'Other:' values:")
                for example in column_other_unmapped_examples[:3]:  # Show first 3
                    print(f"      • Other: {example}")
                if len(column_other_unmapped_examples) > 3:
                    print(f"      ... and {len(column_other_unmapped_examples) - 3} more")
    
    processing_details = {
        'columns_cleaned': True,
        'columns_processed': existing_columns,
        'total_columns_processed': total_columns_processed,
        'total_values_mapped': total_values_mapped,
        'total_other_unmapped': total_other_unmapped,
        'mapping_examples': mapping_examples[:15],  # First 15 examples across all columns
        'other_unmapped_examples': other_unmapped_examples[:15],  # First 15 examples across all columns
        'column_details': column_details
    }
    
    if verbose:
        print(f"  ✓ Cleaned outlet and application columns")
        print(f"  - Columns processed: {total_columns_processed:,}")
        print(f"  - Total values mapped: {total_values_mapped:,}")
        print(f"  - Total 'Other:' unmapped: {total_other_unmapped:,}")
    
    return df_processed, processing_details


def perform_final_cleaning_pipeline(df: pd.DataFrame, use_preliminary: bool = False, verbose: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Perform the complete final cleaning pipeline on a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        use_preliminary (bool): If True, use 'Richard Gaus' reviewer for specific columns
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, dict]: Fully cleaned dataframe with processing details
    """
    if verbose:
        print("Starting final cleaning pipeline...\n")
    
    # Step 1: Filter for reviewer data
    if verbose:
        reviewer_type = "'Consensus' (with Richard Gaus preliminary decisions)" if use_preliminary else "'Consensus'"
        print(f"Step 1: Filtering for {reviewer_type} reviewer...")
    
    df_final, processing_details = filter_reviewer_data(df, use_preliminary=use_preliminary, verbose=verbose)
    
    # Step 2: Add intended_intervention_type column
    if verbose:
        print("\nStep 2: Creating intended_intervention_type column...")
    
    df_final, column_details = add_intended_intervention_type_column(df_final, verbose=verbose)
    
    # Step 3: Add inclusion_criteria column
    if verbose:
        print("\nStep 3: Creating inclusion_criteria column...")
    
    df_final, inclusion_details = add_inclusion_criteria_column(df_final, verbose=verbose)
    
    # Step 4: Add primary_clinical_outcome column
    if verbose:
        print("\nStep 4: Creating primary_clinical_outcome column...")
    
    df_final, outcome_details = add_primary_clinical_outcome_column(df_final, verbose=verbose)
    
    # Step 5: Clean models_employed column
    if verbose:
        print("\nStep 5: Cleaning models_employed column...")
    
    df_final, models_cleaning_details = clean_models_employed_column(df_final, verbose=verbose)
    
    # Step 6: Clean llm_development_approach column
    if verbose:
        print("\nStep 6: Cleaning llm_development_approach column...")
    
    df_final, approach_cleaning_details = clean_llm_development_approach_column(df_final, verbose=verbose)
    
    # Step 7: Clean outlet_field and application type columns
    if verbose:
        print("\nStep 7: Cleaning outlet and application columns...")
    
    df_final, outlet_application_cleaning_details = clean_outlet_and_application_columns(df_final, verbose=verbose)
    
    # Merge processing details
    processing_details['column_creation'] = column_details
    processing_details['inclusion_creation'] = inclusion_details
    processing_details['outcome_creation'] = outcome_details
    processing_details['models_cleaning'] = models_cleaning_details
    processing_details['approach_cleaning'] = approach_cleaning_details
    processing_details['outlet_application_cleaning'] = outlet_application_cleaning_details
    
    if verbose:
        print(f"\n✓ Final cleaning pipeline complete!")
        print(f"  Final dataset: {len(df_final):,} rows")
    
    return df_final, processing_details

