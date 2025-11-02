"""
Clinical panel analysis utilities.

This module provides functions for analyzing human participation in empirical
LLM studies with client-facing applications.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
import re
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
from .utils import create_horizontal_bar_chart, create_time_series_chart, get_color_palette


def filter_clinical_empirical_studies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataset to only include empirical LLM research with client-facing applications.
    
    Args:
        df (pd.DataFrame): Input dataframe with study_type and application_type columns
        
    Returns:
        pd.DataFrame: Filtered dataframe
        
    Raises:
        KeyError: If required columns are not found
    """
    required_columns = ['study_type', 'application_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Filter for empirical LLM studies with client-facing applications
    filtered = df[
        (df['study_type'] == 'Empirical research involving an LLM') &
        (df['application_type'] == 'Client-facing application')
    ].copy()
    
    return filtered


def parse_client_type(client_type_value: str) -> list:
    """
    Parse semicolon-separated client_type values into individual choices.
    
    Args:
        client_type_value (str): Client type string (potentially semicolon-separated)
        
    Returns:
        list: List of individual client type choices
    """
    if pd.isna(client_type_value):
        return []
    
    value_str = str(client_type_value).strip()
    if not value_str:
        return []
    
    # Split by semicolon and clean up
    choices = [choice.strip() for choice in value_str.split(';') if choice.strip()]
    return choices


def has_human_participants(client_type_value: str) -> bool:
    """
    Determine if a study involves human participants based on client_type.
    
    Human participants are present if client_type has any value other than
    "No clients/patients involved" (or is null/empty).
    
    Args:
        client_type_value (str): Client type value(s)
        
    Returns:
        bool: True if human participants are involved, False otherwise
    """
    choices = parse_client_type(client_type_value)
    
    # If no client type specified, assume no human participants
    if not choices:
        return False
    
    # If all choices are "No clients/patients involved", return False
    # Otherwise return True
    return not all(choice == "No clients/patients involved" for choice in choices)


def analyze_human_participation(df: pd.DataFrame) -> Dict:
    """
    Analyze human participation statistics in empirical clinical studies.
    
    Args:
        df (pd.DataFrame): Filtered dataframe of empirical clinical studies
        
    Returns:
        dict: Dictionary containing:
            - total_studies: Total number of studies in filtered dataset
            - studies_with_humans: Number of studies with human participants
            - studies_without_humans: Number of studies without human participants
            - percent_with_humans: Percentage of studies with human participants
            - percent_without_humans: Percentage of studies without human participants
            - sample_sizes: List of sample sizes for studies with human participants
            - sample_size_stats: Statistics about sample sizes
    """
    if 'client_type' not in df.columns:
        raise KeyError("Column 'client_type' not found in dataframe")
    
    total = len(df)
    
    # Determine which studies have human participants
    df_temp = df.copy()
    df_temp['has_humans'] = df_temp['client_type'].apply(has_human_participants)
    
    with_humans = df_temp['has_humans'].sum()
    without_humans = total - with_humans
    
    # Get sample sizes for studies with human participants
    human_studies = df_temp[df_temp['has_humans'] == True].copy()
    sample_sizes = []
    sample_size_info = []
    
    if 'client_count' in df.columns:
        for idx, row in human_studies.iterrows():
            client_count = row.get('client_count')
            client_type = row.get('client_type', '')
            
            # Try to parse client_count as integer
            if pd.notna(client_count):
                try:
                    size = int(float(str(client_count)))
                    if size > 0:
                        sample_sizes.append(size)
                        sample_size_info.append({
                            'study_id': row.get('study_id', 'Unknown'),
                            'title': row.get('title', 'Unknown Title'),
                            'size': size,
                            'client_type': client_type
                        })
                except (ValueError, TypeError):
                    # Handle non-numeric values
                    sample_size_info.append({
                        'study_id': row.get('study_id', 'Unknown'),
                        'title': row.get('title', 'Unknown Title'),
                        'size': str(client_count),
                        'client_type': client_type
                    })
            else:
                # Handle missing client_count data
                sample_size_info.append({
                    'study_id': row.get('study_id', 'Unknown'),
                    'title': row.get('title', 'Unknown Title'),
                    'size': 'Not specified',
                    'client_type': client_type
                })
    
    # Calculate sample size statistics
    sample_size_stats = {}
    if sample_sizes:
        import numpy as np
        sample_size_stats = {
            'count': len(sample_sizes),
            'min': min(sample_sizes),
            'max': max(sample_sizes),
            'mean': np.mean(sample_sizes),
            'median': np.median(sample_sizes),
            'total': sum(sample_sizes)
        }
    
    return {
        'total_studies': total,
        'studies_with_humans': with_humans,
        'studies_without_humans': without_humans,
        'percent_with_humans': (with_humans / total * 100) if total > 0 else 0,
        'percent_without_humans': (without_humans / total * 100) if total > 0 else 0,
        'sample_sizes': sample_sizes,
        'sample_size_info': sample_size_info,
        'sample_size_stats': sample_size_stats
    }


def get_client_type_breakdown(df: pd.DataFrame) -> Dict:
    """
    Get breakdown of all client_type values and their frequencies.
    
    For multiple-choice columns (semicolon-separated), counts each individual choice
    and also tracks unique combinations with study IDs for data review.
    
    Args:
        df (pd.DataFrame): Dataframe containing client_type column
        
    Returns:
        dict: Dictionary containing:
            - total_responses: Total number of non-null client_type entries
            - null_responses: Number of null/empty client_type entries
            - total_choices: Total number of individual choices (after splitting semicolons)
            - unique_choices: Number of unique choice options
            - choice_counts: Dictionary mapping each choice to its count
            - combination_counts: Dictionary mapping each unique combination to its count
            - human_participant_combinations: Combinations that involve human participants
            - combination_details: Dictionary mapping combinations to study IDs and raw values
    """
    if 'client_type' not in df.columns:
        raise KeyError("Column 'client_type' not found in dataframe")
    
    from collections import Counter, defaultdict
    
    all_choices = []
    all_combinations = []
    human_participant_combinations = []
    combination_details = defaultdict(lambda: {'study_ids': [], 'raw_values': [], 'count': 0})
    
    total_responses = df['client_type'].notna().sum()
    null_responses = len(df) - total_responses
    
    # Get study ID column name (could be 'study_id', 'id', or index)
    study_id_col = None
    if 'study_id' in df.columns:
        study_id_col = 'study_id'
    elif 'id' in df.columns:
        study_id_col = 'id'
    
    for idx, row in df.iterrows():
        value = row['client_type']
        if pd.notna(value):
            choices = parse_client_type(value)
            all_choices.extend(choices)
            
            # Store the original combination (sorted for consistency)
            if choices:
                combination = "; ".join(sorted(choices))
                all_combinations.append(combination)
                
                # Store details for data review
                study_id = row[study_id_col] if study_id_col else f"row_{idx}"
                combination_details[combination]['study_ids'].append(study_id)
                combination_details[combination]['raw_values'].append(value)
                combination_details[combination]['count'] += 1
                
                # Check if this combination involves human participants
                if not all(choice == "No clients/patients involved" for choice in choices):
                    human_participant_combinations.append(combination)
    
    choice_counts = Counter(all_choices)
    combination_counts = Counter(all_combinations)
    human_combination_counts = Counter(human_participant_combinations)
    
    return {
        'total_responses': total_responses,
        'null_responses': null_responses,
        'total_choices': len(all_choices),
        'unique_choices': len(choice_counts),
        'choice_counts': dict(choice_counts),
        'combination_counts': dict(combination_counts),
        'human_participant_combinations': dict(human_combination_counts),
        'combination_details': dict(combination_details)
    }


def generate_clinical_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive text report on human participation in clinical studies.
    
    Args:
        df (pd.DataFrame): Input dataframe (will be filtered internally)
        
    Returns:
        str: Formatted text report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("CLINICAL PANEL REPORT")
    report_lines.append("Human Participation in Empirical LLM Studies with Client-Facing Applications")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Filter to clinical empirical studies
    try:
        filtered_df = filter_clinical_empirical_studies(df)
    except KeyError as e:
        return f"Error: {e}"
    
    if filtered_df.empty:
        report_lines.append("No studies found matching criteria:")
        report_lines.append("  - study_type == 'Empirical research involving an LLM'")
        report_lines.append("  - application_type == 'Client-facing application'")
        return "\n".join(report_lines)
    
    report_lines.append(f"Dataset Overview:")
    report_lines.append(f"  Total studies matching criteria: {len(filtered_df)}")
    report_lines.append(f"    (study_type == 'Empirical research involving an LLM' AND")
    report_lines.append(f"     application_type == 'Client-facing application')")
    report_lines.append("")
    
    # Human participation analysis
    report_lines.append("-" * 80)
    report_lines.append("HUMAN PARTICIPATION ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    participation_stats = analyze_human_participation(filtered_df)
    
    report_lines.append(f"Studies involving human participants:")
    report_lines.append(f"  Yes: {participation_stats['studies_with_humans']} "
                       f"({participation_stats['percent_with_humans']:.1f}%)")
    report_lines.append(f"  No:  {participation_stats['studies_without_humans']} "
                       f"({participation_stats['percent_without_humans']:.1f}%)")
    report_lines.append("")
    report_lines.append(f"Note: 'Human participants' means client_type is not 'No clients/patients involved'")
    report_lines.append("")
    
    # Add sample size analysis
    if participation_stats['sample_size_stats']:
        stats = participation_stats['sample_size_stats']
        report_lines.append("Sample Size Statistics (for studies with human participants):")
        report_lines.append(f"  Studies with numeric sample sizes: {stats['count']}")
        report_lines.append(f"  Total participants across all studies: {stats['total']:,}")
        report_lines.append(f"  Sample size range: {stats['min']} - {stats['max']}")
        report_lines.append(f"  Mean sample size: {stats['mean']:.1f}")
        report_lines.append(f"  Median sample size: {stats['median']:.1f}")
        report_lines.append("")
        
        # Show individual sample sizes
        if participation_stats['sample_size_info']:
            report_lines.append("Individual Study Sample Sizes:")
            # Sort by sample size (descending) for numeric values, then by study_id
            sorted_info = sorted(participation_stats['sample_size_info'], 
                               key=lambda x: (isinstance(x['size'], int), -x['size'] if isinstance(x['size'], int) else 0, x['study_id']))
            
            for info in sorted_info:
                size_str = f"{info['size']:,}" if isinstance(info['size'], int) else str(info['size'])
                client_type_short = info['client_type'][:50] + "..." if len(info['client_type']) > 50 else info['client_type']
                
                # For non-numeric sample sizes, show both study ID and title
                if not isinstance(info['size'], int):
                    title_short = info.get('title', 'Unknown Title')[:60] + "..." if len(str(info.get('title', 'Unknown Title'))) > 60 else str(info.get('title', 'Unknown Title'))
                    report_lines.append(f"  Study {info['study_id']} ({title_short}): {size_str} participants ({client_type_short})")
                else:
                    report_lines.append(f"  Study {info['study_id']}: {size_str} participants ({client_type_short})")
            report_lines.append("")
    else:
        report_lines.append("No numeric sample size data available for studies with human participants.")
        report_lines.append("")
    
    # Client type breakdown
    report_lines.append("-" * 80)
    report_lines.append("CLIENT TYPE BREAKDOWN")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    breakdown = get_client_type_breakdown(filtered_df)
    
    report_lines.append(f"Total responses: {breakdown['total_responses']}")
    report_lines.append(f"Null/empty responses: {breakdown['null_responses']}")
    report_lines.append(f"Total individual choices: {breakdown['total_choices']}")
    report_lines.append(f"Unique choice options: {breakdown['unique_choices']}")
    report_lines.append("")
    
    report_lines.append("Individual Choice Counts:")
    # Sort by count (descending)
    sorted_choices = sorted(breakdown['choice_counts'].items(), 
                           key=lambda x: x[1], reverse=True)
    
    for choice, count in sorted_choices:
        percentage = (count / breakdown['total_choices'] * 100) if breakdown['total_choices'] > 0 else 0
        report_lines.append(f"  {choice}: {count} ({percentage:.1f}%)")
    
    report_lines.append("")
    
    # Add unique combinations section
    report_lines.append("-" * 80)
    report_lines.append("UNIQUE CLIENT TYPE COMBINATIONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    report_lines.append("All unique combinations:")
    sorted_combinations = sorted(breakdown['combination_counts'].items(), 
                                key=lambda x: x[1], reverse=True)
    
    for combination, count in sorted_combinations:
        percentage = (count / breakdown['total_responses'] * 100) if breakdown['total_responses'] > 0 else 0
        report_lines.append(f"  \"{combination}\": {count} ({percentage:.1f}%)")
    
    report_lines.append("")
    
    # Add detailed combinations section for data review
    if 'combination_details' in breakdown:
        report_lines.append("-" * 80)
        report_lines.append("DETAILED CLIENT TYPE COMBINATIONS (FOR DATA REVIEW)")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        sorted_combinations = sorted(breakdown['combination_details'].items(), 
                                    key=lambda x: x[1]['count'], reverse=True)
        
        for combination, details in sorted_combinations:
            count = details['count']
            percentage = (count / breakdown['total_responses'] * 100) if breakdown['total_responses'] > 0 else 0
            report_lines.append(f"Combination: \"{combination}\" - {count} studies ({percentage:.1f}%)")
            
            # Show study IDs and raw values
            study_ids = details['study_ids']
            raw_values = details['raw_values']
            
            for i, (study_id, raw_value) in enumerate(zip(study_ids, raw_values)):
                if i < 10:  # Limit to first 10 studies to avoid overly long reports
                    report_lines.append(f"  - {study_id}: \"{raw_value}\"")
                elif i == 10:
                    report_lines.append(f"  - ... and {len(study_ids) - 10} more studies")
                    break
            
            report_lines.append("")
    
    # Add human participant combinations section
    if breakdown['human_participant_combinations']:
        report_lines.append("-" * 80)
        report_lines.append("COMBINATIONS INVOLVING HUMAN PARTICIPANTS")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        sorted_human_combinations = sorted(breakdown['human_participant_combinations'].items(), 
                                          key=lambda x: x[1], reverse=True)
        
        for combination, count in sorted_human_combinations:
            percentage = (count / breakdown['total_responses'] * 100) if breakdown['total_responses'] > 0 else 0
            report_lines.append(f"  \"{combination}\": {count} ({percentage:.1f}%)")
        
        report_lines.append("")
    else:
        report_lines.append("No combinations involving human participants found.")
        report_lines.append("")
    
    # Add validation and assessment analysis
    report_lines.append("-" * 80)
    report_lines.append("VALIDATION AND ASSESSMENT ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Get studies with human participants for this analysis
    filtered_df_temp = filtered_df.copy()
    filtered_df_temp['has_humans'] = filtered_df_temp['client_type'].apply(has_human_participants)
    human_studies_df = filtered_df_temp[filtered_df_temp['has_humans'] == True].copy()
    
    if not human_studies_df.empty:
        # Analysis 1: f1_validated_outcomes_considered == 'y'
        validated_outcomes = human_studies_df[human_studies_df['f1_validated_outcomes_considered'] == 'y']
        report_lines.append(f"Studies with human participants using validated outcomes:")
        report_lines.append(f"  Count: {len(validated_outcomes)} out of {len(human_studies_df)} "
                           f"({len(validated_outcomes)/len(human_studies_df)*100:.1f}%)")
        report_lines.append("")
        
        # Analysis 2: UX assessment in remaining studies
        remaining_studies = human_studies_df[human_studies_df['f1_validated_outcomes_considered'] != 'y']
        ux_assessment_studies = remaining_studies[remaining_studies['ux_assessment_is_present'] == 'Yes']
        report_lines.append(f"Studies without validated outcomes that have UX assessment:")
        report_lines.append(f"  Count: {len(ux_assessment_studies)} out of {len(remaining_studies)} remaining studies "
                           f"({len(ux_assessment_studies)/len(remaining_studies)*100:.1f}% of remaining)" if len(remaining_studies) > 0 else "  Count: 0 (no remaining studies)")
        report_lines.append("")
        
        # Analysis 3: Studies with neither validated outcomes nor UX assessment
        neither_studies = remaining_studies[remaining_studies['ux_assessment_is_present'] != 'Yes']
        
        if not neither_studies.empty:
            report_lines.append(f"Studies with NEITHER validated outcomes NOR UX assessment:")
            report_lines.append(f"  Count: {len(neither_studies)} out of {len(human_studies_df)} total "
                               f"({len(neither_studies)/len(human_studies_df)*100:.1f}%)")
            report_lines.append("")
            
            report_lines.append("Details of studies with neither validation method:")
            for idx, row in neither_studies.iterrows():
                study_name = row.get('title', 'Unknown Title')[:60] + "..." if len(str(row.get('title', 'Unknown Title'))) > 60 else str(row.get('title', 'Unknown Title'))
                study_id = row.get('study_id', 'Unknown ID')
                
                # Get metric names
                metric1 = row.get('metric1_name', 'Not specified')
                metric2 = row.get('metric2_name', 'Not specified') 
                metric3 = row.get('metric3_name', 'Not specified')
                
                report_lines.append(f"  Study {study_id}:")
                report_lines.append(f"    Title: {study_name}")
                report_lines.append(f"    Metric 1: {metric1}")
                report_lines.append(f"    Metric 2: {metric2}")
                report_lines.append(f"    Metric 3: {metric3}")
                report_lines.append("")
        else:
            report_lines.append("All studies with human participants use either validated outcomes or UX assessment.")
            report_lines.append("")
    else:
        report_lines.append("No studies with human participants found for validation analysis.")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def load_figure_config() -> Dict:
    """Load figure configuration from YAML file."""
    config_path = Path(__file__).parent / "figure_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def categorize_sample_size(size: int, categories: Dict[str, str]) -> str:
    """
    Categorize a sample size based on predefined ranges.
    
    Args:
        size (int): Sample size to categorize
        categories (dict): Dictionary mapping category names to range descriptions
        
    Returns:
        str: Category name
    """
    if size == 1:
        return "Case studies"
    elif 2 <= size <= 20:
        return "Very small"
    elif 21 <= size <= 100:
        return "Small"
    elif 101 <= size <= 500:
        return "Medium"
    elif 501 <= size <= 2000:
        return "Large"
    elif size > 2000:
        return "Very large"
    else:
        return "Unknown"


def create_sample_size_figure(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a horizontal bar chart of sample size categories for studies with human participants.
    
    Args:
        df (pd.DataFrame): Pre-filtered dataframe (client-facing empirical LLM studies with clients involved)
        save_path (Optional[str]): Path to save the figure as SVG
        
    Returns:
        plt.Figure: The created figure
    """
    # Load configuration
    config = load_figure_config()
    color_palette = get_color_palette()
    sample_size_categories = config['sample_size_categories']
    
    # Use the pre-filtered data directly
    filtered_df = df
    
    # Get participation analysis to extract sample sizes
    participation_stats = analyze_human_participation(filtered_df)
    sample_sizes = participation_stats['sample_sizes']
    sample_size_info = participation_stats['sample_size_info']
    
    # Calculate the total number of studies with human participants
    total_human_participant_studies = len(sample_size_info)
    
    if not sample_sizes and not sample_size_info:
        # Create empty figure if no data
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, 'No sample size data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Categorize sample sizes
    categorized = [categorize_sample_size(size, sample_size_categories) for size in sample_sizes]
    
    # Count categories
    from collections import Counter
    category_counts = Counter(categorized)
    
    # Count unknown sample sizes (studies with non-numeric or missing sample sizes)
    unknown_count = 0
    for info in sample_size_info:
        if not isinstance(info['size'], int):
            unknown_count += 1
    
    # Define category order (from smallest to largest)
    category_order = ["Case studies", "Very small", "Small", "Medium", "Large", "Very large"]
    
    # Create data for plotting - only include categories that have data
    plot_data = {}
    color_mapping = {}
    for category in category_order:
        if category in category_counts:
            count = category_counts[category]
            range_desc = sample_size_categories[category]
            # Create label combining category and range
            label = f"{category}\n({range_desc})"
            plot_data[label] = count
            color_mapping[label] = color_palette["crimson"]
    
    # Add unknown category if there are any unknown sample sizes
    if unknown_count > 0:
        unknown_label = "Sample size\nnot specified"
        plot_data[unknown_label] = unknown_count
        color_mapping[unknown_label] = color_palette["grey"]
    
    if not plot_data:
        # Create empty figure if no valid categories
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, 'No valid sample size categories found', 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create category order list with proper labels for ordering
    category_order_labels = []
    
    # Add unknown to the beginning of the category order (opposite of case studies)
    if unknown_count > 0:
        category_order_labels.append("Sample size\nnot specified")
    
    for category in category_order:
        if category in category_counts:
            range_desc = sample_size_categories[category]
            label = f"{category}\n({range_desc})"
            category_order_labels.append(label)
    
    # Use the create_horizontal_bar_chart function
    fig = create_horizontal_bar_chart(
        data=plot_data,
        title=f"Human participant sample sizes\n(n={total_human_participant_studies} human participant studies)",
        xlabel="Number of articles",
        ylabel=None,
        figsize=(5, 4),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        category_order=category_order_labels,  # Maintain specific order (smallest to largest)
        label_fontsize=10,
        title_fontsize=14,
        bar_height=0.6
    )
    
    return fig


def create_inclusion_criteria_figure(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a horizontal bar chart of inclusion criteria for studies with human participants.
    
    Args:
        df (pd.DataFrame): Pre-filtered dataframe (client-facing empirical LLM studies with clients involved)
        save_path (Optional[str]): Path to save the figure as SVG
        
    Returns:
        plt.Figure: The created figure
    """
    # Load color palette
    color_palette = get_color_palette()
    
    # Use the pre-filtered data directly
    filtered_df = df
    
    # Check if inclusion_criteria column exists
    if 'inclusion_criteria' not in filtered_df.columns:
        # Create empty figure if column doesn't exist
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, 'inclusion_criteria column not found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Get inclusion criteria data, handling NaN values
    inclusion_data = filtered_df['inclusion_criteria'].dropna()
    
    if inclusion_data.empty:
        # Create empty figure if no data
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, 'No inclusion criteria data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Count occurrences of each inclusion criteria
    from collections import Counter
    criteria_counts = Counter(inclusion_data)
    
    # Define the mapping from raw data to display labels
    criteria_display_mapping = {
        "Unspecified": "Inclusion criteria not specified",
        "Diagnosed mental disorder (ICD/DSM)": "Diagnosed mental disorder (ICD/DSM)",
        "Recruited in in- or outpatient healthcare setting with no specific criteria": "No specific criteria,\nin- or outpatient recruitment setting",
        "Elevated symptoms": "Elevated symptoms",
        "No specific criteria": "No specific criteria,\nnon-clinical recruitment setting"
    }
    
    # Define the display order (top to bottom)
    display_order = [
        "Inclusion criteria not specified",
        "Diagnosed mental disorder (ICD/DSM)",
        "No specific criteria,\nin- or outpatient recruitment setting",
        "Elevated symptoms",
        "No specific criteria,\nnon-clinical recruitment setting"
    ]
    
    # Create ordered data dictionary and color mapping using display labels
    ordered_data = {}
    color_mapping = {}
    
    for display_label in display_order:
        # Find the corresponding raw data key
        raw_key = None
        for raw, display in criteria_display_mapping.items():
            if display == display_label:
                raw_key = raw
                break
        
        if raw_key and raw_key in criteria_counts:
            ordered_data[display_label] = criteria_counts[raw_key]
            # Use grey for "Unspecified", dark_grey for "No specific criteria", crimson for others
            if display_label == "Inclusion criteria not specified":
                color_mapping[display_label] = color_palette["grey"]
            elif display_label == "No specific criteria,\nnon-clinical recruitment setting":
                color_mapping[display_label] = color_palette["dark_grey"]
            else:
                color_mapping[display_label] = color_palette["crimson"]
    
    # Add any remaining criteria not in the predefined mapping
    for raw_criteria, count in criteria_counts.items():
        if raw_criteria not in criteria_display_mapping:
            ordered_data[raw_criteria] = count
            color_mapping[raw_criteria] = color_palette["sky"]
    
    # Use the create_horizontal_bar_chart function
    fig = create_horizontal_bar_chart(
        data=ordered_data,
        title="Use of psychopathology-related inclusion\ncriteria on human participants",
        xlabel="Number of articles",
        ylabel=None,
        figsize=(6, 4),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        category_order=list(ordered_data.keys()),  # Maintain the specific order
        label_fontsize=10,
        title_fontsize=14,
        bar_height=0.6,
        max_label_length=200  # Allow much longer labels for inclusion criteria
    )
    
    return fig


def create_combined_sample_size_inclusion_criteria_figure(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a combined figure with sample size distribution and inclusion criteria charts stacked vertically.
    
    This function reuses the individual chart functions and combines them into subplots,
    following DRY principles and software engineering best practices.
    
    Args:
        df (pd.DataFrame): Pre-filtered dataframe (client-facing empirical LLM studies with clients involved)
        save_path (Optional[str]): Path to save the figure as SVG
        
    Returns:
        plt.Figure: The created figure with two subplots
    """
    # Create figure with two subplots stacked vertically, with more width for long labels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    
    # Generate individual charts without saving them
    sample_size_fig = create_sample_size_figure(df, save_path=None)
    inclusion_criteria_fig = create_inclusion_criteria_figure(df, save_path=None)
    
    # Extract data from the individual figures and recreate in subplots
    
    # === TOP SUBPLOT: Sample Size Distribution ===
    # Get the axes from the sample size figure
    sample_ax = sample_size_fig.axes[0]
    
    # Standard bar height for consistency
    bar_height = 0.6
    
    # Copy the bar chart to our subplot
    for bar in sample_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y() 
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax1.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in sample_ax.texts:
        ax1.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax1.set_ylim(sample_ax.get_ylim())
    ax1.set_yticks(sample_ax.get_yticks())
    ax1.set_yticklabels([t.get_text() for t in sample_ax.get_yticklabels()], fontsize=10)
    ax1.set_title(sample_ax.get_title(), fontsize=14, pad=20)
    
    # Remove x-axis tick labels from top subplot
    ax1.set_xticklabels([])
    
    # Copy styling
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax1.grid(axis='y', visible=False)
    
    # === BOTTOM SUBPLOT: Inclusion Criteria Distribution ===
    # Get the axes from the inclusion criteria figure
    criteria_ax = inclusion_criteria_fig.axes[0]
    
    # Copy the bar chart to our subplot
    for bar in criteria_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax2.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in criteria_ax.texts:
        ax2.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax2.set_ylim(criteria_ax.get_ylim())
    ax2.set_yticks(criteria_ax.get_yticks())
    ax2.set_yticklabels([t.get_text() for t in criteria_ax.get_yticklabels()], fontsize=10)
    ax2.set_title(criteria_ax.get_title(), fontsize=14, pad=20)
    ax2.set_xlabel("Number of Studies", fontsize=12)
    
    # Copy styling
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax2.grid(axis='y', visible=False)
    
    # Determine the maximum x-limit from both charts for consistent scaling
    max_xlim = max(sample_ax.get_xlim()[1], criteria_ax.get_xlim()[1])
    
    # Set the same x-axis limits for both subplots
    ax1.set_xlim(0, max_xlim)
    ax2.set_xlim(0, max_xlim)
    
    # Ensure the bottom subplot has proper x-axis tick labels
    ax2.tick_params(axis='x', which='major', labelsize=10)
    
    # Close the individual figures to free memory
    plt.close(sample_size_fig)
    plt.close(inclusion_criteria_fig)
    
    # Adjust layout with increased spacing between subplots and more left margin for long labels
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, left=0.35)  # Increase vertical spacing and left margin
    
    # Save if path provided
    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    return fig


def create_combined_intervention_control_outcome_figure(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a combined figure with intervention types, control group presence, and primary clinical outcome charts stacked vertically.
    
    This function reuses the individual chart functions and combines them into subplots,
    following DRY principles and software engineering best practices.
    
    Args:
        df (pd.DataFrame): Pre-filtered dataframe (client-facing empirical LLM studies with clients involved)
        save_path (Optional[str]): Path to save the figure as SVG
        
    Returns:
        plt.Figure: The created figure with three subplots
    """
    # Internal configuration for subplot proportions
    # These height ratios control the relative size of each subplot
    # Values represent the relative height allocation for each subplot
    # Example: [3, 1, 2] means intervention_types gets 3/6, control_group gets 1/6, clinical_outcomes gets 2/6 of total height
    # Adjust these values to change proportions:
    # - Increase a value to make that subplot taller
    # - Decrease a value to make that subplot shorter
    # - Equal values (e.g., [1, 1, 1]) create equal-height subplots
    subplot_height_ratios = [3, 0.6, 1.0]  # [intervention_types, control_group, clinical_outcomes]
    
    # Create figure with three subplots stacked vertically with custom height ratios
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, 
        figsize=(6, 10),
        gridspec_kw={'height_ratios': subplot_height_ratios}
    )
    
    # Generate individual charts without saving them
    intervention_fig = create_intervention_type_figure(df, save_path=None)
    control_fig = create_control_group_chart(df, save_path=None)
    outcome_fig = create_primary_clinical_outcome_chart(df, save_path=None)
    
    # Standard bar height for consistency
    bar_height = 0.6
    
    # === TOP SUBPLOT: Intervention Types ===
    # Get the axes from the intervention figure
    intervention_ax = intervention_fig.axes[0]
    
    # Copy the bar chart to our subplot
    for bar in intervention_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y() 
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax1.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in intervention_ax.texts:
        ax1.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax1.set_ylim(intervention_ax.get_ylim())
    ax1.set_yticks(intervention_ax.get_yticks())
    ax1.set_yticklabels([t.get_text() for t in intervention_ax.get_yticklabels()], fontsize=10)
    ax1.set_title(intervention_ax.get_title(), fontsize=14, pad=20)
    
    # Remove x-axis tick labels from top subplot
    ax1.set_xticklabels([])
    
    # Copy styling
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax1.grid(axis='y', visible=False)
    
    # === MIDDLE SUBPLOT: Control Group Presence ===
    # Get the axes from the control group figure
    control_ax = control_fig.axes[0]
    
    # Copy the bar chart to our subplot
    for bar in control_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax2.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in control_ax.texts:
        ax2.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax2.set_ylim(control_ax.get_ylim())
    ax2.set_yticks(control_ax.get_yticks())
    ax2.set_yticklabels([t.get_text() for t in control_ax.get_yticklabels()], fontsize=10)
    ax2.set_title(control_ax.get_title(), fontsize=14, pad=20)
    
    # Remove x-axis tick labels from middle subplot
    ax2.set_xticklabels([])
    
    # Copy styling
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax2.grid(axis='y', visible=False)
    
    # === BOTTOM SUBPLOT: Primary Clinical Outcome ===
    # Get the axes from the outcome figure
    outcome_ax = outcome_fig.axes[0]
    
    # Copy the bar chart to our subplot
    for bar in outcome_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax3.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in outcome_ax.texts:
        ax3.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax3.set_ylim(outcome_ax.get_ylim())
    ax3.set_yticks(outcome_ax.get_yticks())
    ax3.set_yticklabels([t.get_text() for t in outcome_ax.get_yticklabels()], fontsize=10)
    ax3.set_title(outcome_ax.get_title(), fontsize=14, pad=20)
    ax3.set_xlabel("Number of Studies", fontsize=12)
    
    # Copy styling
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax3.grid(axis='y', visible=False)
    
    # Determine the maximum x-limit from all three charts for consistent scaling
    max_xlim = max(
        intervention_ax.get_xlim()[1], 
        control_ax.get_xlim()[1], 
        outcome_ax.get_xlim()[1]
    )
    
    # Set the same x-axis limits for all subplots
    ax1.set_xlim(0, max_xlim)
    ax2.set_xlim(0, max_xlim)
    ax3.set_xlim(0, max_xlim)
    
    # Ensure the bottom subplot has proper x-axis tick labels
    ax3.tick_params(axis='x', which='major', labelsize=10)
    
    # Close the individual figures to free memory
    plt.close(intervention_fig)
    plt.close(control_fig)
    plt.close(outcome_fig)
    
    # Adjust layout with increased spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing between subplots
    
    # Save if path provided
    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    
    return fig


def create_intervention_type_figure(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a horizontal bar chart of intended intervention types for studies with human participants.
    
    Args:
        df (pd.DataFrame): Pre-filtered dataframe (client-facing empirical LLM studies with clients involved)
        save_path (Optional[str]): Path to save the figure as SVG
        
    Returns:
        plt.Figure: The created figure
    """
    # Import here to avoid circular imports
    from .utils import create_horizontal_bar_chart
    
    # Load configuration
    config = load_figure_config()
    color_palette = get_color_palette()
    first_color = list(color_palette.values())[0]  # Use first color from palette
    
    # Use the pre-filtered data directly
    filtered_df = df
    
    # 2. Filter to only studies with human participants
    filtered_df_temp = filtered_df.copy()
    filtered_df_temp['has_humans'] = filtered_df_temp['client_type'].apply(has_human_participants)
    human_studies_df = filtered_df_temp[filtered_df_temp['has_humans'] == True].copy()
    
    # Calculate the total number of studies with human participants
    total_human_participant_studies = len(human_studies_df)
    
    # Get intervention type counts
    intervention_counts = human_studies_df['intended_intervention_type'].value_counts()
    
    if intervention_counts.empty:
        # Create empty figure if no data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No intervention type data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
    
    # Define custom ordering with "Other" at the bottom and custom colors
    intervention_order = []
    color_mapping = {}
    
    # Add all categories except "Other" first (in descending order by count)
    for category in intervention_counts.index:
        if category != "Other":
            intervention_order.append(category)
            # Use dark_grey for "Non-specific support", crimson for others
            if category == "Non-specific support":
                color_mapping[category] = color_palette["dark_grey"]
            else:
                color_mapping[category] = color_palette["crimson"]
    
    # Add "Other" at the end (will appear at bottom of chart)
    if "Other" in intervention_counts.index:
        intervention_order.append("Other")
        color_mapping["Other"] = color_palette["grey"]
    
    # Reorder the data according to our custom order
    ordered_data = {category: intervention_counts[category] for category in intervention_order}
    
    # Create intervention type bar chart using studies with human participants only
    fig = create_horizontal_bar_chart(
        data=ordered_data,
        title=f"Therapeutic modalities\nmodeled in client-facing applications\n(n={total_human_participant_studies} human participant studies)",
        xlabel="Number of articles",
        ylabel=None,
        color=color_mapping,
        figsize=(12, 8),
        save_path=save_path,
        show_percentages=True,
        category_order=intervention_order[::-1],  # Use our custom order
        sort_ascending=False  # Don't sort, use our custom order
    )
    
    return fig


def create_primary_clinical_outcome_chart(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Create a horizontal bar chart showing the distribution of primary clinical outcomes.
    
    This function expects pre-filtered data (client-facing empirical LLM studies with clients involved)
    and creates a chart showing the distribution of primary clinical outcome types.
    
    Args:
        df (pd.DataFrame): Pre-filtered dataframe with primary_clinical_outcome column
        save_path (Optional[str]): Path to save the figure (optional)
        figsize (Tuple[float, float]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created matplotlib figure
        
    Raises:
        KeyError: If required column is not found
        ValueError: If no valid data is found for plotting
    """
    # Check for required column
    if 'primary_clinical_outcome' not in df.columns:
        raise KeyError("Column 'primary_clinical_outcome' not found in dataframe")
    
    # Filter to only rows with filled primary_clinical_outcome (non-empty)
    filtered_df = df[df['primary_clinical_outcome'].notna() & (df['primary_clinical_outcome'] != '')].copy()
    
    if filtered_df.empty:
        raise ValueError("No studies found with primary clinical outcome data")
    
    # Count occurrences of each outcome type
    outcome_counts = filtered_df['primary_clinical_outcome'].value_counts()
    
    color_palette = get_color_palette()
    outcome_colors = {
        'Validated symptom/function scale': color_palette['crimson'],
        'User experience assessment': color_palette['crimson'],
        'Other': color_palette['grey']
    }
    
    # Define preferred order (most rigorous first)
    preferred_order = [
        'Other',
        'User experience assessment',
        'Validated symptom/function scale'
    ]
    
    # Create the horizontal bar chart
    fig = create_horizontal_bar_chart(
        data=outcome_counts,
        title="Primary clinical outcome",
        xlabel="Number of articles",
        ylabel=None,
        figsize=figsize,
        color=outcome_colors,
        category_order=preferred_order,
        show_percentages=True,
        save_path=save_path,
        bar_height=0.6,
        label_fontsize=11
    )
    
    return fig


def create_control_group_chart(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 4)
) -> plt.Figure:
    """
    Create a horizontal bar chart showing the proportion of studies with and without control groups.
    
    This function expects pre-filtered data (client-facing empirical LLM studies with clients involved)
    and analyzes the f2_control_condition_considered column.
    
    Args:
        df (pd.DataFrame): Pre-filtered dataframe with f2_control_condition_considered column
        save_path (Optional[str]): Path to save the figure (optional)
        figsize (Tuple[float, float]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created matplotlib figure
        
    Raises:
        KeyError: If required column is not found
        ValueError: If no valid data is found for plotting
    """
    # Check for required column
    if 'f2_control_condition_considered' not in df.columns:
        raise KeyError("Column 'f2_control_condition_considered' not found in dataframe")
    
    # Filter to only rows with non-null control group data
    filtered_df = df[df['f2_control_condition_considered'].notna()].copy()
    
    if filtered_df.empty:
        raise ValueError("No studies found with control group data")
    
    # Count occurrences and map to meaningful labels
    control_counts = filtered_df['f2_control_condition_considered'].str.lower().value_counts()
    
    # Map values to meaningful labels
    label_mapping = {
        'y': 'Control group present',
        'n': 'No control group',
        'yes': 'Control group present',
        'no': 'No control group'
    }
    
    # Apply mapping and aggregate
    mapped_counts = {}
    for value, count in control_counts.items():
        mapped_label = label_mapping.get(value, f"Unknown ({value})")
        if mapped_label in mapped_counts:
            mapped_counts[mapped_label] += count
        else:
            mapped_counts[mapped_label] = count
    
    if not mapped_counts:
        raise ValueError("No valid control group data found")
    
    color_dict = get_color_palette()
    control_colors = {
        'Control group present': color_dict['crimson'],
        'No control group': color_dict['grey']
    }

    # Define preferred order (control group first)
    preferred_order = [
        'No control group',
        'Control group present'
    ]
    
    # Create the horizontal bar chart
    fig = create_horizontal_bar_chart(
        data=mapped_counts,
        title="Presence of control group",
        xlabel="Number of articles",
        ylabel=None,
        figsize=figsize,
        color=control_colors,
        category_order=preferred_order,
        show_percentages=True,
        save_path=save_path,
        bar_height=0.6,
        label_fontsize=11
    )
    
    return fig



def create_time_series_human_participation_chart(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Create a stacked time series bar chart showing publications over time by human participation status.
    
    This function filters the data to include only client-facing empirical LLM studies,
    then creates a stacked bar chart showing the number of publications per year,
    stacked by whether they include human participants or not.
    
    Args:
        df (pd.DataFrame): Input dataframe with year, application_type, study_type, and client_type columns
        save_path (Optional[str]): Path to save the figure (optional)
        figsize (Tuple[float, float]): Figure size as (width, height)
        
    Returns:
        plt.Figure: The created matplotlib figure
        
    Raises:
        KeyError: If required columns are not found
        ValueError: If no valid data is found for plotting
    """
    # Check for required columns
    required_columns = ['year', 'application_type', 'study_type', 'client_type']
    missing_columns = [col for col in required_columns if col not in df.columns]

    color_dict = get_color_palette()
    participation_colors = {
        'With human participants': color_dict['crimson'],
        'No human participants': color_dict['navy']
    }
    
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    # Filter to client-facing empirical LLM studies (less restrictive than other charts)
    client_facing_mask = df['application_type'] == 'Client-facing application'
    empirical_llm_mask = df['study_type'] == 'Empirical research involving an LLM'
    
    filtered_df = df[client_facing_mask & empirical_llm_mask].copy()
    
    if filtered_df.empty:
        raise ValueError("No client-facing empirical LLM studies found")
    
    # Create human participation category
    filtered_df['has_humans'] = filtered_df['client_type'].apply(has_human_participants)
    filtered_df['human_participation'] = filtered_df['has_humans'].map({
        True: 'With human participants',
        False: 'No human participants'
    })
    
    # Filter out rows with missing year data
    filtered_df = filtered_df[filtered_df['year'].notna()].copy()
    
    if filtered_df.empty:
        raise ValueError("No studies found with valid year data")
    
    # Create the time series chart
    fig = create_time_series_chart(
        df=filtered_df,
        year_column='year',
        category_column='human_participation',
        title="Client-facing applications: publication numbers\nwith and without human participants over time",
        xlabel="Publication year",
        ylabel="Number of articles",
        figsize=figsize,
        colors=participation_colors,
        bar_width=0.8,
        show_counts=True,
        show_totals=True,
        save_path=save_path,
        legend_fontsize=12,
        label_fontsize=11,
        title_fontsize=14
    )
    
    return fig
