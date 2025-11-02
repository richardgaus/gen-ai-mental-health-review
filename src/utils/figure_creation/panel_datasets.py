"""
Panel Datasets Metrics Utilities

This module provides functions for creating visualizations and metrics
related to dataset characteristics in the LLMs in psychotherapy study.
"""

import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from .utils import create_horizontal_bar_chart, get_color_palette


def create_dataset_type_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "dataset_type_high_level_distribution.svg"
) -> str:
    """
    Create a horizontal bar chart showing the distribution of high-level dataset types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information with 'dataset_type_high_level' column
    save_dir : str
        Directory to save the chart
    filename : str, default="dataset_type_high_level_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out rows where dataset_type_high_level is null or empty
    filtered_df = df.dropna(subset=['dataset_type_high_level'])
    filtered_df = filtered_df[filtered_df['dataset_type_high_level'].str.strip() != '']
    
    # Replace literal '\n' strings with actual newline characters for proper label wrapping
    filtered_df = filtered_df.copy()
    filtered_df['dataset_type_high_level'] = filtered_df['dataset_type_high_level'].str.replace('\\n', '\n', regex=False)
    
    # Count dataset types directly without grouping
    dataset_type_counts = filtered_df['dataset_type_high_level'].value_counts()
    
    # Load color palette
    color_palette = get_color_palette()
    
    # Create custom color mapping and category order
    # Put "Other" and "Unspecified" categories at the bottom, other categories sorted by count (descending)
    other_categories = ["Other or unspecified", "Other", "Unspecified"]
    regular_categories = [cat for cat in dataset_type_counts.index if cat not in other_categories]
    
    # Sort regular categories by count (descending)
    regular_categories_sorted = [cat for cat in dataset_type_counts.index if cat in regular_categories]
    
    # Create category order: regular categories (largest first) + other categories at bottom
    # Put "Other" before "Unspecified" at the bottom
    other_at_bottom = []
    for cat in ["Other", "Unspecified", "Other or unspecified"]:
        if cat in dataset_type_counts.index:
            other_at_bottom.append(cat)
    
    category_order = list(reversed(regular_categories_sorted + other_at_bottom))
    
    # Create color mapping
    color_mapping = {}
    for category in dataset_type_counts.index:
        if category == "Other":
            color_mapping[category] = color_palette["dark_grey"]
        elif category in other_categories:
            color_mapping[category] = color_palette["grey"]
        else:
            color_mapping[category] = color_palette["navy"]  # Default navy for regular categories
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=dataset_type_counts,
        title="Types of datasets used in studies for\napplication development and evaluation",
        xlabel="Number of datasets",
        ylabel=None,
        figsize=(6, 6),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        sort_ascending=False,  # We want largest at top, but we'll use custom ordering
        category_order=category_order,
        max_label_length=80,
        label_fontsize=12,
        title_fontsize=16
    )
    
    return save_path


def create_psychopathology_status_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "dataset_psychopathology_status_distribution.svg"
) -> plt.Figure:
    """
    Create a horizontal bar chart showing the distribution of dataset user psychopathology status.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information with 'dataset_user_psychopathology_status' column
    save_dir : str
        Directory to save the chart
    filename : str, default="dataset_psychopathology_status_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    plt.Figure
        The created figure object
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Group categories (including null/empty values)
    def group_psychopathology_status(value):
        if pd.isna(value) or str(value).strip() == '' or str(value).strip().lower() == 'nan':
            return "Not applicable"
        value_str = str(value).strip()
        if value_str in ["Psychopathology", "Unselected"]:
            return value_str
        elif value_str == "Unknown":
            return "Applicable, unspecified"
        else:
            return "Not applicable"
    
    # Apply grouping to all rows (including those with null values)
    df_copy = df.copy()
    df_copy['grouped_status'] = df_copy['dataset_user_psychopathology_status'].apply(group_psychopathology_status)
    
    # Count grouped categories
    status_counts = df_copy['grouped_status'].value_counts()
    
    # Create prettier labels mapping
    label_mapping = {
        "Psychopathology": "Users with psychopathology",
        "Unselected": "General population users",
        "Applicable, unspecified": "User type not specified",
        "Not applicable": "Not applicable to dataset type"
    }
    
    # Apply label mapping to the data
    status_counts_pretty = status_counts.copy()
    status_counts_pretty.index = [label_mapping.get(idx, idx) for idx in status_counts_pretty.index]
    
    # Load color palette and create color mapping (using pretty labels)
    color_palette = get_color_palette()
    color_mapping = {}
    for category in status_counts_pretty.index:
        if category in ["User type not specified", "Not applicable to dataset type"]:
            color_mapping[category] = color_palette["grey"]
        elif category == "General population users":
            color_mapping[category] = color_palette["dark_grey"]
        else:
            color_mapping[category] = color_palette["navy"]  # Default navy
    
    # Create category order: Users with psychopathology -> Unselected users -> Applicable but unspecified -> Not applicable
    ordered_categories = []
    if "Users with psychopathology" in status_counts_pretty.index:
        ordered_categories.append("Users with psychopathology")
    if "General population users" in status_counts_pretty.index:
        ordered_categories.append("General population users")
    if "User type not specified" in status_counts_pretty.index:
        ordered_categories.append("User type not specified")
    if "Not applicable to dataset type" in status_counts_pretty.index:
        ordered_categories.append("Not applicable to dataset type")
    category_order = list(reversed(ordered_categories))
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=status_counts_pretty,
        title="Types of help-seekers in the datasets",
        xlabel="Number of datasets",
        ylabel=None,
        figsize=(10, 6),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16
    )
    
    return fig


def create_responder_type_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "dataset_responder_type_distribution.svg"
) -> plt.Figure:
    """
    Create a horizontal bar chart showing the distribution of dataset responder types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information with 'dataset_responder_type' column
    save_dir : str
        Directory to save the chart
    filename : str, default="dataset_responder_type_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    plt.Figure
        The created figure object
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Group categories (including null/empty values)
    def group_responder_type(value):
        if pd.isna(value) or str(value).strip() == '' or str(value).strip().lower() == 'nan':
            return "Not applicable"
        value_str = str(value).strip()
        if value_str in ["Trained professionals", "Lay people"]:
            return value_str
        elif value_str == "Unknown":
            return "Applicable, unspecified"
        else:
            return "Not applicable"
    
    # Apply grouping to all rows (including those with null values)
    df_copy = df.copy()
    df_copy['grouped_type'] = df_copy['dataset_responder_type'].apply(group_responder_type)
    
    # Count grouped categories
    type_counts = df_copy['grouped_type'].value_counts()
    
    # Create prettier labels mapping
    label_mapping = {
        "Trained professionals": "Trained professionals",
        "Lay people": "Lay people", 
        "Applicable, unspecified": "Responder type not specified",
        "Not applicable": "Not applicable to dataset type"
    }
    
    # Apply label mapping to the data
    type_counts_pretty = type_counts.copy()
    type_counts_pretty.index = [label_mapping.get(idx, idx) for idx in type_counts_pretty.index]
    
    # Load color palette and create color mapping (using pretty labels)
    color_palette = get_color_palette()
    color_mapping = {}
    for category in type_counts_pretty.index:
        if category in ["Responder type not specified", "Not applicable to dataset type"]:
            color_mapping[category] = color_palette["grey"]
        elif category == "Lay people":
            color_mapping[category] = color_palette["dark_grey"]
        else:
            color_mapping[category] = color_palette["navy"]  # Default navy
    
    # Create category order: Trained professionals -> Lay people -> Applicable but unspecified -> Not applicable
    ordered_categories = []
    if "Trained professionals" in type_counts_pretty.index:
        ordered_categories.append("Trained professionals")
    if "Lay people" in type_counts_pretty.index:
        ordered_categories.append("Lay people")
    if "Responder type not specified" in type_counts_pretty.index:
        ordered_categories.append("Responder type not specified")
    if "Not applicable to dataset type" in type_counts_pretty.index:
        ordered_categories.append("Not applicable to dataset type")
    category_order = list(reversed(ordered_categories))
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=type_counts_pretty,
        title="Types of responders in the datasets",
        xlabel="Number of datasets",
        ylabel=None,
        figsize=(10, 6),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16
    )
    
    return fig


def create_dataset_reuse_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "dataset_reuse_distribution.svg"
) -> str:
    """
    Create a horizontal bar chart showing dataset reuse patterns by reference count.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information with 'reference_count' column
    save_dir : str
        Directory to save the chart
    filename : str, default="dataset_reuse_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out rows where reference_count is null
    filtered_df = df.dropna(subset=['reference_count'])
    
    # Create meaningful labels for reference counts
    def create_reuse_label(count):
        if count == 1:
            return "Used once (1 study)"
        elif count == 2:
            return "Used twice (2 studies)"
        else:
            return f"Used {count} times ({count} studies)"
    
    # Create labels and count occurrences
    filtered_df = filtered_df.copy()
    filtered_df['reuse_label'] = filtered_df['reference_count'].apply(create_reuse_label)
    
    # Count how many datasets fall into each reuse category
    reuse_counts = filtered_df['reuse_label'].value_counts()
    
    # Load color palette and create color mapping
    color_palette = get_color_palette()
    color_mapping = {}
    for category in reuse_counts.index:
        if "Used once" in category:
            color_mapping[category] = color_palette["navy"]  # Regular color for single use
        else:
            color_mapping[category] = color_palette["amber"]  # Highlight reused datasets
    
    # Create category order: single use first, then reused datasets by count
    single_use = [cat for cat in reuse_counts.index if "Used once" in cat]
    reused = [cat for cat in reuse_counts.index if "Used once" not in cat]
    
    # Sort reused datasets by the number in the label (ascending)
    def extract_count(label):
        # Extract the number from labels like "Used 5 times (5 studies)"
        import re
        match = re.search(r'Used (\d+) times', label)
        return int(match.group(1)) if match else 0
    
    reused_sorted = sorted(reused, key=extract_count)
    category_order = list(reversed(single_use + reused_sorted))
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=reuse_counts,
        title="Dataset Reuse Across Studies",
        xlabel="Number of Datasets",
        ylabel="Reuse Pattern",
        figsize=(10, 6),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16
    )
    
    return save_path


def create_dataset_language_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "dataset_language_distribution.svg"
) -> str:
    """
    Create a horizontal bar chart showing the distribution of dataset languages.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information with 'dataset_language' column
    save_dir : str
        Directory to save the chart
    filename : str, default="dataset_language_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Transform language labels
    def transform_language_label(value):
        if pd.isna(value) or str(value).strip() == '' or str(value).strip().lower() == 'nan':
            return "Unspecified"
        
        value_str = str(value).strip()
        
        # Handle "Other: unknown" specifically
        if value_str == "Other: unknown":
            return "Unspecified"
        
        # Remove "Other: " prefix from other labels
        if value_str.startswith("Other: "):
            return value_str.replace("Other: ", "")
        
        return value_str
    
    # Apply transformations to all rows
    df_copy = df.copy()
    df_copy['transformed_language'] = df_copy['dataset_language'].apply(transform_language_label)
    
    # Count language occurrences
    initial_counts = df_copy['transformed_language'].value_counts()
    
    # Bin languages that appear only once into "Other"
    def bin_rare_languages(language, count):
        if count == 1 and language != "Unspecified":
            return "Other"
        return language
    
    # Create final language labels by binning rare languages
    language_mapping = {}
    for lang, count in initial_counts.items():
        language_mapping[lang] = bin_rare_languages(lang, count)
    
    df_copy['final_language'] = df_copy['transformed_language'].map(language_mapping)
    language_counts = df_copy['final_language'].value_counts()
    
    # Load color palette and create color mapping
    color_palette = get_color_palette()
    color_mapping = {}
    for category in language_counts.index:
        if category == "Unspecified":
            color_mapping[category] = color_palette["grey"]
        else:
            color_mapping[category] = color_palette["navy"]  # Default navy for all languages including "Other"
    
    # Create category order: specific languages first (by count), then "Other" and "Unspecified" at bottom
    specific_languages = [cat for cat in language_counts.index if cat not in ["Other", "Unspecified"]]
    bottom_categories = []
    for cat in ["Other", "Unspecified"]:  # "Other" before "Unspecified" at bottom
        if cat in language_counts.index:
            bottom_categories.append(cat)
    category_order = list(reversed(specific_languages + bottom_categories))
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=language_counts,
        title="Distribution of Dataset Languages",
        xlabel="Number of Datasets",
        ylabel="Language",
        figsize=(10, 8),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16
    )
    
    return save_path


def create_dataset_public_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "dataset_public_distribution.svg"
) -> str:
    """
    Create a horizontal bar chart showing whether datasets are public or not.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information with 'dataset_is_public' column
    save_dir : str
        Directory to save the chart
    filename : str, default="dataset_public_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Simplify public status labels - bin everything that's not "Yes" into "Not public"
    def simplify_public_status(value):
        if pd.isna(value) or str(value).strip() == '' or str(value).strip().lower() == 'nan':
            return "Private or gated"
        
        value_str = str(value).strip()
        
        if value_str == "Yes":
            return "Publicly available"
        else:
            # Everything else (including "No", "Other:" entries, etc.) goes to "Not public"
            return "Private or gated"
    
    # Apply transformations to all rows
    df_copy = df.copy()
    df_copy['public_status'] = df_copy['dataset_is_public'].apply(simplify_public_status)
    
    # Count public status occurrences
    public_counts = df_copy['public_status'].value_counts()
    
    # Load color palette and create color mapping
    color_palette = get_color_palette()
    color_mapping = {}
    for category in public_counts.index:
        if category == "Publicly available":
            color_mapping[category] = color_palette["mint"]  # Green for public
        elif category == "Private or gated":
            color_mapping[category] = color_palette["crimson"]  # Red for not public
    
    # Create category order: Public first, then Not public
    category_order = []
    for cat in ["Publicly available", "Private or gated"]:
        if cat in public_counts.index:
            category_order.append(cat)
    category_order = list(reversed(category_order))
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=public_counts,
        title="Public availability of datasets",
        xlabel="Number of datasets",
        ylabel=None,
        figsize=(5, 2),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16
    )
    
    return save_path


def generate_datasets_metrics_report(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "5_datasets_metrics_report.txt"
) -> str:
    """
    Generate a comprehensive report on dataset metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information
    save_dir : str
        Directory to save the report
    filename : str, default="5_datasets_metrics_report.txt"
        Filename for the saved report
        
    Returns
    -------
    str
        Path to the saved report file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    report_path = os.path.join(save_dir, filename)
    
    with open(report_path, 'w') as f:
        f.write("DATASETS METRICS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic dataset statistics
        f.write("BASIC STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total studies: {len(df)}\n")
        
        # Dataset type analysis
        f.write("\nHIGH-LEVEL DATASET TYPE DISTRIBUTION\n")
        f.write("-" * 35 + "\n")
        
        # Filter out null/empty dataset types
        filtered_df = df.dropna(subset=['dataset_type_high_level'])
        filtered_df = filtered_df[filtered_df['dataset_type_high_level'].str.strip() != '']
        
        f.write(f"Datasets with high-level type information: {len(filtered_df)}\n")
        f.write(f"Datasets without high-level type information: {len(df) - len(filtered_df)}\n\n")
        
        if len(filtered_df) > 0:
            dataset_type_counts = filtered_df['dataset_type_high_level'].value_counts()
            dataset_type_percentages = (dataset_type_counts / len(filtered_df) * 100).round(1)
            
            f.write("High-level dataset type breakdown:\n")
            for dataset_type, count in dataset_type_counts.items():
                percentage = dataset_type_percentages[dataset_type]
                f.write(f"  {dataset_type}: {count} ({percentage}%)\n")
        
        # Additional dataset characteristics if available
        dataset_columns = [col for col in df.columns if col.startswith('dataset_')]
        
        if len(dataset_columns) > 1:
            f.write(f"\nOTHER DATASET CHARACTERISTICS\n")
            f.write("-" * 30 + "\n")
            
            for col in dataset_columns:
                if col != 'dataset_type' and col in df.columns:
                    non_null_count = df[col].notna().sum()
                    f.write(f"{col}: {non_null_count} studies have information\n")
                    
                    # Show distribution for categorical columns
                    if df[col].dtype == 'object':
                        value_counts = df[col].value_counts()
                        if len(value_counts) <= 10:  # Only show if not too many unique values
                            f.write(f"  Distribution:\n")
                            for value, count in value_counts.items():
                                f.write(f"    {value}: {count}\n")
                    f.write("\n")
    
    return report_path


def create_combined_psychopathology_responder_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "combined_psychopathology_responder_distribution.svg"
) -> str:
    """
    Create a combined figure with dataset types, psychopathology status, and responder type charts stacked vertically.
    
    This function reuses the individual chart functions and combines them into subplots,
    following DRY principles and software engineering best practices.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information
    save_dir : str
        Directory to save the chart
    filename : str, default="combined_psychopathology_responder_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with three subplots stacked vertically using gridspec for better control
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(4, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Prepare dataset type data (replicate logic from create_dataset_type_chart)
    filtered_df_types = df.dropna(subset=['dataset_type_high_level'])
    filtered_df_types = filtered_df_types[filtered_df_types['dataset_type_high_level'].str.strip() != '']
    filtered_df_types = filtered_df_types.copy()
    filtered_df_types['dataset_type_high_level'] = filtered_df_types['dataset_type_high_level'].str.replace('\\n', '\n', regex=False)
    dataset_type_counts = filtered_df_types['dataset_type_high_level'].value_counts()
    
    color_palette = get_color_palette()
    other_categories = ["Other or unspecified", "Other", "Unspecified"]
    regular_categories = [cat for cat in dataset_type_counts.index if cat not in other_categories]
    regular_categories_sorted = [cat for cat in dataset_type_counts.index if cat in regular_categories]
    other_at_bottom = [cat for cat in ["Other", "Unspecified", "Other or unspecified"] if cat in dataset_type_counts.index]
    category_order_types = list(reversed(regular_categories_sorted + other_at_bottom))
    
    color_mapping_types = {}
    for category in dataset_type_counts.index:
        if category == "Other":
            color_mapping_types[category] = color_palette["dark_grey"]
        elif category in other_categories:
            color_mapping_types[category] = color_palette["grey"]
        else:
            color_mapping_types[category] = color_palette["navy"]
    
    # Create dataset type chart as figure object
    dataset_type_fig = create_horizontal_bar_chart(
        data=dataset_type_counts,
        title="Types of datasets",
        xlabel="Number of datasets",
        ylabel=None,
        figsize=(6, 12),
        color=color_mapping_types,
        show_percentages=True,
        save_path=None,
        sort_ascending=False,
        category_order=category_order_types,
        max_label_length=80,
        label_fontsize=12,
        title_fontsize=14
    )
    
    psychopathology_fig = create_psychopathology_status_chart(df, save_dir, "temp_psycho.svg")
    responder_fig = create_responder_type_chart(df, save_dir, "temp_responder.svg")
    
    # Extract data from the individual figures and recreate in subplots
    
    # === TOP SUBPLOT: Dataset Types Distribution ===
    # Get the axes from the dataset type figure
    dataset_type_ax = dataset_type_fig.axes[0]
    
    # Standard bar height for consistency
    bar_height = 0.6
    
    # Copy the bar chart to our subplot
    for bar in dataset_type_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y() 
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax1.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in dataset_type_ax.texts:
        ax1.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax1.set_ylim(dataset_type_ax.get_ylim())
    ax1.set_yticks(dataset_type_ax.get_yticks())
    ax1.set_yticklabels([t.get_text() for t in dataset_type_ax.get_yticklabels()], fontsize=10)
    ax1.set_title(dataset_type_ax.get_title(), fontsize=14, pad=20)
    
    # Remove x-axis tick labels from top subplot
    ax1.set_xticklabels([])
    
    # Copy styling
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax1.grid(axis='y', visible=False)
    
    # === MIDDLE SUBPLOT: Psychopathology Status Distribution ===
    # Get the axes from the psychopathology figure
    psycho_ax = psychopathology_fig.axes[0]
    
    # Copy the bar chart to our subplot
    for bar in psycho_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax2.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in psycho_ax.texts:
        ax2.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax2.set_ylim(psycho_ax.get_ylim())
    ax2.set_yticks(psycho_ax.get_yticks())
    ax2.set_yticklabels([t.get_text() for t in psycho_ax.get_yticklabels()], fontsize=10)
    ax2.set_title(psycho_ax.get_title(), fontsize=14, pad=20)
    
    # Remove x-axis tick labels from middle subplot
    ax2.set_xticklabels([])
    
    # Copy styling
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax2.grid(axis='y', visible=False)
    
    # === BOTTOM SUBPLOT: Responder Type Distribution ===
    # Get the axes from the responder type figure
    responder_ax = responder_fig.axes[0]
    
    # Copy the bar chart to our subplot
    for bar in responder_ax.patches:
        # Get bar properties
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        
        # Create new bar in our subplot with consistent height
        ax3.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in responder_ax.texts:
        ax3.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Copy axis properties
    ax3.set_ylim(responder_ax.get_ylim())
    ax3.set_yticks(responder_ax.get_yticks())
    ax3.set_yticklabels([t.get_text() for t in responder_ax.get_yticklabels()], fontsize=10)
    ax3.set_title(responder_ax.get_title(), fontsize=14, pad=20)
    ax3.set_xlabel(responder_ax.get_xlabel(), fontsize=12)
    
    # Copy styling
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax3.grid(axis='y', visible=False)
    
    # Align x-axes by setting the same xlim
    max_xlim = max(dataset_type_ax.get_xlim()[1], psycho_ax.get_xlim()[1], responder_ax.get_xlim()[1])
    ax1.set_xlim(0, max_xlim)
    ax2.set_xlim(0, max_xlim)
    ax3.set_xlim(0, max_xlim)
    
    # Layout is already controlled by gridspec, no need for additional adjustments
    
    # Close the individual figures to free memory
    plt.close(dataset_type_fig)
    plt.close(psychopathology_fig)
    plt.close(responder_fig)
    
    # Clean up temp files
    temp_dataset_type_path = os.path.join(save_dir, "temp_dataset_type.svg")
    temp_psycho_path = os.path.join(save_dir, "temp_psycho.svg")
    temp_responder_path = os.path.join(save_dir, "temp_responder.svg")
    if os.path.exists(temp_dataset_type_path):
        os.remove(temp_dataset_type_path)
    if os.path.exists(temp_psycho_path):
        os.remove(temp_psycho_path)
    if os.path.exists(temp_responder_path):
        os.remove(temp_responder_path)
    
    # Save the combined figure
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def create_reused_datasets_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "reused_datasets_distribution.svg"
) -> str:
    """
    Create a horizontal bar chart showing only datasets that were used more than once.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset information with 'dataset_id' and 'reference_count' columns
    save_dir : str
        Directory to save the chart
    filename : str, default="reused_datasets_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter to only datasets used more than once
    filtered_df = df.dropna(subset=['reference_count'])
    reused_datasets = filtered_df[filtered_df['reference_count'] > 1]
    
    if len(reused_datasets) == 0:
        # If no reused datasets, create an empty chart with a message
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No datasets were used more than once', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path
    
    # Create a series with dataset_id as index and reference_count as values
    reused_counts = reused_datasets.set_index('dataset_id')['reference_count']
    
    # Sort by count (descending) for better visualization
    reused_counts = reused_counts.sort_values(ascending=True)  # ascending=True for horizontal bars (bottom to top)
    
    # Create prettier labels for dataset names
    dataset_pretty_labels = {
        'counsel_chat': 'CounselChat',
        'psy_qa': 'PsyQA',
        'amod': 'Amod/mental_health_\ncounseling_conversations',
        'alexander_street_press': 'Alexander Street Press Counseling\nand Psychotherapy Transcripts',
        'perez_rosa_motivational_interviewing': 'Pérez-Rosas Motivational\nInterviewing Dataset',
        'hope': 'HOPE',
        'empathetic_dialogues': 'Facebook EmpatheticDialogues',
        'jigsaw_toxic_comments': 'Jigsaw Toxic Comments'
    }
    
    # Rename the index to use prettier labels
    reused_counts_pretty = reused_counts.copy()
    reused_counts_pretty.index = [dataset_pretty_labels.get(idx, idx) for idx in reused_counts_pretty.index]
    
    # Calculate total datasets for context
    total_datasets = len(filtered_df)
    datasets_used_once = total_datasets - len(reused_datasets)
    
    # Add an empty row for the note text (with empty y-label)
    note_text = f"Note: An additional {datasets_used_once} datasets\nwere used in only one study each"
    reused_counts_pretty[""] = 0  # Add empty row with empty label and 0 value
    
    # Create the chart using our proven function
    save_path = os.path.join(save_dir, filename)
    
    # Get color palette and use navy color
    color_palette = get_color_palette()
    
    fig = create_horizontal_bar_chart(
        data=reused_counts_pretty,
        title="Datasets used more than once:\nNumber of articles using each dataset",
        xlabel="Number of articles",
        ylabel=None,
        figsize=(5, 6),
        color=color_palette["navy"],
        show_percentages=False,  # Don't show percentages since this is absolute counts
        save_path=None,  # We'll handle saving manually to adjust layout
        max_label_length=80,
        label_fontsize=11,
        title_fontsize=14,
        return_stats=False
    )
    
    # Get the axes to adjust layout and add note text
    ax = fig.get_axes()[0]
    
    # Reset y-axis labels to default alignment (right-aligned, which works better)
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('right')
    
    # Remove the bottom bar (the empty one) and its label
    bars = ax.patches
    if len(bars) > 0:
        bars[0].set_visible(False)  # Hide the first bar (bottom bar, which is the empty one)
    
    # Remove the "0" label for the empty bar
    for text in ax.texts:
        if text.get_text() == "0":
            text.set_visible(False)
    
    # Remove the tick mark for the empty row (bottom tick)
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    if len(yticks) > 0:
        # Hide the bottom tick mark by setting it to empty
        new_yticks = yticks[1:]  # Remove first (bottom) tick
        new_yticklabels = yticklabels[1:]  # Remove first (bottom) tick label
        ax.set_yticks(new_yticks)
        ax.set_yticklabels([label.get_text() for label in new_yticklabels])
    
    # Add the note text in place of the removed bar
    # Position it at y=0 (bottom row) where the bar would have been
    ax.text(0.5, 0, note_text, 
            fontsize=12, style='italic', color='black',
            ha='left', va='center')
    
    # Adjust the subplot to give more space for y-axis labels
    plt.subplots_adjust(left=0.4)  # Increase left margin significantly
    
    # Save the figure
    fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def create_datasets_metrics_panel(
    datasets_file: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Create a complete datasets metrics panel with charts and reports.
    
    Parameters
    ----------
    datasets_file : str
        Path to the CSV file containing dataset information
    output_dir : str
        Directory to save all outputs
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing paths to created files and summary statistics
    """
    # Load the data
    df = pd.read_csv(datasets_file)
    
    # Create the dataset type chart
    chart_path = create_dataset_type_chart(df, output_dir)
    
    # Create the psychopathology status chart
    psychopathology_fig = create_psychopathology_status_chart(df, output_dir)
    psychopathology_chart_path = os.path.join(output_dir, "dataset_psychopathology_status_distribution.svg")
    
    # Create the responder type chart
    responder_fig = create_responder_type_chart(df, output_dir)
    responder_chart_path = os.path.join(output_dir, "dataset_responder_type_distribution.svg")
    
    # Create the dataset reuse chart
    reuse_chart_path = create_dataset_reuse_chart(df, output_dir)
    
    # Create the focused reused datasets chart
    reused_datasets_chart_path = create_reused_datasets_chart(df, output_dir)
    
    # Create the dataset language chart
    language_chart_path = create_dataset_language_chart(df, output_dir)
    
    # Create the dataset public chart
    public_chart_path = create_dataset_public_chart(df, output_dir)
    
    # Create the combined psychopathology and responder type chart
    combined_psycho_responder_chart_path = create_combined_psychopathology_responder_chart(df, output_dir)
    
    # Generate the report
    report_path = generate_datasets_metrics_report(df, output_dir)
    
    # Calculate summary statistics
    filtered_df = df.dropna(subset=['dataset_type_high_level'])
    filtered_df = filtered_df[filtered_df['dataset_type_high_level'].str.strip() != '']
    
    summary = {
        'total_datasets': len(df),
        'datasets_with_high_level_type': len(filtered_df),
        'unique_high_level_types': filtered_df['dataset_type_high_level'].nunique() if len(filtered_df) > 0 else 0,
        'most_common_high_level_type': filtered_df['dataset_type_high_level'].mode().iloc[0] if len(filtered_df) > 0 else None,
        'chart_path': chart_path,
        'psychopathology_chart_path': psychopathology_chart_path,
        'responder_chart_path': responder_chart_path,
        'reuse_chart_path': reuse_chart_path,
        'reused_datasets_chart_path': reused_datasets_chart_path,
        'language_chart_path': language_chart_path,
        'public_chart_path': public_chart_path,
        'combined_psycho_responder_chart_path': combined_psycho_responder_chart_path,
        'report_path': report_path
    }
    
    return summary
