"""
Panel Measurements Metrics Utilities

This module provides functions for creating visualizations and metrics
related to measurement characteristics in the LLMs in psychotherapy study.
"""

import pandas as pd
import os
import yaml
from typing import Optional, Dict, Any
from .utils import create_horizontal_bar_chart, get_color_palette


def create_metric_supercategories_chart(
    df: pd.DataFrame,
    save_dir: str,
    total_studies_without_participants: int,
    filename: str = "metric_supercategories_distribution.svg"
) -> str:
    """
    Create a horizontal bar chart showing the distribution of metric supercategories by article count.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the measurements information with 'metric_supercategory' and 'reference_title' columns
    save_dir : str
        Directory to save the chart
    total_studies_without_participants : int
        Total number of studies without human participants (for percentage calculation denominator)
    filename : str, default="metric_supercategories_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out rows where metric_supercategory is null or empty
    filtered_df = df.dropna(subset=['metric_supercategory'])
    filtered_df = filtered_df[filtered_df['metric_supercategory'].str.strip() != '']
    
    # Count unique articles per metric supercategory
    # Group by metric_supercategory and count unique reference_title values
    supercategory_article_counts = (
        filtered_df.groupby('metric_supercategory')['reference_title']
        .nunique()
        .sort_values(ascending=False)
    )
    
    # Filter out supercategories that appear in only one article
    supercategory_article_counts = supercategory_article_counts[supercategory_article_counts > 1]
    
    # Create display labels mapping
    display_labels = {
        'reference_similarity': 'Comparison against\nground-truth reference',
        'human_rating': 'Human rating',
        'linguistic_analysis': 'Linguistic properties',
        'automatic_rating_empathy_or_sentiment': 'Automatic rating of\nempathy or sentiment',
        'automatic_rating_safety_or_bias': 'Automatic rating of\nsafety or bias',
        'automatic_rating_other_response_quality': 'Automatic rating of\nother response quality'
    }
    
    # Apply display labels
    supercategory_article_counts.index = supercategory_article_counts.index.map(
        lambda x: display_labels.get(x, x)
    )
    
    # Load color palette
    color_palette = get_color_palette()
    
    # Create category order: largest first (already sorted by count descending)
    category_order = list(reversed(supercategory_article_counts.index.tolist()))
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    # Use the provided total studies count for percentage calculation
    # This includes studies without any measurements
    title = f"How were applications evaluated in studies\nwithout human participants (n = {total_studies_without_participants})?"
    
    fig = create_horizontal_bar_chart(
        data=supercategory_article_counts,
        title=title,
        xlabel="Number of Articles",
        ylabel="",
        figsize=(6, 6),
        color=color_palette["navy"],
        show_percentages=True,
        save_path=save_path,
        sort_ascending=False,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16,
        percentage_total=total_studies_without_participants
    )
    
    return save_path


def create_benchmark_quality_chart(
    df: pd.DataFrame,
    save_dir: str,
    total_studies_without_participants: int,
    filename: str = "benchmark_quality_distribution.svg"
) -> str:
    """
    Create a horizontal bar chart showing the distribution of benchmark quality by study.
    
    Aggregation logic per study:
    - High quality: if study has at least one measurement with high quality benchmark
    - Low quality: if study has at least one low quality benchmark but no high quality
    - No benchmark: if study has no benchmarks (all measurements marked as "no benchmark")
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the measurements information with 'benchmark_quality' and 'reference_title' columns
    save_dir : str
        Directory to save the chart
    total_studies_without_participants : int
        Total number of studies without human participants (for percentage calculation denominator)
    filename : str, default="benchmark_quality_distribution.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter out rows where benchmark_quality is null or empty
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    # Normalize benchmark quality values
    def normalize_benchmark_quality(value):
        if pd.isna(value) or str(value).strip() == '':
            return 'no benchmark'
        
        value_str = str(value).strip().lower()
        
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        elif value_str in ['no benchmark', 'no_benchmark']:
            return 'no benchmark'
        else:
            # Any other value defaults to 'no benchmark'
            return 'no benchmark'
    
    # Apply normalization
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    # Aggregate by study (reference_title)
    study_benchmark_quality = {}
    
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        
        # Apply aggregation logic
        if 'high' in benchmark_qualities:
            study_benchmark_quality[study] = 'Human support-provider'
        elif 'low' in benchmark_qualities:
            study_benchmark_quality[study] = 'Another technical application'
        else:
            study_benchmark_quality[study] = 'No comparison method'
    
    # Count studies by aggregated benchmark quality
    quality_counts = pd.Series(study_benchmark_quality.values()).value_counts()
    
    # Load color palette and create color mapping
    color_palette = get_color_palette()
    color_mapping = {}
    for category in quality_counts.index:
        if category == 'Human support-provider':
            color_mapping[category] = color_palette["sky"]  # Sky blue for high quality
        elif category == 'Another technical application':
            color_mapping[category] = color_palette["rose"]  # Rose for low quality
        elif category == 'No comparison method':
            color_mapping[category] = color_palette["grey"]  # Grey for no benchmark
    
    # Create category order: High quality, Low quality, No benchmark
    category_order = []
    for cat in ['Human support-provider', 'Another technical application', 'No comparison method']:
        if cat in quality_counts.index:
            category_order.append(cat)
    category_order = list(reversed(category_order))
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    # Use the provided total studies count for percentage calculation
    # This includes studies without any measurements
    fig = create_horizontal_bar_chart(
        data=quality_counts,
        title="Comparison methods that studies used\nto benchmark their applications",
        xlabel="Number of articles",
        ylabel=None,
        figsize=(6, 3),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        sort_ascending=False,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16,
        percentage_total=total_studies_without_participants
    )
    
    return save_path


def create_benchmark_quality_stacked_chart(
    df: pd.DataFrame,
    save_dir: str,
    total_studies_without_participants: int,
    filename: str = "benchmark_quality_distribution_stacked.svg"
) -> str:
    """
    Create a horizontal bar chart showing benchmark quality with performance stacked.
    
    For Human support-provider and Another technical application, the bars are split
    by performance (better, similar, worse). No comparison method remains as a single bar.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the measurements information
    save_dir : str
        Directory to save the chart
    total_studies_without_participants : int
        Total number of studies without human participants (for percentage calculation denominator)
    filename : str, default="benchmark_quality_distribution_stacked.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get the performance data for high and low quality benchmarks
    high_quality_data = _get_performance_data_high_quality(df)
    low_quality_data = _get_performance_data_low_quality(df)
    
    # Get counts for no comparison method
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    def normalize_benchmark_quality(value):
        value_str = str(value).strip().lower()
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        else:
            return 'no benchmark'
    
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    # Count studies with no benchmark
    no_benchmark_studies = set()
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        if 'high' not in benchmark_qualities and 'low' not in benchmark_qualities:
            no_benchmark_studies.add(study)
    
    no_benchmark_count = len(no_benchmark_studies)
    
    # Prepare data for plotting (reversed order - top to bottom)
    categories = ['No comparison method', 'Another technical application', 'Human support-provider']
    
    # Extract performance counts
    better_high = high_quality_data.get('Better than benchmark', 0)
    similar_high = high_quality_data.get('Similar to benchmark', 0)
    worse_high = high_quality_data.get('Worse than benchmark', 0)
    
    better_low = low_quality_data.get('Better than benchmark', 0)
    similar_low = low_quality_data.get('Similar to benchmark', 0)
    worse_low = low_quality_data.get('Worse than benchmark', 0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 3))
    
    # Get color palette
    color_palette = get_color_palette()
    colors_performance = {
        'better': color_palette["mint"],
        'similar': color_palette["amber"],
        'worse': color_palette["crimson"]
    }
    
    # Y positions (reversed)
    y_positions = np.arange(len(categories))
    bar_height = 0.6
    
    # Plot single bar for No comparison method (now at position 0, bottom)
    ax.barh(y_positions[0], no_benchmark_count, bar_height, color=color_palette["grey"], alpha=0.8)
    
    # Plot stacked bars for Another technical application (now at position 1, middle)
    ax.barh(y_positions[1], worse_low, bar_height, color=colors_performance['worse'], alpha=0.8)
    ax.barh(y_positions[1], similar_low, bar_height, left=worse_low, 
            color=colors_performance['similar'], alpha=0.8)
    ax.barh(y_positions[1], better_low, bar_height, left=worse_low + similar_low, 
            color=colors_performance['better'], alpha=0.8)
    
    # Plot stacked bars for Human support-provider (now at position 2, top)
    ax.barh(y_positions[2], worse_high, bar_height, color=colors_performance['worse'], alpha=0.8)
    ax.barh(y_positions[2], similar_high, bar_height, left=worse_high, 
            color=colors_performance['similar'], alpha=0.8)
    ax.barh(y_positions[2], better_high, bar_height, left=worse_high + similar_high, 
            color=colors_performance['better'], alpha=0.8)
    
    # Calculate total studies shown in this chart
    total_high = worse_high + similar_high + better_high
    total_low = worse_low + similar_low + better_low
    
    # Use the provided total studies count for percentage calculation
    # This includes studies without any measurements
    # Add value label for No comparison method (now at position 0)
    pct_no = round(no_benchmark_count / total_studies_without_participants * 100)
    ax.text(no_benchmark_count + 1, y_positions[0], f"{int(no_benchmark_count)} ({pct_no}%)", 
            ha='left', va='center', fontsize=10)
    
    # Add value labels for Another technical application (now at position 1)
    if worse_low > 0:
        ax.text(worse_low / 2, y_positions[1], str(int(worse_low)), 
                ha='center', va='center', fontsize=10)
    if similar_low > 0:
        ax.text(worse_low + similar_low / 2, y_positions[1], str(int(similar_low)), 
                ha='center', va='center', fontsize=10)
    if better_low > 0:
        ax.text(worse_low + similar_low + better_low / 2, y_positions[1], str(int(better_low)), 
                ha='center', va='center', fontsize=10)
    
    # Add total and percentage for Another technical application
    pct_low = round(total_low / total_studies_without_participants * 100)
    ax.text(total_low + 1, y_positions[1], f"{int(total_low)} ({pct_low}%)", 
            ha='left', va='center', fontsize=10)
    
    # Add value labels for Human support-provider (now at position 2, top)
    if worse_high > 0:
        ax.text(worse_high / 2, y_positions[2], str(int(worse_high)), 
                ha='center', va='center', fontsize=10)
    if similar_high > 0:
        ax.text(worse_high + similar_high / 2, y_positions[2], str(int(similar_high)), 
                ha='center', va='center', fontsize=10)
    if better_high > 0:
        ax.text(worse_high + similar_high + better_high / 2, y_positions[2], str(int(better_high)), 
                ha='center', va='center', fontsize=10)
    
    # Add total and percentage for Human support-provider
    pct_high = round(total_high / total_studies_without_participants * 100)
    ax.text(total_high + 1, y_positions[2], f"{int(total_high)} ({pct_high}%)", 
            ha='left', va='center', fontsize=10)
    
    # Set labels and title
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories, fontsize=11)
    ax.set_xlabel("Number of articles", fontsize=12)
    
    # Set title centered on the figure instead of just the axes
    title = f"Comparison methods that studies without human participants\n(n = {total_studies_without_participants}) used to benchmark their applications"
    fig.suptitle(title, fontsize=14, y=0.95, ha='center')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax.grid(axis='y', visible=False)
    
    # Set x-axis limits
    max_x = max(total_high, total_low, no_benchmark_count)
    ax.set_xlim(0, max_x * 1.15)
    
    # Add legend below the figure
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_performance['better'], alpha=0.8, label='Better'),
        Patch(facecolor=colors_performance['similar'], alpha=0.8, label='Equal'),
        Patch(facecolor=colors_performance['worse'], alpha=0.8, label='Worse')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=True,
              title='Performance of application', title_fontsize=10,
              fancybox=True, shadow=True, framealpha=0.9)
    
    # Adjust layout with space for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def create_performance_high_quality_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "performance_high_quality_benchmarks.svg"
) -> str:
    """
    Create a horizontal bar chart showing performance vs benchmark for studies
    with high quality benchmarks.
    
    For each study in the high quality group, we select measurements with high quality
    benchmarks and display the best performance (b > s > w).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the measurements information
    save_dir : str
        Directory to save the chart
    filename : str, default="performance_high_quality_benchmarks.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # First, identify which studies are in the high quality group
    # (using the same logic as create_benchmark_quality_chart)
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    def normalize_benchmark_quality(value):
        if pd.isna(value) or str(value).strip() == '':
            return 'no benchmark'
        value_str = str(value).strip().lower()
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        else:
            return 'no benchmark'
    
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    # Identify high quality studies
    high_quality_studies = set()
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        if 'high' in benchmark_qualities:
            high_quality_studies.add(study)
    
    # For each high quality study, get measurements with high quality benchmarks
    def normalize_performance(value):
        if pd.isna(value) or str(value).strip() == '':
            return None
        value_str = str(value).strip().lower()
        if value_str in ['b', 'better']:
            return 'better'
        elif value_str in ['s', 'similar']:
            return 'similar'
        elif value_str in ['w', 'worse']:
            return 'worse'
        else:
            return None
    
    # Performance ranking (better > similar > worse)
    performance_rank = {'better': 3, 'similar': 2, 'worse': 1}
    
    study_best_performance = {}
    for study in high_quality_studies:
        # Get measurements with high quality benchmarks for this study
        study_high_measurements = filtered_df[
            (filtered_df['reference_title'] == study) &
            (filtered_df['normalized_benchmark_quality'] == 'high')
        ]
        
        # Normalize performance values
        performances = study_high_measurements['performance_vs_benchmark'].apply(normalize_performance)
        performances = performances.dropna()
        
        if len(performances) > 0:
            # Get the best performance
            best_performance = max(performances, key=lambda x: performance_rank.get(x, 0))
            study_best_performance[study] = best_performance
    
    # Count studies by best performance
    performance_counts = pd.Series(study_best_performance.values()).value_counts()
    
    # Ensure all three categories are present (even if count is 0)
    for category in ['better', 'similar', 'worse']:
        if category not in performance_counts:
            performance_counts[category] = 0
    
    # Create display labels
    display_labels = {
        'better': 'Better than benchmark',
        'similar': 'Similar to benchmark',
        'worse': 'Worse than benchmark'
    }
    
    performance_counts.index = performance_counts.index.map(display_labels)
    
    # Load color palette and create color mapping
    color_palette = get_color_palette()
    color_mapping = {
        'Better than benchmark': color_palette["mint"],
        'Similar to benchmark': color_palette["amber"],
        'Worse than benchmark': color_palette["crimson"]
    }
    
    # Create category order
    category_order = ['Worse than benchmark', 'Similar to benchmark', 'Better than benchmark']
    
    # Get total high quality studies for percentage calculation
    total_high_quality_studies = len(high_quality_studies)
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=performance_counts,
        title="Performance vs High Quality Benchmarks",
        xlabel="Number of Studies",
        ylabel="Performance",
        figsize=(10, 5),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        sort_ascending=False,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16,
        percentage_total=total_high_quality_studies
    )
    
    return save_path


def create_performance_low_quality_chart(
    df: pd.DataFrame,
    save_dir: str,
    filename: str = "performance_low_quality_benchmarks.svg"
) -> str:
    """
    Create a horizontal bar chart showing performance vs benchmark for studies
    with low quality benchmarks.
    
    For each study in the low quality group, we select measurements with low quality
    benchmarks and display the best performance (b > s > w).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the measurements information
    save_dir : str
        Directory to save the chart
    filename : str, default="performance_low_quality_benchmarks.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # First, identify which studies are in the low quality group
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    def normalize_benchmark_quality(value):
        if pd.isna(value) or str(value).strip() == '':
            return 'no benchmark'
        value_str = str(value).strip().lower()
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        else:
            return 'no benchmark'
    
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    # Identify low quality studies (has low but not high)
    low_quality_studies = set()
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        if 'low' in benchmark_qualities and 'high' not in benchmark_qualities:
            low_quality_studies.add(study)
    
    # For each low quality study, get measurements with low quality benchmarks
    def normalize_performance(value):
        if pd.isna(value) or str(value).strip() == '':
            return None
        value_str = str(value).strip().lower()
        if value_str in ['b', 'better']:
            return 'better'
        elif value_str in ['s', 'similar']:
            return 'similar'
        elif value_str in ['w', 'worse']:
            return 'worse'
        else:
            return None
    
    # Performance ranking (better > similar > worse)
    performance_rank = {'better': 3, 'similar': 2, 'worse': 1}
    
    study_best_performance = {}
    for study in low_quality_studies:
        # Get measurements with low quality benchmarks for this study
        study_low_measurements = filtered_df[
            (filtered_df['reference_title'] == study) &
            (filtered_df['normalized_benchmark_quality'] == 'low')
        ]
        
        # Normalize performance values
        performances = study_low_measurements['performance_vs_benchmark'].apply(normalize_performance)
        performances = performances.dropna()
        
        if len(performances) > 0:
            # Get the best performance
            best_performance = max(performances, key=lambda x: performance_rank.get(x, 0))
            study_best_performance[study] = best_performance
    
    # Count studies by best performance
    performance_counts = pd.Series(study_best_performance.values()).value_counts()
    
    # Ensure all three categories are present (even if count is 0)
    for category in ['better', 'similar', 'worse']:
        if category not in performance_counts:
            performance_counts[category] = 0
    
    # Create display labels
    display_labels = {
        'better': 'Better than benchmark',
        'similar': 'Similar to benchmark',
        'worse': 'Worse than benchmark'
    }
    
    performance_counts.index = performance_counts.index.map(display_labels)
    
    # Load color palette and create color mapping
    color_palette = get_color_palette()
    color_mapping = {
        'Better than benchmark': color_palette["mint"],
        'Similar to benchmark': color_palette["amber"],
        'Worse than benchmark': color_palette["crimson"]
    }
    
    # Create category order
    category_order = ['Worse than benchmark', 'Similar to benchmark', 'Better than benchmark']
    
    # Get total low quality studies for percentage calculation
    total_low_quality_studies = len(low_quality_studies)
    
    # Create the chart
    save_path = os.path.join(save_dir, filename)
    
    fig = create_horizontal_bar_chart(
        data=performance_counts,
        title="Performance vs Low Quality Benchmarks",
        xlabel="Number of Studies",
        ylabel="Performance",
        figsize=(10, 5),
        color=color_mapping,
        show_percentages=True,
        save_path=save_path,
        sort_ascending=False,
        category_order=category_order,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16,
        percentage_total=total_low_quality_studies
    )
    
    return save_path


def create_combined_performance_chart(
    df: pd.DataFrame,
    save_dir: str,
    total_studies_without_participants: int,
    filename: str = "performance_combined_quality_benchmarks.svg"
) -> str:
    """
    Create a combined figure with performance charts for high and low quality benchmarks 
    stacked vertically with aligned x-axis and a shared legend.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the measurements information
    save_dir : str
        Directory to save the chart
    total_studies_without_participants : int
        Total number of studies without human participants (for percentage calculation denominator)
    filename : str, default="performance_combined_quality_benchmarks.svg"
        Filename for the saved chart
        
    Returns
    -------
    str
        Path to the saved chart file
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 5))
    
    # Generate individual charts without saving them
    high_quality_fig = create_horizontal_bar_chart(
        data=_get_performance_data_high_quality(df),
        title="Performance against human support-providers",
        xlabel="Number of articles",
        ylabel=None,
        figsize=(10, 5),
        color=_get_performance_color_mapping(),
        show_percentages=True,
        save_path=None,
        sort_ascending=False,
        category_order=['Worse than benchmark', 'Similar to benchmark', 'Better than benchmark'],
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16,
        percentage_total=_get_high_quality_study_count(df)
    )
    
    low_quality_fig = create_horizontal_bar_chart(
        data=_get_performance_data_low_quality(df),
        title="Performance against another technical application",
        xlabel="Number of articles",
        ylabel=None,
        figsize=(10, 5),
        color=_get_performance_color_mapping(),
        show_percentages=True,
        save_path=None,
        sort_ascending=False,
        category_order=['Worse than benchmark', 'Similar to benchmark', 'Better than benchmark'],
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16,
        percentage_total=_get_low_quality_study_count(df)
    )
    
    # Get the axes from the individual figures
    high_ax = high_quality_fig.axes[0]
    low_ax = low_quality_fig.axes[0]
    
    bar_height = 0.6
    
    # === TOP SUBPLOT: High Quality Benchmarks ===
    # Copy bars
    for bar in high_ax.patches:
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        ax1.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in high_ax.texts:
        ax1.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Set title and y-axis properties
    ax1.set_title(high_ax.get_title(), fontsize=14, pad=20)
    ax1.set_ylim(high_ax.get_ylim())
    ax1.set_yticks([])  # Remove y-ticks
    ax1.set_yticklabels([])  # Remove y-tick labels
    ax1.set_xticklabels([])
    
    # Styling - keep y-axis spine visible
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(True)  # Keep y-axis line
    ax1.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax1.grid(axis='y', visible=False)
    
    # === BOTTOM SUBPLOT: Low Quality Benchmarks ===
    # Copy bars
    for bar in low_ax.patches:
        x = bar.get_x()
        y = bar.get_y()
        width = bar.get_width()
        height = bar.get_height()
        color = bar.get_facecolor()
        ax2.barh(y + height/2, width, height=bar_height, color=color, alpha=0.8)
    
    # Copy text annotations
    for text in low_ax.texts:
        ax2.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize())
    
    # Set title and y-axis properties
    ax2.set_title(low_ax.get_title(), fontsize=14, pad=20)
    ax2.set_xlabel("Number of Studies", fontsize=12)
    ax2.set_ylim(low_ax.get_ylim())
    ax2.set_yticks([])  # Remove y-ticks
    ax2.set_yticklabels([])  # Remove y-tick labels
    
    # Styling - keep y-axis spine visible
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(True)  # Keep y-axis line
    ax2.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax2.grid(axis='y', visible=False)
    
    # Align x-axes
    max_xlim = max(high_ax.get_xlim()[1], low_ax.get_xlim()[1])
    ax1.set_xlim(0, max_xlim)
    ax2.set_xlim(0, max_xlim)
    
    # Ensure bottom subplot has x-axis tick labels
    ax2.tick_params(axis='x', which='major', labelsize=10)
    
    # Create legend in the upper right of the top chart
    color_palette = get_color_palette()
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_palette["mint"], alpha=0.8, label='Better than benchmark'),
        Patch(facecolor=color_palette["amber"], alpha=0.8, label='Similar to benchmark'),
        Patch(facecolor=color_palette["crimson"], alpha=0.8, label='Worse than benchmark')
    ]
    
    # Place legend in upper right of the top subplot
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.9)
    
    # Close individual figures
    plt.close(high_quality_fig)
    plt.close(low_quality_fig)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save figure
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def _get_performance_color_mapping():
    """Helper function to get consistent color mapping for performance charts."""
    color_palette = get_color_palette()
    return {
        'Better than benchmark': color_palette["mint"],
        'Similar to benchmark': color_palette["amber"],
        'Worse than benchmark': color_palette["crimson"]
    }


def _get_performance_data_high_quality(df: pd.DataFrame) -> pd.Series:
    """Helper function to extract performance data for high quality benchmarks."""
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    def normalize_benchmark_quality(value):
        value_str = str(value).strip().lower()
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        else:
            return 'no benchmark'
    
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    high_quality_studies = set()
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        if 'high' in benchmark_qualities:
            high_quality_studies.add(study)
    
    def normalize_performance(value):
        if pd.isna(value) or str(value).strip() == '':
            return None
        value_str = str(value).strip().lower()
        if value_str in ['b', 'better']:
            return 'better'
        elif value_str in ['s', 'similar']:
            return 'similar'
        elif value_str in ['w', 'worse']:
            return 'worse'
        else:
            return None
    
    performance_rank = {'better': 3, 'similar': 2, 'worse': 1}
    
    study_best_performance = {}
    for study in high_quality_studies:
        study_high_measurements = filtered_df[
            (filtered_df['reference_title'] == study) &
            (filtered_df['normalized_benchmark_quality'] == 'high')
        ]
        performances = study_high_measurements['performance_vs_benchmark'].apply(normalize_performance)
        performances = performances.dropna()
        
        if len(performances) > 0:
            best_performance = max(performances, key=lambda x: performance_rank.get(x, 0))
            study_best_performance[study] = best_performance
    
    performance_counts = pd.Series(study_best_performance.values()).value_counts()
    
    for category in ['better', 'similar', 'worse']:
        if category not in performance_counts:
            performance_counts[category] = 0
    
    display_labels = {
        'better': 'Better than benchmark',
        'similar': 'Similar to benchmark',
        'worse': 'Worse than benchmark'
    }
    
    performance_counts.index = performance_counts.index.map(display_labels)
    return performance_counts


def _get_performance_data_low_quality(df: pd.DataFrame) -> pd.Series:
    """Helper function to extract performance data for low quality benchmarks."""
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    def normalize_benchmark_quality(value):
        value_str = str(value).strip().lower()
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        else:
            return 'no benchmark'
    
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    low_quality_studies = set()
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        if 'low' in benchmark_qualities and 'high' not in benchmark_qualities:
            low_quality_studies.add(study)
    
    def normalize_performance(value):
        if pd.isna(value) or str(value).strip() == '':
            return None
        value_str = str(value).strip().lower()
        if value_str in ['b', 'better']:
            return 'better'
        elif value_str in ['s', 'similar']:
            return 'similar'
        elif value_str in ['w', 'worse']:
            return 'worse'
        else:
            return None
    
    performance_rank = {'better': 3, 'similar': 2, 'worse': 1}
    
    study_best_performance = {}
    for study in low_quality_studies:
        study_low_measurements = filtered_df[
            (filtered_df['reference_title'] == study) &
            (filtered_df['normalized_benchmark_quality'] == 'low')
        ]
        performances = study_low_measurements['performance_vs_benchmark'].apply(normalize_performance)
        performances = performances.dropna()
        
        if len(performances) > 0:
            best_performance = max(performances, key=lambda x: performance_rank.get(x, 0))
            study_best_performance[study] = best_performance
    
    performance_counts = pd.Series(study_best_performance.values()).value_counts()
    
    for category in ['better', 'similar', 'worse']:
        if category not in performance_counts:
            performance_counts[category] = 0
    
    display_labels = {
        'better': 'Better than benchmark',
        'similar': 'Similar to benchmark',
        'worse': 'Worse than benchmark'
    }
    
    performance_counts.index = performance_counts.index.map(display_labels)
    return performance_counts


def _get_high_quality_study_count(df: pd.DataFrame) -> int:
    """Helper function to get count of high quality studies."""
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    def normalize_benchmark_quality(value):
        value_str = str(value).strip().lower()
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        else:
            return 'no benchmark'
    
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    high_quality_studies = set()
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        if 'high' in benchmark_qualities:
            high_quality_studies.add(study)
    
    return len(high_quality_studies)


def _get_low_quality_study_count(df: pd.DataFrame) -> int:
    """Helper function to get count of low quality studies."""
    filtered_df = df.dropna(subset=['benchmark_quality'])
    filtered_df = filtered_df[filtered_df['benchmark_quality'].str.strip() != '']
    
    def normalize_benchmark_quality(value):
        value_str = str(value).strip().lower()
        if value_str in ['h', 'high']:
            return 'high'
        elif value_str in ['l', 'low']:
            return 'low'
        else:
            return 'no benchmark'
    
    filtered_df = filtered_df.copy()
    filtered_df['normalized_benchmark_quality'] = filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
    
    low_quality_studies = set()
    for study in filtered_df['reference_title'].unique():
        study_data = filtered_df[filtered_df['reference_title'] == study]
        benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
        if 'low' in benchmark_qualities and 'high' not in benchmark_qualities:
            low_quality_studies.add(study)
    
    return len(low_quality_studies)


def generate_measurements_metrics_report(
    df: pd.DataFrame,
    save_dir: str,
    total_studies_without_participants: int,
    filename: str = "5-2_measurements_metrics_report.txt"
) -> str:
    """
    Generate a comprehensive report on measurements metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the measurements information
    save_dir : str
        Directory to save the report
    total_studies_without_participants : int
        Total number of studies without human participants (for percentage calculation denominator)
    filename : str, default="5-2_measurements_metrics_report.txt"
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
        f.write("MEASUREMENTS METRICS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic measurements statistics
        f.write("BASIC STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total studies without human participants: {total_studies_without_participants}\n")
        f.write(f"Articles with measurements: {df['reference_title'].nunique()}\n")
        f.write(f"Articles without measurements: {total_studies_without_participants - df['reference_title'].nunique()}\n")
        f.write(f"Total measurements: {len(df)}\n")
        
        # Metric supercategory analysis
        f.write("\nMETRIC SUPERCATEGORY DISTRIBUTION\n")
        f.write("-" * 35 + "\n")
        
        # Filter out null/empty metric supercategories
        filtered_df = df.dropna(subset=['metric_supercategory'])
        filtered_df = filtered_df[filtered_df['metric_supercategory'].str.strip() != '']
        
        f.write(f"Measurements with supercategory information: {len(filtered_df)}\n")
        f.write(f"Measurements without supercategory information: {len(df) - len(filtered_df)}\n\n")
        
        if len(filtered_df) > 0:
            # Count by measurements
            supercategory_measurement_counts = filtered_df['metric_supercategory'].value_counts()
            supercategory_measurement_percentages = (supercategory_measurement_counts / len(filtered_df) * 100).round(1)
            
            # Count by unique articles
            supercategory_article_counts = (
                filtered_df.groupby('metric_supercategory')['reference_title']
                .nunique()
                .sort_values(ascending=False)
            )
            
            f.write("Metric supercategory breakdown by measurements:\n")
            for supercategory, count in supercategory_measurement_counts.items():
                percentage = supercategory_measurement_percentages[supercategory]
                f.write(f"  {supercategory}: {count} measurements ({percentage}%)\n")
            
            f.write(f"\nMetric supercategory breakdown by articles:\n")
            f.write(f"(Percentages calculated using total studies without participants: n={total_studies_without_participants})\n")
            for supercategory, count in supercategory_article_counts.items():
                percentage = round(count / total_studies_without_participants * 100, 1)
                f.write(f"  {supercategory}: {count} articles ({percentage}%)\n")
        
        # Metric category analysis
        f.write(f"\nMETRIC CATEGORY ANALYSIS\n")
        f.write("-" * 25 + "\n")
        
        if 'metric_category' in df.columns:
            category_filtered_df = df.dropna(subset=['metric_category'])
            category_filtered_df = category_filtered_df[category_filtered_df['metric_category'].str.strip() != '']
            
            f.write(f"Measurements with category information: {len(category_filtered_df)}\n")
            
            if len(category_filtered_df) > 0:
                category_counts = category_filtered_df['metric_category'].value_counts()
                f.write(f"Unique metric categories: {len(category_counts)}\n")
                f.write(f"Top 10 metric categories by measurement count:\n")
                for category, count in category_counts.head(10).items():
                    f.write(f"  {category}: {count}\n")
        
        # Benchmark quality analysis
        f.write(f"\nBENCHMARK QUALITY ANALYSIS\n")
        f.write("-" * 27 + "\n")
        
        if 'benchmark_quality' in df.columns:
            benchmark_filtered_df = df.dropna(subset=['benchmark_quality'])
            benchmark_filtered_df = benchmark_filtered_df[benchmark_filtered_df['benchmark_quality'].str.strip() != '']
            
            f.write(f"Measurements with benchmark quality information: {len(benchmark_filtered_df)}\n")
            
            if len(benchmark_filtered_df) > 0:
                benchmark_counts = benchmark_filtered_df['benchmark_quality'].value_counts()
                f.write(f"Benchmark quality distribution:\n")
                for quality, count in benchmark_counts.items():
                    percentage = round(count / len(benchmark_filtered_df) * 100, 1)
                    f.write(f"  {quality}: {count} ({percentage}%)\n")
        
        # Performance vs benchmark analysis
        f.write(f"\nPERFORMANCE VS BENCHMARK ANALYSIS\n")
        f.write("-" * 35 + "\n")
        
        if 'performance_vs_benchmark' in df.columns:
            perf_filtered_df = df.dropna(subset=['performance_vs_benchmark'])
            perf_filtered_df = perf_filtered_df[perf_filtered_df['performance_vs_benchmark'].str.strip() != '']
            
            f.write(f"Measurements with performance comparison information: {len(perf_filtered_df)}\n")
            
            if len(perf_filtered_df) > 0:
                perf_counts = perf_filtered_df['performance_vs_benchmark'].value_counts()
                f.write(f"Performance vs benchmark distribution:\n")
                for performance, count in perf_counts.items():
                    percentage = round(count / len(perf_filtered_df) * 100, 1)
                    f.write(f"  {performance}: {count} ({percentage}%)\n")
        
        # Benchmark quality aggregation by study analysis
        f.write(f"\nBENCHMARK QUALITY BY STUDY ANALYSIS\n")
        f.write("-" * 37 + "\n")
        
        if 'benchmark_quality' in df.columns:
            # Apply the same aggregation logic as in create_benchmark_quality_chart
            benchmark_filtered_df = df.dropna(subset=['benchmark_quality'])
            benchmark_filtered_df = benchmark_filtered_df[benchmark_filtered_df['benchmark_quality'].str.strip() != '']
            
            f.write(f"Studies with benchmark quality information: {benchmark_filtered_df['reference_title'].nunique()}\n")
            f.write(f"Studies without benchmark quality information: {df['reference_title'].nunique() - benchmark_filtered_df['reference_title'].nunique()}\n\n")
            
            if len(benchmark_filtered_df) > 0:
                # Normalize benchmark quality values
                def normalize_benchmark_quality(value):
                    if pd.isna(value) or str(value).strip() == '':
                        return 'no benchmark'
                    
                    value_str = str(value).strip().lower()
                    
                    if value_str in ['h', 'high']:
                        return 'high'
                    elif value_str in ['l', 'low']:
                        return 'low'
                    elif value_str in ['no benchmark', 'no_benchmark']:
                        return 'no benchmark'
                    else:
                        return 'no benchmark'
                
                # Apply normalization
                benchmark_filtered_df = benchmark_filtered_df.copy()
                benchmark_filtered_df['normalized_benchmark_quality'] = benchmark_filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
                
                # Aggregate by study (reference_title)
                study_benchmark_quality = {}
                
                for study in benchmark_filtered_df['reference_title'].unique():
                    study_data = benchmark_filtered_df[benchmark_filtered_df['reference_title'] == study]
                    benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
                    
                    # Apply aggregation logic
                    if 'high' in benchmark_qualities:
                        study_benchmark_quality[study] = 'High quality'
                    elif 'low' in benchmark_qualities:
                        study_benchmark_quality[study] = 'Low quality'
                    else:
                        study_benchmark_quality[study] = 'No benchmark'
                
                # Count studies by aggregated benchmark quality
                quality_counts = pd.Series(study_benchmark_quality.values()).value_counts()
                total_studies_with_data = len(study_benchmark_quality)
                
                f.write(f"Benchmark quality aggregation logic per study:\n")
                f.write(f"  - High quality: if study has at least one measurement with high quality benchmark\n")
                f.write(f"  - Low quality: if study has at least one low quality benchmark but no high quality\n")
                f.write(f"  - No benchmark: if study has no benchmarks (all measurements marked as 'no benchmark')\n\n")
                
                f.write(f"Aggregated benchmark quality distribution by study:\n")
                for quality, count in quality_counts.items():
                    percentage = round(count / total_studies_with_data * 100, 1)
                    f.write(f"  {quality}: {count} studies ({percentage}%)\n")
                
                f.write(f"\nRaw benchmark quality distribution by measurements:\n")
                raw_quality_counts = benchmark_filtered_df['benchmark_quality'].value_counts()
                for quality, count in raw_quality_counts.items():
                    percentage = round(count / len(benchmark_filtered_df) * 100, 1)
                    f.write(f"  {quality}: {count} measurements ({percentage}%)\n")
        
        # Performance vs benchmark analysis by quality group
        f.write(f"\nPERFORMANCE VS BENCHMARK BY QUALITY GROUP\n")
        f.write("-" * 42 + "\n")
        
        if 'benchmark_quality' in df.columns and 'performance_vs_benchmark' in df.columns:
            # Apply the same logic as in the chart functions
            perf_filtered_df = df.dropna(subset=['benchmark_quality'])
            perf_filtered_df = perf_filtered_df[perf_filtered_df['benchmark_quality'].str.strip() != '']
            
            def normalize_benchmark_quality(value):
                if pd.isna(value) or str(value).strip() == '':
                    return 'no benchmark'
                value_str = str(value).strip().lower()
                if value_str in ['h', 'high']:
                    return 'high'
                elif value_str in ['l', 'low']:
                    return 'low'
                else:
                    return 'no benchmark'
            
            def normalize_performance(value):
                if pd.isna(value) or str(value).strip() == '':
                    return None
                value_str = str(value).strip().lower()
                if value_str in ['b', 'better']:
                    return 'better'
                elif value_str in ['s', 'similar']:
                    return 'similar'
                elif value_str in ['w', 'worse']:
                    return 'worse'
                else:
                    return None
            
            perf_filtered_df = perf_filtered_df.copy()
            perf_filtered_df['normalized_benchmark_quality'] = perf_filtered_df['benchmark_quality'].apply(normalize_benchmark_quality)
            
            # Performance ranking (better > similar > worse)
            performance_rank = {'better': 3, 'similar': 2, 'worse': 1}
            
            # Identify high quality studies
            high_quality_studies = set()
            low_quality_studies = set()
            
            for study in perf_filtered_df['reference_title'].unique():
                study_data = perf_filtered_df[perf_filtered_df['reference_title'] == study]
                benchmark_qualities = study_data['normalized_benchmark_quality'].unique()
                if 'high' in benchmark_qualities:
                    high_quality_studies.add(study)
                elif 'low' in benchmark_qualities:
                    low_quality_studies.add(study)
            
            f.write(f"Performance aggregation logic per study:\n")
            f.write(f"  - For high quality group: select measurements with high quality benchmarks, show best performance\n")
            f.write(f"  - For low quality group: select measurements with low quality benchmarks, show best performance\n")
            f.write(f"  - Performance ranking: better > similar > worse\n\n")
            
            # Analyze high quality studies
            f.write(f"HIGH QUALITY BENCHMARKS GROUP:\n")
            f.write(f"Studies in high quality group: {len(high_quality_studies)}\n")
            
            high_study_best = {}
            for study in high_quality_studies:
                study_high_measurements = perf_filtered_df[
                    (perf_filtered_df['reference_title'] == study) &
                    (perf_filtered_df['normalized_benchmark_quality'] == 'high')
                ]
                performances = study_high_measurements['performance_vs_benchmark'].apply(normalize_performance)
                performances = performances.dropna()
                
                if len(performances) > 0:
                    best_performance = max(performances, key=lambda x: performance_rank.get(x, 0))
                    high_study_best[study] = best_performance
            
            f.write(f"Studies with performance data: {len(high_study_best)}\n")
            
            if len(high_study_best) > 0:
                high_counts = pd.Series(high_study_best.values()).value_counts()
                f.write(f"Performance distribution:\n")
                for perf, count in high_counts.items():
                    percentage = round(count / len(high_quality_studies) * 100, 1)
                    f.write(f"  {perf} than benchmark: {count} studies ({percentage}%)\n")
            
            # Analyze low quality studies
            f.write(f"\nLOW QUALITY BENCHMARKS GROUP:\n")
            f.write(f"Studies in low quality group: {len(low_quality_studies)}\n")
            
            low_study_best = {}
            for study in low_quality_studies:
                study_low_measurements = perf_filtered_df[
                    (perf_filtered_df['reference_title'] == study) &
                    (perf_filtered_df['normalized_benchmark_quality'] == 'low')
                ]
                performances = study_low_measurements['performance_vs_benchmark'].apply(normalize_performance)
                performances = performances.dropna()
                
                if len(performances) > 0:
                    best_performance = max(performances, key=lambda x: performance_rank.get(x, 0))
                    low_study_best[study] = best_performance
            
            f.write(f"Studies with performance data: {len(low_study_best)}\n")
            
            if len(low_study_best) > 0:
                low_counts = pd.Series(low_study_best.values()).value_counts()
                f.write(f"Performance distribution:\n")
                for perf, count in low_counts.items():
                    percentage = round(count / len(low_quality_studies) * 100, 1)
                    f.write(f"  {perf} than benchmark: {count} studies ({percentage}%)\n")
    
    return report_path


def create_measurements_metrics_panel(
    measurements_file: str,
    output_dir: str,
    total_studies_without_participants: int
) -> Dict[str, Any]:
    """
    Create a complete measurements metrics panel with charts and reports.
    
    Parameters
    ----------
    measurements_file : str
        Path to the CSV file containing measurements information
    output_dir : str
        Directory to save all outputs
    total_studies_without_participants : int
        Total number of studies without human participants (for percentage calculation denominator)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing paths to created files and summary statistics
    """
    # Load the data
    df = pd.read_csv(measurements_file)
    
    # Create the metric supercategories chart
    chart_path = create_metric_supercategories_chart(df, output_dir, total_studies_without_participants)
    
    # Create the benchmark quality chart
    benchmark_chart_path = create_benchmark_quality_chart(df, output_dir, total_studies_without_participants)
    benchmark_stacked_chart_path = create_benchmark_quality_stacked_chart(df, output_dir, total_studies_without_participants)
    
    # Create the performance charts
    performance_high_chart_path = create_performance_high_quality_chart(df, output_dir)
    performance_low_chart_path = create_performance_low_quality_chart(df, output_dir)
    performance_combined_chart_path = create_combined_performance_chart(df, output_dir, total_studies_without_participants)
    
    # Generate the report
    report_path = generate_measurements_metrics_report(df, output_dir, total_studies_without_participants)
    
    # Calculate summary statistics
    filtered_df = df.dropna(subset=['metric_supercategory'])
    filtered_df = filtered_df[filtered_df['metric_supercategory'].str.strip() != '']
    
    # Count unique articles per supercategory
    supercategory_article_counts = (
        filtered_df.groupby('metric_supercategory')['reference_title']
        .nunique()
        .sort_values(ascending=False)
    )
    
    # Filter out supercategories that appear in only one article
    supercategory_article_counts = supercategory_article_counts[supercategory_article_counts > 1]
    
    summary = {
        'total_measurements': len(df),
        'unique_articles': df['reference_title'].nunique(),
        'total_studies_without_participants': total_studies_without_participants,
        'articles_without_measurements': total_studies_without_participants - df['reference_title'].nunique(),
        'unique_supercategories': len(supercategory_article_counts),
        'most_common_supercategory': supercategory_article_counts.index[0] if len(supercategory_article_counts) > 0 else None,
        'chart_path': chart_path,
        'benchmark_chart_path': benchmark_chart_path,
        'benchmark_stacked_chart_path': benchmark_stacked_chart_path,
        'performance_high_chart_path': performance_high_chart_path,
        'performance_low_chart_path': performance_low_chart_path,
        'performance_combined_chart_path': performance_combined_chart_path,
        'report_path': report_path
    }
    
    return summary
