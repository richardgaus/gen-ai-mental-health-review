"""
Overview panel creation utilities.

This module contains functions to create comprehensive reports and visualizations
showing the distribution of application types and their subtypes in LLM psychotherapy research.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from .utils import create_time_series_chart, get_color_palette


def generate_overview_report(df: pd.DataFrame, excluded_population_surveys: bool = False, original_count: int = None) -> str:
    """
    Generate a comprehensive overview report on application types and subtypes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the dataset (potentially filtered)
    excluded_population_surveys : bool, default=False
        Whether population surveys were excluded from the analysis
    original_count : int, optional
        Original number of studies before filtering (if applicable)
    
    Returns
    -------
    str
        Formatted text report
    """
    from datetime import datetime
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("LLM APPLICATIONS IN PSYCHOTHERAPY RESEARCH - OVERVIEW REPORT")
    report_lines.append("Application Types and Subtypes Distribution Analysis")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset overview
    total_studies = len(df)
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 80)
    
    if excluded_population_surveys and original_count is not None:
        excluded_count = original_count - total_studies
        report_lines.append(f"Total studies analyzed: {total_studies}")
        report_lines.append(f"Original dataset size: {original_count}")
        report_lines.append(f"Population surveys excluded: {excluded_count}")
        report_lines.append("")
        report_lines.append("Filtering criteria:")
        report_lines.append("  ✗ Excluded: study_type == 'Population survey'")
        report_lines.append("  ✓ Included: All other study types")
    else:
        report_lines.append(f"Total studies in dataset: {total_studies}")
        report_lines.append("")
        report_lines.append("Analysis scope: All studies (no exclusions applied)")
    
    report_lines.append("")
    
    # Application type distribution
    app_type_counts = df['application_type'].value_counts()
    
    report_lines.append("APPLICATION TYPE DISTRIBUTION")
    report_lines.append("-" * 80)
    
    for app_type, count in app_type_counts.items():
        percentage = (count / total_studies) * 100
        report_lines.append(f"{app_type}: {count} studies ({percentage:.1f}%)")
    
    report_lines.append("")
    
    # Detailed analysis for Client-facing applications
    client_facing_df = df[df['application_type'] == 'Client-facing application']
    if not client_facing_df.empty:
        report_lines.append("CLIENT-FACING APPLICATION SUBTYPES")
        report_lines.append("-" * 80)
        report_lines.append(f"Total client-facing applications: {len(client_facing_df)}")
        report_lines.append("")
        
        # Analyze client-facing subtypes
        client_subtypes = client_facing_df['application_subtype_client_facing'].value_counts()
        
        report_lines.append("Subtype breakdown:")
        for subtype, count in client_subtypes.items():
            if pd.notna(subtype):
                percentage = (count / len(client_facing_df)) * 100
                report_lines.append(f"  • {subtype}: {count} studies ({percentage:.1f}%)")
        
        # Check for missing subtypes
        missing_subtypes = client_facing_df['application_subtype_client_facing'].isna().sum()
        if missing_subtypes > 0:
            percentage = (missing_subtypes / len(client_facing_df)) * 100
            report_lines.append(f"  • [Missing/Unspecified]: {missing_subtypes} studies ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Additional analysis for client-facing apps
        report_lines.append("Client-facing application characteristics:")
        
        # Study types within client-facing
        study_types = client_facing_df['study_type'].value_counts()
        report_lines.append("  Study types:")
        for study_type, count in study_types.items():
            if pd.notna(study_type):
                percentage = (count / len(client_facing_df)) * 100
                report_lines.append(f"    - {study_type}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
    
    # Detailed analysis for Therapist guidance applications
    therapist_df = df[df['application_type'] == 'Therapist guidance application']
    if not therapist_df.empty:
        report_lines.append("THERAPIST GUIDANCE APPLICATION SUBTYPES")
        report_lines.append("-" * 80)
        report_lines.append(f"Total therapist guidance applications: {len(therapist_df)}")
        report_lines.append("")
        
        # Analyze therapist-facing subtypes
        therapist_subtypes = therapist_df['application_subtype_therapist_facing'].value_counts()
        
        report_lines.append("Subtype breakdown:")
        for subtype, count in therapist_subtypes.items():
            if pd.notna(subtype):
                percentage = (count / len(therapist_df)) * 100
                report_lines.append(f"  • {subtype}: {count} studies ({percentage:.1f}%)")
        
        # Check for missing subtypes
        missing_subtypes = therapist_df['application_subtype_therapist_facing'].isna().sum()
        if missing_subtypes > 0:
            percentage = (missing_subtypes / len(therapist_df)) * 100
            report_lines.append(f"  • [Missing/Unspecified]: {missing_subtypes} studies ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Additional analysis for therapist guidance apps
        report_lines.append("Therapist guidance application characteristics:")
        
        # Study types within therapist guidance
        study_types = therapist_df['study_type'].value_counts()
        report_lines.append("  Study types:")
        for study_type, count in study_types.items():
            if pd.notna(study_type):
                percentage = (count / len(therapist_df)) * 100
                report_lines.append(f"    - {study_type}: {count} ({percentage:.1f}%)")
        
        report_lines.append("")
    
    # Analysis of other application types
    other_app_types = [app_type for app_type in app_type_counts.index 
                      if app_type not in ['Client-facing application', 'Therapist guidance application']]
    
    if other_app_types:
        report_lines.append("OTHER APPLICATION TYPES")
        report_lines.append("-" * 80)
        
        for app_type in other_app_types:
            if pd.notna(app_type):
                app_df = df[df['application_type'] == app_type]
                count = len(app_df)
                percentage = (count / total_studies) * 100
                
                report_lines.append(f"{app_type}: {count} studies ({percentage:.1f}%)")
                
                # Study types within this application type
                study_types = app_df['study_type'].value_counts()
                if len(study_types) > 1:
                    report_lines.append("  Study types:")
                    for study_type, st_count in study_types.items():
                        if pd.notna(study_type):
                            st_percentage = (st_count / count) * 100
                            report_lines.append(f"    - {study_type}: {st_count} ({st_percentage:.1f}%)")
                
                report_lines.append("")
    
    # Cross-tabulation analysis
    report_lines.append("CROSS-TABULATION: APPLICATION TYPE vs STUDY TYPE")
    report_lines.append("-" * 80)
    
    # Create cross-tabulation
    crosstab = pd.crosstab(df['application_type'], df['study_type'], margins=True)
    
    # Format the cross-tabulation for the report
    report_lines.append("Application Type \\ Study Type:")
    report_lines.append("")
    
    # Header row
    header = "Application Type".ljust(35)
    for col in crosstab.columns:
        if col != 'All':
            header += f"{str(col)[:20]:>22}"
    header += f"{'Total':>22}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Data rows
    for idx in crosstab.index:
        if idx != 'All':
            row = f"{str(idx)[:34]:34}"
            for col in crosstab.columns:
                if col != 'All':
                    value = crosstab.loc[idx, col]
                    row += f"{value:>22}"
            # Add total
            total = crosstab.loc[idx, 'All']
            row += f"{total:>22}"
            report_lines.append(row)
    
    # Total row
    total_row = "TOTAL".ljust(34)
    for col in crosstab.columns:
        if col != 'All':
            value = crosstab.loc['All', col]
            total_row += f"{value:>22}"
    total_row += f"{crosstab.loc['All', 'All']:>22}"
    report_lines.append("-" * len(header))
    report_lines.append(total_row)
    
    report_lines.append("")
    
    # Summary insights
    report_lines.append("SUMMARY INSIGHTS")
    report_lines.append("-" * 80)
    
    # Most common application type
    most_common_app = app_type_counts.index[0]
    most_common_count = app_type_counts.iloc[0]
    most_common_pct = (most_common_count / total_studies) * 100
    
    report_lines.append(f"• Most common application type: {most_common_app}")
    report_lines.append(f"  ({most_common_count} studies, {most_common_pct:.1f}% of total)")
    
    # Client-facing vs therapist-facing comparison
    client_count = app_type_counts.get('Client-facing application', 0)
    therapist_count = app_type_counts.get('Therapist guidance application', 0)
    
    if client_count > 0 and therapist_count > 0:
        ratio = client_count / therapist_count
        report_lines.append(f"• Client-facing to therapist guidance ratio: {ratio:.1f}:1")
        report_lines.append(f"  ({client_count} client-facing vs {therapist_count} therapist guidance)")
    elif client_count > 0:
        report_lines.append(f"• {client_count} client-facing applications, no therapist guidance applications")
    elif therapist_count > 0:
        report_lines.append(f"• {therapist_count} therapist guidance applications, no client-facing applications")
    
    # Empirical research focus
    empirical_count = len(df[df['study_type'] == 'Empirical research involving an LLM'])
    empirical_pct = (empirical_count / total_studies) * 100
    report_lines.append(f"• Empirical LLM research: {empirical_count} studies ({empirical_pct:.1f}% of total)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def get_application_statistics(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics about application types and subtypes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    
    Returns
    -------
    Dict
        Dictionary containing various statistics
    """
    
    stats = {
        'total_studies': len(df),
        'application_types': df['application_type'].value_counts().to_dict(),
        'client_facing_subtypes': {},
        'therapist_facing_subtypes': {}
    }
    
    # Client-facing subtypes
    client_df = df[df['application_type'] == 'Client-facing application']
    if not client_df.empty:
        stats['client_facing_subtypes'] = client_df['application_subtype_client_facing'].value_counts().to_dict()
    
    # Therapist-facing subtypes
    therapist_df = df[df['application_type'] == 'Therapist guidance application']
    if not therapist_df.empty:
        stats['therapist_facing_subtypes'] = therapist_df['application_subtype_therapist_facing'].value_counts().to_dict()
    
    return stats


def create_outlet_field_time_series(
    df: pd.DataFrame,
    title: str = "Publications by Outlet Field Over Time",
    xlabel: str = "Year",
    ylabel: str = "Number of Publications",
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create a time series chart showing publication trends by outlet field categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing year and outlet_field columns
    title : str, default="Publications by Outlet Field Over Time"
        Chart title
    xlabel : str, default="Year"
        X-axis label
    ylabel : str, default="Number of Publications"
        Y-axis label
    figsize : Tuple[float, float], default=(12, 8)
        Figure size as (width, height)
    save_path : Optional[str], default=None
        Path to save the figure. If None, don't save
    
    Returns
    -------
    None
        Displays the chart and optionally saves it
    """
    
    # Create a copy for processing
    df_processed = df.copy()

    # Get the color palette
    color_dict = get_color_palette()
    colors = {
        'Clinical / Health Sciences': color_dict['navy'],
        'Computational / Engineering': color_dict['amber'],
        'Other': color_dict['grey']
    }
    
    # Define the field categorization function
    def categorize_outlet_field(field):
        if pd.isna(field):
            return "Other"
        
        field_str = str(field).lower()
        
        # Clinical / Health Sciences
        clinical_keywords = ['psychology', 'psychiatry', 'medicine', 'digital health']
        if any(keyword in field_str for keyword in clinical_keywords):
            return "Clinical / Health Sciences"
        
        # Computational / Engineering
        computational_keywords = ['computer science', 'hci']
        if any(keyword in field_str for keyword in computational_keywords):
            return "Computational / Engineering"
        
        # Everything else
        return "Other"
    
    # Apply categorization
    df_processed['outlet_field_category'] = df_processed['outlet_field'].apply(categorize_outlet_field)
    
    # Filter out rows without valid years
    df_processed = df_processed.dropna(subset=['year'])
    df_processed['year'] = df_processed['year'].astype(int)
    
    # Create the time series chart
    fig = create_time_series_chart(
        df=df_processed,
        year_column='year',
        category_column='outlet_field_category',
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        colors=colors,
        save_path=save_path,
        show_counts=True,
        show_totals=True,
        trend_line_start_year=None,  # No trend line
        legend_fontsize=14,
        label_fontsize=14,
        title_fontsize=16
    )
    
    return fig


def get_outlet_field_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about outlet field categorization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    
    Returns
    -------
    Dict
        Dictionary containing outlet field statistics
    """
    
    # Apply the same categorization as in the chart
    def categorize_outlet_field(field):
        if pd.isna(field):
            return "Other"
        
        field_str = str(field).lower()
        
        # Clinical / Health Sciences
        clinical_keywords = ['psychology', 'psychiatry', 'medicine', 'digital health']
        if any(keyword in field_str for keyword in clinical_keywords):
            return "Clinical / Health Sciences"
        
        # Computational / Engineering
        computational_keywords = ['computer science', 'hci']
        if any(keyword in field_str for keyword in computational_keywords):
            return "Computational / Engineering"
        
        # Everything else
        return "Other"
    
    df['outlet_field_category'] = df['outlet_field'].apply(categorize_outlet_field)
    
    category_counts = df['outlet_field_category'].value_counts()
    original_field_counts = df['outlet_field'].value_counts()
    
    stats = {
        'total_studies': len(df),
        'category_distribution': category_counts.to_dict(),
        'original_field_distribution': original_field_counts.to_dict(),
        'categorization_mapping': {}
    }
    
    # Create mapping of original fields to categories
    for field in df['outlet_field'].unique():
        if pd.notna(field):
            category = categorize_outlet_field(field)
            if category not in stats['categorization_mapping']:
                stats['categorization_mapping'][category] = []
            stats['categorization_mapping'][category].append(field)
    
    return stats
