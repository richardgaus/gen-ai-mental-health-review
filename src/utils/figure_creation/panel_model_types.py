"""
Panel for creating model types visualization.

This module contains functions to create horizontal bar charts showing the distribution
of language models employed in research studies.
"""

from typing import Optional, Tuple
import pandas as pd
from collections import Counter
from .utils import create_horizontal_bar_chart, create_time_series_chart, get_color_palette


def create_model_types_panel(
    df: pd.DataFrame,
    models_column: str = "models_employed",
    title: str = "Language Models Employed in Studies",
    xlabel: str = "Number of Studies",
    ylabel: str = "Model Type",
    figsize: Tuple[float, float] = (10, 8),
    top_n: int = 6,
    exclude_other: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Create a horizontal bar chart showing the distribution of language models employed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the models_employed column
    models_column : str, default="models_employed"
        Name of the column containing semicolon-separated model names
    title : str, default="Language Models Employed in Studies"
        Chart title
    xlabel : str, default="Number of Studies"
        X-axis label
    ylabel : str, default="Model Type"
        Y-axis label
    figsize : Tuple[float, float], default=(10, 8)
        Figure size as (width, height)
    top_n : int, default=6
        Number of top models to show
    exclude_other : bool, default=True
        Whether to exclude "Other" category from the results
    save_path : Optional[str], default=None
        Path to save the figure. If None, don't save
    
    Returns
    -------
    None
        Displays the chart and optionally saves it
    
    Examples
    --------
    >>> df = pd.read_csv('data/processed/final_data.csv')
    >>> create_model_types_panel(df, save_path='results/model_types.png')
    """
    
    # Extract and count model types
    model_counts = _extract_and_count_models(
        df, 
        models_column=models_column, 
        exclude_other=exclude_other
    )
    colors = get_color_palette()
    
    # Get top N models
    top_models = dict(model_counts.most_common(top_n))
    
    # Remove " family" from model labels (except for Mistral and Claude)
    def clean_label(label):
        # Keep " family" for Mistral and Claude models
        if "Mistral" in label or "Claude" in label:
            return label
        # Remove " family" for all other models
        return label.replace(" family", "")
    
    top_models = {
        clean_label(label): count 
        for label, count in top_models.items()
    }
    
    # Create the horizontal bar chart
    fig = create_horizontal_bar_chart(
        data=top_models,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        color=colors["navy"],  # Using the default bluish green color
        show_percentages=True,
        save_path=save_path,
        sort_ascending=True,
        max_label_length=50,
        label_fontsize=12,
        title_fontsize=16
    )
    
    return fig


def create_llm_approach_panel(
    df: pd.DataFrame,
    approach_column: str = "llm_development_approach",
    title: str = "LLM Development Approaches in Studies",
    xlabel: str = "Number of Studies",
    ylabel: str = "Development Approach",
    figsize: Tuple[float, float] = (12, 8),
    top_n: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Create a horizontal bar chart showing the distribution of LLM development approaches.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the llm_development_approach column
    approach_column : str, default="llm_development_approach"
        Name of the column containing LLM development approaches
    title : str, default="LLM Development Approaches in Studies"
        Chart title
    xlabel : str, default="Number of Studies"
        X-axis label
    ylabel : str, default="Development Approach"
        Y-axis label
    figsize : Tuple[float, float], default=(12, 8)
        Figure size as (width, height)
    top_n : int, default=8
        Number of top approaches to show
    save_path : Optional[str], default=None
        Path to save the figure. If None, don't save
    
    Returns
    -------
    None
        Displays the chart and optionally saves it
    """
    
    # Count approach frequencies
    approach_counts = df[approach_column].value_counts().head(top_n)
    colors = get_color_palette()
    
    # Create the horizontal bar chart
    fig = create_horizontal_bar_chart(
        data=approach_counts,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        color=colors["navy"],  # Using orange color for distinction
        show_percentages=True,
        save_path=save_path,
        sort_ascending=True,
        max_label_length=60,
        label_fontsize=11,
        title_fontsize=16
    )
    
    return fig


def create_open_vs_closed_weight_time_series(
    df: pd.DataFrame,
    year_column: str = "year",
    on_premise_column: str = "p1_on_premise_model_considered",
    title: str = "Open vs Closed Weight Model Usage Over Time",
    xlabel: str = "Year",
    ylabel: str = "Number of Studies",
    figsize: Tuple[float, float] = (12, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Create a time series chart showing open vs closed weight model usage trends.
    
    Open-weight models are defined as studies where p1_on_premise_model_considered == 'y'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing year and on-premise model consideration data
    year_column : str, default="year"
        Name of the column containing year values
    on_premise_column : str, default="p1_on_premise_model_considered"
        Name of the column indicating on-premise model consideration
    title : str, default="Open vs Closed Weight Model Usage Over Time"
        Chart title
    xlabel : str, default="Year"
        X-axis label
    ylabel : str, default="Number of Studies"
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
    
    # Create a copy of the dataframe for processing
    df_processed = df.copy()
    
    # Clean and categorize the on-premise model consideration
    def categorize_model_weight(value):
        if pd.isna(value):
            return "Unknown"
        value_str = str(value).lower().strip()
        if value_str in ['y', 'y (some)']:
            return "Using some open-weight models"
        elif value_str == 'n':
            return "Using only closed-weight models"
        else:
            return "Unknown"
    
    df_processed['model_weight_category'] = df_processed[on_premise_column].apply(categorize_model_weight)
    
    # Filter out unknown categories for cleaner visualization
    df_processed = df_processed[df_processed['model_weight_category'] != 'Unknown']
    
    # Ensure year is numeric and filter out invalid years
    df_processed[year_column] = pd.to_numeric(df_processed[year_column], errors='coerce')
    df_processed = df_processed.dropna(subset=[year_column])
    df_processed[year_column] = df_processed[year_column].astype(int)
    
    # Define colors for the categories
    color_palette = get_color_palette()
    colors = {
        'Using some open-weight models': color_palette["mint"],
        'Using only closed-weight models': color_palette["amber"]
    }
    
    # Create the time series chart
    fig = create_time_series_chart(
        df=df_processed,
        year_column=year_column,
        category_column='model_weight_category',
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


def _extract_and_count_models(
    df: pd.DataFrame, 
    models_column: str = "models_employed",
    exclude_other: bool = True
) -> Counter:
    """
    Extract individual model names from semicolon-separated values and count them.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    models_column : str, default="models_employed"
        Name of the column containing semicolon-separated model names
    exclude_other : bool, default=True
        Whether to exclude "Other" category from the results
    
    Returns
    -------
    Counter
        Counter object with model names as keys and counts as values
    """
    
    model_counter = Counter()
    
    # Process each row's models_employed value
    for models_str in df[models_column].dropna():
        if isinstance(models_str, str):
            # Split by semicolon and clean up whitespace
            individual_models = [model.strip() for model in models_str.split(';')]
            
            # Filter out empty strings and optionally "Other"
            individual_models = [
                model for model in individual_models 
                if model and (not exclude_other or model.lower() != 'other')
            ]
            
            # Count each model
            model_counter.update(individual_models)
    
    return model_counter


def get_model_statistics(
    df: pd.DataFrame,
    models_column: str = "models_employed",
    exclude_other: bool = True
) -> dict:
    """
    Get detailed statistics about model usage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    models_column : str, default="models_employed"
        Name of the column containing semicolon-separated model names
    exclude_other : bool, default=True
        Whether to exclude "Other" category from the results
    
    Returns
    -------
    dict
        Dictionary containing various statistics about model usage
    """
    
    model_counts = _extract_and_count_models(
        df, 
        models_column=models_column, 
        exclude_other=exclude_other
    )
    
    total_model_mentions = sum(model_counts.values())
    unique_models = len(model_counts)
    studies_with_models = df[models_column].notna().sum()
    
    stats = {
        'total_model_mentions': total_model_mentions,
        'unique_models': unique_models,
        'studies_with_models': studies_with_models,
        'most_common_model': model_counts.most_common(1)[0] if model_counts else None,
        'model_distribution': dict(model_counts.most_common()),
        'average_models_per_study': total_model_mentions / studies_with_models if studies_with_models > 0 else 0
    }
    
    return stats


def generate_model_types_report(
    df: pd.DataFrame,
    models_column: str = "models_employed",
    top_n: int = 6,
    exclude_other: bool = True
) -> str:
    """
    Generate a comprehensive report on language model usage in studies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should be pre-filtered for empirical LLM studies)
    models_column : str, default="models_employed"
        Name of the column containing semicolon-separated model names
    top_n : int, default=6
        Number of top models to include in detailed analysis
    exclude_other : bool, default=True
        Whether to exclude "Other" category from the results
    
    Returns
    -------
    str
        Formatted text report
    """
    from datetime import datetime
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("LANGUAGE MODELS EMPLOYED IN EMPIRICAL LLM STUDIES")
    report_lines.append("Model Usage Analysis for Psychotherapy Research")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Get statistics
    stats = get_model_statistics(df, models_column=models_column, exclude_other=exclude_other)
    model_counts = _extract_and_count_models(df, models_column=models_column, exclude_other=exclude_other)
    
    # Overview section
    report_lines.append("OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total studies analyzed: {len(df)}")
    report_lines.append(f"Studies with model information: {stats['studies_with_models']}")
    report_lines.append(f"Total model mentions: {stats['total_model_mentions']}")
    report_lines.append(f"Unique models identified: {stats['unique_models']}")
    report_lines.append(f"Average models per study: {stats['average_models_per_study']:.2f}")
    report_lines.append("")
    
    if exclude_other:
        report_lines.append("Note: 'Other' category models excluded from analysis")
        report_lines.append("")
    
    # Top models section
    top_models = list(stats['model_distribution'].items())[:top_n]
    report_lines.append(f"TOP {top_n} MOST FREQUENTLY USED MODELS")
    report_lines.append("-" * 80)
    
    for i, (model, count) in enumerate(top_models, 1):
        percentage = (count / stats['total_model_mentions']) * 100
        report_lines.append(f"{i:2d}. {model}")
        report_lines.append(f"    Studies: {count} ({percentage:.1f}% of all model mentions)")
        report_lines.append("")
    
    # Model family analysis
    report_lines.append("MODEL FAMILY ANALYSIS")
    report_lines.append("-" * 80)
    
    # Group by families
    family_counts = {}
    for model, count in stats['model_distribution'].items():
        # Simple family grouping based on model names
        if 'GPT-4' in model or 'GPT-4o' in model:
            family_counts['GPT-4 Family'] = family_counts.get('GPT-4 Family', 0) + count
        elif 'GPT-3.5' in model:
            family_counts['GPT-3.5 Family'] = family_counts.get('GPT-3.5 Family', 0) + count
        elif 'GPT-3' in model:
            family_counts['GPT-3 Family'] = family_counts.get('GPT-3 Family', 0) + count
        elif 'GPT-2' in model:
            family_counts['GPT-2 Family'] = family_counts.get('GPT-2 Family', 0) + count
        elif 'ChatGPT' in model:
            family_counts['ChatGPT (unspecified)'] = family_counts.get('ChatGPT (unspecified)', 0) + count
        elif 'Llama' in model:
            family_counts['Llama Family'] = family_counts.get('Llama Family', 0) + count
        elif 'T5' in model:
            family_counts['T5 Family'] = family_counts.get('T5 Family', 0) + count
        elif 'BERT' in model:
            family_counts['BERT Family'] = family_counts.get('BERT Family', 0) + count
        elif 'Mistral' in model:
            family_counts['Mistral Family'] = family_counts.get('Mistral Family', 0) + count
        else:
            family_counts['Other Models'] = family_counts.get('Other Models', 0) + count
    
    # Sort families by count
    sorted_families = sorted(family_counts.items(), key=lambda x: x[1], reverse=True)
    
    for family, count in sorted_families:
        percentage = (count / stats['total_model_mentions']) * 100
        report_lines.append(f"{family}: {count} mentions ({percentage:.1f}%)")
    report_lines.append("")
    
    # Detailed model inventory
    report_lines.append("COMPLETE MODEL INVENTORY")
    report_lines.append("-" * 80)
    report_lines.append("All models mentioned in the dataset (ordered by frequency):")
    report_lines.append("")
    
    for i, (model, count) in enumerate(stats['model_distribution'].items(), 1):
        percentage = (count / stats['total_model_mentions']) * 100
        report_lines.append(f"{i:2d}. {model} ({count} studies, {percentage:.1f}%)")
    
    report_lines.append("")
    
    # Studies using multiple models
    multi_model_studies = []
    for _, row in df.iterrows():
        models_str = row.get(models_column, '')
        if pd.notna(models_str) and isinstance(models_str, str):
            models_list = [m.strip() for m in models_str.split(';') if m.strip()]
            if exclude_other:
                models_list = [m for m in models_list if m.lower() != 'other']
            if len(models_list) > 1:
                study_id = row.get('study_id', 'Unknown')
                title = row.get('title', 'Unknown title')
                multi_model_studies.append((study_id, title[:60] + "..." if len(title) > 60 else title, models_list))
    
    if multi_model_studies:
        report_lines.append("STUDIES USING MULTIPLE MODELS")
        report_lines.append("-" * 80)
        report_lines.append(f"Number of studies using multiple models: {len(multi_model_studies)}")
        report_lines.append("")
        
        for study_id, title, models in multi_model_studies[:10]:  # Show top 10
            report_lines.append(f"Study ID: {study_id}")
            report_lines.append(f"Title: {title}")
            report_lines.append(f"Models: {'; '.join(models)}")
            report_lines.append("")
        
        if len(multi_model_studies) > 10:
            report_lines.append(f"... and {len(multi_model_studies) - 10} more studies")
            report_lines.append("")
    
    # Temporal analysis if year data is available
    if 'year' in df.columns:
        report_lines.append("TEMPORAL TRENDS")
        report_lines.append("-" * 80)
        
        # Year-wise model usage
        yearly_data = {}
        for _, row in df.iterrows():
            year = row.get('year')
            models_str = row.get(models_column, '')
            
            if pd.notna(year) and pd.notna(models_str) and isinstance(models_str, str):
                year = int(float(year)) if isinstance(year, (int, float)) else year
                if year not in yearly_data:
                    yearly_data[year] = Counter()
                
                models_list = [m.strip() for m in models_str.split(';') if m.strip()]
                if exclude_other:
                    models_list = [m for m in models_list if m.lower() != 'other']
                
                yearly_data[year].update(models_list)
        
        for year in sorted(yearly_data.keys()):
            total_mentions = sum(yearly_data[year].values())
            report_lines.append(f"{year}: {total_mentions} model mentions")
            
            # Show top 3 models for this year
            top_3 = yearly_data[year].most_common(3)
            for model, count in top_3:
                report_lines.append(f"  • {model}: {count}")
            report_lines.append("")
    
    # LLM Development Approach Analysis
    report_lines.append("LLM DEVELOPMENT APPROACH ANALYSIS")
    report_lines.append("-" * 80)
    
    if 'llm_development_approach' in df.columns:
        approach_counts = df['llm_development_approach'].value_counts()
        total_approaches = approach_counts.sum()
        
        report_lines.append(f"Total studies with approach information: {total_approaches}")
        report_lines.append("")
        
        report_lines.append("Top development approaches:")
        for i, (approach, count) in enumerate(approach_counts.head(8).items(), 1):
            percentage = (count / total_approaches) * 100
            # Truncate long approach names for display
            display_approach = approach[:70] + "..." if len(str(approach)) > 70 else approach
            report_lines.append(f"{i:2d}. {display_approach}")
            report_lines.append(f"    Studies: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # Categorize approaches
        prompting_only = approach_counts.get('Only prompting', 0)
        fine_tuning_only = approach_counts.get('Only fine-tuning', 0)
        fine_tuning_plus = sum(count for approach, count in approach_counts.items() 
                              if 'Fine-tuning' in str(approach) and 'other modules' in str(approach))
        prompting_plus = sum(count for approach, count in approach_counts.items() 
                            if 'Prompting' in str(approach) and 'other modules' in str(approach))
        other_approaches = sum(count for approach, count in approach_counts.items() 
                              if 'Other' in str(approach))
        
        report_lines.append("Approach categories:")
        if prompting_only > 0:
            report_lines.append(f"  • Prompting only: {prompting_only} studies ({(prompting_only/total_approaches)*100:.1f}%)")
        if fine_tuning_only > 0:
            report_lines.append(f"  • Fine-tuning only: {fine_tuning_only} studies ({(fine_tuning_only/total_approaches)*100:.1f}%)")
        if fine_tuning_plus > 0:
            report_lines.append(f"  • Fine-tuning + other modules: {fine_tuning_plus} studies ({(fine_tuning_plus/total_approaches)*100:.1f}%)")
        if prompting_plus > 0:
            report_lines.append(f"  • Prompting + other modules: {prompting_plus} studies ({(prompting_plus/total_approaches)*100:.1f}%)")
        if other_approaches > 0:
            report_lines.append(f"  • Other/Custom approaches: {other_approaches} studies ({(other_approaches/total_approaches)*100:.1f}%)")
        
        report_lines.append("")
    else:
        report_lines.append("LLM development approach column not found in dataset.")
        report_lines.append("")
    
    # ChatGPT models analysis
    report_lines.append("CHATGPT MODELS ANALYSIS")
    report_lines.append("-" * 80)
    
    # Define ChatGPT model categories
    chatgpt_models = {
        'GPT-4 / GPT-4o family': 0,
        'GPT-3.5 family': 0,
        'ChatGPT, model unspecified': 0
    }
    
    # Count mentions for each ChatGPT model category
    for model, count in stats['model_distribution'].items():
        if 'GPT-4' in model or 'GPT-4o' in model:
            chatgpt_models['GPT-4 / GPT-4o family'] += count
        elif 'GPT-3.5' in model:
            chatgpt_models['GPT-3.5 family'] += count
        elif 'ChatGPT' in model and 'unspecified' in model:
            chatgpt_models['ChatGPT, model unspecified'] += count
    
    # Count unique studies that used any ChatGPT model
    chatgpt_study_count = 0
    for _, row in df.iterrows():
        models_str = row.get(models_column, '')
        if pd.notna(models_str) and isinstance(models_str, str):
            models_list = [m.strip() for m in models_str.split(';') if m.strip()]
            if exclude_other:
                models_list = [m for m in models_list if m.lower() != 'other']
            
            # Check if any model in this study is a ChatGPT model
            has_chatgpt = any(
                'GPT-4' in model or 'GPT-4o' in model or 'GPT-3.5' in model or 
                ('ChatGPT' in model and 'unspecified' in model)
                for model in models_list
            )
            if has_chatgpt:
                chatgpt_study_count += 1
    
    total_chatgpt_mentions = sum(chatgpt_models.values())
    chatgpt_study_percentage = (chatgpt_study_count / len(df)) * 100 if len(df) > 0 else 0
    chatgpt_mention_percentage = (total_chatgpt_mentions / stats['total_model_mentions']) * 100 if stats['total_model_mentions'] > 0 else 0
    
    report_lines.append(f"Studies employing any ChatGPT model: {chatgpt_study_count} out of {len(df)} studies ({chatgpt_study_percentage:.1f}%)")
    report_lines.append(f"Total ChatGPT model mentions: {total_chatgpt_mentions} out of {stats['total_model_mentions']} mentions ({chatgpt_mention_percentage:.1f}%)")
    report_lines.append("")
    
    report_lines.append("Breakdown by ChatGPT model type:")
    for model_type, count in chatgpt_models.items():
        if count > 0:
            percentage = (count / stats['total_model_mentions']) * 100
            report_lines.append(f"  • {model_type}: {count} mentions ({percentage:.1f}%)")
    report_lines.append("")
    
    # Open vs Closed Weight Models Analysis
    report_lines.append("OPEN VS CLOSED WEIGHT MODELS ANALYSIS")
    report_lines.append("-" * 80)
    
    if 'p1_on_premise_model_considered' in df.columns:
        # Categorize models by weight type
        def categorize_model_weight(value):
            if pd.isna(value):
                return "Unknown"
            value_str = str(value).lower().strip()
            if value_str in ['y', 'y (some)']:
                return "Open-weight"
            elif value_str == 'n':
                return "Closed-weight"
            else:
                return "Unknown"
        
        df['model_weight_category'] = df['p1_on_premise_model_considered'].apply(categorize_model_weight)
        weight_counts = df['model_weight_category'].value_counts()
        
        total_with_weight_info = weight_counts.get('Open-weight', 0) + weight_counts.get('Closed-weight', 0)
        open_weight_count = weight_counts.get('Open-weight', 0)
        closed_weight_count = weight_counts.get('Closed-weight', 0)
        unknown_count = weight_counts.get('Unknown', 0)
        
        if total_with_weight_info > 0:
            open_percentage = (open_weight_count / total_with_weight_info) * 100
            closed_percentage = (closed_weight_count / total_with_weight_info) * 100
            
            report_lines.append(f"Studies with weight classification: {total_with_weight_info}")
            report_lines.append(f"Open-weight models: {open_weight_count} studies ({open_percentage:.1f}%)")
            report_lines.append(f"Closed-weight models: {closed_weight_count} studies ({closed_percentage:.1f}%)")
            if unknown_count > 0:
                report_lines.append(f"Unknown/unspecified: {unknown_count} studies")
            report_lines.append("")
            
            # Temporal analysis if year data is available
            if 'year' in df.columns:
                df_with_years = df.dropna(subset=['year'])
                if not df_with_years.empty:
                    yearly_weight_analysis = df_with_years.groupby(['year', 'model_weight_category']).size().unstack(fill_value=0)
                    
                    if 'Open-weight' in yearly_weight_analysis.columns or 'Closed-weight' in yearly_weight_analysis.columns:
                        report_lines.append("Temporal trends (open vs closed weight models):")
                        for year in sorted(yearly_weight_analysis.index):
                            open_count = yearly_weight_analysis.loc[year].get('Open-weight', 0)
                            closed_count = yearly_weight_analysis.loc[year].get('Closed-weight', 0)
                            total_year = open_count + closed_count
                            
                            if total_year > 0:
                                year_int = int(year) if isinstance(year, float) else year
                                open_pct = (open_count / total_year) * 100
                                report_lines.append(f"  {year_int}: {open_count} open-weight, {closed_count} closed-weight ({open_pct:.0f}% open)")
                        report_lines.append("")
        else:
            report_lines.append("No studies found with weight classification information.")
            report_lines.append("")
    else:
        report_lines.append("On-premise model consideration column not found in dataset.")
        report_lines.append("")
    
    # Summary and insights
    report_lines.append("SUMMARY INSIGHTS")
    report_lines.append("-" * 80)
    
    if stats['most_common_model']:
        model_name, count = stats['most_common_model']
        percentage = (count / stats['total_model_mentions']) * 100
        report_lines.append(f"• {model_name} is the most frequently used model ({count} studies, {percentage:.1f}%)")
    
    report_lines.append(f"• {chatgpt_study_count} studies ({chatgpt_study_percentage:.1f}%) employed any ChatGPT model")
    
    # Check for OpenAI dominance
    openai_models = ['GPT-4', 'GPT-3.5', 'GPT-3', 'GPT-2', 'ChatGPT']
    openai_count = sum(count for model, count in stats['model_distribution'].items() 
                      if any(openai_model in model for openai_model in openai_models))
    openai_percentage = (openai_count / stats['total_model_mentions']) * 100
    report_lines.append(f"• OpenAI models account for {openai_count} mentions ({openai_percentage:.1f}% of total)")
    
    # Check for open-source vs closed models
    open_source_keywords = ['Llama', 'T5', 'BERT', 'Mistral']
    open_source_count = sum(count for model, count in stats['model_distribution'].items() 
                           if any(keyword in model for keyword in open_source_keywords))
    if open_source_count > 0:
        open_source_percentage = (open_source_count / stats['total_model_mentions']) * 100
        report_lines.append(f"• Open-source models account for {open_source_count} mentions ({open_source_percentage:.1f}% of total)")
    
    report_lines.append(f"• {len(multi_model_studies)} studies used multiple different models")
    
    if stats['average_models_per_study'] > 1.2:
        report_lines.append("• Studies tend to use multiple models for comparison or different tasks")
    elif stats['average_models_per_study'] < 1.1:
        report_lines.append("• Studies typically focus on a single model")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)
