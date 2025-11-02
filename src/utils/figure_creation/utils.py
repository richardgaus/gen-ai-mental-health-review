from typing import Union, Dict, List, Tuple, Optional
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import colorsys


def _is_color_dark(hex_color: str, threshold: float = 0.5) -> bool:
    """
    Determine if a hex color is dark based on its luminance.
    
    Parameters
    ----------
    hex_color : str
        Hex color code (e.g., '#FF0000' or 'FF0000')
    threshold : float, default=0.5
        Luminance threshold below which color is considered dark
        
    Returns
    -------
    bool
        True if color is dark, False if light
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    # Calculate luminance using standard formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    return luminance < threshold


def create_horizontal_bar_chart(
    data: Union[pd.Series, pd.DataFrame, Dict, Counter, List[Tuple]], 
    title: str = "Horizontal Bar Chart",
    xlabel: str = "Count",
    ylabel: str = "Categories", 
    figsize: Tuple[float, float] = (8, 6),
    color: Union[str, Dict[str, str]] = "#009E73",
    top_n: Optional[int] = None,
    show_percentages: bool = True,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    sort_ascending: bool = True,
    category_order: Optional[List[str]] = None,
    max_label_length: int = 40,
    label_fontsize: int = 10,
    title_fontsize: int = 14,
    title_offset: float = 0.5,
    return_stats: bool = False,
    bar_height: float = 0.6,
    percentage_total: Optional[int] = None
) -> Union[plt.Figure, Tuple[plt.Figure, Dict]]:
    """
    Create a clean, professional horizontal bar chart with consistent styling.
    
    Parameters
    ----------
    data : Union[pd.Series, pd.DataFrame, Dict, Counter, List[Tuple]]
        Input data in various formats:
        - pd.Series: index as categories, values as counts
        - pd.DataFrame: must have exactly 2 columns (category, count)
        - Dict: keys as categories, values as counts
        - Counter: categories and their counts
        - List[Tuple]: list of (category, count) tuples
    title : str, default="Horizontal Bar Chart"
        Chart title
    xlabel : str, default="Count"
        X-axis label
    ylabel : str, default="Categories"
        Y-axis label
    figsize : Tuple[float, float], default=(8, 6)
        Figure size as (width, height)
    color : Union[str, Dict[str, str]], default="#009E73"
        Bar color(s). Can be:
        - Single color string (e.g., "#009E73") for all bars
        - Dictionary mapping category names to colors
        - Dictionary with semantic keys for automatic color matching
    top_n : Optional[int], default=None
        Show only top N categories by count. If None, show all
    show_percentages : bool, default=True
        Whether to show percentages in bar labels
    save_path : Optional[str], default=None
        Path to save the figure. If None, don't save
    ax : Optional[plt.Axes], default=None
        Matplotlib axes to plot on. If None, create new figure
    sort_ascending : bool, default=True
        Sort bars in ascending order (largest at top). Ignored if category_order is provided
    category_order : Optional[List[str]], default=None
        Custom order for categories (bottom to top). Categories not in the list 
        will be appended in original order
    max_label_length : int, default=40
        Maximum length for category labels before truncation
    label_fontsize : int, default=10
        Font size for bar labels and axis labels
    title_fontsize : int, default=14
        Font size for title
    title_offset : float, default=0.5
        Horizontal position of title (0=left, 0.5=center, 1=right)
    return_stats : bool, default=False
        Whether to return summary statistics along with the figure
    bar_height : float, default=0.6
        Height of each bar
    percentage_total : Optional[int], default=None
        Total to use for percentage calculation. If None, uses sum of data counts
    
    Returns
    -------
    Union[plt.Figure, Tuple[plt.Figure, Dict]]
        Figure object, or tuple of (Figure, stats_dict) if return_stats=True
    
    Examples
    --------
    >>> # Single color (original behavior)
    >>> fig = create_horizontal_bar_chart(data, color="#009E73")
    
    >>> # Direct category mapping
    >>> colors = {
    ...     'Category A': '#0072B2',
    ...     'Category B': '#E69F00', 
    ...     'Category C': '#009E73'
    ... }
    >>> fig = create_horizontal_bar_chart(data, color=colors)
    
    >>> # Semantic mapping for Yes/No data
    >>> binary_colors = {'yes': '#0072B2', 'no': '#E69F00'}
    >>> fig = create_horizontal_bar_chart(data, color=binary_colors)
    
    >>> # Custom category ordering
    >>> order = ['No specific criteria', 'Some symptoms', 'Diagnosed disorder']
    >>> fig = create_horizontal_bar_chart(data, category_order=order)
    """
    # Okabe-Ito color palette (colorblind-friendly)
    OKABE_ITO = {
        "orange": "#E69F00",
        "sky_blue": "#56B4E9", 
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
        "black": "#000000"
    }
    
    # Semantic keyword mappings for color matching
    SEMANTIC_KEYWORDS = {
        'yes': ['yes', 'present', 'included', 'control group present'],
        'no': ['no', 'absent', 'not', 'none', 'no control', 'no ux'],
        'low': ['no specific', 'none', 'no criteria'],
        'medium': ['some', 'symptom', 'moderate'],
        'high': ['diagnosed', 'disorder', 'icd', 'dsm']
    }
    
    # Color fallback hierarchy for semantic matching
    SEMANTIC_FALLBACKS = {
        'yes': ['yes', 'positive', 'present'],
        'no': ['no', 'negative', 'absent'],
        'low': ['low', 'least_selective'],
        'medium': ['medium', 'mid_selective'],
        'high': ['high', 'most_selective']
    }
    
    def _parse_data() -> pd.DataFrame:
        """Parse input data into a standardized DataFrame."""
        if isinstance(data, pd.Series):
            return pd.DataFrame({'category': data.index, 'count': data.values})
        
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 2:
                raise ValueError("DataFrame must have exactly 2 columns: (category, count)")
            df = data.copy()
            df.columns = ['category', 'count']
            return df.reset_index(drop=True)
        
        if isinstance(data, (dict, Counter)):
            return pd.DataFrame(list(data.items()), columns=['category', 'count'])
        
        if isinstance(data, list) and data and isinstance(data[0], tuple):
            return pd.DataFrame(data, columns=['category', 'count'])
        
        raise ValueError(
            "Unsupported data format. Use pd.Series, pd.DataFrame, dict, "
            "Counter, or list of tuples."
        )
    
    def _apply_ordering(df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom ordering or sorting to the DataFrame."""
        if category_order is not None:
            # Create order mapping with missing categories at the end
            order_map = {cat: idx for idx, cat in enumerate(category_order)}
            missing_cats = [cat for cat in df['category'] if cat not in order_map]
            order_map.update({cat: len(category_order) + idx 
                            for idx, cat in enumerate(missing_cats)})
            
            df['_order'] = df['category'].map(order_map)
            return df.sort_values('_order').drop('_order', axis=1).reset_index(drop=True)
        
        return df.sort_values('count', ascending=sort_ascending).reset_index(drop=True)
    
    def _get_bar_color(category_name: str) -> str:
        """Determine bar color based on category name and color configuration."""
        # Single color for all bars
        if isinstance(color, str):
            return color
        
        if not isinstance(color, dict):
            return OKABE_ITO['bluish_green']
        
        # Direct category match
        if category_name in color:
            return color[category_name]
        
        # Semantic matching
        category_lower = category_name.lower()
        for semantic_key, keywords in SEMANTIC_KEYWORDS.items():
            if any(keyword in category_lower for keyword in keywords):
                # Try fallback keys in order
                for fallback_key in SEMANTIC_FALLBACKS.get(semantic_key, [semantic_key]):
                    if fallback_key in color:
                        return color[fallback_key]
                
                # Use default color for this semantic type
                default_colors = {
                    'yes': OKABE_ITO['blue'],
                    'no': OKABE_ITO['orange'],
                    'low': OKABE_ITO['sky_blue'],
                    'medium': OKABE_ITO['bluish_green'],
                    'high': OKABE_ITO['vermillion']
                }
                return default_colors.get(semantic_key, OKABE_ITO['bluish_green'])
        
        # Fallback to first color in dict or default
        return next(iter(color.values())) if color else OKABE_ITO['bluish_green']
    
    def _truncate_label(label: str, max_length: int) -> str:
        """Truncate label if it exceeds maximum length."""
        label_str = str(label)
        return label_str if len(label_str) <= max_length else f"{label_str[:max_length-3]}..."
    
    # Parse and prepare data
    df = _parse_data()
    original_total = df['count'].sum()
    
    # Filter to top N if specified
    if top_n is not None:
        df = df.nlargest(top_n, 'count')
    
    # Apply ordering/sorting
    df = _apply_ordering(df)
    
    # Calculate percentages based on original total or provided total
    total_for_percentage = percentage_total if percentage_total is not None else original_total
    df['percentage'] = (df['count'] / total_for_percentage) * 100
    
    # Determine colors for each bar
    bar_colors = [_get_bar_color(cat) for cat in df['category']]
    
    # Create or use existing axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        owns_figure = True
    else:
        fig = ax.get_figure()
        owns_figure = False
    
    # Create horizontal bars
    y_positions = range(len(df))
    ax.barh(y_positions, df['count'], color=bar_colors, alpha=0.8, height=bar_height)
    
    # Add value labels
    max_count = df['count'].max() if not df.empty else 1
    for i, (count, pct) in enumerate(zip(df['count'], df['percentage'])):
        label = f"{count} ({pct:.0f}%)" if show_percentages else f"{count}"
        ax.text(count + max_count * 0.02, i, label, 
                ha='left', va='center', fontsize=label_fontsize)
    
    # Configure axes
    ax.set_xlabel(xlabel, fontsize=label_fontsize + 2)
    ax.set_ylabel(ylabel, fontsize=label_fontsize + 2)
    ax.set_title(title, fontsize=title_fontsize, loc='center', pad=20, x=title_offset)
    
    # Set y-axis labels with truncation
    truncated_labels = [_truncate_label(cat, max_label_length) for cat in df['category']]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(truncated_labels, fontsize=label_fontsize)
    
    # Set x-axis limits with padding
    ax.set_xlim(0, max_count * 1.3)
    
    # Apply styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', visible=True, linewidth=0.3, alpha=0.7)
    ax.grid(axis='y', visible=False)
    
    # Adjust layout if we created the figure
    if owns_figure:
        plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Return with statistics if requested
    if return_stats:
        stats = {
            'total_items': len(_parse_data()),
            'displayed_items': len(df),
            'total_count': original_total,
            'top_category': df.iloc[-1]['category'] if not df.empty else None,
            'top_count': df.iloc[-1]['count'] if not df.empty else 0,
            'top_percentage': df.iloc[-1]['percentage'] if not df.empty else 0,
            'data_summary': df.to_dict('records'),
            'color_mapping': dict(zip(df['category'], bar_colors))
        }
        return fig, stats
    
    return fig


def create_time_series_chart(
    df: pd.DataFrame,
    year_column: str,
    category_column: str,
    title: str = "Publications by Year",
    xlabel: str = "Year",
    ylabel: str = "Number of Publications",
    figsize: Tuple[float, float] = (8, 6),
    colors: Optional[Dict[str, str]] = None,
    bar_width: float = 0.8,
    show_counts: bool = True,
    show_totals: bool = True,
    trend_line_start_year: Optional[int] = None,
    save_path: Optional[str] = None,
    legend_fontsize: int = 16,
    label_fontsize: int = 16,
    title_fontsize: int = 20
) -> plt.Figure:
    """
    Create a stacked bar chart showing time series data with category breakdown.
    
    This function creates a professional time series visualization with:
    - Stacked bars showing category distribution over time
    - Optional trend line for specified years
    - Count labels on bars and totals above bars
    - Empty bars highlighted for years with no data
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing year and category columns
    year_column : str
        Name of column containing year values
    category_column : str
        Name of column containing category values
    title : str, default="Publications by Year"
        Chart title
    xlabel : str, default="Year"
        X-axis label
    ylabel : str, default="Number of Publications"
        Y-axis label
    figsize : Tuple[float, float], default=(8, 6)
        Figure size as (width, height)
    colors : Optional[Dict[str, str]], default=None
        Dictionary mapping category names to colors. If None, uses Okabe-Ito palette
    bar_width : float, default=0.8
        Width of bars (0-1)
    show_counts : bool, default=True
        Whether to show counts within each bar segment
    show_totals : bool, default=True
        Whether to show total counts above bars
    trend_line_start_year : Optional[int], default=None
        If specified, draws a trend line starting from this year
    save_path : Optional[str], default=None
        Path to save the figure. If None, don't save
    legend_fontsize : int, default=16
        Font size for legend
    label_fontsize : int, default=16
        Font size for axis labels
    title_fontsize : int, default=20
        Font size for title
        
    Returns
    -------
    plt.Figure
        The created figure object
        
    Examples
    --------
    >>> # Basic usage
    >>> fig = create_time_series_chart(
    ...     df, 
    ...     year_column='Year',
    ...     category_column='includes_human_participants'
    ... )
    
    >>> # With custom colors and trend line
    >>> colors = {
    ...     'With Participants': '#D55E00',
    ...     'Without Participants': '#56B4E9'
    ... }
    >>> fig = create_time_series_chart(
    ...     df,
    ...     year_column='Year',
    ...     category_column='participant_status',
    ...     colors=colors,
    ...     trend_line_start_year=2019
    ... )
    """
    # Okabe-Ito color palette (colorblind-friendly)
    OKABE_ITO = {
        "vermillion": "#D55E00",
        "sky_blue": "#56B4E9",
        "orange": "#E69F00",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "reddish_purple": "#CC79A7"
    }
    
    # Group by year and category, then count
    year_counts = df.groupby([year_column, category_column]).size().unstack(fill_value=0)
    
    # Ensure all years from min to max are included
    if not year_counts.empty:
        all_years = range(int(year_counts.index.min()), int(year_counts.index.max()) + 1)
        year_counts = year_counts.reindex(all_years, fill_value=0)
    
    # Set default colors if not provided
    if colors is None:
        category_names = year_counts.columns.tolist()
        default_colors = [OKABE_ITO["vermillion"], OKABE_ITO["sky_blue"], 
                         OKABE_ITO["orange"], OKABE_ITO["bluish_green"]]
        colors = {cat: default_colors[i % len(default_colors)] 
                 for i, cat in enumerate(category_names)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked bars
    bar_colors = [colors.get(col, OKABE_ITO["sky_blue"]) for col in year_counts.columns]
    year_counts.plot(
        kind='bar',
        stacked=True,
        color=bar_colors,
        ax=ax,
        width=bar_width,
        edgecolor='none',
        legend=False
    )
    
    # Add empty bars for years with 0 publications
    for i, (year, row) in enumerate(year_counts.iterrows()):
        total = row.sum()
        if total == 0:
            ax.bar(i, 0.1, width=bar_width, color='#E0E0E0', 
                  edgecolor='#E0E0E0', linewidth=0)
            ax.text(i, 0.6, '0', ha='center', color='black')
    
    # Add total counts above bars
    if show_totals:
        for i, (year, row) in enumerate(year_counts.iterrows()):
            total = row.sum()
            if total > 0:
                ax.text(i, total + 2, f'{int(total)}', ha='center')
    
    # Add counts within each segment
    if show_counts:
        for i, (year, row) in enumerate(year_counts.iterrows()):
            bottom = 0
            for j, (category, count) in enumerate(row.items()):
                if count >= 3:  # Only show annotations for counts >= 2
                    # Get the color for this category
                    category_color = colors.get(category, OKABE_ITO["sky_blue"])
                    
                    # Determine text color based on background darkness
                    text_color = 'white' if _is_color_dark(category_color) else 'black'
                    
                    ax.text(i, bottom + count/2, f"{int(count)}", 
                           ha='center', va='center', color=text_color)
                
                if count > 0:  # Still need to update bottom for all segments
                    bottom += count
    
    # Add trend line if specified
    if trend_line_start_year is not None:
        trend_data = [(i, row.sum()) for i, (year, row) in enumerate(year_counts.iterrows())
                     if year >= trend_line_start_year and row.sum() > 0]
        if trend_data:
            x_vals, y_vals = zip(*trend_data)
            ax.plot(x_vals, y_vals, color='#999999', linewidth=1, 
                   linestyle='-', zorder=-1)
    
    # Style the plot
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True, linewidth=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    
    # Configure x-axis
    plt.xticks(rotation=0)
    
    # Add legend
    ax.legend(title=None, fontsize=legend_fontsize)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def get_color_palette() -> Dict[str, str]:
    """
    Load and return the color palette from figure_config.yaml.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping color names to hex color codes
    """
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'figure_config.yaml')
    
    # Load the YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config['color_palette']