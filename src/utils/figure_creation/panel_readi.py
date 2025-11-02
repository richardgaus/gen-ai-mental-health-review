"""
READI Framework panel creation utilities.

This module contains functions to create visualizations showing the adoption
of READI (Responsible, Equitable, Accessible, Deployable, Interpretable) 
framework components in client-facing LLM applications.
"""

from typing import Optional, Tuple, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import yaml
from pathlib import Path

from .utils import get_color_palette


def load_figure_config() -> Dict:
    """Load configuration from figure_config.yaml."""
    config_path = Path(__file__).parent / "figure_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_readi_panel(
    df: pd.DataFrame,
    title: str = "READI Framework Adoption in Client-Facing LLM Applications",
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a READI framework adoption visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing client-facing application studies
    title : str, default="READI Framework Adoption in Client-Facing LLM Applications"
        Chart title
    figsize : Tuple[float, float], default=(12, 10)
        Figure size as (width, height)
    save_path : Optional[str], default=None
        Path to save the figure. If None, don't save
    
    Returns
    -------
    plt.Figure
        The created figure object
    """
    
    # Load configuration
    config = load_figure_config()
    color_palette = get_color_palette()
    human_participants_only = config.get('readi_only_human_participants', [])
    
    # Define the READI categories and components
    categories_data = [
        {
            "group": "Safety",
            "color": color_palette["crimson"],  # Red
            "items": [
                {
                    "code": "S-1",
                    "column": "s1_risk_detection_considered",
                    "title": "Risk-detection",
                    "description": "Explicit detection of acute risk (e.g., self-harm, suicide)"
                },
                {
                    "code": "S-2", 
                    "column": "s2_content_safety_considered",
                    "title": "Content-safety evaluation",
                    "description": "Automatic screening of LLM outputs for counter-therapeutic traits"
                }
            ]
        },
        {
            "group": "Privacy",
            "color": color_palette["amber"],  # Orange
            "items": [
                {
                    "code": "P-1",
                    "column": "p1_on_premise_model_considered",
                    "title": "On-premise-capable model",
                    "description": "Use of LLM with open weights that can be deployed locally"
                },
                {
                    "code": "P-2",
                    "column": "p2_privacy_awareness_considered",
                    "title": "Privacy/confidentiality awareness",
                    "description": "Data protection measures for real-world deployment discussed"
                }
            ]
        },
        {
            "group": "Equity", 
            "color": color_palette["mint"],  # Green
            "items": [
                {
                    "code": "E-1",
                    "column": "e1_demographics_reporting_considered",
                    "title": "Demographic reporting",
                    "description": "Demographics of participants/dataset reported"
                },
                {
                    "code": "E-2",
                    "column": "e2_outcomes_by_demographics_considered",
                    "title": "Subgroup outcomes", 
                    "description": "Results broken down by demographic groups"
                }
            ]
        },
        {
            "group": "Engagement",
            "color": color_palette["sky"],  # Blue
            "items": [
                {
                    "code": "G-1",
                    "column": "g1_early_discontinuation_considered",
                    "title": "Early-discontinuation data*",
                    "description": "Under-use or less-than-intended utilization reported"
                },
                {
                    "code": "G-2",
                    "column": "g2_overuse_considered",
                    "title": "Over-use reported/prevented*",
                    "description": "Excessive use monitored/reported or usage cap present"
                }
            ]
        },
        {
            "group": "Effectiveness",
            "color": color_palette["navy"],  # Dark blue
            "items": [
                {
                    "code": "F-1", 
                    "column": "f1_validated_outcomes_considered",
                    "title": "Validated clinical measures*",
                    "description": "Recognised symptom/function scales with stated reliability/validity"
                },
                {
                    "code": "F-2",
                    "column": "f2_control_condition_considered",
                    "title": "Control condition*",
                    "description": "Control condition present (incl. inactive/waitlist)"
                }
            ]
        },
        {
            "group": "Implementation",
            "color": color_palette["purple"],  # Purple
            "items": [
                {
                    "code": "I-1",
                    "column": "i1_multilevel_feasibility_considered",
                    "title": "Diverse stakeholder feasibility*",
                    "description": "Feedback from ≥2 stakeholder levels (e.g., patients & clinicians)"
                },
                {
                    "code": "I-2", 
                    "column": "i2_healthcare_integration_considered",
                    "title": "Healthcare integration",
                    "description": "Workflow integration, regulatory compliance, or cost discussed"
                }
            ]
        }
    ]
    
    # Add asterisks to items that are only evaluated for human participants
    for category in categories_data:
        for item in category["items"]:
            if item["column"] in human_participants_only:
                if not item["title"].endswith("*"):
                    item["title"] += "*"
    
    # Calculate counts and proportions
    total_studies = len(df)
    
    # For human participants only columns, filter the dataframe
    def get_count_and_total(column_name: str) -> Tuple[int, int]:
        if column_name in human_participants_only:
            # Filter for studies with human participants
            human_df = df[df['client_type'] != 'No clients/patients involved']
            if len(human_df) == 0:
                return 0, 0
            yes_count = human_df[column_name].astype(str).str.lower().isin(['y', 'yes']).sum()
            return yes_count, len(human_df)
        else:
            # Use all studies
            yes_count = df[column_name].astype(str).str.lower().isin(['y', 'yes']).sum()
            return yes_count, total_studies
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Remove default axes and styling
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 13)  # Adjust for 12 items
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Variables for positioning
    y_position = 12
    row_height = 0.9
    category_spacing = 0.3
    prev_group = None
    
    # Draw data rows
    for category in categories_data:
        category_start_y = y_position
        
        for i, item in enumerate(category["items"]):
            count, item_total = get_count_and_total(item["column"])
            proportion = count / item_total if item_total > 0 else 0
            
            # Add category spacing between groups
            if prev_group != category["group"]:
                if prev_group is not None:
                    y_position -= category_spacing
                    category_start_y = y_position
                prev_group = category["group"]
            
            # Category group label and box (only for first item in group)
            if i == 0:
                category_height = len(category["items"]) * row_height
                
                # Large category background covering all items
                cat_rect = Rectangle(
                    (2, category_start_y - category_height + 0.55), 15, category_height - 0.1,
                    facecolor=category["color"], alpha=0.2,
                    edgecolor='none'
                )
                ax.add_patch(cat_rect)
                
                # Center the text vertically within the category box
                category_center_y = category_start_y - (category_height / 2) + 0.5
                ax.text(9.5, category_center_y, category["group"], 
                       fontsize=12, fontweight='bold', color=category["color"],
                       va='center', ha='center')
            
            # Component code and title
            ax.text(20, y_position + 0.15, f"{item['code']}: {item['title']}", 
                   fontsize=11, fontweight='bold', color='black',
                   va='center', ha='left')
            
            # Description
            ax.text(20, y_position - 0.15, item['description'],
                   fontsize=9, color='#666666',
                   va='center', ha='left')
            
            # Progress bar background
            bar_bg = Rectangle((58, y_position - 0.2), 27, 0.4,
                              facecolor='#E8E8E8', edgecolor='none')
            ax.add_patch(bar_bg)
            
            # Progress bar fill
            bar_fill = Rectangle((58, y_position - 0.2), 27 * proportion, 0.4,
                               facecolor=category["color"], alpha=0.8, edgecolor='none')
            ax.add_patch(bar_fill)
            
            # Percentage text at the end of the bar
            bar_end_x = 58 + 27 * proportion
            ax.text(bar_end_x + 0.5, y_position, f'{proportion:.0%}',
                   fontsize=11, fontweight='normal', color='black',
                   va='center', ha='left')
            
            # Count text with asterisk if human participants only
            if item["column"] in human_participants_only:
                count_text = f'({count}/{item_total})*'
            else:
                count_text = f'({count}/{item_total})'
            
            ax.text(88, y_position, count_text,
                   fontsize=11, color='black',
                   va='center', ha='left')
            
            y_position -= row_height
    
    # Add column headers
    header_y = 12.8
    ax.text(9.5, header_y, 'COMPONENT', fontsize=12, fontweight='bold', 
            color='black', ha='center')
    ax.text(20, header_y, 'CODING CRITERION', fontsize=12, fontweight='bold',
            color='black', ha='left')
    ax.text(72.5, header_y, 'ADOPTION RATE', fontsize=12, fontweight='bold',
            color='black', ha='center')
    
    # Add scale reference
    for pct in [0, 25, 50, 75, 100]:
        x_pos = 58 + 27 * (pct/100)
        ax.text(x_pos, header_y - 0.5, f'{pct}%',
               fontsize=9, ha='center', color='black')
        if pct in [0, 100]:
            ax.plot([x_pos, x_pos], [header_y - 0.6, 0.5], 
                   color='black', linewidth=0.5, linestyle='--', alpha=0.7)
    
    # Title only (removed subtitle)
    ax.text(50, 13.5, title, 
            ha='center', fontsize=16, fontweight='bold', color='black')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_readi_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive report on READI framework adoption.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing client-facing application studies
    
    Returns
    -------
    str
        Formatted text report
    """
    from datetime import datetime
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("READI FRAMEWORK ADOPTION REPORT")
    report_lines.append("Client-Facing Empirical LLM Studies in Psychotherapy Research")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Load configuration
    config = load_figure_config()
    human_participants_only = config.get('readi_only_human_participants', [])
    
    # Overview
    total_studies = len(df)
    human_studies = len(df[df['client_type'] != 'No clients/patients involved'])
    
    report_lines.append("OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"Total client-facing empirical LLM studies: {total_studies}")
    report_lines.append(f"Studies with human participants: {human_studies}")
    report_lines.append("")
    report_lines.append("Filtering criteria:")
    report_lines.append("  ✓ Client-facing application")
    report_lines.append("  ✓ Empirical research involving an LLM")
    report_lines.append("")
    report_lines.append("Note: Components marked with * are only evaluated for studies")
    report_lines.append("with human participants (client_type != 'No clients/patients involved')")
    report_lines.append("")
    
    # Define READI components
    readi_components = [
        ("S-1", "s1_risk_detection_considered", "Risk-detection", "Safety"),
        ("S-2", "s2_content_safety_considered", "Content-safety evaluation", "Safety"),
        ("P-1", "p1_on_premise_model_considered", "On-premise-capable model", "Privacy"),
        ("P-2", "p2_privacy_awareness_considered", "Privacy/confidentiality awareness", "Privacy"),
        ("E-1", "e1_demographics_reporting_considered", "Demographic reporting", "Equity"),
        ("E-2", "e2_outcomes_by_demographics_considered", "Subgroup outcomes", "Equity"),
        ("G-1", "g1_early_discontinuation_considered", "Early-discontinuation data", "Engagement"),
        ("G-2", "g2_overuse_considered", "Over-use reported/prevented", "Engagement"),
        ("F-1", "f1_validated_outcomes_considered", "Validated clinical measures", "Effectiveness"),
        ("F-2", "f2_control_condition_considered", "Control condition", "Effectiveness"),
        ("I-1", "i1_multilevel_feasibility_considered", "Multilevel feasibility", "Implementation"),
        ("I-2", "i2_healthcare_integration_considered", "Healthcare integration", "Implementation")
    ]
    
    # Calculate adoption rates
    adoption_data = []
    for code, column, title, category in readi_components:
        if column in human_participants_only:
            # Filter for studies with human participants
            eval_df = df[df['client_type'] != 'No clients/patients involved']
            eval_total = len(eval_df)
            asterisk = "*"
        else:
            eval_df = df
            eval_total = total_studies
            asterisk = ""
        
        if eval_total > 0:
            yes_count = eval_df[column].astype(str).str.lower().isin(['y', 'yes']).sum()
            adoption_rate = yes_count / eval_total
        else:
            yes_count = 0
            adoption_rate = 0
        
        adoption_data.append({
            'code': code,
            'title': title + asterisk,
            'category': category,
            'count': yes_count,
            'total': eval_total,
            'rate': adoption_rate
        })
    
    # Overall statistics
    all_rates = [item['rate'] for item in adoption_data]
    overall_avg = np.mean(all_rates)
    
    # Calculate per-study READI item counts
    study_readi_counts = []
    for _, row in df.iterrows():
        count = 0
        for code, column, title, category in readi_components:
            if column in human_participants_only:
                # Only count if study has human participants
                if row.get('client_type', '') != 'No clients/patients involved':
                    if str(row.get(column, '')).lower() in ['y', 'yes']:
                        count += 1
            else:
                # Count for all studies
                if str(row.get(column, '')).lower() in ['y', 'yes']:
                    count += 1
        study_readi_counts.append(count)
    
    # Calculate statistics
    mean_items = np.mean(study_readi_counts)
    median_items = np.median(study_readi_counts)
    min_items = min(study_readi_counts)
    max_items = max(study_readi_counts)
    zero_items_count = sum(1 for count in study_readi_counts if count == 0)
    
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Average adoption rate across all components: {overall_avg:.1%}")
    report_lines.append(f"Highest adoption rate: {max(all_rates):.1%}")
    report_lines.append(f"Lowest adoption rate: {min(all_rates):.1%}")
    report_lines.append("")
    
    report_lines.append("PER-STUDY READI ITEM STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Mean READI items met per study: {mean_items:.2f}")
    report_lines.append(f"Median READI items met per study: {median_items:.1f}")
    report_lines.append(f"Range of READI items met: {min_items} - {max_items}")
    report_lines.append(f"Studies meeting zero READI items: {zero_items_count}/{total_studies} ({zero_items_count/total_studies:.1%})")
    report_lines.append("")
    
    # Top 10 studies with most READI criteria met
    study_data = []
    for idx, (_, row) in enumerate(df.iterrows()):
        count = study_readi_counts[idx]
        study_id = row.get('study_id', 'Unknown')
        title = row.get('title', 'Unknown title')
        study_data.append((count, study_id, title))
    
    # Sort by READI count (descending) and take top 10
    top_studies = sorted(study_data, key=lambda x: x[0], reverse=True)[:10]
    
    report_lines.append("TOP 10 STUDIES BY READI CRITERIA MET")
    report_lines.append("-" * 80)
    
    for i, (count, study_id, title) in enumerate(top_studies, 1):
        # Truncate title if too long
        display_title = title[:80] + "..." if len(title) > 80 else title
        report_lines.append(f"{i:2d}. {count} criteria met - {study_id}")
        report_lines.append(f"    {display_title}")
        report_lines.append("")
    
    report_lines.append("")
    
    # Component-wise breakdown
    report_lines.append("COMPONENT-WISE ADOPTION RATES")
    report_lines.append("-" * 80)
    
    # Sort by adoption rate (descending)
    sorted_components = sorted(adoption_data, key=lambda x: x['rate'], reverse=True)
    
    for item in sorted_components:
        report_lines.append(f"{item['code']} - {item['title']}: {item['count']}/{item['total']} ({item['rate']:.1%})")
    
    report_lines.append("")
    
    # Category-wise analysis
    report_lines.append("CATEGORY-WISE ANALYSIS")
    report_lines.append("-" * 80)
    
    categories = {}
    for item in adoption_data:
        if item['category'] not in categories:
            categories[item['category']] = []
        categories[item['category']].append(item)
    
    for category, items in categories.items():
        avg_rate = np.mean([item['rate'] for item in items])
        report_lines.append(f"\n{category.upper()} (Average: {avg_rate:.1%})")
        for item in items:
            report_lines.append(f"  {item['code']} - {item['title']}: {item['count']}/{item['total']} ({item['rate']:.1%})")
    
    report_lines.append("")
    
    # High and low adoption analysis
    high_adoption = [item for item in adoption_data if item['rate'] > 0.5]
    low_adoption = [item for item in adoption_data if item['rate'] < 0.25]
    
    report_lines.append("ADOPTION PATTERNS")
    report_lines.append("-" * 80)
    
    if high_adoption:
        report_lines.append("\nComponents with >50% adoption:")
        for item in sorted(high_adoption, key=lambda x: x['rate'], reverse=True):
            report_lines.append(f"  {item['code']} - {item['title']}: {item['rate']:.1%}")
    else:
        report_lines.append("\nNo components have >50% adoption")
    
    if low_adoption:
        report_lines.append("\nComponents with <25% adoption:")
        for item in sorted(low_adoption, key=lambda x: x['rate']):
            report_lines.append(f"  {item['code']} - {item['title']}: {item['rate']:.1%}")
    else:
        report_lines.append("\nNo components have <25% adoption")
    
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    if overall_avg < 0.3:
        report_lines.append("• Overall READI adoption is low. Consider developing guidelines")
        report_lines.append("  and tools to support researchers in implementing these components.")
    elif overall_avg < 0.6:
        report_lines.append("• READI adoption is moderate. Focus on improving adoption of")
        report_lines.append("  lower-performing components while maintaining current strengths.")
    else:
        report_lines.append("• READI adoption is relatively high. Continue current practices")
        report_lines.append("  and focus on achieving consistency across all components.")
    
    if low_adoption:
        report_lines.append("\n• Priority areas for improvement:")
        for item in sorted(low_adoption, key=lambda x: x['rate'])[:3]:
            report_lines.append(f"  - {item['title']} ({item['rate']:.1%} adoption)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)
