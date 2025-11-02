"""
Report generation utilities for data cleaning scripts.

This module provides functions to generate standardized summary reports
for data processing steps in the LLMs in Psychotherapy research pipeline.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def generate_processing_report(
    script_name: str,
    input_file: str,
    output_file: str,
    processing_steps: Dict[str, Any],
    summary_stats: Dict[str, Any],
    execution_time: Optional[float] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a standardized processing report for data cleaning scripts.
    
    Args:
        script_name (str): Name of the script that performed the processing
        input_file (str): Path to input file
        output_file (str): Path to output file
        processing_steps (Dict[str, Any]): Dictionary describing processing steps performed
        summary_stats (Dict[str, Any]): Summary statistics about the data transformation
        execution_time (Optional[float]): Execution time in seconds
        additional_info (Optional[Dict[str, Any]]): Additional script-specific information
        
    Returns:
        Dict[str, Any]: Complete processing report
    """
    report = {
        "metadata": {
            "script_name": script_name,
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time
        },
        "files": {
            "input_file": str(input_file),
            "output_file": str(output_file)
        },
        "processing_steps": processing_steps,
        "summary_statistics": summary_stats
    }
    
    if additional_info:
        report["additional_info"] = additional_info
    
    return report


def save_report_to_file(report: Dict[str, Any], output_dir: str, filename: str, save_json: bool = True) -> str:
    """
    Save a processing report to a file in the specified directory.
    
    Args:
        report (Dict[str, Any]): Processing report dictionary
        output_dir (str): Directory to save the report
        filename (str): Filename for the report (without extension)
        save_json (bool): Whether to save JSON file (default True)
        
    Returns:
        str: Path to the saved report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Always save as human-readable text
    txt_path = output_path / f"{filename}.txt"
    with open(txt_path, 'w') as f:
        f.write(format_report_as_text(report))
    
    # Optionally save as JSON
    if save_json:
        json_path = output_path / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        return str(json_path)
    
    return str(txt_path)


def format_report_as_text(report: Dict[str, Any]) -> str:
    """
    Format a processing report as human-readable text.
    
    Args:
        report (Dict[str, Any]): Processing report dictionary
        
    Returns:
        str: Formatted text report
    """
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append(f"DATA PROCESSING REPORT: {report['metadata']['script_name']}")
    lines.append("=" * 80)
    lines.append("")
    
    # Metadata
    lines.append("EXECUTION METADATA:")
    lines.append(f"  Timestamp: {report['metadata']['timestamp']}")
    if report['metadata'].get('execution_time_seconds'):
        lines.append(f"  Execution Time: {report['metadata']['execution_time_seconds']:.2f} seconds")
    lines.append("")
    
    # Files
    lines.append("FILES:")
    lines.append(f"  Input:  {report['files']['input_file']}")
    lines.append(f"  Output: {report['files']['output_file']}")
    lines.append("")
    
    # Processing Steps
    lines.append("PROCESSING STEPS:")
    for step_name, step_info in report['processing_steps'].items():
        lines.append(f"  {step_name}:")
        if isinstance(step_info, dict):
            for key, value in step_info.items():
                lines.append(f"    - {key}: {value}")
        else:
            lines.append(f"    - {step_info}")
    lines.append("")
    
    # Summary Statistics
    lines.append("SUMMARY STATISTICS:")
    for stat_name, stat_value in report['summary_statistics'].items():
        if isinstance(stat_value, dict):
            lines.append(f"  {stat_name}:")
            for key, value in stat_value.items():
                lines.append(f"    - {key}: {value:,}" if isinstance(value, (int, float)) else f"    - {key}: {value}")
        else:
            formatted_value = f"{stat_value:,}" if isinstance(stat_value, (int, float)) else str(stat_value)
            lines.append(f"  {stat_name}: {formatted_value}")
    lines.append("")
    
    # Additional Info
    if 'additional_info' in report:
        lines.append("ADDITIONAL INFORMATION:")
        for info_name, info_value in report['additional_info'].items():
            if isinstance(info_value, dict):
                lines.append(f"  {info_name}:")
                for key, value in info_value.items():
                    lines.append(f"    - {key}: {value}")
            elif isinstance(info_value, list):
                lines.append(f"  {info_name}:")
                for item in info_value:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  {info_name}: {info_value}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("End of Report")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def get_dataframe_summary(df: pd.DataFrame, name: str = "dataset") -> Dict[str, Any]:
    """
    Generate summary statistics for a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        name (str): Name to use for the dataset in the summary
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    summary = {
        f"{name}_rows": len(df),
        f"{name}_columns": len(df.columns),
        f"{name}_memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }
    
    # Add column info
    if len(df.columns) > 0:
        summary[f"{name}_column_names"] = list(df.columns)
        
        # Data types summary
        dtype_counts = df.dtypes.value_counts().to_dict()
        summary[f"{name}_data_types"] = {str(k): int(v) for k, v in dtype_counts.items()}
        
        # Missing values summary
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            missing_summary = {col: int(count) for col, count in missing_counts.items() if count > 0}
            summary[f"{name}_missing_values"] = missing_summary
    
    return summary


def compare_dataframes(df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare two dataframes and generate a comparison summary.
    
    Args:
        df_before (pd.DataFrame): Dataframe before processing
        df_after (pd.DataFrame): Dataframe after processing
        
    Returns:
        Dict[str, Any]: Comparison summary
    """
    comparison = {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_changed": len(df_after) - len(df_before),
        "columns_before": len(df_before.columns),
        "columns_after": len(df_after.columns),
        "columns_changed": len(df_after.columns) - len(df_before.columns)
    }
    
    # Column changes
    cols_before = set(df_before.columns)
    cols_after = set(df_after.columns)
    
    added_columns = cols_after - cols_before
    removed_columns = cols_before - cols_after
    
    if added_columns:
        comparison["columns_added"] = list(added_columns)
    if removed_columns:
        comparison["columns_removed"] = list(removed_columns)
    
    return comparison
