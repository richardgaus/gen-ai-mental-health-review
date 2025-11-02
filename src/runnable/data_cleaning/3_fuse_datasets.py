#!/usr/bin/env python3
"""
Dataset fusion script.

This script fuses the transformed 1st search data with the 2nd search intermediate data
to create a combined dataset. It handles ID column conflicts and preserves all columns.

Usage:
    python 3_fuse_datasets.py [--first_search PATH] [--second_search PATH] [--output PATH] [--verbose]

Example:
    python 3_fuse_datasets.py --first_search data/intermediate/covidence_export_1st_search_20250921_intermediate_transformed_for_fusion.csv --second_search data/intermediate/intermediate_data.csv --output data/intermediate/fused_data.csv --verbose
"""

import argparse
import sys
import time
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from utils.data_cleaning.report_generator import generate_processing_report, save_report_to_file, get_dataframe_summary, compare_dataframes


def add_search_prefixes(df: pd.DataFrame, search_type: str) -> pd.DataFrame:
    """
    Add search type prefixes to ID columns to prevent conflicts.
    
    Args:
        df (pd.DataFrame): Input dataframe
        search_type (str): Either "1st" or "2nd" to identify search type
        
    Returns:
        pd.DataFrame: Dataframe with modified ID columns
    """
    df_modified = df.copy()
    
    # Define ID columns that need prefixes
    id_columns = ['study_id', 'covidence_id']
    
    for col in id_columns:
        if col in df_modified.columns:
            # Add prefix to the values, handling NaN values
            df_modified[col] = df_modified[col].apply(
                lambda x: f"{search_type}_search_{x}" if pd.notna(x) else x
            )
            print(f"✓ Added '{search_type}_search_' prefix to {col} column")
    
    return df_modified


def align_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align columns between two dataframes, ensuring both have the same columns.
    
    Args:
        df1 (pd.DataFrame): First dataframe (1st search)
        df2 (pd.DataFrame): Second dataframe (2nd search) 
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Aligned dataframes with same columns
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    # Get all unique columns
    all_columns = cols1 | cols2
    
    # Add missing columns to each dataframe
    df1_aligned = df1.copy()
    df2_aligned = df2.copy()
    
    # Add missing columns to df1
    missing_in_df1 = cols2 - cols1
    for col in missing_in_df1:
        df1_aligned[col] = pd.NA
        
    # Add missing columns to df2  
    missing_in_df2 = cols1 - cols2
    for col in missing_in_df2:
        df2_aligned[col] = pd.NA
    
    # Ensure column order is consistent
    column_order = sorted(all_columns)
    df1_aligned = df1_aligned[column_order]
    df2_aligned = df2_aligned[column_order]
    
    print(f"✓ Column alignment completed:")
    print(f"  - Total columns: {len(all_columns)}")
    print(f"  - Added to 1st search: {len(missing_in_df1)} columns")
    print(f"  - Added to 2nd search: {len(missing_in_df2)} columns")
    
    if missing_in_df1:
        print(f"  - Columns added to 1st search:")
        for col in sorted(missing_in_df1):
            print(f"    • {col}")
    
    return df1_aligned, df2_aligned


def remove_duplicate_studies(df1: pd.DataFrame, df2: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Remove studies from df2 (2nd search) that have the same title as studies in df1 (1st search).
    
    Args:
        df1 (pd.DataFrame): First dataset (1st search)
        df2 (pd.DataFrame): Second dataset (2nd search)
        verbose (bool): Enable verbose logging
        
    Returns:
        tuple: (df1_unchanged, df2_deduplicated, deduplication_info)
    """
    if verbose:
        print("Checking for duplicate studies between 1st and 2nd search datasets...")
    
    # Check if Title column exists in both datasets
    if 'Title' not in df1.columns or 'Title' not in df2.columns:
        if verbose:
            print("Warning: 'Title' column not found in one or both datasets. Skipping deduplication.")
        return df1, df2, {'duplicates_found': 0, 'duplicates_removed': 0, 'duplicate_titles': []}
    
    # Get unique titles from 1st search (normalize for comparison)
    df1_titles = set(df1['Title'].dropna().str.strip().str.lower())
    
    if verbose:
        print(f"  - 1st search contains {len(df1_titles)} unique study titles")
        print(f"  - 2nd search contains {len(df2)} rows before deduplication")
    
    # Find duplicates in 2nd search
    df2_copy = df2.copy()
    df2_copy['title_normalized'] = df2_copy['Title'].fillna('').str.strip().str.lower()
    
    # Identify rows in df2 that have titles matching df1
    duplicate_mask = df2_copy['title_normalized'].isin(df1_titles)
    duplicates_found = duplicate_mask.sum()
    
    # Get the actual duplicate titles for reporting
    duplicate_titles = df2_copy[duplicate_mask]['Title'].dropna().unique().tolist()
    
    # Remove duplicates from df2
    df2_deduplicated = df2_copy[~duplicate_mask].drop('title_normalized', axis=1).copy()
    
    deduplication_info = {
        'duplicates_found': duplicates_found,
        'duplicates_removed': duplicates_found,
        'duplicate_titles': duplicate_titles,
        'rows_before': len(df2),
        'rows_after': len(df2_deduplicated)
    }
    
    if verbose:
        print(f"✓ Duplicate study detection complete:")
        print(f"  - Duplicate studies found: {duplicates_found}")
        print(f"  - Rows removed from 2nd search: {duplicates_found}")
        print(f"  - 2nd search rows after deduplication: {len(df2_deduplicated)}")
        
        if duplicate_titles and verbose:
            print(f"  - Duplicate study titles removed:")
            for title in duplicate_titles[:5]:  # Show first 5 titles
                print(f"    • {title}")
            if len(duplicate_titles) > 5:
                print(f"    ... and {len(duplicate_titles) - 5} more")
    
    return df1, df2_deduplicated, deduplication_info


def fuse_datasets(df1: pd.DataFrame, df2: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Fuse two datasets by concatenating them vertically, after removing duplicates.
    
    Args:
        df1 (pd.DataFrame): First dataset (1st search)
        df2 (pd.DataFrame): Second dataset (2nd search)
        verbose (bool): Enable verbose logging
        
    Returns:
        tuple: (fused_dataset, deduplication_info)
    """
    # Step 1: Remove duplicate studies from 2nd search
    if verbose:
        print("Step 1: Removing duplicate studies from 2nd search...")
    df1_clean, df2_clean, deduplication_info = remove_duplicate_studies(df1, df2, verbose=verbose)
    
    # Step 2: Add search type prefixes to prevent ID conflicts
    if verbose:
        print("\nStep 2: Adding search type prefixes to ID columns...")
    df1_prefixed = add_search_prefixes(df1_clean, "1st")
    df2_prefixed = add_search_prefixes(df2_clean, "2nd")
    
    # Step 3: Align columns
    if verbose:
        print("\nStep 3: Aligning columns between datasets...")
    df1_aligned, df2_aligned = align_columns(df1_prefixed, df2_prefixed)
    
    # Step 4: Concatenate the datasets
    if verbose:
        print("\nStep 4: Concatenating datasets...")
    df_fused = pd.concat([df1_aligned, df2_aligned], ignore_index=True)
    
    if verbose:
        print(f"✓ Datasets successfully fused:")
        print(f"  - 1st search rows: {len(df1_aligned)}")
        print(f"  - 2nd search rows: {len(df2_aligned)}")
        print(f"  - Total fused rows: {len(df_fused)}")
        print(f"  - Total columns: {len(df_fused.columns)}")
    
    return df_fused, deduplication_info


def validate_fusion(df_original1: pd.DataFrame, df_original2: pd.DataFrame, df_fused: pd.DataFrame) -> dict:
    """
    Validate the fusion results.
    
    Args:
        df_original1 (pd.DataFrame): Original 1st search data
        df_original2 (pd.DataFrame): Original 2nd search data
        df_fused (pd.DataFrame): Fused dataset
        
    Returns:
        dict: Validation results
    """
    validation = {
        'row_count_correct': len(df_fused) == len(df_original1) + len(df_original2),
        'expected_rows': len(df_original1) + len(df_original2),
        'actual_rows': len(df_fused),
        'column_preservation': True,
        'id_conflicts_resolved': True
    }
    
    # Check if all original columns are preserved
    all_original_columns = set(df_original1.columns) | set(df_original2.columns)
    fused_columns = set(df_fused.columns)
    
    missing_columns = all_original_columns - fused_columns
    if missing_columns:
        validation['column_preservation'] = False
        validation['missing_columns'] = list(missing_columns)
    
    # Check for ID conflicts (should not have any duplicate IDs after prefixing)
    if 'study_id' in df_fused.columns:
        unique_study_ids = df_fused['study_id'].dropna().nunique()
        total_study_ids = df_fused['study_id'].dropna().count()
        if unique_study_ids != total_study_ids:
            validation['id_conflicts_resolved'] = False
    
    return validation


def main():
    """Main function to run the dataset fusion pipeline."""
    parser = argparse.ArgumentParser(
        description="Fuse 1st and 2nd search datasets"
    )
    
    parser.add_argument(
        "--first_search",
        type=str,
        default="data/intermediate/covidence_export_1st_search_20250921_intermediate_transformed_for_fusion.csv",
        help="Path to the transformed 1st search CSV file (relative to current working directory)"
    )
    
    parser.add_argument(
        "--second_search", 
        type=str,
        default="data/intermediate/intermediate_data.csv",
        help="Path to the 2nd search intermediate CSV file (relative to current working directory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/intermediate/fused_data.csv",
        help="Path for the output fused dataset CSV file (relative to current working directory)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle paths (resolve relative to current working directory)
    input_path1 = Path(args.first_search).resolve()
    input_path2 = Path(args.second_search).resolve()
    output_path = Path(args.output).resolve()
    
    if args.verbose:
        print(f"1st search file: {input_path1}")
        print(f"2nd search file: {input_path2}")
        print(f"Output file: {output_path}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load both datasets
        if args.verbose:
            print("\nLoading datasets...")
        
        if not input_path1.exists():
            raise FileNotFoundError(f"1st search file not found: {input_path1}")
        
        if not input_path2.exists():
            raise FileNotFoundError(f"2nd search file not found: {input_path2}")
        
        df1 = pd.read_csv(input_path1)
        df2 = pd.read_csv(input_path2)
        
        if args.verbose:
            print(f"✓ Loaded 1st search data: {len(df1):,} rows, {len(df1.columns):,} columns")
            print(f"✓ Loaded 2nd search data: {len(df2):,} rows, {len(df2.columns):,} columns")
        
        # Store originals for validation
        df1_original = df1.copy()
        df2_original = df2.copy()
        
        # Fuse the datasets
        if args.verbose:
            print(f"\nFusing datasets...")
        
        df_fused, deduplication_info = fuse_datasets(df1, df2, verbose=args.verbose)
        
        # Validate fusion
        if args.verbose:
            print(f"\nValidating fusion...")
            
        # Adjust validation for deduplication - use original df1 and deduplicated df2 count
        expected_rows_after_dedup = len(df1_original) + deduplication_info['rows_after']
        validation = validate_fusion(df1_original, df2_original, df_fused)
        # Override the expected rows calculation to account for deduplication
        validation['expected_rows'] = expected_rows_after_dedup
        validation['row_count_correct'] = len(df_fused) == expected_rows_after_dedup
        
        if args.verbose:
            print(f"✓ Validation results:")
            print(f"  - Row count correct: {validation['row_count_correct']}")
            print(f"  - Expected rows: {validation['expected_rows']:,}")
            print(f"  - Actual rows: {validation['actual_rows']:,}")
            print(f"  - Column preservation: {validation['column_preservation']}")
            print(f"  - ID conflicts resolved: {validation['id_conflicts_resolved']}")
            
            if not validation['column_preservation']:
                print(f"  - Missing columns: {validation.get('missing_columns', [])}")
        
        # Save the fused dataset
        if args.verbose:
            print(f"\nSaving fused dataset to {output_path}...")
        
        df_fused.to_csv(output_path, index=False)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate processing report
        if args.verbose:
            print(f"\nGenerating processing report...")
        
        # Calculate study counts for both datasets
        df1_studies = df1_original['covidence_id'].nunique() if 'covidence_id' in df1_original.columns else len(df1_original)
        df2_studies = df2_original['covidence_id'].nunique() if 'covidence_id' in df2_original.columns else len(df2_original)
        fused_studies = df_fused['covidence_id'].nunique() if 'covidence_id' in df_fused.columns else len(df_fused)
        
        # Get column alignment details
        cols1 = set(df1_original.columns)
        cols2 = set(df2_original.columns)
        missing_in_df1 = cols2 - cols1
        missing_in_df2 = cols1 - cols2
        
        # Create detailed column alignment description
        alignment_description = f"Aligned columns between datasets - added {len(missing_in_df1)} columns to 1st search data and {len(missing_in_df2)} columns to 2nd search data (filled with NA values)"
        if missing_in_df1:
            alignment_description += f". Columns added to 1st search: {', '.join(sorted(missing_in_df1))}"
        if missing_in_df2:
            alignment_description += f". Columns added to 2nd search: {', '.join(sorted(missing_in_df2))}"
        if not missing_in_df1 and not missing_in_df2:
            alignment_description = "Column alignment completed - both datasets already had identical column sets, no columns needed to be added"
        
        # Create deduplication description
        if deduplication_info['duplicates_found'] > 0:
            duplicate_titles_text = f"Duplicate study titles removed from 2nd search: {', '.join(deduplication_info['duplicate_titles'][:3])}" if len(deduplication_info['duplicate_titles']) <= 3 else f"Duplicate study titles removed from 2nd search: {', '.join(deduplication_info['duplicate_titles'][:3])} and {len(deduplication_info['duplicate_titles']) - 3} more"
            deduplication_description = f"Removed {deduplication_info['duplicates_found']} duplicate studies from 2nd search (studies with same titles as 1st search). {duplicate_titles_text}"
        else:
            deduplication_description = "No duplicate studies found between 1st and 2nd search datasets"
        
        processing_steps = {
            "step_1_data_loading": f"Loaded 1st search data ({len(df1_original)} rows, {len(df1_original.columns)} columns) and 2nd search data ({len(df2_original)} rows, {len(df2_original.columns)} columns)",
            "step_2_deduplication": deduplication_description,
            "step_3_prefix_addition": f"Added search type prefixes ('1st_search_' and '2nd_search_') to ID columns (study_id, covidence_id) to prevent conflicts during fusion",
            "step_4_column_alignment": alignment_description,
            "step_5_dataset_fusion": f"Concatenated datasets vertically to create fused dataset with {len(df_fused)} total rows and {len(df_fused.columns)} columns",
            "step_6_validation": f"Validated fusion results - confirmed row count preservation ({validation['row_count_correct']}) and column preservation ({validation['column_preservation']})"
        }
        
        # Create simplified summary statistics
        summary_stats = {
            "first_search_studies": df1_studies,
            "first_search_rows": len(df1_original),
            "first_search_columns": len(df1_original.columns),
            "second_search_studies_original": df2_studies,
            "second_search_rows_original": len(df2_original),
            "second_search_columns": len(df2_original.columns),
            "duplicate_studies_removed": deduplication_info['duplicates_found'],
            "second_search_rows_after_dedup": deduplication_info['rows_after'],
            "fused_studies": fused_studies,
            "fused_rows": len(df_fused),
            "fused_columns": len(df_fused.columns),
            "columns_added_to_first": len(missing_in_df1),
            "columns_added_to_second": len(missing_in_df2),
            "fusion_successful": validation['row_count_correct'] and validation['column_preservation'],
            "execution_time_seconds": execution_time
        }
        
        report = generate_processing_report(
            script_name="3_fuse_datasets.py",
            input_file=f"{input_path1} + {input_path2}",
            output_file=output_path,
            processing_steps=processing_steps,
            summary_stats=summary_stats,
            execution_time=execution_time
        )
        
        # Save report
        reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
        report_filename = "3_fuse_datasets"
        report_path = save_report_to_file(report, str(reports_dir), report_filename, save_json=False)
        
        if args.verbose:
            print(f"✓ Successfully saved fused dataset to {output_path}")
            print(f"✓ Processing report saved to {report_path}")
            print(f"✓ Fusion complete!")
            print(f"\nFused dataset: {len(df_fused):,} rows with {len(df_fused.columns):,} columns")
            print(f"Execution time: {execution_time:.2f} seconds")
        else:
            print(f"Dataset fusion complete. Output saved to: {output_path}")
            print(f"Processing report saved to: {report_path}")
            
    except Exception as e:
        print(f"Error during fusion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
