#!/usr/bin/env python3
"""
Interrater agreement calculation script for LLMs in Psychotherapy research.

This script calculates Cohen's kappa coefficient for measuring agreement
between two reviewers (excluding consensus) on specified columns.
The complete analysis is saved as a single report in results/reports_data_processing/.

Usage:
    python 6_calculate_interrater_agreement.py --input PATH [--verbose]

Example:
    python 6_calculate_interrater_agreement.py --input data/intermediate/fused_data.csv --verbose
"""

import argparse
import sys
import time
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from utils.interrater_agreement import calculate_kappa_for_columns, load_kappa_config, generate_kappa_report
from utils.data_cleaning.report_generator import generate_processing_report, save_report_to_file


def main():
    """Main function to run the interrater agreement calculation."""
    parser = argparse.ArgumentParser(
        description="Calculate Cohen's kappa interrater agreement for specified columns"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file (relative to current working directory)"
    )
    
    # Removed --output argument since we only generate one report in results/reports_data_processing/
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle input path (resolve relative to current working directory)
    input_path = Path(args.input).resolve()
    
    # Output will be saved to results/reports_data_processing/
    reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
    output_path = reports_dir / "6_calculate_interrater_agreement.txt"
    
    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output report: {output_path}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load the data
        if args.verbose:
            print("Loading data...")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load data - handle both indexed and non-indexed CSV files
        try:
            # Try loading with covidence_id as index first
            df = pd.read_csv(input_path, index_col='covidence_id')
            if args.verbose:
                print(f"Loaded {len(df):,} rows with {len(df.columns):,} columns (covidence_id as index)")
        except:
            # Fallback to regular loading
            df = pd.read_csv(input_path)
            if args.verbose:
                print(f"Loaded {len(df):,} rows with {len(df.columns):,} columns")
        
        df_original = df.copy()
        
        # Check for required columns
        if 'reviewer_name' not in df.columns:
            raise KeyError("Column 'reviewer_name' not found in dataframe. This script requires reviewer data.")
        
        # Load configuration
        if args.verbose:
            print("\\nLoading Cohen's kappa configuration...")
        
        single_choice_columns, multi_choice_columns, numerical_columns = load_kappa_config()
        
        if not single_choice_columns:
            raise ValueError("No columns specified for Cohen's kappa calculation in configuration")
        
        # Print column counts clearly
        print(f"\n📊 COLUMN TYPES IDENTIFIED:")
        print(f"  • Single-choice categorical columns: {len(single_choice_columns)}")
        print(f"  • Multi-choice categorical columns: {len(multi_choice_columns)}")
        print(f"  • Numerical columns: {len(numerical_columns)}")
        
        if args.verbose:
            print(f"\n✓ Configuration loaded:")
            print(f"  - Sample single-choice columns: {single_choice_columns[:5]}")
            if multi_choice_columns:
                print(f"  - Sample multi-choice columns: {list(multi_choice_columns.keys())[:3]}")
            if numerical_columns:
                print(f"  - Sample numerical columns: {numerical_columns[:3]}")
        
        # Check reviewers
        reviewers = df['reviewer_name'].unique()
        non_consensus_reviewers = [r for r in reviewers if r != 'Consensus']
        
        if args.verbose:
            print(f"\\n📊 Reviewer Analysis:")
            print(f"  - Total reviewers: {len(reviewers)}")
            print(f"  - Non-consensus reviewers: {len(non_consensus_reviewers)}")
            print(f"  - Reviewers: {list(reviewers)}")
        
        if len(non_consensus_reviewers) < 2:
            raise ValueError(f"Need at least 2 non-consensus reviewers for agreement calculation. Found: {non_consensus_reviewers}")
        
        # Filter to keep only rows that have both "Richard Gaus" and "Reviewer Two"
        if args.verbose:
            print(f"\\n🔍 Filtering for rows with both required reviewers...")
        
        # Get unique study identifiers (assuming index or a specific column identifies studies)
        if hasattr(df.index, 'name') and df.index.name:
            study_id_col = df.index.name
            study_ids = df.index.unique()
        elif 'covidence_id' in df.columns:
            study_id_col = 'covidence_id'
            study_ids = df['covidence_id'].unique()
        else:
            # Fallback: assume first column is study identifier
            study_id_col = df.columns[0]
            study_ids = df[study_id_col].unique()
        
        # Find studies that have both required reviewers
        required_reviewers = {"Richard Gaus", "Reviewer Two"}
        studies_to_keep = []
        studies_removed_info = []
        
        for study_id in study_ids:
            if hasattr(df.index, 'name') and df.index.name:
                study_data = df[df.index == study_id]
            else:
                study_data = df[df[study_id_col] == study_id]
            
            study_reviewers = set(study_data['reviewer_name'].unique())
            
            # Keep study if it has both required reviewers
            if required_reviewers.issubset(study_reviewers):
                studies_to_keep.append(study_id)
            else:
                # Capture information about removed studies
                study_title = study_data['title'].iloc[0] if 'title' in study_data.columns and len(study_data) > 0 else "Title not available"
                missing_reviewers = required_reviewers - study_reviewers
                studies_removed_info.append({
                    'study_id': study_id,
                    'title': study_title,
                    'present_reviewers': list(study_reviewers),
                    'missing_reviewers': list(missing_reviewers)
                })
        
        # Filter the dataframe to keep only studies with both reviewers
        if hasattr(df.index, 'name') and df.index.name:
            df_filtered = df[df.index.isin(studies_to_keep)]
        else:
            df_filtered = df[df[study_id_col].isin(studies_to_keep)]
        
        rows_before = len(df)
        rows_after = len(df_filtered)
        studies_before = len(study_ids)
        studies_after = len(studies_to_keep)
        
        if args.verbose:
            print(f"✓ Filtering complete:")
            print(f"  - Studies before filtering: {studies_before}")
            print(f"  - Studies after filtering: {studies_after}")
            print(f"  - Studies removed: {studies_before - studies_after}")
            print(f"  - Rows before filtering: {rows_before:,}")
            print(f"  - Rows after filtering: {rows_after:,}")
            print(f"  - Rows removed: {rows_before - rows_after:,}")
            
            if studies_removed_info:
                print(f"\n📋 Studies removed (missing required reviewers):")
                for i, study_info in enumerate(studies_removed_info, 1):
                    print(f"  {i}. {study_info['study_id']}")
                    print(f"     Title: {study_info['title']}")
                    print(f"     Present reviewers: {', '.join(study_info['present_reviewers'])}")
                    print(f"     Missing reviewers: {', '.join(study_info['missing_reviewers'])}")
                    print()
        
        # Update df to use the filtered version
        df = df_filtered
        
        if len(df) == 0:
            raise ValueError("No studies found with both 'Richard Gaus' and 'Reviewer Two' reviewers")
        
        # Calculate Cohen's kappa for all specified columns
        if args.verbose:
            print(f"\\n🔢 Calculating Cohen's kappa for {len(single_choice_columns)} columns...")
        
        results, overall_stats, multi_choice_results = calculate_kappa_for_columns(df, single_choice_columns, multi_choice_columns, verbose=args.verbose)
        
        # Generate kappa report content (but don't save separately)
        if args.verbose:
            print(f"\\n📝 Generating interrater agreement analysis...")
        
        kappa_report_content = generate_kappa_report(results, overall_stats, multi_choice_results, None,  # Don't save to file
                                                     single_choice_count=len(single_choice_columns), 
                                                     numerical_count=len(numerical_columns))
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate processing report
        if args.verbose:
            print(f"\nGenerating processing report...")
        
        # Create removed studies summary for report
        removed_studies_summary = "; ".join([f"{info['study_id']} ({info['title'][:50]}...)" for info in studies_removed_info]) if studies_removed_info else "None"
        
        processing_steps = {
            "step_1_data_loading": f"Loaded dataset with reviewer information - {len(df_original)} rows and {len(df_original.columns)} columns",
            "step_2_configuration_loading": f"Loaded Cohen's kappa configuration identifying {len(single_choice_columns)} single-choice, {len(multi_choice_columns)} multi-choice, and {len(numerical_columns)} numerical columns",
            "step_3_dual_reviewer_filtering": f"Filtered to keep only studies with both 'Richard Gaus' and 'Reviewer Two' reviewers - removed {studies_before - studies_after} studies ({rows_before - rows_after:,} rows), keeping {studies_after} studies ({rows_after:,} rows). Removed studies: {removed_studies_summary}",
            "step_4_kappa_calculation": f"Calculated Cohen's kappa for {len(single_choice_columns)} single-choice categorical columns using scikit-learn implementation",
            "step_5_multi_choice_analysis": f"Analyzed {len(multi_choice_columns)} multi-choice categorical columns using specialized multi-label kappa calculation" if multi_choice_columns else "No multi-choice columns found for analysis",
            "step_6_report_generation": f"Generated detailed interrater agreement report with kappa scores, AC1 coefficients, and agreement rates for all analyzed columns"
        }
        
        # Create simplified summary statistics
        valid_kappa_results = len([r for r in results.values() if r['kappa'] is not None]) if results else 0
        valid_multi_choice_results = len([r for r in multi_choice_results.values() if r['overall_kappa'] is not None]) if multi_choice_results else 0
        
        summary_stats = {
            "input_studies": studies_before,
            "input_rows": rows_before,
            "input_columns": len(df_original.columns),
            "output_studies": studies_after,
            "output_rows": rows_after,
            "output_columns": len(df.columns),
            "studies_removed": studies_before - studies_after,
            "rows_removed": rows_before - rows_after,
            "reviewers_analyzed": len([r for r in df['reviewer_name'].unique() if r != 'Consensus']) if 'reviewer_name' in df.columns else 0,
            "single_choice_columns_analyzed": len(single_choice_columns),
            "multi_choice_columns_analyzed": len(multi_choice_columns),
            "numerical_columns_identified": len(numerical_columns),
            "valid_kappa_results": valid_kappa_results,
            "valid_multi_choice_results": valid_multi_choice_results,
            "overall_agreement_rate": overall_stats.get('overall_agreement_rate', 0) if overall_stats else 0,
            "execution_time_seconds": execution_time
        }
        
        # Include kappa report content in additional_info
        additional_info = {
            "interrater_agreement_analysis": kappa_report_content
        }
        
        report = generate_processing_report(
            script_name="6_calculate_interrater_agreement.py",
            input_file=input_path,
            output_file=output_path,
            processing_steps=processing_steps,
            summary_stats=summary_stats,
            execution_time=execution_time,
            additional_info=additional_info
        )
        
        # Save processing report
        reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
        report_filename = "6_calculate_interrater_agreement"
        processing_report_path = save_report_to_file(report, str(reports_dir), report_filename, save_json=False)
        
        if args.verbose:
            print(f"✓ Successfully saved interrater agreement report to {processing_report_path}")
            print(f"✓ Interrater agreement calculation complete!")
            print(f"Execution time: {execution_time:.2f} seconds")
            
            # Print summary
            valid_kappa_results = [r for r in results.values() if not pd.isna(r['kappa'])]
            valid_ac1_results = [r for r in results.values() if not pd.isna(r['ac1'])]
            valid_agreement_results = [r for r in results.values() if not pd.isna(r['agreement_rate'])]
            
            print(f"\n📈 Summary:")
            print(f"  - Single-choice categorical columns analyzed: {len(results)}")
            print(f"  - Valid kappa scores: {len(valid_kappa_results)}")
            print(f"  - Valid AC1 scores: {len(valid_ac1_results)}")
            print(f"  - Valid agreement rates: {len(valid_agreement_results)}")
            
            # Multi-choice summary
            valid_mc_results = [r for r in multi_choice_results.values() if not pd.isna(r['overall_kappa'])]
            if multi_choice_results:
                print(f"  - Multi-choice categorical columns analyzed: {len(multi_choice_results)}")
                print(f"  - Valid multi-choice kappa scores: {len(valid_mc_results)}")
            
            # Numerical columns summary (for information only)
            if numerical_columns:
                print(f"  - Numerical columns identified: {len(numerical_columns)} (not analyzed for kappa)")
            
            # Overall agreement
            if overall_stats.get('status') == 'success':
                print(f"\n🎯 Overall Agreement Across All Fields:")
                print(f"  - Total comparisons: {overall_stats['total_comparisons']:,}")
                print(f"  - Overall agreement rate: {overall_stats['overall_agreement_rate']:.1%}")
                
                overall_ac1 = overall_stats.get('overall_ac1', np.nan)
                overall_kappa = overall_stats.get('overall_kappa', np.nan)
                if not pd.isna(overall_ac1):
                    print(f"  - Overall AC1: {overall_ac1:.3f}")
                if not pd.isna(overall_kappa):
                    print(f"  - Overall Cohen's kappa: {overall_kappa:.3f}")
            
            # Field-by-field statistics
            if valid_agreement_results:
                agreement_values = [r['agreement_rate'] for r in valid_agreement_results]
                print(f"\n📉 Field-by-Field Agreement:")
                print(f"  - Mean agreement per field: {pd.Series(agreement_values).mean():.1%}")
                print(f"  - Agreement range: {pd.Series(agreement_values).min():.1%} to {pd.Series(agreement_values).max():.1%}")
            
            if valid_ac1_results:
                ac1_values = [r['ac1'] for r in valid_ac1_results]
                print(f"\n🎆 Field-by-Field AC1:")
                print(f"  - Mean AC1 per field: {pd.Series(ac1_values).mean():.3f}")
                print(f"  - AC1 range: {pd.Series(ac1_values).min():.3f} to {pd.Series(ac1_values).max():.3f}")
            
            if valid_kappa_results:
                kappa_values = [r['kappa'] for r in valid_kappa_results]
                print(f"\n📈 Field-by-Field Kappa:")
                print(f"  - Mean kappa per field: {pd.Series(kappa_values).mean():.3f}")
                print(f"  - Kappa range: {pd.Series(kappa_values).min():.3f} to {pd.Series(kappa_values).max():.3f}")
            
            if valid_mc_results:
                mc_kappa_values = [r['overall_kappa'] for r in valid_mc_results]
                print(f"\n🔀 Multi-Choice Kappa:")
                print(f"  - Mean overall kappa: {pd.Series(mc_kappa_values).mean():.3f}")
                print(f"  - Kappa range: {pd.Series(mc_kappa_values).min():.3f} to {pd.Series(mc_kappa_values).max():.3f}")
        else:
            print(f"Interrater agreement calculation complete. Report saved to: {processing_report_path}")
            
    except Exception as e:
        print(f"Error during calculation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
