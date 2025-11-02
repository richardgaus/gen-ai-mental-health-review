#!/usr/bin/env python3
"""
Final data cleaning script for LLMs in Psychotherapy research.

This script performs final cleaning operations on CSV files including:
- Filtering for consensus reviewer data only (or using preliminary decisions from Richard Gaus for specific columns)
- Optionally removing population survey studies (when --remove_population_surveys flag is used)
- Creating intended_intervention_type column from intervention_type
- Creating inclusion_criteria column from client_type
- Creating primary_clinical_outcome column for client-facing empirical studies
- Cleaning models_employed column by parsing "Other:" entries and mapping to standardized names
- Cleaning llm_development_approach column by mapping values and collapsing multi-label entries
- Cleaning outlet_field and application type columns by mapping values to standardized categories

Usage:
    python 7_perform_final_cleaning.py --input PATH [--output DIR] [--verbose] [--use-preliminary-decisions] [--remove_population_surveys]

Example:
    python 7_perform_final_cleaning.py --input data/intermediate/fused_data_cleaned.csv --verbose
    python 7_perform_final_cleaning.py --input data/intermediate/fused_data_cleaned.csv --output data/final
    python 7_perform_final_cleaning.py --input data/intermediate/fused_data_cleaned.csv --use-preliminary-decisions --verbose
    python 7_perform_final_cleaning.py --input data/intermediate/fused_data_cleaned.csv --remove_population_surveys --verbose
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from utils.data_cleaning.final_cleaning_utils import perform_final_cleaning_pipeline
from utils.data_cleaning.report_generator import generate_processing_report, save_report_to_file


def remove_population_surveys(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove all studies where study_type is 'Population survey'.
    
    Args:
        df (pd.DataFrame): Input dataframe with study_type column
        verbose (bool): Enable verbose logging
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Filtered dataframe and processing details dictionary
    """
    # Check for required column
    if 'study_type' not in df.columns:
        raise KeyError("Column 'study_type' not found in dataframe")
    
    # Store original counts
    original_rows = len(df)
    original_studies = df.index.nunique() if hasattr(df.index, 'nunique') else len(df)
    
    # Filter out population surveys
    df_filtered = df[df['study_type'] != 'Population survey'].copy()
    
    # Calculate what was removed
    final_rows = len(df_filtered)
    final_studies = df_filtered.index.nunique() if hasattr(df_filtered.index, 'nunique') else len(df_filtered)
    rows_removed = original_rows - final_rows
    studies_removed = original_studies - final_studies
    
    if verbose:
        print(f"✓ Removed population survey studies:")
        print(f"  - Original studies: {original_studies:,}")
        print(f"  - Studies after filtering: {final_studies:,}")
        print(f"  - Studies removed: {studies_removed:,}")
        print(f"  - Original rows: {original_rows:,}")
        print(f"  - Final rows: {final_rows:,}")
        print(f"  - Rows removed: {rows_removed:,}")
    
    processing_details = {
        'original_rows': original_rows,
        'original_studies': original_studies,
        'final_rows': final_rows,
        'final_studies': final_studies,
        'rows_removed': rows_removed,
        'studies_removed': studies_removed
    }
    
    return df_filtered, processing_details


def main():
    """Main function to run the final cleaning pipeline."""
    parser = argparse.ArgumentParser(
        description="Perform final cleaning operations on CSV data"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file (relative to current working directory)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/final",
        help="Directory path to save output file (default: data/final). File will be saved as final_data.csv"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--use-preliminary-decisions",
        action="store_true",
        help="Use preliminary decisions from 'Richard Gaus' reviewer instead of 'Consensus' for specific columns"
    )
    
    parser.add_argument(
        "--remove_population_surveys",
        action="store_true",
        help="Remove all studies where study_type is 'Population survey'"
    )
    
    args = parser.parse_args()
    
    # Handle input path (resolve relative to current working directory)
    input_path = Path(args.input).resolve()
    
    # Handle output path - treat as directory and append filename
    output_dir = Path(args.output).resolve()
    output_path = output_dir / "final_data.csv"
    
    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load the data
        if args.verbose:
            print("Loading data...")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Try loading with covidence_id as index first
        try:
            df = pd.read_csv(input_path, index_col='covidence_id')
            if args.verbose:
                print(f"Loaded {len(df):,} rows with {len(df.columns):,} columns (covidence_id as index)")
        except:
            # Fallback to regular loading
            df = pd.read_csv(input_path)
            if args.verbose:
                print(f"Loaded {len(df):,} rows with {len(df.columns):,} columns")
        
        df_original = df.copy()
        
        # Perform the complete final cleaning pipeline with optional population survey removal
        if args.verbose:
            print("\n" + "=" * 60)
        
        # Import the individual functions from the pipeline to create a custom pipeline
        from utils.data_cleaning.final_cleaning_utils import (
            filter_reviewer_data, add_intended_intervention_type_column,
            add_inclusion_criteria_column, add_primary_clinical_outcome_column,
            clean_models_employed_column, clean_llm_development_approach_column,
            clean_outlet_and_application_columns
        )
        
        # Step 1: Filter for reviewer data (consensus filtering)
        if args.verbose:
            reviewer_type = "'Consensus' (with Richard Gaus preliminary decisions)" if getattr(args, 'use_preliminary_decisions', False) else "'Consensus'"
            print(f"Step 1: Filtering for {reviewer_type} reviewer...")
        
        df_final, processing_details = filter_reviewer_data(
            df, 
            use_preliminary=getattr(args, 'use_preliminary_decisions', False), 
            verbose=args.verbose
        )
        
        # Step 2: Remove population surveys if requested (right after consensus filtering)
        population_survey_details = None
        if args.remove_population_surveys:
            if args.verbose:
                print(f"\nStep 2: Removing population survey studies...")
            
            df_final, population_survey_details = remove_population_surveys(df_final, verbose=args.verbose)
        
        # Continue with the rest of the pipeline (steps renumbered accordingly)
        step_offset = 1 if args.remove_population_surveys else 0
        
        # Step 2/3: Add intended_intervention_type column
        if args.verbose:
            print(f"\nStep {2 + step_offset}: Creating intended_intervention_type column...")
        
        df_final, column_details = add_intended_intervention_type_column(df_final, verbose=args.verbose)
        
        # Step 3/4: Add inclusion_criteria column
        if args.verbose:
            print(f"\nStep {3 + step_offset}: Creating inclusion_criteria column...")
        
        df_final, inclusion_details = add_inclusion_criteria_column(df_final, verbose=args.verbose)
        
        # Step 4/5: Add primary_clinical_outcome column
        if args.verbose:
            print(f"\nStep {4 + step_offset}: Creating primary_clinical_outcome column...")
        
        df_final, outcome_details = add_primary_clinical_outcome_column(df_final, verbose=args.verbose)
        
        # Step 5/6: Clean models_employed column
        if args.verbose:
            print(f"\nStep {5 + step_offset}: Cleaning models_employed column...")
        
        df_final, models_cleaning_details = clean_models_employed_column(df_final, verbose=args.verbose)
        
        # Step 6/7: Clean llm_development_approach column
        if args.verbose:
            print(f"\nStep {6 + step_offset}: Cleaning llm_development_approach column...")
        
        df_final, approach_cleaning_details = clean_llm_development_approach_column(df_final, verbose=args.verbose)
        
        # Step 7/8: Clean outlet_field and application type columns
        if args.verbose:
            print(f"\nStep {7 + step_offset}: Cleaning outlet and application columns...")
        
        df_final, outlet_application_cleaning_details = clean_outlet_and_application_columns(df_final, verbose=args.verbose)
        
        # Merge processing details
        processing_details['column_creation'] = column_details
        processing_details['inclusion_creation'] = inclusion_details
        processing_details['outcome_creation'] = outcome_details
        processing_details['models_cleaning'] = models_cleaning_details
        processing_details['approach_cleaning'] = approach_cleaning_details
        processing_details['outlet_application_cleaning'] = outlet_application_cleaning_details
        
        # Add population survey removal details if it was performed
        if population_survey_details:
            processing_details['population_survey_removal'] = population_survey_details
        
        if args.verbose:
            print("=" * 60 + "\n")
        
        # Save the cleaned data
        if args.verbose:
            print(f"Saving cleaned data to {output_path}...")
        
        df_final.to_csv(output_path)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Generate processing report
        if args.verbose:
            print(f"\nGenerating processing report...")
        
        # Build processing steps description
        processing_steps = {
            "step_1_data_loading": f"Loaded dataset with {processing_details['original_rows']:,} rows and {len(df_original.columns)} columns",
            "step_2_consensus_filtering": f"Filtered to keep only '{processing_details.get('reviewer_used', 'Consensus')}' reviewer rows - removed {processing_details['rows_removed']:,} rows from other reviewers ({', '.join([f'{k}: {v}' for k, v in processing_details['reviewer_counts'].items() if k != 'Consensus'])}). Final dataset: {processing_details['final_rows']:,} rows"
        }
        
        # Add population survey removal step if it was performed
        step_counter = 3
        if 'population_survey_removal' in processing_details:
            pop_details = processing_details['population_survey_removal']
            processing_steps[f"step_{step_counter}_population_survey_removal"] = f"Removed {pop_details['studies_removed']:,} population survey studies (removed {pop_details['rows_removed']:,} rows). Studies remaining: {pop_details['final_studies']:,} (down from {pop_details['original_studies']:,})"
            step_counter += 1
        
        # Add column creation step if it was performed
        if 'column_creation' in processing_details and processing_details['column_creation']['column_created']:
            col_details = processing_details['column_creation']
            step_desc = f"Created 'intended_intervention_type' column from 'intervention_type' column. "
            step_desc += f"Processed {col_details['unique_original_values']:,} unique original values into {col_details['unique_transformed_values']:,} transformed values. "
            step_desc += f"Applied {col_details['transformations_applied']:,} different transformation rules."
            if col_details['example_transformations']:
                step_desc += f" Examples: {'; '.join(col_details['example_transformations'][:3])}"
            processing_steps[f"step_{step_counter}_column_creation"] = step_desc
        elif 'column_creation' in processing_details:
            processing_steps[f"step_{step_counter}_column_creation"] = f"Skipped 'intended_intervention_type' column creation: {processing_details['column_creation'].get('reason', 'Unknown reason')}"
        step_counter += 1
        
        # Add inclusion criteria creation step if it was performed
        if 'inclusion_creation' in processing_details and processing_details['inclusion_creation']['column_created']:
            inclusion_details = processing_details['inclusion_creation']
            step_desc = f"Created 'inclusion_criteria' column from 'client_type' column. "
            step_desc += f"Processed {inclusion_details['unique_original_values']:,} unique original values into {inclusion_details['unique_transformed_values']:,} transformed values. "
            step_desc += f"Applied {inclusion_details['transformations_applied']:,} different transformation rules. "
            criteria_counts = inclusion_details['criteria_distribution']
            if criteria_counts:
                counts_str = ', '.join([f"{k}: {v}" for k, v in criteria_counts.items()])
                step_desc += f"Distribution: {counts_str}."
            if inclusion_details['example_transformations']:
                step_desc += f" Examples: {'; '.join(inclusion_details['example_transformations'][:3])}"
            processing_steps[f"step_{step_counter}_inclusion_creation"] = step_desc
        elif 'inclusion_creation' in processing_details:
            processing_steps[f"step_{step_counter}_inclusion_creation"] = f"Skipped 'inclusion_criteria' column creation: {processing_details['inclusion_creation'].get('reason', 'Unknown reason')}"
        step_counter += 1
        
        # Add primary clinical outcome step if it was performed
        if 'outcome_creation' in processing_details and processing_details['outcome_creation']['column_created']:
            outcome_details = processing_details['outcome_creation']
            step_desc = f"Created 'primary_clinical_outcome' column for client-facing empirical LLM studies with clients involved. "
            step_desc += f"Filled {outcome_details['total_filled_rows']:,} rows out of {outcome_details['total_eligible_rows']:,} eligible rows. "
            outcome_counts = outcome_details['outcome_counts']
            if outcome_counts:
                counts_str = ', '.join([f"{k}: {v}" for k, v in outcome_counts.items() if k != ''])
                step_desc += f"Distribution: {counts_str}."
            processing_steps[f"step_{step_counter}_outcome_creation"] = step_desc
        elif 'outcome_creation' in processing_details:
            processing_steps[f"step_{step_counter}_outcome_creation"] = f"Skipped 'primary_clinical_outcome' column creation: {processing_details['outcome_creation'].get('reason', 'Unknown reason')}"
        step_counter += 1
        
        # Add models cleaning step if it was performed
        if 'models_cleaning' in processing_details and processing_details['models_cleaning']['column_cleaned']:
            models_details = processing_details['models_cleaning']
            step_desc = f"Cleaned 'models_employed' column by first removing {models_details['total_non_gen_models_removed']:,} non-generative models, "
            step_desc += f"then parsing 'Other:' entries and mapping them using configuration. "
            step_desc += f"Processed {models_details['total_entries_processed']:,} entries, mapped {models_details['total_models_mapped']:,} models, "
            step_desc += f"set {models_details['total_models_unmapped']:,} unmapped models to 'Other'. "
            
            # Add details about non-generative models removed
            if models_details.get('non_gen_removed_examples'):
                unique_removed = sorted(list(set(models_details['non_gen_removed_examples'])))
                removed_sample = ', '.join(unique_removed[:5])
                if len(unique_removed) > 5:
                    removed_sample += f" (and {len(unique_removed) - 5} more)"
                step_desc += f"Non-generative models removed: {removed_sample}. "
            
            # Add details about mappings applied
            if models_details.get('mapping_examples'):
                mapping_sample = ', '.join(models_details['mapping_examples'][:3])
                if len(models_details['mapping_examples']) > 3:
                    mapping_sample += f" (and {len(models_details['mapping_examples']) - 3} more)"
                step_desc += f"Model mappings: {mapping_sample}. "
            
            # Add details about unmapped models (showing all instances including duplicates)
            if models_details.get('unmapped_models_with_duplicates'):
                # Show all instances including duplicates
                unmapped_with_dupes = models_details['unmapped_models_with_duplicates']
                unmapped_sample = ', '.join(unmapped_with_dupes[:8])  # Show more since duplicates are meaningful
                if len(unmapped_with_dupes) > 8:
                    unmapped_sample += f" (and {len(unmapped_with_dupes) - 8} more instances)"
                step_desc += f"Unmapped models set to 'Other': {unmapped_sample}. "
            elif models_details['unmapped_models']:
                # Fallback to unique list if duplicates not available
                unmapped_sample = ', '.join(models_details['unmapped_models'][:5])
                if len(models_details['unmapped_models']) > 5:
                    unmapped_sample += f" (and {len(models_details['unmapped_models']) - 5} more)"
                step_desc += f"Unmapped models set to 'Other': {unmapped_sample}. "
            
            # Add distribution of final models_employed column
            if 'models_employed' in df_final.columns:
                # Calculate model distribution
                all_models = []
                for models_entry in df_final['models_employed'].dropna():
                    if models_entry and str(models_entry).strip():
                        models = [model.strip() for model in str(models_entry).split(';') if model.strip()]
                        all_models.extend(models)
                
                if all_models:
                    from collections import Counter
                    model_counts = Counter(all_models)
                    top_models = model_counts.most_common(5)
                    dist_sample = ', '.join([f"{model}: {count}" for model, count in top_models])
                    step_desc += f"Final model distribution (top 5): {dist_sample}."
            
            processing_steps[f"step_{step_counter}_models_cleaning"] = step_desc
        elif 'models_cleaning' in processing_details:
            processing_steps[f"step_{step_counter}_models_cleaning"] = f"Skipped 'models_employed' column cleaning: {processing_details['models_cleaning'].get('reason', 'Unknown reason')}"
        step_counter += 1
        
        # Add llm_development_approach cleaning step if it was performed
        if 'approach_cleaning' in processing_details and processing_details['approach_cleaning']['column_cleaned']:
            approach_details = processing_details['approach_cleaning']
            step_desc = f"Cleaned 'llm_development_approach' column by mapping {approach_details['total_values_mapped']:,} values, "
            step_desc += f"collapsing {approach_details['total_entries_collapsed']:,} multi-label entries using priority rules, "
            step_desc += f"and final categorizing {approach_details['total_entries_final_categorized']:,} entries to main categories. "
            step_desc += f"Processed {approach_details['total_entries_processed']:,} entries with {approach_details['mapping_rules_available']:,} mapping rules available. "
            
            # Add mapping examples
            if approach_details.get('mapping_examples'):
                mapping_sample = ', '.join(approach_details['mapping_examples'][:3])
                if len(approach_details['mapping_examples']) > 3:
                    mapping_sample += f" (and {len(approach_details['mapping_examples']) - 3} more)"
                step_desc += f"Value mappings: {mapping_sample}. "
            
            # Add collapse examples
            if approach_details.get('collapse_examples'):
                collapse_sample = ', '.join(approach_details['collapse_examples'][:3])
                if len(approach_details['collapse_examples']) > 3:
                    collapse_sample += f" (and {len(approach_details['collapse_examples']) - 3} more)"
                step_desc += f"Multi-label collapses: {collapse_sample}. "
            
            # Add final categorization examples
            if approach_details.get('final_categorization_examples'):
                final_sample = ', '.join(approach_details['final_categorization_examples'][:3])
                if len(approach_details['final_categorization_examples']) > 3:
                    final_sample += f" (and {len(approach_details['final_categorization_examples']) - 3} more)"
                step_desc += f"Final categorizations: {final_sample}."
            
            processing_steps[f"step_{step_counter}_approach_cleaning"] = step_desc
        elif 'approach_cleaning' in processing_details:
            processing_steps[f"step_{step_counter}_approach_cleaning"] = f"Skipped 'llm_development_approach' column cleaning: {processing_details['approach_cleaning'].get('reason', 'Unknown reason')}"
        step_counter += 1
        
        # Add outlet and application cleaning step if it was performed
        if 'outlet_application_cleaning' in processing_details and processing_details['outlet_application_cleaning']['columns_cleaned']:
            outlet_app_details = processing_details['outlet_application_cleaning']
            step_desc = f"Cleaned outlet and application columns by mapping {outlet_app_details['total_values_mapped']:,} values "
            step_desc += f"and setting {outlet_app_details['total_other_unmapped']:,} unmapped 'Other:' values to 'Other' "
            step_desc += f"across {outlet_app_details['total_columns_processed']:,} columns: {', '.join(outlet_app_details['columns_processed'])}. "
            
            # Add column-specific details
            if outlet_app_details.get('column_details'):
                column_summaries = []
                for col, details in outlet_app_details['column_details'].items():
                    summary = f"{col} ({details['values_mapped']} mapped"
                    if details['other_unmapped_count'] > 0:
                        summary += f", {details['other_unmapped_count']} 'Other:' unmapped"
                    summary += ")"
                    column_summaries.append(summary)
                step_desc += f"Details: {', '.join(column_summaries)}. "
            
            # Add mapping examples
            if outlet_app_details.get('mapping_examples'):
                mapping_sample = ', '.join(outlet_app_details['mapping_examples'][:3])
                if len(outlet_app_details['mapping_examples']) > 3:
                    mapping_sample += f" (and {len(outlet_app_details['mapping_examples']) - 3} more)"
                step_desc += f"Mapping examples: {mapping_sample}. "
            
            # Add unmapped "Other: " examples
            if outlet_app_details.get('other_unmapped_examples'):
                unmapped_sample = ', '.join(outlet_app_details['other_unmapped_examples'][:3])
                if len(outlet_app_details['other_unmapped_examples']) > 3:
                    unmapped_sample += f" (and {len(outlet_app_details['other_unmapped_examples']) - 3} more)"
                step_desc += f"Unmapped 'Other:' examples: {unmapped_sample}."
            
            processing_steps[f"step_{step_counter}_outlet_application_cleaning"] = step_desc
        elif 'outlet_application_cleaning' in processing_details:
            processing_steps[f"step_{step_counter}_outlet_application_cleaning"] = f"Skipped outlet and application column cleaning: {processing_details['outlet_application_cleaning'].get('reason', 'Unknown reason')}"
        
        # Calculate column changes
        columns_added = len(df_final.columns) - len(df_original.columns)
        columns_removed = 0  # We don't remove columns in this script
        
        summary_stats = {
            "input_studies": processing_details['original_studies'],
            "input_rows": processing_details['original_rows'],
            "input_columns": len(df_original.columns),
            "output_studies": processing_details['final_studies'],
            "output_rows": processing_details['final_rows'],
            "output_columns": len(df_final.columns),
            "studies_removed": processing_details['studies_removed'],
            "rows_removed": processing_details['rows_removed'],
            "columns_added": columns_added,
            "columns_removed": columns_removed,
            "index_column": df_final.index.name if hasattr(df_final.index, 'name') else None,
            "execution_time_seconds": execution_time
        }
        
        # Add column creation details to summary stats if available
        if 'column_creation' in processing_details and processing_details['column_creation']['column_created']:
            col_details = processing_details['column_creation']
            summary_stats.update({
                "intended_intervention_type_created": True,
                "unique_intervention_types_original": col_details['unique_original_values'],
                "unique_intervention_types_transformed": col_details['unique_transformed_values'],
                "transformation_rules_applied": col_details['transformations_applied']
            })
        else:
            summary_stats["intended_intervention_type_created"] = False
        
        # Add inclusion criteria creation details to summary stats if available
        if 'inclusion_creation' in processing_details and processing_details['inclusion_creation']['column_created']:
            inclusion_details = processing_details['inclusion_creation']
            summary_stats.update({
                "inclusion_criteria_created": True,
                "unique_client_types_original": inclusion_details['unique_original_values'],
                "unique_inclusion_criteria_transformed": inclusion_details['unique_transformed_values'],
                "inclusion_transformation_rules_applied": inclusion_details['transformations_applied'],
                "inclusion_criteria_distribution": inclusion_details['criteria_distribution']
            })
        else:
            summary_stats["inclusion_criteria_created"] = False
        
        # Add primary clinical outcome details to summary stats if available
        if 'outcome_creation' in processing_details and processing_details['outcome_creation']['column_created']:
            outcome_details = processing_details['outcome_creation']
            summary_stats.update({
                "primary_clinical_outcome_created": True,
                "eligible_rows_for_outcome": outcome_details['total_eligible_rows'],
                "rows_with_outcome_filled": outcome_details['total_filled_rows'],
                "outcome_distribution": outcome_details['outcome_counts'],
                "studies_with_other_outcome": outcome_details['other_studies']
            })
        else:
            summary_stats["primary_clinical_outcome_created"] = False
        
        # Add models cleaning details to summary stats if available
        if 'models_cleaning' in processing_details and processing_details['models_cleaning']['column_cleaned']:
            models_details = processing_details['models_cleaning']
            
            # Calculate final model distribution
            final_model_distribution = {}
            if 'models_employed' in df_final.columns:
                all_models = []
                for models_entry in df_final['models_employed'].dropna():
                    if models_entry and str(models_entry).strip():
                        models = [model.strip() for model in str(models_entry).split(';') if model.strip()]
                        all_models.extend(models)
                
                if all_models:
                    from collections import Counter
                    model_counts = Counter(all_models)
                    final_model_distribution = dict(model_counts)
            
            summary_stats.update({
                "models_employed_cleaned": True,
                "non_generative_models_removed": models_details['total_non_gen_models_removed'],
                "non_gen_models_removed_list": sorted(list(set(models_details.get('non_gen_removed_examples', [])))),
                "models_mapped_count": models_details['total_models_mapped'],
                "mapping_examples_applied": models_details.get('mapping_examples', []),
                "models_unmapped_count": models_details['total_models_unmapped'],
                "unmapped_models_set_to_other": models_details.get('unmapped_models_with_duplicates', models_details.get('unmapped_models', [])),
                "final_models_distribution": final_model_distribution,
                "unique_models_after_cleaning": len(final_model_distribution)
            })
        else:
            summary_stats["models_employed_cleaned"] = False
        
        # Add llm_development_approach cleaning details to summary stats if available
        if 'approach_cleaning' in processing_details and processing_details['approach_cleaning']['column_cleaned']:
            approach_details = processing_details['approach_cleaning']
            
            # Calculate final approach distribution
            final_approach_distribution = {}
            if 'llm_development_approach' in df_final.columns:
                all_approaches = []
                for approach_entry in df_final['llm_development_approach'].dropna():
                    if approach_entry and str(approach_entry).strip():
                        approaches = [approach.strip() for approach in str(approach_entry).split(';') if approach.strip()]
                        all_approaches.extend(approaches)
                
                if all_approaches:
                    from collections import Counter
                    approach_counts = Counter(all_approaches)
                    final_approach_distribution = dict(approach_counts)
            
            summary_stats.update({
                "llm_development_approach_cleaned": True,
                "approach_values_mapped": approach_details['total_values_mapped'],
                "approach_entries_collapsed": approach_details['total_entries_collapsed'],
                "approach_entries_final_categorized": approach_details['total_entries_final_categorized'],
                "approach_mapping_examples": approach_details.get('mapping_examples', []),
                "approach_collapse_examples": approach_details.get('collapse_examples', []),
                "approach_final_categorization_examples": approach_details.get('final_categorization_examples', []),
                "final_approach_distribution": final_approach_distribution,
                "unique_approaches_after_cleaning": len(final_approach_distribution)
            })
        else:
            summary_stats["llm_development_approach_cleaned"] = False
        
        # Add outlet and application cleaning details to summary stats if available
        if 'outlet_application_cleaning' in processing_details and processing_details['outlet_application_cleaning']['columns_cleaned']:
            outlet_app_details = processing_details['outlet_application_cleaning']
            
            summary_stats.update({
                "outlet_application_columns_cleaned": True,
                "outlet_app_columns_processed": outlet_app_details['columns_processed'],
                "outlet_app_total_values_mapped": outlet_app_details['total_values_mapped'],
                "outlet_app_total_other_unmapped": outlet_app_details['total_other_unmapped'],
                "outlet_app_mapping_examples": outlet_app_details.get('mapping_examples', []),
                "outlet_app_other_unmapped_examples": outlet_app_details.get('other_unmapped_examples', []),
                "outlet_app_column_details": outlet_app_details.get('column_details', {})
            })
        else:
            summary_stats["outlet_application_columns_cleaned"] = False
        
        report = generate_processing_report(
            script_name="7_perform_final_cleaning.py",
            input_file=input_path,
            output_file=output_path,
            processing_steps=processing_steps,
            summary_stats=summary_stats,
            execution_time=execution_time
        )
        
        # Save report
        reports_dir = Path(__file__).parent.parent.parent.parent / "results" / "reports_data_processing"
        report_filename = "7_perform_final_cleaning"
        report_path = save_report_to_file(report, str(reports_dir), report_filename, save_json=False)
        
        if args.verbose:
            print(f"✓ Successfully saved cleaned data to {output_path}")
            print(f"✓ Processing report saved to {report_path}")
            print(f"✓ Final cleaning complete!")
            print(f"\nFinal dataset: {len(df_final):,} rows with {len(df_final.columns):,} columns")
            if hasattr(df_final.index, 'name') and df_final.index.name:
                print(f"Index: {df_final.index.name} ({len(df_final.index):,} entries)")
            print(f"Execution time: {execution_time:.2f} seconds")
        else:
            print(f"Final cleaning complete. Output saved to: {output_path}")
            print(f"Processing report saved to: {report_path}")
            
    except Exception as e:
        print(f"Error during cleaning: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

