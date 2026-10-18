#!/usr/bin/env python3
"""
Sensitivity analysis by participation mode (Supplementary Table S5).

This script compares the study characteristics shown in Figure 5 and the
READI-derived criteria between all articles evaluating client-facing
applications with human participants and the subset in which participants
used the application (excluding articles in which participants only rated
LLM-generated outputs; participation_mode, added in data cleaning step 10).

Usage:
    python 8_sensitivity_participation_mode.py [--input FILENAME] [--output FOLDER]
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.figure_creation.panel_clinical import has_human_participants, analyze_human_participation, categorize_sample_size

SAMPLE_SIZE_LABELS = {  # Figure 5B
    "Case studies": "Case study (n = 1)", "Very small": "Very small (2–20)", "Small": "Small (21–100)",
    "Medium": "Medium (101–500)", "Large": "Large (501–2,000)", "Very large": "Very large (> 2,000)",
}
INCLUSION_LABELS = {  # Figure 5B
    "Unspecified": "Inclusion criteria not specified",
    "Diagnosed mental disorder (ICD/DSM)": "Diagnosed mental disorder (ICD/DSM)",
    "Recruited in in- or outpatient healthcare setting with no specific criteria": "No specific criteria, in- or outpatient recruitment setting",
    "Elevated symptoms": "Elevated symptoms",
    "No specific criteria": "No specific criteria, non-clinical recruitment setting",
}
OUTCOME_ORDER = ["Validated symptom/function scale", "User experience assessment", "Other"]  # Figure 5C
READI_CRITERIA = {  # Figure 4 labels; definitions in Supplementary Notes S5.1
    "S-1: Risk-detection": "s1_risk_detection_considered",
    "S-2: Content-safety evaluation": "s2_content_safety_considered",
    "P-1: On-premise-capable model": "p1_on_premise_model_considered",
    "P-2: Privacy/confidentiality awareness": "p2_privacy_awareness_considered",
    "E-1: Demographic reporting": "e1_demographics_reporting_considered",
    "E-2: Subgroup outcomes": "e2_outcomes_by_demographics_considered",
    "G-1: Early-discontinuation data": "g1_early_discontinuation_considered",
    "G-2: Over-use reported/prevented": "g2_overuse_considered",
    "F-1: Validated clinical measures": "f1_validated_outcomes_considered",
    "F-2: Control condition": "f2_control_condition_considered",
    "I-1: Diverse stakeholder feasibility": "i1_multilevel_feasibility_considered",
    "I-2: Healthcare integration": "i2_healthcare_integration_considered",
}


def count(mask: pd.Series, total: int) -> str:
    """Format a count with its percentage of the total."""
    return f"{int(mask.sum())} ({100 * mask.sum() / total:.0f}%)"


def is_yes(column: pd.Series) -> pd.Series:
    return column.fillna("").str.strip().str.lower() == "y"


def summarize(df: pd.DataFrame, interventions: list) -> dict:
    """Compute the Figure 5 characteristics and READI criteria for one group of articles."""
    n = len(df)
    rows = {}

    sizes = analyze_human_participation(df)["sample_sizes"]
    categories = pd.Series([categorize_sample_size(s, {}) for s in sizes])
    rows[("Sample size", "Median (range)")] = f"{pd.Series(sizes).median():g} ({min(sizes)}–{max(sizes):,})"
    rows[("Sample size", "Fewer than 100 participants")] = count(pd.Series([s < 100 for s in sizes]), len(sizes))
    for category, label in SAMPLE_SIZE_LABELS.items():
        rows[("Sample size", label)] = count(categories == category, len(sizes))

    for value, label in INCLUSION_LABELS.items():
        rows[("Inclusion criteria", label)] = count(df["inclusion_criteria"] == value, n)

    for intervention in interventions:
        rows[("Intervention type", intervention)] = count(df["intended_intervention_type"] == intervention, n)

    rows[("Control group", "Control group present")] = count(is_yes(df["f2_control_condition_considered"]), n)

    for outcome in OUTCOME_ORDER:
        rows[("Primary outcome", outcome)] = count(df["primary_clinical_outcome"] == outcome, n)

    met = sum(is_yes(df[column]).astype(int) for column in READI_CRITERIA.values())
    for label, column in READI_CRITERIA.items():
        rows[("READI-derived criteria", label)] = count(is_yes(df[column]), n)
    rows[("READI-derived criteria", "Criteria met (of 12), mean (median, range)")] = f"{met.mean():.1f} ({met.median():g}, {met.min()}–{met.max()})"
    rows[("READI-derived criteria", "No criterion met")] = count(met == 0, n)
    return rows


def in_table_1(df: pd.DataFrame) -> pd.Series:
    """Criteria of the controlled clinical studies in Table 1 (sample size > 10, control group, validated outcome)."""
    size = pd.to_numeric(df["client_count"], errors="coerce")
    return (size > 10) & is_yes(df["f2_control_condition_considered"]) & is_yes(df["f1_validated_outcomes_considered"])


def main():
    """Main function to run the sensitivity analysis."""
    parser = argparse.ArgumentParser(description="Sensitivity analysis by participation mode (Supplementary Table S5)")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/final_data.csv",
        help="Path to input CSV file (default: data/processed/final_data.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/8_sensitivity_participation_mode",
        help="Output folder (default: results/8_sensitivity_participation_mode)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "participation_mode" not in df.columns:
        sys.exit("Error: column participation_mode not found; run data cleaning step 10 first.")

    is_human = (df["application_type"] == "Client-facing application") \
        & (df["study_type"] == "Empirical research involving an LLM") \
        & df["client_type"].apply(has_human_participants)
    all_human = df[is_human]
    used_only = all_human[all_human["participation_mode"] == "used"]

    # Intervention types in the order of Figure 5C (descending frequency, "Other" last)
    interventions = [i for i in all_human["intended_intervention_type"].value_counts().index if i != "Other"] + ["Other"]
    table = pd.DataFrame({
        f"All articles with human participants (n = {len(all_human)})": summarize(all_human, interventions),
        f"Participants used the application (n = {len(used_only)})": summarize(used_only, interventions),
    })
    table.index.names = ["Section", "Characteristic"]

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / "table_s5_sensitivity_participation_mode.csv")

    excluded = all_human[all_human["participation_mode"] == "rated"]
    report = ["SENSITIVITY ANALYSIS BY PARTICIPATION MODE", "=" * 80, "",
              "Excluded articles (participants rated LLM-generated outputs without using an application):"]
    report += [f"  {row.study_id}: {row.title} (n = {row.client_count})" for row in excluded.itertuples()]
    report += ["", f"Excluded articles meeting the Table 1 criteria: {int(in_table_1(excluded).sum())} "
               f"(all articles: {int(in_table_1(all_human).sum())})",
               f"Articles without reported sample size: "
               f"{', '.join(all_human.loc[pd.to_numeric(all_human['client_count'], errors='coerce').isna(), 'study_id'])}",
               "", table.to_string(), ""]
    (output_folder / "8_sensitivity_participation_mode_report.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
