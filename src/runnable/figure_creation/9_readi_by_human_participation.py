#!/usr/bin/env python3
"""
READI-derived criteria by human participation (Results, Figure 4 paragraph).

This script reports READI adoption separately for articles evaluating client-facing
applications with and without human participants. Criteria marked as applying only to
articles with human participants (readi_only_human_participants in figure_config.yaml;
asterisked in Figure 4) count only for articles with human participants, as in Figure 4.
Per group, it reports the number of applicable criteria, the criteria met per article
(mean, median, range), the share of applicable criteria met, the articles meeting no
criterion and the adoption of each criterion. As a check, the share of criteria met is
also reported over the criteria that apply to both groups.

Usage:
    python 9_readi_by_human_participation.py [--input FILENAME] [--output FOLDER]
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.figure_creation.panel_clinical import has_human_participants
from utils.figure_creation.panel_readi import load_figure_config

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


def is_yes(column: pd.Series) -> pd.Series:
    return column.fillna("").astype(str).str.strip().str.lower().isin(["y", "yes"])


def count(mask: pd.Series, total: int) -> str:
    """Format a count with its percentage of the total."""
    return f"{int(mask.sum())} ({100 * mask.sum() / total:.0f}%)"


def summarize(df: pd.DataFrame, criteria: dict, shared: dict) -> dict:
    """Compute READI adoption for one group of articles over its applicable criteria."""
    n = len(df)
    met = sum(is_yes(df[column]).astype(int) for column in criteria.values())
    met_shared = sum(is_yes(df[column]).astype(int) for column in shared.values())
    rows = {
        ("Overview", "Articles"): str(n),
        ("Overview", "Applicable criteria"): str(len(criteria)),
        ("Criteria met", "Mean (median, range)"): f"{met.mean():.1f} ({met.median():g}, {met.min()}–{met.max()})",
        ("Criteria met", "Share of applicable criteria met"): f"{100 * met.mean() / len(criteria):.0f}%",
        ("Criteria met", "No criterion met"): count(met == 0, n),
        ("Criteria met", f"Share of the {len(shared)} shared criteria met"): f"{100 * met_shared.mean() / len(shared):.0f}%",
    }
    for label, column in READI_CRITERIA.items():
        rows[("Adoption per criterion", label)] = count(is_yes(df[column]), n) if label in criteria else "n/a"
    return rows


def main():
    """Main function to report READI adoption by human participation."""
    parser = argparse.ArgumentParser(description="READI-derived criteria by human participation")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/final_data.csv",
        help="Path to input CSV file (default: data/processed/final_data.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/9_readi_by_human_participation",
        help="Output folder (default: results/9_readi_by_human_participation)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    client_facing = df[(df["application_type"] == "Client-facing application")
                       & (df["study_type"] == "Empirical research involving an LLM")]
    is_human = client_facing["client_type"].apply(has_human_participants)
    with_human, without_human = client_facing[is_human], client_facing[~is_human]

    human_only = set(load_figure_config().get("readi_only_human_participants", []))
    shared = {label: column for label, column in READI_CRITERIA.items() if column not in human_only}
    table = pd.DataFrame({
        f"With human participants (n = {len(with_human)})": summarize(with_human, READI_CRITERIA, shared),
        f"Without human participants (n = {len(without_human)})": summarize(without_human, shared, shared),
    })
    table.index.names = ["Section", "Characteristic"]

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_folder / "readi_by_human_participation.csv")

    report = ["READI-DERIVED CRITERIA BY HUMAN PARTICIPATION", "=" * 80, "",
              f"Articles evaluating client-facing applications: {len(client_facing)}",
              f"Criteria applying only to articles with human participants: "
              f"{', '.join(label.split(':')[0] for label, column in READI_CRITERIA.items() if column in human_only)}",
              "", table.to_string(), ""]
    (output_folder / "9_readi_by_human_participation_report.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
