#!/usr/bin/env python3
"""
Add classifications made during peer review to final_data.csv.

This script adds two classifications to final_data.csv as new columns. All other
columns of final_data.csv are left unchanged.

1. participation_mode (Supplementary Notes S5.3): for articles evaluating client-facing
   applications with human participants, whether participants used the application
   ("used") or only rated LLM-generated outputs ("rated"). Read from an Excel file with
   one row per article.
2. design_purpose and design_purpose_basis (Supplementary Notes S5.2): for articles
   evaluating client-facing applications, the design-purpose group ("General-purpose only",
   "Purpose-built", "Companion") and whether it was derived from the charted development
   approach or from dual coding. Combined from the charted development approach (Step 1),
   the Covidence export of the dual coding (Step 2) and the companion screen coding sheet
   with the hits of the full-text search (Step 3).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.figure_creation.panel_clinical import has_human_participants
from utils.data_cleaning.design_purpose_utils import (load_step_2_coding, load_companion_coding, classify_design_purpose,
                                                      GENERAL_PURPOSE_ONLY, PURPOSE_BUILT, COMPANION)

PARTICIPATION_MODES = {"used", "rated"}


def add_participation_mode(df: pd.DataFrame, coding: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the participation mode coding into the final data.

    Raises an error if the coded articles do not match exactly the articles
    evaluating client-facing applications with human participants, or if the
    coding is incomplete.
    """
    is_human = (df["application_type"] == "Client-facing application") & df["client_type"].apply(has_human_participants)
    expected_ids = set(df.loc[is_human, "study_id"])
    coded_ids = set(coding["study_id"])

    if coding["study_id"].duplicated().any():
        raise ValueError(f"Duplicate study_ids in coding: {sorted(coding.loc[coding['study_id'].duplicated(), 'study_id'])}")
    if coded_ids != expected_ids:
        raise ValueError(f"Coded study_ids do not match articles with human participants. "
                         f"Missing: {sorted(expected_ids - coded_ids)}; unexpected: {sorted(coded_ids - expected_ids)}")
    invalid = coding[~coding["participation_mode"].isin(PARTICIPATION_MODES)]
    if len(invalid):
        raise ValueError(f"Invalid participation_mode for: {invalid['study_id'].tolist()}")
    missing_coder = coding[(coding["classified_by"] == "") | (coding["verified_by"] == "")]
    if len(missing_coder):
        raise ValueError(f"Missing classified_by or verified_by for: {missing_coder['study_id'].tolist()}")

    df = df.drop(columns=["participation_mode"], errors="ignore")  # re-running gives the same result
    df["participation_mode"] = df["study_id"].map(dict(zip(coding["study_id"], coding["participation_mode"]))).fillna("")
    return df


def add_design_purpose(df: pd.DataFrame, design_purpose: pd.DataFrame) -> pd.DataFrame:
    """Merge the design purpose classification into the final data (empty for other application types)."""
    df = df.drop(columns=["design_purpose", "design_purpose_basis"], errors="ignore")  # re-running gives the same result
    for column in ["design_purpose", "design_purpose_basis"]:
        df[column] = df["study_id"].map(design_purpose[column]).fillna("")
    return df


def generate_report(df: pd.DataFrame, coding: pd.DataFrame, input_path: Path, coding_path: Path,
                    design_purpose_paths: list) -> str:
    """Generate a text report of the added classifications."""
    lines = ["Peer Review Classifications Report", "=" * 50, "",
             f"Input/output file: {input_path}", f"Participation mode coding: {coding_path}",
             f"Design purpose coding: {', '.join(str(p) for p in design_purpose_paths)}", "",
             "Participation mode (articles evaluating client-facing applications with human participants)",
             f"  Articles coded: {len(coding)}"]
    for mode in ["used", "rated"]:
        lines.append(f"  {mode}: {(df['participation_mode'] == mode).sum()}")
    lines.append(f"  Classified by: {', '.join(sorted(coding['classified_by'].unique()))}; "
                 f"verified by: {', '.join(sorted(coding['verified_by'].unique()))}")
    lines += ["", "  Articles with participation_mode = rated:"]
    for _, row in coding[coding["participation_mode"] == "rated"].iterrows():
        lines.append(f"    {row['study_id']} ({row['first_author']} {row['year']}): {row['notes']}")

    client_facing = df[df["application_type"] == "Client-facing application"]
    human = client_facing["client_type"].apply(has_human_participants)
    groups = [GENERAL_PURPOSE_ONLY, PURPOSE_BUILT, COMPANION]
    lines += ["", "Design purpose (articles evaluating client-facing applications)",
              f"  {'Group':<22}{'All':>8}{'With human participants':>26}{'Without human participants':>29}"]
    for group in groups + ["Total"]:
        mask = client_facing["design_purpose"].isin(groups) if group == "Total" else client_facing["design_purpose"] == group
        lines.append(f"  {group:<22}{mask.sum():>8}{(mask & human).sum():>26}{(mask & ~human).sum():>29}")
    lines.append("  Basis: " + ", ".join(f"{basis} {n}" for basis, n in client_facing["design_purpose_basis"].value_counts().items()))
    for column, label in [("llm_development_approach", "development approach"),
                          ("application_subtype_client_facing", "client-facing application subtype (interaction format)")]:
        table = pd.crosstab(client_facing[column].replace("", "(not specified)"), client_facing["design_purpose"],
                            margins=True, margins_name="Total")[groups + ["Total"]]
        lines += ["", f"  Design purpose by {label}:"] + ["    " + line for line in table.to_string().splitlines()]
    lines += ["", "  Articles classified as companion:"]
    for _, row in client_facing[client_facing["design_purpose"] == COMPANION].iterrows():
        lines.append(f"    {row['study_id']}: {row['title']}")
    return "\n".join(lines) + "\n"


def main():
    """Main function to add peer review classifications."""
    parser = argparse.ArgumentParser(description="Add classifications made during peer review to final_data.csv.")
    parser.add_argument(
        "--input",
        required=True,
        help="Relative path to final_data.csv (columns are added in place)"
    )
    parser.add_argument(
        "--participation-mode",
        required=True,
        help="Relative path to the Excel file with the participation mode coding"
    )
    parser.add_argument(
        "--design-purpose-coding",
        required=True,
        help="Relative path to the Covidence export of the design purpose dual coding (Step 2)"
    )
    parser.add_argument(
        "--companion-coding",
        required=True,
        help="Relative path to the Excel file with the companion screen coding (Step 3)"
    )
    parser.add_argument(
        "--companion-hits",
        required=True,
        help="Relative path to the hits of the full-text search for companion applications (Step 3)"
    )

    args = parser.parse_args()

    try:
        # Convert relative paths to absolute paths
        project_root = Path(__file__).parent.parent.parent.parent
        input_path = project_root / args.input
        coding_path = project_root / args.participation_mode
        design_purpose_paths = [project_root / args.design_purpose_coding, project_root / args.companion_coding,
                                project_root / args.companion_hits]

        print(f"Input/output file: {input_path}")
        print(f"Participation mode coding: {coding_path}")

        # Read everything as strings so that the existing columns are written back unchanged
        df = pd.read_csv(input_path, index_col=0, dtype=str, keep_default_na=False)
        coding = pd.read_excel(coding_path, sheet_name="Coding", dtype=str).fillna("")
        coding = coding.apply(lambda col: col.str.strip())

        df = add_participation_mode(df, coding)

        step_2 = load_step_2_coding(design_purpose_paths[0], df)
        companion = load_companion_coding(design_purpose_paths[1])
        hits = pd.read_csv(design_purpose_paths[2], dtype=str, keep_default_na=False)
        df = add_design_purpose(df, classify_design_purpose(df, step_2, companion, set(hits["study_id"])))
        df.to_csv(input_path)

        report = generate_report(df, coding, input_path, coding_path, design_purpose_paths)
        reports_folder = project_root / "results" / "reports_data_processing"
        reports_folder.mkdir(parents=True, exist_ok=True)
        report_path = reports_folder / "10_add_peer_review_classifications.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        print(report)
        print(f"Successfully added peer review classifications to: {input_path}")
        print(f"Report saved to: {report_path}")

    except Exception as e:
        print(f"Error while adding peer review classifications: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
