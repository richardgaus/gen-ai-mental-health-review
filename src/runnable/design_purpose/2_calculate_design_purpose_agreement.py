#!/usr/bin/env python3
"""
Inter-rater agreement for the design purpose classification (Supplementary Notes S5.2.2 and S5.2.3).

This script calculates percentage agreement and Cohen's kappa between the two reviewers for
    Step 2: the three yes/no items coded for the articles with a prompting-only or other
            development approach (Covidence export), and
    Step 3: the verification of the articles with hits in the full-text search for companion
            applications (companion screen coding sheet).
It also compares the result of the full-text search with the companion item of the dual-coded
articles (Step 3 vs Step 2).
Kappa is calculated with the same function as for the data charting (data cleaning step 6).
The report is saved to results/reports_data_processing/design_purpose_agreement.txt.

Usage:
    python 2_calculate_design_purpose_agreement.py [--input FILE] [--design-purpose-coding FILE] [--companion-coding FILE]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.interrater_agreement import calculate_cohens_kappa
from utils.data_cleaning.design_purpose_utils import ITEMS, load_step_2_coding, load_companion_coding


def agreement_lines(label: str, reviewer_1: pd.Series, reviewer_2: pd.Series, names: pd.Series) -> list:
    """Format agreement, kappa, positive codes and disagreements for one item."""
    result = calculate_cohens_kappa(reviewer_1.reset_index(drop=True), reviewer_2.reset_index(drop=True), label)
    n_positive = [int((r == "yes").sum()) for r in (reviewer_1, reviewer_2)]
    lines = [f"{label}",
             f"  Agreement: {result['agreements']} of {result['n_pairs']} ({100 * result['agreement_rate']:.1f}%)",
             f"  Cohen's kappa: {result['kappa']:.2f}",
             f"  Coded yes: Reviewer 1 {n_positive[0]}, Reviewer 2 {n_positive[1]}"]
    if min(n_positive) <= 1:
        lines.append("  Note: kappa is not reported for this item in the supplement (at most one article coded yes)")
    disagreements = names[(reviewer_1 != reviewer_2).values]
    lines.append(f"  Disagreements: {'; '.join(disagreements) if len(disagreements) else 'none'}")
    return lines + [""]


def main():
    """Main function to calculate the inter-rater agreement of the design purpose classification."""
    parser = argparse.ArgumentParser(description="Inter-rater agreement for the design purpose classification")
    parser.add_argument("--input", default="data/processed/final_data.csv",
                        help="Path to final_data.csv (default: data/processed/final_data.csv)")
    parser.add_argument("--design-purpose-coding", default="data/unprocessed/covidence_export_design_purpose_20261016.csv",
                        help="Covidence export of the Step 2 coding")
    parser.add_argument("--companion-coding", default="data/unprocessed/companion_screen_coding.xlsx",
                        help="Coding sheet of the Step 3 verification")
    parser.add_argument("--output", default="results/reports_data_processing/design_purpose_agreement.txt",
                        help="Path of the report")
    args = parser.parse_args()

    final_df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    step_2 = load_step_2_coding(args.design_purpose_coding, final_df)
    companion = load_companion_coding(args.companion_coding)

    lines = ["DESIGN PURPOSE CLASSIFICATION: INTER-RATER AGREEMENT", "=" * 80, ""]

    # Step 2: one row per article, items of Reviewer 1 and Reviewer 2 side by side
    wide = step_2[step_2["reviewer"] != "Consensus"].pivot(index="study_id", columns="reviewer", values=list(ITEMS))
    titles = step_2.drop_duplicates("study_id").set_index("study_id")["title"]
    names = pd.Series([f"{i} ({titles[i][:60]})" for i in wide.index])
    lines += [f"Step 2: independent coding of {len(wide)} articles (Covidence export: {args.design_purpose_coding})", ""]
    for item, label in ITEMS.items():
        lines += agreement_lines(label, wide[(item, "Reviewer 1")], wide[(item, "Reviewer 2")], names)
    pooled_1 = pd.concat([wide[(item, "Reviewer 1")] for item in ITEMS])
    pooled_2 = pd.concat([wide[(item, "Reviewer 2")] for item in ITEMS])
    pooled = calculate_cohens_kappa(pooled_1.reset_index(drop=True), pooled_2.reset_index(drop=True), "pooled")
    identical = sum((wide[(item, "Reviewer 1")] == wide[(item, "Reviewer 2")]) for item in ITEMS) == len(ITEMS)
    lines += [f"Pooled over the three items: {pooled['agreements']} of {pooled['n_pairs']} ({100 * pooled['agreement_rate']:.1f}%), "
              f"Cohen's kappa {pooled['kappa']:.2f}",
              f"Articles with identical codes on all three items: {int(identical.sum())} of {len(wide)}", ""]

    consensus = step_2[step_2["reviewer"] == "Consensus"]
    combinations = consensus[list(ITEMS)].apply(lambda r: " + ".join(ITEMS[i] for i in ITEMS if r[i] == "yes"), axis=1)
    lines += ["Consensus combinations:"] + [f"  {c}: {n}" for c, n in combinations.value_counts().items()] + [""]

    # Step 3: verification of the articles with search hits
    wide = companion.pivot(index="study_id", columns="reviewer", values="companion")
    names = pd.Series(companion.drop_duplicates("study_id").set_index("study_id").loc[wide.index, "first_author_year"].values)
    lines += [f"Step 3: verification of {len(wide)} articles with hits in the full-text search ({args.companion_coding})", ""]
    lines += agreement_lines("Evaluates a companion application", wide["Reviewer 1"], wide["Reviewer 2"], names)
    lines += [f"Consensus yes: {', '.join(names[(wide['Consensus'] == 'yes').values]) or 'none'}", ""]

    # Step 3 vs Step 2: search result for the dual-coded articles compared with their companion item
    step_2_companion = consensus.set_index("study_id")["companion"]
    search_companion = step_2_companion.index.map(lambda i: wide["Consensus"].get(i, "no"))  # no hit = no
    search_companion = pd.Series(search_companion, index=step_2_companion.index)
    both = step_2_companion[(step_2_companion == "yes") & (search_companion == "yes")].index
    n_agree = int((step_2_companion == search_companion).sum())
    lines += [f"Step 3 vs Step 2: companion item of the {len(step_2_companion)} dual-coded articles",
              "  Search result (consensus; no hit = no) vs Step 2 consensus",
              f"  Agreement: {n_agree} of {len(step_2_companion)} ({100 * n_agree / len(step_2_companion):.1f}%)",
              f"  Companion in both: {'; '.join(f'{i} ({titles[i][:60]})' for i in both) or 'none'}",
              f"  Disagreements: {'; '.join(step_2_companion.index[step_2_companion != search_companion]) or 'none'}",
              f"  Of the {len(step_2_companion)}, articles with search hits: {int(step_2_companion.index.isin(wide.index).sum())}"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
