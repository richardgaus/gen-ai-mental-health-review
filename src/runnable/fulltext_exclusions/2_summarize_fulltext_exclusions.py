#!/usr/bin/env python3
"""
Reasons for the full-text exclusions (Figure 1, box "Records excluded").

Figure 1 reports 258 full texts screened and 172 records retained (the studies entering data
charting, see data cleaning steps 3 and 4), so 86 records were excluded. This script assigns one
exclusion reason to each of them, combining three sources:
    1. the full-text exclusions recorded in Covidence (output of script 1),
    2. the reasons assigned on 18 October 2026 (data/unprocessed/fulltext_exclusion_reasons_assigned_20261018.xlsx):
       rows whose title matches a Covidence exclusion without a recorded reason ("no reason") replace
       that reason; the other rows are excluded records not among the Covidence exclusions,
    3. records charted in both Covidence reviews: the copy in the 2nd search was removed as a duplicate
       (same title rule as data cleaning step 3).
The Covidence reasons are then grouped into the reasons shown in Figure 1.

Outputs (results/fulltext_exclusions/):
    fulltext_exclusions_by_record.csv: one row per excluded record with its reason and Figure 1 group
    fulltext_exclusion_reasons.csv: number of records per Figure 1 group and per Covidence reason
    fulltext_exclusions_report.txt: summary and the text of the Figure 1 box

Usage:
    python 2_summarize_fulltext_exclusions.py [--excluded FILE] [--assigned FILE] [--output FOLDER]
"""

import argparse
from pathlib import Path

import pandas as pd

FULL_TEXTS_SCREENED = 258  # Figure 1
NO_REASON = "no reason"
DUPLICATE = "duplicate record"

# Figure 1 groups (in display order) and the Covidence reasons they contain
REASON_GROUPS = {
    "Preprint, thesis or book": ["preprint", "thesis", "book chapter"],
    "No empirical evaluation": ["No evaluation", "opinion/conceptual"],
    "No mental health support focus": ["not specifically psychotherapy/mental health support"],
    "No LLM used": ["no LLM used"],
    "Not in English": ["non-english"],
    "Full text not retrievable": ["not retrieved"],
    "Duplicate record": [DUPLICATE],
}


def normalize(title: pd.Series) -> pd.Series:
    return title.fillna("").str.strip().str.lower()


def add_assigned_reasons(excluded: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    """Replace missing Covidence reasons by the assigned ones and add the records missing from Covidence."""
    excluded = excluded.copy()
    excluded["reason_source"] = "covidence"
    titles = normalize(excluded["title"])
    added = []
    for _, row in assigned.iterrows():
        match = titles == row["title"].strip().lower()
        if match.sum() > 1:
            raise ValueError(f"Assigned title matches several excluded records: {row['title']}")
        if match.sum() == 1:
            if excluded.loc[match, "exclusion_reason"].item() != NO_REASON:
                raise ValueError(f"Assigned title matches a record that already has a reason: {row['title']}")
            excluded.loc[match, ["exclusion_reason", "reason_source"]] = [row["exclusion_reason"], "assigned_20261018"]
        else:
            added.append({"study": row["study"], "title": row["title"], "doi": row["doi"],
                          "exclusion_reason": row["exclusion_reason"], "record_source": "assigned_20261018",
                          "reason_source": "assigned_20261018"})
    return pd.concat([excluded, pd.DataFrame(added)], ignore_index=True)


def find_duplicates(first_search: pd.DataFrame, second_search: pd.DataFrame) -> pd.DataFrame:
    """Records charted in both Covidence reviews; the 2nd search copy was removed (data cleaning step 3)."""
    duplicates = second_search[normalize(second_search["Title"]).isin(set(normalize(first_search["Title"])))]
    duplicates = duplicates.drop_duplicates("Covidence #")
    return pd.DataFrame({"covidence_number": "#" + duplicates["Covidence #"].str.lstrip("#"),
                         "study": duplicates["Study ID"], "title": duplicates["Title"],
                         "exclusion_reason": DUPLICATE, "record_source": "covidence_export_2nd_search_20251030",
                         "reason_source": "data_cleaning_step_3"})


def main():
    """Main function to summarize the full-text exclusion reasons."""
    parser = argparse.ArgumentParser(description="Reasons for the full-text exclusions (Figure 1)")
    parser.add_argument("--excluded", default="data/intermediate/covidence_fulltext_excluded.csv",
                        help="Full-text exclusions recorded in Covidence (output of script 1)")
    parser.add_argument("--assigned", default="data/unprocessed/fulltext_exclusion_reasons_assigned_20261018.xlsx",
                        help="Exclusion reasons assigned on 18 October 2026")
    parser.add_argument("--first-search", default="data/unprocessed/covidence_export_1st_search_20251030.csv",
                        help="Covidence data extraction export of the 1st search")
    parser.add_argument("--second-search", default="data/unprocessed/covidence_export_2nd_search_20251030.csv",
                        help="Covidence data extraction export of the 2nd search")
    parser.add_argument("--retained", default="data/intermediate/fused_data_cleaned.csv",
                        help="Records retained after full-text screening (output of data cleaning step 4)")
    parser.add_argument("--output", default="results/fulltext_exclusions", help="Output folder")
    args = parser.parse_args()

    excluded = pd.read_csv(args.excluded, dtype=str).fillna("")
    assigned = pd.read_excel(args.assigned, sheet_name="Coding", dtype=str).fillna("")
    first_search = pd.read_csv(args.first_search, dtype=str)
    second_search = pd.read_csv(args.second_search, dtype=str)
    retained = pd.read_csv(args.retained, dtype=str)["covidence_id"].nunique()

    records = add_assigned_reasons(excluded, assigned)
    records = pd.concat([records, find_duplicates(first_search, second_search)], ignore_index=True).fillna("")

    # Checks: one reason per record, every reason in a Figure 1 group, total matches Figure 1
    if (records["exclusion_reason"] == NO_REASON).any():
        raise ValueError(f"Records without reason: {records.loc[records['exclusion_reason'] == NO_REASON, 'title'].tolist()}")
    if normalize(records["title"]).duplicated().any():
        raise ValueError(f"Records listed twice: {records.loc[normalize(records['title']).duplicated(), 'title'].tolist()}")
    group_of = {reason: group for group, reasons in REASON_GROUPS.items() for reason in reasons}
    unmapped = sorted(set(records["exclusion_reason"]) - set(group_of))
    if unmapped:
        raise ValueError(f"Reasons not in a Figure 1 group: {unmapped}")
    if len(records) != FULL_TEXTS_SCREENED - retained:
        raise ValueError(f"{len(records)} excluded records, but Figure 1 implies {FULL_TEXTS_SCREENED} - {retained} = "
                         f"{FULL_TEXTS_SCREENED - retained}")
    records["figure_1_group"] = records["exclusion_reason"].map(group_of)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records["group_order"] = records["figure_1_group"].map({group: i for i, group in enumerate(REASON_GROUPS)})
    records = records.sort_values(["group_order", "exclusion_reason", "title"]).drop(columns="group_order")
    records.to_csv(output / "fulltext_exclusions_by_record.csv", index=False, encoding="utf-8")

    counts = (records.groupby(["figure_1_group", "exclusion_reason"], sort=False).size()
              .rename("n").reset_index())
    counts.insert(1, "group_n", counts["figure_1_group"].map(records["figure_1_group"].value_counts()))
    counts.to_csv(output / "fulltext_exclusion_reasons.csv", index=False)

    box = [f"Records excluded (n = {len(records)}):"]
    box += [f"• {group} (n = {(records['figure_1_group'] == group).sum()})" for group in REASON_GROUPS]
    lines = ["FULL-TEXT EXCLUSION REASONS (Figure 1)", "=" * 80, "",
             f"Full texts screened: {FULL_TEXTS_SCREENED}", f"Records retained (data cleaning step 4): {retained}",
             f"Records excluded: {len(records)}", "",
             "Sources of the excluded records:"]
    lines += [f"  {source}: {n}" for source, n in records["record_source"].value_counts().items()]
    lines += ["", "Sources of the exclusion reasons:"]
    lines += [f"  {source}: {n}" for source, n in records["reason_source"].value_counts().items()]
    lines += ["", "Reasons by Figure 1 group:", counts.to_string(index=False), "", "Figure 1 box:"] + box
    (output / "fulltext_exclusions_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
