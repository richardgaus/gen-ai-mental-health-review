"""
Utilities for the design purpose classification (Supplementary Notes S5.2).

The classification was added during peer review and proceeds in three steps:
    Step 1: articles whose charted development approach involves fine-tuning or additional modules
            are classified as evaluating a purpose-built application (from final_data.csv).
    Step 2: the remaining articles ("Only prompting" or "Other") were coded independently by two
            reviewers on three yes/no items (Covidence export).
    Step 3: all articles were searched for companion applications; every article with a hit was
            verified by two reviewers (companion screen coding sheet).
"""

import re

import pandas as pd

CLIENT_FACING = "Client-facing application"
STEP_1_APPROACHES = {"Only fine-tuning", "Fine-tuning + other modules", "Prompting + other modules"}
STEP_2_APPROACHES = {"Only prompting", "Other"}

# Items of the Step 2 coding, as named in the Covidence export
ITEMS = {
    "general_purpose": "General-purpose LLM",
    "purpose_built": "Purpose-built application",
    "companion": "Companion application",
}
REVIEWERS = ["Reviewer 1", "Reviewer 2", "Consensus"]

# Reporting groups (mutually exclusive)
GENERAL_PURPOSE_ONLY = "General-purpose only"
PURPOSE_BUILT = "Purpose-built"
COMPANION = "Companion"

COMPANION_COLUMNS = {
    "Evaluates AI companion application – Reviewer 1": "Reviewer 1",
    "Evaluates AI companion application – Reviewer 2": "Reviewer 2",
    "Consensus": "Consensus",
}


def normalize_title(title: str) -> str:
    """Lower-case a title and remove everything except letters and digits (for matching)."""
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def load_step_2_coding(export_path, final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the Covidence export of the Step 2 coding as one row per article and reviewer.

    Returns a DataFrame with the columns study_id, title, reviewer and one yes/no column per item.
    Articles are matched to final_data.csv by title, because the Covidence numbers of this
    classification differ from the study_ids.
    """
    export = pd.read_csv(export_path, dtype=str, keep_default_na=False)
    if set(export["Reviewer Name"]) != set(REVIEWERS):
        raise ValueError(f"Unexpected reviewer names: {sorted(set(export['Reviewer Name']))}")

    titles = final_df.assign(key=final_df["title"].map(normalize_title))
    titles = titles[titles["application_type"] == CLIENT_FACING]
    if titles["key"].duplicated().any():
        raise ValueError("Titles of client-facing articles are not unique after normalization")
    study_ids = dict(zip(titles["key"], titles["study_id"]))

    export["study_id"] = export["Title"].map(normalize_title).map(study_ids)
    unmatched = export.loc[export["study_id"].isna(), "Title"].unique()
    if len(unmatched):
        raise ValueError(f"Titles not found among client-facing articles: {list(unmatched)}")

    coding = pd.DataFrame({"study_id": export["study_id"], "title": export["Title"], "reviewer": export["Reviewer Name"]})
    categories = export["Application category"].str.split(";").apply(lambda values: {v.strip() for v in values if v.strip()})
    unknown = set().union(*categories) - set(ITEMS.values())
    if unknown:
        raise ValueError(f"Unknown application categories: {sorted(unknown)}")
    if (categories.apply(len) == 0).any():
        raise ValueError(f"Empty application category for: {export.loc[categories.apply(len) == 0, 'Title'].tolist()}")
    for item, label in ITEMS.items():
        coding[item] = categories.apply(lambda values: "yes" if label in values else "no")

    counts = coding.groupby("study_id")["reviewer"].apply(sorted)
    incomplete = counts[counts.apply(lambda r: r != sorted(REVIEWERS))]
    if len(incomplete):
        raise ValueError(f"Articles without exactly one code per reviewer and consensus: {incomplete.index.tolist()}")
    return coding


def load_companion_coding(coding_path) -> pd.DataFrame:
    """
    Load the Step 3 coding sheet as one row per article and reviewer.

    Returns a DataFrame with the columns study_id, first_author_year, reviewer and companion (yes/no/unclear).
    """
    sheet = pd.read_excel(coding_path, sheet_name="Coding", dtype=str).fillna("")
    sheet = sheet.apply(lambda col: col.str.strip())
    if sheet["Study ID"].duplicated().any():
        raise ValueError(f"Duplicate study IDs in companion coding: {sheet.loc[sheet['Study ID'].duplicated(), 'Study ID'].tolist()}")
    coding = sheet.melt(id_vars=["Study ID", "First author, year"], value_vars=list(COMPANION_COLUMNS),
                        var_name="reviewer", value_name="companion")
    coding = coding.rename(columns={"Study ID": "study_id", "First author, year": "first_author_year"})
    coding["reviewer"] = coding["reviewer"].map(COMPANION_COLUMNS)
    coding["companion"] = coding["companion"].str.lower()
    invalid = coding[~coding["companion"].isin({"yes", "no", "unclear"})]
    if len(invalid):
        raise ValueError(f"Missing or invalid companion codes: {invalid[['study_id', 'reviewer', 'companion']].values.tolist()}")
    return coding


def classify_design_purpose(final_df: pd.DataFrame, step_2: pd.DataFrame, companion: pd.DataFrame,
                            hit_study_ids: set) -> pd.DataFrame:
    """
    Combine Steps 1-3 into the design purpose of each article evaluating a client-facing application.

    Returns a DataFrame indexed by study_id with the columns design_purpose and design_purpose_basis.
    Raises an error if the coded articles do not match the articles expected for each step.
    """
    client_facing = final_df[final_df["application_type"] == CLIENT_FACING]
    approach = client_facing.set_index("study_id")["llm_development_approach"]
    unexpected = set(approach) - STEP_1_APPROACHES - STEP_2_APPROACHES
    if unexpected:
        raise ValueError(f"Unexpected development approaches: {sorted(unexpected)}")
    step_1_ids = set(approach[approach.isin(STEP_1_APPROACHES)].index)
    step_2_ids = set(approach[approach.isin(STEP_2_APPROACHES)].index)

    step_2 = step_2[step_2["reviewer"] == "Consensus"].set_index("study_id")
    if set(step_2.index) != step_2_ids:
        raise ValueError(f"Step 2 coding does not match the articles with a prompting-only or other development approach. "
                         f"Missing: {sorted(step_2_ids - set(step_2.index))}; unexpected: {sorted(set(step_2.index) - step_2_ids)}")

    companion = companion[companion["reviewer"] == "Consensus"].set_index("study_id")["companion"]
    if set(companion.index) != hit_study_ids:
        raise ValueError(f"Companion coding does not match the articles with search hits. "
                         f"Missing: {sorted(hit_study_ids - set(companion.index))}; unexpected: {sorted(set(companion.index) - hit_study_ids)}")
    if (companion == "unclear").any():
        raise ValueError(f"Consensus 'unclear' for: {companion[companion == 'unclear'].index.tolist()}")

    result = pd.DataFrame(index=sorted(step_1_ids | step_2_ids))
    result.index.name = "study_id"
    result["design_purpose_basis"] = ["development approach" if i in step_1_ids else "dual coding" for i in result.index]
    purpose_built = {i: i in step_1_ids or step_2.at[i, "purpose_built"] == "yes" for i in result.index}
    # Companion status comes from the full-text search (Step 3) for all articles; articles without hits are not companion
    is_companion = {i: companion.get(i, "no") == "yes" for i in result.index}

    # Step 2 companion item and Step 3 must agree for the dual-coded articles
    mismatch = [i for i in step_2_ids if (step_2.at[i, "companion"] == "yes") != is_companion[i]]
    if mismatch:
        raise ValueError(f"Step 2 and Step 3 disagree on companion status for: {sorted(mismatch)}")

    # Reporting groups: companion takes precedence over purpose-built, which takes precedence over general-purpose only
    result["design_purpose"] = [COMPANION if is_companion[i] else PURPOSE_BUILT if purpose_built[i] else GENERAL_PURPOSE_ONLY
                                for i in result.index]
    general_purpose_only = result.index[result["design_purpose"] == GENERAL_PURPOSE_ONLY]
    not_general_purpose = [i for i in general_purpose_only if step_2.at[i, "general_purpose"] != "yes"]
    if not_general_purpose:
        raise ValueError(f"Articles with no design purpose (no item coded yes): {not_general_purpose}")
    return result[["design_purpose", "design_purpose_basis"]]
