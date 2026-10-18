#!/usr/bin/env python3
"""
Extract the full-text exclusions of the 2nd search review in Covidence (Figure 1).

The full-text exclusions and their reasons were recorded in Covidence. Later, the excluded
records had to be moved back to screening, which removed the recorded reasons. Before that,
the page "Excluded - Full text review" was saved as HTML (30 September 2025); it is the only
remaining record of the reasons. Two further excluded records, not on the saved page, were
exported from Covidence separately (CSV export of the excluded records, reason in the Notes column).

This script combines both sources into one CSV file with one row per excluded record:
covidence_number, study, title, authors, source, year, doi, excluded_on, exclusion_reason,
record_source. Reasons are copied as recorded in Covidence ("no reason" where none was given).
Abstracts, tags and all other page content are not extracted.

Note: the saved HTML page and the Covidence export are not distributed with this repository
because they contain personal account data, so this script can only be re-run with local copies
of both files. Its output is included as data/intermediate/covidence_fulltext_excluded.csv.

Usage:
    python 1_extract_fulltext_exclusions.py --html FILE --export FILE [--extraction-export FILE] [--output FILE]
"""

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

COLUMNS = ["covidence_number", "study", "title", "authors", "source", "year", "doi",
           "excluded_on", "exclusion_reason", "record_source"]
PAGE_SOURCE = "covidence_page_20250930"
EXPORT_SOURCE = "covidence_export_20261018"


def text(element) -> str:
    return re.sub(r"\s+", " ", element.get_text(" ", strip=True)) if element else ""


def year_from_label(label: str) -> str:
    match = re.search(r"\b((?:19|20)\d\d)$", label)
    return match.group(1) if match else ""


def parse_page(path: Path) -> pd.DataFrame:
    """Read the excluded records (one table row each) from the saved Covidence page."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    rows = []
    for info in soup.find_all("td", class_="info"):
        row = info.find_parent("tr")
        header = text(info.find("div", class_="study-header"))
        number, _, label = header.partition(" - ")
        references = info.find_all("div", class_="reference")
        if len(references) != 1:
            raise ValueError(f"{number}: expected one reference, found {len(references)}")
        reference = references[0]
        doi = reference.find("p", class_="ref-ids")
        doi = doi.find("a") if doi else None
        rows.append({
            "covidence_number": number.strip(),
            "study": label.strip(),
            "title": text(reference.find(class_="title")),
            "authors": text(reference.find("p", class_="authors")),
            "source": text(reference.find("p", class_="journal")),
            "year": year_from_label(label.strip()),
            "doi": text(doi),
            "excluded_on": datetime.strptime(text(row.find("td", class_="date")), "%d/%m/%y").strftime("%Y-%m-%d"),
            "exclusion_reason": text(row.find("td", class_="reason")),
            "record_source": PAGE_SOURCE,
        })

    # The page header gives the number of excluded records, e.g. "Excluded references 81"
    label = soup.find(string=re.compile(r"^\s*Excluded references\s*$"))
    expected = int(label.find_next(string=re.compile(r"\d")).strip()) if label else None
    if expected is not None and expected != len(rows):
        raise ValueError(f"Page header lists {expected} excluded records, but {len(rows)} were parsed")
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_export(path: Path) -> pd.DataFrame:
    """Read the excluded records from the Covidence CSV export (reason in the Notes column)."""
    export = pd.read_csv(path, dtype=str).fillna("")
    rows = []
    for _, record in export.iterrows():
        match = re.match(r"Exclusion reason:\s*(.*?);", record["Notes"])
        if not match:
            raise ValueError(f"{record['Covidence #']}: no exclusion reason in Notes: {record['Notes']!r}")
        rows.append({
            "covidence_number": record["Covidence #"],
            "study": record["Study"],
            "title": record["Title"],
            "authors": record["Authors"],
            "source": record["Journal"],
            "year": record["Published Year"],
            "doi": record["DOI"],
            "excluded_on": "",  # not included in the export
            "exclusion_reason": match.group(1).strip(),
            "record_source": EXPORT_SOURCE,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Main function to extract the full-text exclusions."""
    parser = argparse.ArgumentParser(description="Extract the full-text exclusions of the 2nd search review")
    parser.add_argument("--html", required=True,
                        help="Saved Covidence page 'Excluded - Full text review' (30 September 2025)")
    parser.add_argument("--export", required=True,
                        help="Covidence CSV export of the excluded records not on the saved page")
    parser.add_argument("--extraction-export", default="data/unprocessed/covidence_export_2nd_search_20251030.csv",
                        help="Covidence data extraction export of the 2nd search, used as a check")
    parser.add_argument("--output", default="data/intermediate/covidence_fulltext_excluded.csv",
                        help="Path of the output CSV file")
    args = parser.parse_args()

    page = parse_page(Path(args.html))
    export = parse_export(Path(args.export))
    excluded = pd.concat([page, export], ignore_index=True)

    duplicated = excluded["covidence_number"][excluded["covidence_number"].duplicated()]
    if len(duplicated):
        raise ValueError(f"Records in both sources: {sorted(duplicated)}")
    if (excluded["exclusion_reason"] == "").any():
        raise ValueError("Records without exclusion reason text: "
                         f"{excluded.loc[excluded['exclusion_reason'] == '', 'covidence_number'].tolist()}")
    extracted = set(pd.read_csv(args.extraction_export, dtype=str)["Covidence #"])
    also_extracted = sorted(set(excluded["covidence_number"]) & extracted)
    if also_extracted:
        raise ValueError(f"Excluded records also in the data extraction export: {also_extracted}")

    excluded["sort_key"] = excluded["covidence_number"].str.lstrip("#").astype(int)
    excluded = excluded.sort_values("sort_key").drop(columns="sort_key")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(args.output, index=False, encoding="utf-8")

    lines = ["FULL-TEXT EXCLUSIONS (2nd search review)", "=" * 80, "",
             f"Saved page: {Path(args.html).name}", f"  SHA-256: {sha256(Path(args.html))}",
             f"Export: {Path(args.export).name}", f"  SHA-256: {sha256(Path(args.export))}", "",
             f"Records from the saved page: {len(page)}", f"Records from the export: {len(export)}",
             f"Total excluded records: {len(excluded)}", "", "Exclusion reasons (as recorded):"]
    lines += [f"  {reason}: {n}" for reason, n in excluded["exclusion_reason"].value_counts().items()]
    print("\n".join(lines))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
