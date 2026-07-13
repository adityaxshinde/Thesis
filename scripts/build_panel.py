"""
Builds a company-quarter panel from the raw SEC EDGAR Financial Statement
Data Sets in data/raw/{quarter}/{sub.txt,num.txt}.

For each quarter folder, parses sub.txt (one row per XBRL submission) and
num.txt (one row per reported numeric fact), joins them on accession
number (adsh), and extracts 10 raw line items per company per fiscal
quarter. XBRL tag names vary across filers/years, so each concept is
looked up through an ordered list of tag variants, taking the first
non-null match.

Output: data/processed/panel.parquet
"""

import os
import glob
import statistics

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "raw"))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "processed"))
OUT_PATH = os.path.join(PROCESSED_DIR, "panel.parquet")

VALID_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}

# concept -> (ordered tag variants, required qtrs)
# qtrs=1 -> single-quarter flow value (income statement lines)
# qtrs=0 -> point-in-time value (balance sheet lines)
CONCEPTS = {
    "revenues": (
        ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"],
        1,
    ),
    "operating_income_loss": (["OperatingIncomeLoss"], 1),
    "net_income_loss": (["NetIncomeLoss"], 1),
    "assets_current": (["AssetsCurrent"], 0),
    "liabilities_current": (["LiabilitiesCurrent"], 0),
    "assets": (["Assets"], 0),
    "liabilities": (["Liabilities"], 0),
    "stockholders_equity": (
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        0,
    ),
    "cash": (["CashAndCashEquivalentsAtCarryingValue"], 0),
    "long_term_debt": (["LongTermDebt"], 0),
    "short_term_debt": (["ShortTermBorrowings", "DebtCurrent"], 0),
}

LINE_ITEM_COLUMNS = [
    "revenues",
    "operating_income_loss",
    "net_income_loss",
    "assets_current",
    "liabilities_current",
    "assets",
    "liabilities",
    "stockholders_equity",
    "cash",
    "total_debt",
]

ALL_TAGS = sorted({tag for tags, _ in CONCEPTS.values() for tag in tags})

NUM_CHUNKSIZE = 500_000


def find_quarters():
    dirs = sorted(
        d for d in glob.glob(os.path.join(RAW_DIR, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "sub.txt"))
    )
    return [(os.path.basename(d), d) for d in dirs]


def load_sub(quarter_dir):
    path = os.path.join(quarter_dir, "sub.txt")
    usecols = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "prevrpt"]
    dtype = {
        "adsh": "string",
        "cik": "int64",
        "name": "string",
        "form": "string",
        "fy": "string",
        "fp": "string",
    }
    df = pd.read_csv(
        path, sep="\t", usecols=usecols, dtype=dtype, encoding="utf-8", low_memory=False
    )
    df = df[df["form"].isin(VALID_FORMS)]
    df = df[df["prevrpt"] == 0]
    df = df.dropna(subset=["period"])
    df["period"] = df["period"].astype("int64")
    # one row per (cik, period): keep the most recently filed submission
    df = df.sort_values("filed").drop_duplicates(subset=["cik", "period"], keep="last")
    return df


def load_num_filtered(quarter_dir):
    path = os.path.join(quarter_dir, "num.txt")
    usecols = ["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"]
    dtype = {
        "adsh": "string",
        "tag": "string",
        "uom": "string",
        "segments": "string",
        "coreg": "string",
    }
    chunks = []
    reader = pd.read_csv(
        path, sep="\t", usecols=usecols, dtype=dtype, encoding="utf-8",
        chunksize=NUM_CHUNKSIZE, low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[chunk["tag"].isin(ALL_TAGS)]
        chunk = chunk[chunk["uom"] == "USD"]
        chunk = chunk[chunk["segments"].isna() | (chunk["segments"] == "")]
        chunk = chunk[chunk["coreg"].isna() | (chunk["coreg"] == "")]
        chunk = chunk[chunk["qtrs"].isin([0, 1])]
        if not chunk.empty:
            chunks.append(chunk[["adsh", "tag", "ddate", "qtrs", "value"]])
    if not chunks:
        return pd.DataFrame(columns=["adsh", "tag", "ddate", "qtrs", "value"])
    return pd.concat(chunks, ignore_index=True)


def first_available(num_df, tags, qtrs_required):
    """For each adsh, return the value of the first tag (in priority order)
    that has a non-null value at the required qtrs duration."""
    d = num_df[(num_df["qtrs"] == qtrs_required) & (num_df["tag"].isin(tags))]
    if d.empty:
        return pd.Series(dtype="float64", name="value")
    d = d.dropna(subset=["value"])
    if d.empty:
        return pd.Series(dtype="float64", name="value")
    priority = {tag: i for i, tag in enumerate(tags)}
    d = d.copy()
    d["_priority"] = d["tag"].map(priority)
    d = d.sort_values(["adsh", "_priority"])
    d = d.drop_duplicates(subset="adsh", keep="first")
    return d.set_index("adsh")["value"]


def build_quarter_panel(quarter, quarter_dir):
    sub = load_sub(quarter_dir)
    if sub.empty:
        return None

    num = load_num_filtered(quarter_dir)
    if num.empty:
        merged = sub.copy()
        for col in LINE_ITEM_COLUMNS:
            merged[col] = pd.NA
        return merged

    # keep only facts whose period end (ddate) matches the submission's
    # reported balance sheet date, i.e. the current period, not a
    # comparative prior period also present in the same filing
    period_map = sub.set_index("adsh")["period"]
    num = num[num["adsh"].isin(period_map.index)]
    num["ddate"] = num["ddate"].astype("int64")
    num = num[num["adsh"].map(period_map) == num["ddate"]]

    concept_series = {}
    for concept, (tags, qtrs_required) in CONCEPTS.items():
        concept_series[concept] = first_available(num, tags, qtrs_required)

    wide = pd.DataFrame(concept_series)
    ltd = wide["long_term_debt"]
    std = wide["short_term_debt"]
    wide["total_debt"] = ltd.add(std, fill_value=0)
    wide.loc[ltd.isna() & std.isna(), "total_debt"] = pd.NA

    wide = wide.reindex(columns=LINE_ITEM_COLUMNS)
    wide.index.name = "adsh"

    merged = sub.merge(wide, how="left", left_on="adsh", right_index=True)
    return merged


def to_fiscal_quarter(period_series):
    dt = pd.to_datetime(period_series.astype(str), format="%Y%m%d", errors="coerce")
    return dt.dt.to_period("Q").astype(str)


def flag_abnormal_counts(quarter_labels, counts):
    n = len(quarter_labels)
    flags = []
    for i in range(n):
        neighbours = [
            counts[j] for j in range(max(0, i - 2), min(n, i + 3))
            if j != i
        ]
        if len(neighbours) < 2:
            flags.append("")
            continue
        neighbour_median = statistics.median(neighbours)
        if neighbour_median > 0 and counts[i] < 0.5 * neighbour_median:
            pct = 100 * counts[i] / neighbour_median
            flags.append(f"LOW ({pct:.0f}% of neighbour median)")
        else:
            flags.append("")
    return flags


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    quarters = find_quarters()
    print(f"Found {len(quarters)} quarter folders in {RAW_DIR}\n")

    pieces = []
    per_quarter_counts = []
    for idx, (quarter, quarter_dir) in enumerate(quarters, 1):
        print(f"[{idx}/{len(quarters)}] {quarter} ... ", end="", flush=True)
        panel = build_quarter_panel(quarter, quarter_dir)
        if panel is None or panel.empty:
            print("no usable rows")
            per_quarter_counts.append(0)
            continue
        n_companies = panel["cik"].nunique()
        print(f"{len(panel)} filings, {n_companies} companies")
        per_quarter_counts.append(n_companies)
        pieces.append(panel)

    panel = pd.concat(pieces, ignore_index=True)
    panel["fiscal_quarter"] = to_fiscal_quarter(panel["period"])

    panel = panel[
        ["cik", "name", "sic", "fiscal_quarter", "period", "form", "fy", "fp", "filed", "adsh"]
        + LINE_ITEM_COLUMNS
    ]

    panel.to_parquet(OUT_PATH, index=False)

    print("\n" + "=" * 90)
    print("PANEL SUMMARY")
    print("=" * 90)
    print(f"Shape: {panel.shape[0]} rows x {panel.shape[1]} columns")
    print(f"Date range (fiscal_quarter): {panel['fiscal_quarter'].min()} .. {panel['fiscal_quarter'].max()}")
    print(f"Unique companies (CIK): {panel['cik'].nunique()}")

    print("\nCoverage by line item (% of rows with a non-null value):")
    for col in LINE_ITEM_COLUMNS:
        pct = 100 * panel[col].notna().mean()
        print(f"  {col:<25} {pct:6.2f}%")

    print("\nCompany count per fiscal quarter, with abnormal-drop flags:")
    by_q = panel.groupby("fiscal_quarter")["cik"].nunique().sort_index()
    labels = by_q.index.tolist()
    counts = by_q.values.tolist()
    flags = flag_abnormal_counts(labels, counts)
    flagged_any = False
    for label, count, flag in zip(labels, counts, flags):
        marker = f"  <-- {flag}" if flag else ""
        if flag:
            flagged_any = True
        print(f"  {label:<10} {count:>8}{marker}")
    if not flagged_any:
        print("  No fiscal quarters flagged for abnormal company-count drops.")

    print(f"\nSaved panel to: {OUT_PATH}")


if __name__ == "__main__":
    main()
