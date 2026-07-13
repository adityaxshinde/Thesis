"""
Post-processing pass on data/processed/panel.parquet.

1. Filters to fiscal_quarter 2012Q1..2026Q1, dropping the pre-2012 rows
   (period misparsed from a handful of filings -> XBRL date errors).
2. Re-extracts total_debt from the raw EDGAR num.txt files using an
   expanded set of XBRL tag variants (see LT_TAGS / ST_TAGS below), to
   raise total_debt coverage beyond the original LongTermDebt /
   ShortTermBorrowings-DebtCurrent-only lookup in build_panel.py.
3. Computes six ratios per company-quarter:
     operating_margin, current_ratio, debt_to_assets, revenue_growth,
     roe, cash_to_assets
   Edge-case rules (applied uniformly via safe_ratio()):
     - denominator <= 0 or missing -> ratio is NaN (never divide)
     - result outside the bounds table -> winsorized (clipped) to bounds
4. Overwrites data/processed/panel.parquet with raw line items + ratios.
"""

import glob
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "raw"))
PANEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "processed", "panel.parquet"))

MIN_QUARTER = "2012Q1"
MAX_QUARTER = "2026Q1"

# Long-term-like and short-term-like debt tag variants, in lookup priority
# order. LongTermDebt / ShortTermBorrowings / DebtCurrent were already used
# by build_panel.py; the rest are the additional variants requested to
# raise coverage. NotesPayable and DebtInstrumentCarryingAmount are
# generic/ambiguous tags but are typically used for term (long-lived) debt
# instruments, so they're grouped with the long-term component.
LT_TAGS = [
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "LongTermNotesPayable",
    "NotesPayable",
    "DebtInstrumentCarryingAmount",
]
ST_TAGS = [
    "ShortTermBorrowings",
    "DebtCurrent",
    "CommercialPaper",
    "LineOfCreditFacilityMaximumBorrowingCapacity",
]
ALL_DEBT_TAGS = sorted(set(LT_TAGS) | set(ST_TAGS))

NUM_CHUNKSIZE = 500_000

RATIO_BOUNDS = {
    "operating_margin": (-5, 5),
    "current_ratio": (0, 50),
    "debt_to_assets": (0, 10),
    "revenue_growth": (-1, 10),
    "roe": (-10, 10),
    "cash_to_assets": (0, 1),
}


def find_quarter_dirs():
    dirs = sorted(
        d for d in glob.glob(os.path.join(RAW_DIR, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "num.txt"))
    )
    return dirs


def first_available(num_df, tags):
    """For each adsh, the value of the first tag (priority order) with a
    non-null point-in-time (qtrs=0) value."""
    d = num_df[(num_df["qtrs"] == 0) & (num_df["tag"].isin(tags))]
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


def extract_debt_for_quarter(quarter_dir, adsh_set, period_map):
    path = os.path.join(quarter_dir, "num.txt")
    usecols = ["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"]
    dtype = {
        "adsh": "string",
        "tag": "string",
        "uom": "string",
        "segments": "string",
        "coreg": "string",
    }
    lt_pieces, st_pieces = [], []
    reader = pd.read_csv(
        path, sep="\t", usecols=usecols, dtype=dtype, encoding="utf-8",
        chunksize=NUM_CHUNKSIZE, low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[chunk["tag"].isin(ALL_DEBT_TAGS)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["adsh"].isin(adsh_set)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["uom"] == "USD"]
        chunk = chunk[chunk["segments"].isna() | (chunk["segments"] == "")]
        chunk = chunk[chunk["coreg"].isna() | (chunk["coreg"] == "")]
        chunk = chunk[chunk["qtrs"] == 0]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["ddate"] = chunk["ddate"].astype("int64")
        chunk = chunk[chunk["adsh"].map(period_map) == chunk["ddate"]]
        if chunk.empty:
            continue
        lt_pieces.append(chunk[chunk["tag"].isin(LT_TAGS)][["adsh", "tag", "qtrs", "value"]])
        st_pieces.append(chunk[chunk["tag"].isin(ST_TAGS)][["adsh", "tag", "qtrs", "value"]])

    lt_df = pd.concat(lt_pieces, ignore_index=True) if lt_pieces else pd.DataFrame(columns=["adsh", "tag", "qtrs", "value"])
    st_df = pd.concat(st_pieces, ignore_index=True) if st_pieces else pd.DataFrame(columns=["adsh", "tag", "qtrs", "value"])
    lt_series = first_available(lt_df, LT_TAGS)
    st_series = first_available(st_df, ST_TAGS)
    return lt_series, st_series


def safe_ratio(numerator, denominator):
    """Divide only where denominator > 0; else NaN. Never divides by <=0."""
    denom = denominator.where(denominator > 0)
    return numerator / denom


def main():
    print(f"Loading panel from {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    n_before = len(panel)

    # --- 1. filter to 2012Q1..2026Q1 -------------------------------------
    in_range = (panel["fiscal_quarter"] >= MIN_QUARTER) & (panel["fiscal_quarter"] <= MAX_QUARTER)
    n_dropped = (~in_range).sum()
    panel = panel[in_range].reset_index(drop=True)
    print(f"Dropped {n_dropped} pre-2012/out-of-range rows (likely XBRL date errors).")
    print(f"Rows remaining: {len(panel)} (was {n_before})")

    old_debt_coverage = 100 * panel["total_debt"].notna().mean()

    # --- 2. re-extract total_debt with expanded tag variants -------------
    adsh_set = set(panel["adsh"])
    period_map = panel.set_index("adsh")["period"]
    quarter_dirs = find_quarter_dirs()
    print(f"\nRe-extracting debt tags from {len(quarter_dirs)} raw quarter folders ...")

    lt_all, st_all = [], []
    for idx, qdir in enumerate(quarter_dirs, 1):
        qname = os.path.basename(qdir)
        lt_series, st_series = extract_debt_for_quarter(qdir, adsh_set, period_map)
        lt_all.append(lt_series)
        st_all.append(st_series)
        print(f"  [{idx}/{len(quarter_dirs)}] {qname}: lt={len(lt_series)} st={len(st_series)}", flush=True)

    lt_combined = pd.concat(lt_all)
    st_combined = pd.concat(st_all)
    lt_combined = lt_combined[~lt_combined.index.duplicated(keep="first")]
    st_combined = st_combined[~st_combined.index.duplicated(keep="first")]

    total_debt_new = lt_combined.add(st_combined, fill_value=0)
    both_missing = lt_combined.reindex(total_debt_new.index).isna() & st_combined.reindex(total_debt_new.index).isna()
    total_debt_new.loc[both_missing] = pd.NA

    panel["total_debt"] = panel["adsh"].map(total_debt_new).astype("float64")

    new_debt_coverage = 100 * panel["total_debt"].notna().mean()
    print(f"\ntotal_debt coverage: {old_debt_coverage:.2f}% -> {new_debt_coverage:.2f}% after tag expansion")

    # --- 3. compute ratios -------------------------------------------------
    panel = panel.sort_values(["cik", "period"]).reset_index(drop=True)

    panel["operating_margin"] = safe_ratio(panel["operating_income_loss"], panel["revenues"])
    panel["current_ratio"] = safe_ratio(panel["assets_current"], panel["liabilities_current"])
    panel["debt_to_assets"] = safe_ratio(panel["total_debt"], panel["assets"])
    panel["roe"] = safe_ratio(panel["net_income_loss"], panel["stockholders_equity"])
    panel["cash_to_assets"] = safe_ratio(panel["cash"], panel["assets"])

    prior_revenues = panel.groupby("cik")["revenues"].shift(1)
    panel["revenue_growth"] = safe_ratio(panel["revenues"] - prior_revenues, prior_revenues)

    for col, (lo, hi) in RATIO_BOUNDS.items():
        panel[col] = panel[col].clip(lower=lo, upper=hi)

    # --- 4. report -----------------------------------------------------
    print("\n" + "=" * 90)
    print("RATIO COVERAGE & SUMMARY STATS")
    print("=" * 90)
    for col in RATIO_BOUNDS:
        s = panel[col]
        coverage = 100 * s.notna().mean()
        print(f"\n{col}  (coverage: {coverage:.2f}%)")
        if s.notna().any():
            print(
                f"  mean={s.mean():.4f}  median={s.median():.4f}  "
                f"p25={s.quantile(0.25):.4f}  p75={s.quantile(0.75):.4f}"
            )
        else:
            print("  no non-null values")

    print(f"\nRows remaining after 2012Q1-2026Q1 filter: {len(panel)}")
    print(f"total_debt coverage after tag expansion: {new_debt_coverage:.2f}% (was {old_debt_coverage:.2f}%)")

    panel.to_parquet(PANEL_PATH, index=False)
    print(f"\nSaved updated panel to: {PANEL_PATH}")


if __name__ == "__main__":
    main()
