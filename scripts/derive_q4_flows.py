"""Derive missing Q4 income-statement flows for 10-K rows (Session 5).

Pipeline position: build_panel.py -> fix_panel.py -> derive_q4_flows.py
-> build_label.py.

Why: 10-Ks report ANNUAL flow values (qtrs=4). build_panel.py keeps only
qtrs in {0, 1}, so a 10-K row gets quarterly revenues / operating income /
net income only if the filer separately tagged Q4-specific quarterly data.
The SEC eliminated that disclosure requirement (Reg S-K Item 302(a)
amendment, adopted Nov 2020, effective early 2021), so 10-K quarterly-flow
coverage collapsed from ~19% (2019Q4) to ~2% (2022Q4). Since the at_risk
label needs a 3-quarter operating-margin chain (T-1, T, T+1), weak Q4
coverage wipes out Q3, Q4, and Q1 labels -- post-2021, only Q2 quarters
survived labeling in volume.

Fix: for each 10-K (fp == FY) row missing a flow value,
    Q4 flow = annual (qtrs=4, from the 10-K itself)
              - sum of the same fiscal year's Q1 + Q2 + Q3 quarterly flows
              (from the 10-Q rows already in the panel).
All inputs are known at the 10-K's filing time, so this introduces no
lookahead relative to the existing rows.

Steps:
1. Scan all raw num.txt files for qtrs=4 values of the three flow concepts
   (same tag variants, USD / no-segments / no-coreg / ddate == period
   filters as build_panel.py), restricted to the panel's 10-K adshs.
2. Sum Q1-Q3 quarterly flows per (cik, fy) from the panel's 10-Q rows;
   require all three quarters present, each ending within 400 days before
   the 10-K's period end (guards against fiscal-year changes / stale fy).
3. Fill ONLY missing flow values on FY rows; never overwrite reported ones.
   Derived revenues < 0 are discarded (restatement artifacts).
4. Recompute the dependent ratios panel-wide with fix_panel.py's rules
   (safe_ratio + winsorization bounds): operating_margin, roe,
   revenue_growth. Balance-sheet ratios are unaffected.
5. Report coverage before/after and overwrite data/processed/panel.parquet.
"""

import glob
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "raw"))
PANEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "processed", "panel.parquet"))

# concept -> ordered tag variants (annual duration, qtrs=4); same variants
# and priority as build_panel.py CONCEPTS.
FLOW_CONCEPTS = {
    "revenues": ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "operating_income_loss": ["OperatingIncomeLoss"],
    "net_income_loss": ["NetIncomeLoss"],
}
ALL_FLOW_TAGS = sorted({t for tags in FLOW_CONCEPTS.values() for t in tags})

MAX_DAYS_QTR_BEFORE_FYEND = 400  # Q1 ends ~270 days before FY end; allow slack

NUM_CHUNKSIZE = 500_000

# Ratios to recompute after the fill; bounds identical to fix_panel.py.
RECOMPUTE_BOUNDS = {
    "operating_margin": (-5, 5),
    "roe": (-10, 10),
    "revenue_growth": (-1, 10),
}


def find_quarter_dirs():
    return sorted(
        d for d in glob.glob(os.path.join(RAW_DIR, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "num.txt"))
    )


def first_available(num_df, tags):
    """Per adsh, value of the first tag (priority order) with a non-null value."""
    d = num_df[num_df["tag"].isin(tags)].dropna(subset=["value"])
    if d.empty:
        return pd.Series(dtype="float64", name="value")
    priority = {tag: i for i, tag in enumerate(tags)}
    d = d.copy()
    d["_priority"] = d["tag"].map(priority)
    d = d.sort_values(["adsh", "_priority"])
    d = d.drop_duplicates(subset="adsh", keep="first")
    return d.set_index("adsh")["value"]


def extract_annual_for_quarter(quarter_dir, adsh_set, period_map):
    """qtrs=4 flow facts for the target adshs, current period only."""
    path = os.path.join(quarter_dir, "num.txt")
    usecols = ["adsh", "tag", "ddate", "qtrs", "uom", "segments", "coreg", "value"]
    dtype = {
        "adsh": "string",
        "tag": "string",
        "uom": "string",
        "segments": "string",
        "coreg": "string",
    }
    pieces = []
    reader = pd.read_csv(
        path, sep="\t", usecols=usecols, dtype=dtype, encoding="utf-8",
        chunksize=NUM_CHUNKSIZE, low_memory=False,
    )
    for chunk in reader:
        chunk = chunk[chunk["tag"].isin(ALL_FLOW_TAGS)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["qtrs"] == 4]
        chunk = chunk[chunk["adsh"].isin(adsh_set)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["uom"] == "USD"]
        chunk = chunk[chunk["segments"].isna() | (chunk["segments"] == "")]
        chunk = chunk[chunk["coreg"].isna() | (chunk["coreg"] == "")]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["ddate"] = chunk["ddate"].astype("int64")
        chunk = chunk[chunk["adsh"].map(period_map) == chunk["ddate"]]
        if not chunk.empty:
            pieces.append(chunk[["adsh", "tag", "value"]])
    if not pieces:
        return pd.DataFrame(columns=["adsh", "tag", "value"])
    return pd.concat(pieces, ignore_index=True)


def safe_ratio(numerator, denominator):
    """Divide only where denominator > 0; else NaN. Same as fix_panel.py."""
    denom = denominator.where(denominator > 0)
    return numerator / denom


def main():
    panel = pd.read_parquet(PANEL_PATH)
    print(f"Loaded panel: {len(panel):,} rows")

    is_fy = (panel["fp"] == "FY") & panel["form"].isin(["10-K", "10-K/A"])
    fy_rows = panel[is_fy]
    print(f"10-K FY rows: {len(fy_rows):,}")

    # --- 1. extract annual (qtrs=4) flows for all 10-K FY adshs ----------
    adsh_set = set(fy_rows["adsh"])
    period_map = fy_rows.set_index("adsh")["period"]
    quarter_dirs = find_quarter_dirs()
    print(f"\nScanning {len(quarter_dirs)} raw quarter folders for qtrs=4 flows ...")
    pieces = []
    for idx, qdir in enumerate(quarter_dirs, 1):
        piece = extract_annual_for_quarter(qdir, adsh_set, period_map)
        pieces.append(piece)
        print(f"  [{idx}/{len(quarter_dirs)}] {os.path.basename(qdir)}: {len(piece)} facts", flush=True)
    annual_facts = pd.concat(pieces, ignore_index=True)

    annual = pd.DataFrame({
        concept: first_available(annual_facts, tags)
        for concept, tags in FLOW_CONCEPTS.items()
    })
    print(f"\nAnnual values extracted for {len(annual):,} adshs:")
    for c in FLOW_CONCEPTS:
        print(f"  {c:<25} {annual[c].notna().sum():>7,}")

    # --- 2. quarterly Q1-Q3 sums per (cik, fy) ----------------------------
    fy_end = pd.to_datetime(fy_rows["period"].astype(str), format="%Y%m%d", errors="coerce")
    fy_key = pd.DataFrame({
        "adsh": fy_rows["adsh"].values,
        "cik": fy_rows["cik"].values,
        "fy": fy_rows["fy"].values,
        "fy_end": fy_end.values,
    })

    q_rows = panel[panel["fp"].isin(["Q1", "Q2", "Q3"])].copy()
    q_rows["q_end"] = pd.to_datetime(q_rows["period"].astype(str), format="%Y%m%d", errors="coerce")

    merged = fy_key.merge(
        q_rows[["cik", "fy", "fp", "q_end"] + list(FLOW_CONCEPTS)],
        on=["cik", "fy"], how="inner",
    )
    delta_days = (merged["fy_end"] - merged["q_end"]).dt.days
    merged = merged[(delta_days > 0) & (delta_days <= MAX_DAYS_QTR_BEFORE_FYEND)]
    # one row per (adsh, fp): duplicates would double-count a quarter
    merged = merged.drop_duplicates(subset=["adsh", "fp"], keep="last")

    filled_counts = {}
    for concept in FLOW_CONCEPTS:
        g = merged.dropna(subset=[concept]).groupby("adsh")[concept]
        sums = g.sum()
        counts = g.size()
        q123 = sums[counts == 3]  # need all three quarters
        derived = annual[concept].reindex(q123.index) - q123
        derived = derived.dropna()
        if concept == "revenues":
            n_neg = int((derived < 0).sum())
            if n_neg:
                print(f"  discarding {n_neg} negative derived revenues")
            derived = derived[derived >= 0]

        # --- 3. fill only missing values on FY rows -----------------------
        target = panel.loc[is_fy, ["adsh", concept]].set_index("adsh")
        fill = derived.reindex(target.index)
        fill = fill[target[concept].isna() & fill.notna()]
        idx = panel.index[is_fy][panel.loc[is_fy, "adsh"].isin(fill.index)]
        panel.loc[idx, concept] = panel.loc[idx, "adsh"].map(fill)
        filled_counts[concept] = len(fill)

    print("\nDerived-and-filled Q4 flow values (missing -> filled):")
    for c, n in filled_counts.items():
        print(f"  {c:<25} {n:>7,}   coverage now {100 * panel[c].notna().mean():.1f}%")

    # --- 4. recompute dependent ratios (fix_panel.py rules) ---------------
    panel = panel.sort_values(["cik", "period"]).reset_index(drop=True)
    panel["operating_margin"] = safe_ratio(panel["operating_income_loss"], panel["revenues"])
    panel["roe"] = safe_ratio(panel["net_income_loss"], panel["stockholders_equity"])
    prior_revenues = panel.groupby("cik")["revenues"].shift(1)
    panel["revenue_growth"] = safe_ratio(panel["revenues"] - prior_revenues, prior_revenues)
    for col, (lo, hi) in RECOMPUTE_BOUNDS.items():
        panel[col] = panel[col].clip(lower=lo, upper=hi)

    # --- 5. report per-calendar-quarter operating_margin coverage ---------
    print("\noperating_margin coverage by fiscal quarter (post-fill):")
    q = panel["fiscal_quarter"].astype(str)
    cov = panel.groupby(q)["operating_margin"].apply(lambda s: 100 * s.notna().mean())
    n_rows = panel.groupby(q).size()
    for label in cov.index:
        print(f"  {label:<8} {n_rows[label]:>6,} rows  {cov[label]:5.1f}%")

    panel.to_parquet(PANEL_PATH, index=False)
    print(f"\nSaved patched panel ({len(panel):,} rows) -> {PANEL_PATH}")


if __name__ == "__main__":
    main()
