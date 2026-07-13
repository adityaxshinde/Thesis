"""Build the at-risk label column + sector-percentile features (Sessions 4-4b).

Steps:
1. Leverage-feature migration (idempotent, Session 4b audit):
   - rename debt_to_assets (total_debt / assets) -> debt_to_assets_narrow
     (supplementary column; total_debt coverage is only ~45%).
   - add liabilities_to_assets = liabilities / assets (clipped to fixed
     bounds (0, 10)) as the PRIMARY leverage feature (~80% coverage).
   fix_panel.py produces this schema directly on any future re-run from raw.
2. Dedupe (cik, fiscal_quarter): keep latest-filed row per group (resolves
   amendment pairs and calendar-quarter bucketing collisions in fiscal_quarter,
   see build_panel.py's to_fiscal_quarter()).
3. Sort by cik, fiscal_quarter ascending.
4. op_margin_trend = operating_margin - prior quarter's operating_margin (same cik).
5. next_op_margin_trend = op_margin_trend shifted -1 within cik group (T+1's value).
   next_op_margin_trend and at_risk are LABEL-ONLY columns -- never use them,
   or any other T+1-derived column, as a model feature.
6. Sector-percentile features: for six quarter-T ratios, the company's
   percentile rank within its (fiscal_quarter, sic) peer group. Uses ONLY
   quarter-T cross-sectional data -- today's relative standing, no future info.
7. at_risk: bottom-quartile flag of next_op_margin_trend within each
   (fiscal_quarter, sic) peer group. A row is labelable only if it has a
   valid op_margin_trend (feature), a valid next_op_margin_trend (label
   input), a non-null sic, and >= 4 labelable companies in its peer group.
8. Print the attrition waterfall, then save panel + labeled_panel.
"""
import pandas as pd

PANEL_PATH = "data/processed/panel.parquet"
LABELED_PATH = "data/processed/labeled_panel.parquet"

LIAB_TO_ASSETS_BOUNDS = (0, 10)  # fixed constants, matches fix_panel.py RATIO_BOUNDS

# Quarter-T ratios that get a sector-percentile feature column.
PCTILE_COLS = [
    "operating_margin",
    "current_ratio",
    "liabilities_to_assets",
    "revenue_growth",
    "roe",
    "cash_to_assets",
]


def main():
    df = pd.read_parquet(PANEL_PATH)
    n_loaded = len(df)

    # --- Step 1: leverage-feature migration (idempotent) ---
    if "debt_to_assets" in df.columns:
        df = df.rename(columns={"debt_to_assets": "debt_to_assets_narrow"})
        print("Renamed debt_to_assets -> debt_to_assets_narrow (supplementary).")
    if "liabilities_to_assets" not in df.columns:
        lo, hi = LIAB_TO_ASSETS_BOUNDS
        denom = df["assets"].where(df["assets"] > 0)
        df["liabilities_to_assets"] = (df["liabilities"] / denom).clip(lo, hi)
        print(f"Added liabilities_to_assets = liabilities / assets, "
              f"clipped to fixed bounds {LIAB_TO_ASSETS_BOUNDS} "
              f"(coverage: {100 * df['liabilities_to_assets'].notna().mean():.2f}%).")

    # --- Step 2: dedupe (cik, fiscal_quarter): keep latest filed row ---
    df = df.sort_values(["cik", "fiscal_quarter", "filed"], ascending=True)
    df = df.drop_duplicates(subset=["cik", "fiscal_quarter"], keep="last")
    n_dedup = len(df)
    print(f"Deduped (cik, fiscal_quarter): {n_loaded} -> {n_dedup} rows "
          f"({n_loaded - n_dedup} dropped)")

    assert df.duplicated(subset=["cik", "fiscal_quarter"]).sum() == 0, \
        "duplicates remain after dedup"

    # --- Step 3: sort ascending by cik, fiscal_quarter ---
    df = df.sort_values(["cik", "fiscal_quarter"]).reset_index(drop=True)

    # --- Step 4: op_margin_trend = change vs prior quarter, same company ---
    g = df.groupby("cik")
    prior_operating_margin = g["operating_margin"].shift(1)
    df["op_margin_trend"] = df["operating_margin"] - prior_operating_margin

    # --- Step 5: forward-shift within cik group -> next_op_margin_trend ---
    df["next_op_margin_trend"] = df.groupby("cik")["op_margin_trend"].shift(-1)

    # --- Step 6: sector-percentile features (quarter-T data only) ---
    # rank(pct=True) within the (fiscal_quarter, sic) cross-section: purely
    # contemporaneous peer comparison, no T+1 values involved. NaN ratios and
    # NaN sic stay NaN.
    by_sector_quarter = df.groupby(["fiscal_quarter", "sic"])
    for col in PCTILE_COLS:
        df[f"{col}_sector_pctile"] = by_sector_quarter[col].rank(pct=True)
    print(f"Added {len(PCTILE_COLS)} sector-percentile feature columns: "
          + ", ".join(f"{c}_sector_pctile" for c in PCTILE_COLS))

    # --- Step 7: sector-relative bottom-quartile label ---
    has_trend = df["op_margin_trend"].notna()
    has_next = df["next_op_margin_trend"].notna()
    has_sic = df["sic"].notna()
    valid = has_trend & has_next & has_sic

    def flag_bottom_quartile(s: pd.Series) -> pd.Series:
        if s.notna().sum() < 4:
            return pd.Series(pd.NA, index=s.index, dtype="Int64")
        q25 = s.quantile(0.25)
        return (s <= q25).astype("Int64")

    df["at_risk"] = pd.array([pd.NA] * len(df), dtype="Int64")
    grouped = df.loc[valid].groupby(["fiscal_quarter", "sic"])["next_op_margin_trend"]
    flags = grouped.transform(flag_bottom_quartile)
    df.loc[valid, "at_risk"] = flags.astype("Int64")

    labeled_mask = df["at_risk"].notna()
    n_labeled = int(labeled_mask.sum())

    # --- Step 8: attrition waterfall ---
    n0 = n_dedup
    n1 = int(has_trend.sum())
    n2 = int((has_trend & has_next).sum())
    n3 = int(valid.sum())

    def row(stage, count, lost):
        print(f"  {stage:<58} {count:>9,}  {lost:>8,}  {100 * count / n0:6.1f}%")

    print()
    print("=" * 90)
    print("ATTRITION WATERFALL")
    print("=" * 90)
    print(f"  {'stage':<58} {'rows':>9}  {'lost':>8}  {'remain':>7}")
    row("panel rows (post Session 3, deduped, pre-labeling)", n0, 0)
    row("- no valid op_margin_trend (first quarter per company)", n1, n0 - n1)
    row("- no valid next_op_margin_trend (last quarter per co.)", n2, n1 - n2)
    row("- missing SIC code", n3, n2 - n3)
    row("- SIC-quarter peer group had < 4 labelable companies", n_labeled, n3 - n_labeled)
    row("= final labeled rows", n_labeled, 0)
    print("=" * 90)

    base_rate = df.loc[labeled_mask, "at_risk"].mean()
    print(f"\nOverall base rate of at_risk=1: {base_rate:.4f}")

    # --- Step 9: save ---
    df.to_parquet(PANEL_PATH, index=False)
    print(f"\nSaved updated panel ({len(df)} rows x {df.shape[1]} cols) -> {PANEL_PATH}")

    labeled_out = df[labeled_mask].reset_index(drop=True)
    labeled_out.to_parquet(LABELED_PATH, index=False)
    print(f"Saved labeled_panel ({len(labeled_out)} rows x {labeled_out.shape[1]} cols) -> {LABELED_PATH}")

    print("\nFinal columns:")
    for c in df.columns:
        print(f"  {c}")


if __name__ == "__main__":
    main()
