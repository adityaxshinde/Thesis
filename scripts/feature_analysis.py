"""Feature analysis for the baseline models (Session 6).

1. Reproducibility: retrain HGB with random_state=42 (RF already seeded),
   same 13 features and 2022Q1 chronological split; overwrite saved model.
2. Multicollinearity: training-set correlation matrix, flag |r| > 0.7.
3. Permutation importance (scoring=average_precision) on the full test set,
   HGB vs RF side by side.
4. Temporal stability: permutation importance on the two halves of the test
   window (2022Q1-2023Q4 vs 2024Q1 onward), HGB only.

Outputs: data/models/hgb_baseline.joblib (overwritten),
outputs/feature_correlation.csv, outputs/permutation_importance.csv,
outputs/importance_by_period.csv
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from train_baselines import (
    FEATURES, LABEL, LABELED_PATH, MODELS_DIR, OUTPUTS_DIR, SPLIT_QUARTER,
    precision_at_top_k,
)

PERIOD_SPLIT = "2024Q1"  # second half of the test window starts here
PREV_PR_AUC = 0.456
PREV_P10 = 0.538
N_REPEATS = 5
SEED = 42


def perm_table(model, X, y, label):
    print(f"  permutation importance: {label} ({len(y):,} rows) ...", flush=True)
    r = permutation_importance(
        model, X, y, scoring="average_precision",
        n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1,
    )
    return pd.Series(r.importances_mean, index=FEATURES)


def ranked(s):
    return s.rank(ascending=False).astype(int)


def main():
    df = pd.read_parquet(LABELED_PATH)
    sub = df.dropna(subset=FEATURES + [LABEL]).copy()
    fq = sub["fiscal_quarter"].astype(str)
    train, test = sub[fq < SPLIT_QUARTER], sub[fq >= SPLIT_QUARTER]
    X_tr, y_tr = train[FEATURES], train[LABEL].astype("int64")
    X_te, y_te = test[FEATURES], test[LABEL].astype("int64")
    fq_te = fq[test.index]
    print(f"Labeled panel: {len(df):,} rows; complete-feature rows: {len(sub):,}")
    print(f"Split at {SPLIT_QUARTER}: train {len(train):,} / test {len(test):,}")
    print(f"Test quarters: {fq_te.min()} .. {fq_te.max()}")

    # --- 1. reproducibility: seeded HGB -----------------------------------
    print("\n[1] Retraining HGB with random_state=42 ...", flush=True)
    hgb = HistGradientBoostingClassifier(random_state=SEED)
    hgb.fit(X_tr, y_tr)
    s_hgb = hgb.predict_proba(X_te)[:, 1]
    ap = average_precision_score(y_te, s_hgb)
    p10, k = precision_at_top_k(y_te, s_hgb)
    print(f"  PR-AUC:            {ap:.4f}  (previous unseeded run: {PREV_PR_AUC})")
    print(f"  precision@top-10%: {p10:.4f}  (previous unseeded run: {PREV_P10})  "
          f"[k={k:,}]")
    hgb_path = os.path.join(MODELS_DIR, "hgb_baseline.joblib")
    joblib.dump(hgb, hgb_path)
    print(f"  overwrote {hgb_path}")

    rf_path = os.path.join(MODELS_DIR, "rf_baseline.joblib")
    rf = joblib.load(rf_path)
    print(f"  loaded RF (already seeded) from {rf_path}")

    # --- 2. multicollinearity ---------------------------------------------
    print("\n[2] Training-set feature correlation matrix")
    corr = X_tr.corr()
    corr.to_csv(os.path.join(OUTPUTS_DIR, "feature_correlation.csv"))
    pairs = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) > 0.7:
                pairs.append({"feature_a": a, "feature_b": b, "correlation": r})
    flagged = pd.DataFrame(pairs).sort_values(
        "correlation", key=lambda s: s.abs(), ascending=False
    ) if pairs else pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])
    print(f"  pairs with |r| > 0.7: {len(flagged)}")
    if len(flagged):
        print(flagged.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    raw_pct = [(f, f + "_sector_pctile") for f in FEATURES
               if f + "_sector_pctile" in FEATURES]
    print("\n  raw ratio vs its sector percentile:")
    for a, b in raw_pct:
        print(f"    {a:<24s} vs pctile: r = {corr.loc[a, b]:+.3f}")

    # --- 3. permutation importance, full test set --------------------------
    print("\n[3] Permutation importance on full test set "
          f"(scoring=average_precision, n_repeats={N_REPEATS})")
    imp_hgb = perm_table(hgb, X_te, y_te, "HGB")
    imp_rf = perm_table(rf, X_te, y_te, "RF")
    full = pd.DataFrame({
        "hgb_importance": imp_hgb, "hgb_rank": ranked(imp_hgb),
        "rf_importance": imp_rf, "rf_rank": ranked(imp_rf),
    }).sort_values("hgb_importance", ascending=False)
    full.index.name = "feature"
    full.to_csv(os.path.join(OUTPUTS_DIR, "permutation_importance.csv"))
    print("\n" + full.to_string(float_format=lambda v: f"{v:.5f}"))

    # --- 4. temporal stability (HGB) ---------------------------------------
    print(f"\n[4] Temporal stability: test halves split at {PERIOD_SPLIT} (HGB)")
    early = fq_te < PERIOD_SPLIT
    X_e, y_e = X_te[early], y_te[early]
    X_l, y_l = X_te[~early], y_te[~early]
    print(f"  early: {fq_te[early].min()}..{fq_te[early].max()} "
          f"({len(y_e):,} rows, base rate {y_e.mean():.4f})")
    print(f"  late:  {fq_te[~early].min()}..{fq_te[~early].max()} "
          f"({len(y_l):,} rows, base rate {y_l.mean():.4f})")
    imp_e = perm_table(hgb, X_e, y_e, "early half")
    imp_l = perm_table(hgb, X_l, y_l, "late half")
    periods = pd.DataFrame({
        "importance_2022_2023": imp_e, "rank_2022_2023": ranked(imp_e),
        "importance_2024_on": imp_l, "rank_2024_on": ranked(imp_l),
    }).sort_values("importance_2022_2023", ascending=False)
    periods.index.name = "feature"
    periods.to_csv(os.path.join(OUTPUTS_DIR, "importance_by_period.csv"))
    print("\n" + periods.to_string(float_format=lambda v: f"{v:.5f}"))

    print("\nSaved: outputs/feature_correlation.csv, "
          "outputs/permutation_importance.csv, outputs/importance_by_period.csv")


if __name__ == "__main__":
    main()
