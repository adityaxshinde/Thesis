"""Baseline models for the at-risk classification task (Session 5).

- 13 quarter-T features (6 ratios + op_margin_trend + 6 sector percentiles)
- label: at_risk (bottom-quartile next-quarter sector-relative margin trend)
- chronological split: train < 2022Q1 <= test (never random on panel data)
- metrics: PR-AUC (average precision) and precision@top-10%, plus lift over
  a naive constant-probability baseline
- models: HistGradientBoostingClassifier (sklearn defaults) and
  RandomForestClassifier(n_estimators=300, random_state=42) -- no tuning,
  clean defaults-first baseline.

Outputs: data/models/{hgb,rf}_baseline.joblib, outputs/pr_curve.png,
outputs/model_metrics.csv
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LABELED_PATH = os.path.join(ROOT, "data", "processed", "labeled_panel.parquet")
MODELS_DIR = os.path.join(ROOT, "data", "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

FEATURES = [
    "operating_margin", "current_ratio", "liabilities_to_assets",
    "revenue_growth", "roe", "cash_to_assets", "op_margin_trend",
    "operating_margin_sector_pctile", "current_ratio_sector_pctile",
    "liabilities_to_assets_sector_pctile", "revenue_growth_sector_pctile",
    "roe_sector_pctile", "cash_to_assets_sector_pctile",
]
LABEL = "at_risk"
LEAK_COLS = ["next_op_margin_trend", "at_risk"]  # label-side, never features
SPLIT_QUARTER = "2022Q1"
TOP_FRAC = 0.10

# dataviz reference palette (light mode)
C_SURFACE = "#fcfcfb"
C_HGB = "#2a78d6"      # categorical slot 1 (blue)
C_RF = "#1baf7a"       # categorical slot 2 (aqua)
C_BASELINE = "#898781"  # muted ink
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"


def precision_at_top_k(y_true, scores, frac=TOP_FRAC):
    k = max(1, int(np.ceil(frac * len(y_true))))
    order = np.argsort(-scores, kind="stable")
    return float(np.asarray(y_true)[order[:k]].mean()), k


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    df = pd.read_parquet(LABELED_PATH)
    print(f"Loaded labeled panel: {len(df):,} rows x {df.shape[1]} cols")
    assert all(c in df.columns for c in FEATURES + LEAK_COLS), "missing columns"
    assert not any(c in FEATURES for c in LEAK_COLS), "label column leaked into features"
    print(f"Features ({len(FEATURES)}): T-only ratios/trend/percentiles; "
          f"excluded label-side columns: {LEAK_COLS}")

    # --- 1. feature matrix: drop rows with any NaN feature ---------------
    sub = df.dropna(subset=FEATURES + [LABEL]).copy()
    print(f"\nRows after dropping NaN features: {len(sub):,} "
          f"(dropped {len(df) - len(sub):,})")

    # --- 2. chronological split ------------------------------------------
    fq = sub["fiscal_quarter"].astype(str)
    train, test = sub[fq < SPLIT_QUARTER], sub[fq >= SPLIT_QUARTER]
    X_tr, y_tr = train[FEATURES], train[LABEL].astype("int64")
    X_te, y_te = test[FEATURES], test[LABEL].astype("int64")
    base_tr, base_te = y_tr.mean(), y_te.mean()
    print(f"\nSplit at {SPLIT_QUARTER} (chronological):")
    print(f"  train: {len(train):,} rows, base rate {base_tr:.4f}")
    print(f"  test:  {len(test):,} rows, base rate {base_te:.4f}")
    print("\nTest rows per quarter:")
    print(test.groupby(fq[test.index]).size().to_string())

    # --- 3. naive baseline: constant prob = train base rate ---------------
    # AP of a constant score equals the test positive rate; precision@top-k
    # under uniform ties is the test base rate in expectation.
    const_scores = np.full(len(y_te), base_tr)
    ap_naive = average_precision_score(y_te, const_scores)
    p10_naive = float(base_te)
    print(f"\nNaive baseline: PR-AUC={ap_naive:.4f}  precision@top-10%={p10_naive:.4f}")

    # --- 4. train ----------------------------------------------------------
    models = {
        "HGB": HistGradientBoostingClassifier(),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    }
    scores = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...", flush=True)
        model.fit(X_tr, y_tr)
        scores[name] = model.predict_proba(X_te)[:, 1]

    # --- 5. evaluate --------------------------------------------------------
    rows = []
    for name in ["HGB", "RF"]:
        ap = average_precision_score(y_te, scores[name])
        p10, k = precision_at_top_k(y_te, scores[name])
        rows.append({"model": name, "pr_auc": ap, "precision_at_top10pct": p10,
                     "pr_auc_lift": ap / ap_naive, "p10_lift": p10 / p10_naive})
    metrics = pd.DataFrame(
        [{"model": "naive_baseline", "pr_auc": ap_naive,
          "precision_at_top10pct": p10_naive, "pr_auc_lift": 1.0, "p10_lift": 1.0}]
        + rows
    )
    _, k = precision_at_top_k(y_te, scores["HGB"])
    print(f"\n(top-10% of test = {k:,} rows)")
    print("\n" + "=" * 72)
    print("METRIC COMPARISON (test set)")
    print("=" * 72)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- 6. save -------------------------------------------------------------
    joblib.dump(models["HGB"], os.path.join(MODELS_DIR, "hgb_baseline.joblib"))
    joblib.dump(models["RF"], os.path.join(MODELS_DIR, "rf_baseline.joblib"))
    metrics.to_csv(os.path.join(OUTPUTS_DIR, "model_metrics.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    fig.patch.set_facecolor(C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    # direct-label anchors: mid-curve, opposite sides, since the two curves
    # nearly coincide (aqua is sub-3:1 on light surface -> relief rule)
    label_spec = {"HGB": (0.35, (0, 10)), "RF": (0.65, (0, -16))}
    for name, color in [("HGB", C_HGB), ("RF", C_RF)]:
        prec, rec, _ = precision_recall_curve(y_te, scores[name])
        ap = metrics.loc[metrics["model"] == name, "pr_auc"].iloc[0]
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{name} (AP={ap:.3f})")
        anchor_rec, offset = label_spec[name]
        anchor_prec = float(np.interp(anchor_rec, rec[::-1], prec[::-1]))
        ax.annotate(name, xy=(anchor_rec, anchor_prec), xytext=offset,
                    textcoords="offset points", color=color, ha="center",
                    fontsize=10, fontweight="bold")
    ax.axhline(ap_naive, color=C_BASELINE, linewidth=1.5, linestyle=(0, (4, 3)),
               label=f"naive baseline (AP={ap_naive:.3f})")
    ax.set_xlabel("Recall", color=C_INK2)
    ax.set_ylabel("Precision", color=C_INK2)
    ax.set_title("Precision-recall: at-risk next-quarter margin deterioration\n"
                 f"test = {SPLIT_QUARTER} onward, {len(test):,} company-quarters",
                 color=C_INK, fontsize=11, loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color=C_GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(C_AXIS)
    ax.tick_params(colors=C_INK2)
    legend = ax.legend(loc="upper right", frameon=False)
    for text in legend.get_texts():
        text.set_color(C_INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "pr_curve.png"),
                facecolor=C_SURFACE, bbox_inches="tight")
    print(f"\nSaved: {MODELS_DIR}\\hgb_baseline.joblib, rf_baseline.joblib")
    print(f"Saved: {OUTPUTS_DIR}\\pr_curve.png, model_metrics.csv")


if __name__ == "__main__":
    main()
