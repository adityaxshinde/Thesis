# Sector-Adjusted Financial Deterioration Scoring

Predicting whether a company's operating-margin trend will fall into the
bottom quartile relative to its SIC-code sector peers in the following
quarter, using SEC EDGAR structured (XBRL) data.

Pipeline order:

```
scripts/download_edgar_data.py   # fetch raw quarterly EDGAR zips
scripts/build_panel.py           # raw filings -> company-quarter panel
scripts/fix_panel.py             # date filter, debt re-extraction, ratios
scripts/derive_q4_flows.py       # fill missing Q4 flows on 10-K rows
scripts/build_label.py           # trend features, sector percentiles, label
scripts/train_baselines.py       # chronological split, HGB + RF baselines
scripts/feature_analysis.py      # seeding, correlations, permutation importance
```

## Methodology decisions

### 1. Data source and window

The raw data are the SEC's **Financial Statement Data Sets** (DERA), one zip
archive per calendar quarter of filings received by EDGAR. Each archive
contains `sub.txt` (one row per XBRL submission) and `num.txt` (one row per
reported numeric fact). Fifty-seven quarterly archives are downloaded,
**2012 Q1 through 2026 Q1**, with a descriptive `User-Agent` header
(name + email) on every request, as SEC automated-access policy requires.
Downloads are retried up to three times with linear backoff, written
atomically via a `.partial` temp file, and a post-download check flags any
quarter whose `sub.txt` row count falls below 50% of the median of its
neighbouring quarters (within two quarters on each side).

Two boundary caveats:

- **2026 Q1 is a partial quarter.** A fiscal quarter's filings arrive in the
  EDGAR archive of the calendar quarter in which they were *filed*, which is
  typically the following one. Fiscal quarters ending in 2026 Q1 are
  therefore still filling in: many of those filings will appear in the
  2026 Q2 archive, which the SEC has not yet published. Combined with the
  forward-shifted label (which needs quarter T+1's data), usable
  feature–label pairs extend only through **2025 Q4**.
- **Why the sample does not reach back to 2008.** The Financial Statement
  Data Sets only begin in 2009 — there is no structured XBRL data for the
  2008 crisis period in this source. Moreover, the XBRL mandate was phased
  in by filer size, and smaller reporting companies were not required to
  tag their financials until fiscal periods ending in mid-2011. Starting the
  sample at 2012 Q1 avoids a cross-section that is skewed toward large
  filers in the early years.

### 2. Panel construction

The unit of observation is one row per **company (CIK) per fiscal quarter**.
For each quarterly archive, `build_panel.py`:

- keeps only forms **10-K, 10-K/A, 10-Q, 10-Q/A** with `prevrpt == 0`
  (i.e., not superseded by a later report within that archive);
- within the archive, deduplicates on `(cik, period)` keeping the most
  recently *filed* submission;
- joins `num.txt` facts to submissions on accession number (`adsh`),
  keeping only USD-denominated, consolidated facts (no `segments`, no
  `coreg`) whose date (`ddate`) equals the submission's reported period end
  — this excludes comparative prior-period figures restated inside the same
  filing;
- extracts ten raw line items. Because XBRL tag names vary across filers
  and years, each concept is looked up through an ordered list of tag
  variants, taking the first non-null match (e.g., revenues resolves
  through `Revenues` → `SalesRevenueNet` →
  `RevenueFromContractWithCustomerExcludingAssessedTax`). Income-statement
  flows require quarterly duration (`qtrs = 1`); balance-sheet items are
  point-in-time (`qtrs = 0`).

The fiscal-quarter key is the calendar quarter containing the period end
date. Because amendments can be filed in a *later* calendar quarter's
archive than the original filing, and because calendar-quarter bucketing
can collide, the concatenated panel still contained **745 duplicate
`(cik, fiscal_quarter)` rows**; `build_label.py` resolves these by keeping
the latest-filed row per pair, so every company-quarter is unique before
any feature or label depending on within-company ordering is computed.

A small number of rows carried period dates before 2012 — XBRL date-entry
errors in a handful of filings, misparsed into implausible fiscal quarters.
`fix_panel.py` drops everything outside the 2012 Q1 – 2026 Q1 window.

### 3. Feature engineering

**Six core ratios**, computed per company-quarter from the extracted line
items:

| Ratio | Formula |
|---|---|
| `operating_margin` | operating income (loss) / revenues |
| `current_ratio` | current assets / current liabilities |
| `liabilities_to_assets` | total liabilities / total assets |
| `revenue_growth` | (revenues_T − revenues_prev) / revenues_prev, within company |
| `roe` | net income (loss) / stockholders' equity |
| `cash_to_assets` | cash and equivalents / total assets |

All divisions go through a single `safe_ratio()` rule: **if the denominator
is missing or ≤ 0, the ratio is NaN** — there is no division by zero or by
negative denominators (so, e.g., ROE is undefined rather than sign-flipped
for negative-equity firms). `revenue_growth` uses the company's prior panel
row (sorted by period), i.e., the previous *available* quarter; gaps in a
company's filing history are not bridged or interpolated. In practice this
is a minor limitation: only 0.34% of non-null `revenue_growth` values and
0.31% of non-null `op_margin_trend` values are computed across a gap
(i.e., the prior available row is not the immediately preceding fiscal
quarter), falling to 0.26% within the labeled modeling set.

**The leverage decision.** Two leverage measures exist in the panel:

- `liabilities_to_assets` = `Liabilities` / `Assets` is the **primary**
  leverage feature. Both tags are near-universally reported, giving
  **78.88% coverage** (identical to two decimals in the full panel and in
  the labeled modeling set; verified directly from the saved parquet
  files).
- `debt_to_assets_narrow` = reconstructed total debt / `Assets` is kept as
  a **supplementary** column. Total debt has to be reconstructed from
  heterogeneous XBRL debt tags — a long-term component (first match among
  `LongTermDebt`, `LongTermDebtNoncurrent`,
  `LongTermDebtAndCapitalLeaseObligations`, `LongTermNotesPayable`,
  `NotesPayable`, `DebtInstrumentCarryingAmount`) plus a short-term
  component (`ShortTermBorrowings`, `DebtCurrent`, `CommercialPaper`),
  summed with missing components treated as zero unless both are missing.
  (`LineOfCreditFacilityMaximumBorrowingCapacity` was initially included
  in the short-term list but removed: it measures the *capacity* of a
  credit facility, not an outstanding borrowing, so it does not belong in
  a debt-outstanding ratio.) Even after this tag expansion, coverage
  reaches only **58.77% in the labeled modeling set** (44.98% across the
  full panel; both verified directly from the saved parquet files), and
  the tag set mixes concepts of varying precision. A leverage feature
  missing for 41% of labeled rows would dominate complete-case attrition,
  so the broad-coverage liabilities ratio is used for modeling and the
  narrow debt ratio is retained for robustness checks only (it is not
  among the 13 model features).

**Six sector-percentile features.** For each of the six core ratios, the
company's percentile rank (`rank(pct=True)`, average ties) is computed
**within its `(fiscal_quarter, SIC)` peer group**, the peer group being
recomputed fresh each quarter. This is leak-safe by construction: the rank
at quarter T uses only the quarter-T cross-section — it compares companies
to their contemporaneous peers and involves no future values and no
statistics estimated on later data. Rows with a missing ratio or missing
SIC code simply get a missing percentile.

**`op_margin_trend`** — the quarter-over-quarter change in operating
margin, `operating_margin_T − operating_margin_{T−1}` within company — is
itself a feature. It uses only quarter T and earlier data (T-only in the
sense of being fully known at T), and it is the same quantity whose T+1
value defines the label, making it the natural "momentum" predictor.

**Winsorization.** Each ratio is clipped to **fixed, hardcoded bounds**:

| Ratio | Bounds |
|---|---|
| `operating_margin` | (−5, 5) |
| `current_ratio` | (0, 50) |
| `liabilities_to_assets` | (0, 10) |
| `debt_to_assets_narrow` | (0, 10) |
| `revenue_growth` | (−1, 10) |
| `roe` | (−10, 10) |
| `cash_to_assets` | (0, 1) |

These are constants in the code, **not data-derived percentiles**. Clipping
at, say, the sample's 1st/99th percentile would embed statistics of the full
sample — including test-period observations — into every training-period
feature value; fixed a-priori bounds cannot leak information across the
chronological split.

The model uses **13 features**: the six core ratios, `op_margin_trend`, and
the six sector percentiles.

### 4. The Q4 coverage fix

10-Ks report **annual** income-statement flows (`qtrs = 4`), and the panel
extraction keeps only `qtrs ∈ {0, 1}`, so a 10-K row receives quarterly
revenues / operating income / net income only if the filer separately
tagged fourth-quarter figures. Historically many did, because Regulation
S-K **Item 302(a)** required selected quarterly financial data in the
annual report. The SEC **eliminated that requirement** (amendment adopted
November 2020, effective early 2021), and quarterly-flow coverage on 10-K
rows collapsed from roughly **19% in 2019 Q4 to about 2% in 2022 Q4**.

The damage compounds through the label: `at_risk` needs a three-quarter
operating-margin chain (T−1, T, T+1), so a missing Q4 margin destroys the
labels of Q3 (missing T+1), Q4 (missing T), and Q1 (missing T−1). Post-2021,
only Q2 observations survived labeling in volume — a severe seasonal
selection bias in exactly the most recent part of the sample.

`derive_q4_flows.py` repairs this by deriving Q4 flows arithmetically:

> Q4 flow = annual flow (qtrs = 4, from the 10-K itself)
> − (Q1 + Q2 + Q3 quarterly flows from the same fiscal year's 10-Q rows)

Safeguards, in order:

- Annual facts are extracted with the same tag variants, priority order,
  and USD / no-segment / no-coregistrant / `ddate == period` filters as the
  original extraction, restricted to the panel's 10-K (`fp == "FY"`) rows.
- The Q1–Q3 sum requires **all three quarters present** for that
  `(cik, fiscal year)`, each ending within **400 days** before the 10-K's
  period end — a guard against fiscal-year changes and stale `fy` values —
  with duplicates per `(adsh, fiscal period)` dropped so no quarter is
  double-counted.
- Derived values **fill only missing** flow fields on FY rows; a reported
  Q4 figure is never overwritten.
- Derived **negative revenues are discarded** as restatement artifacts.
- The dependent ratios (`operating_margin`, `roe`, `revenue_growth`) are
  then recomputed panel-wide under the same `safe_ratio` and winsorization
  rules; balance-sheet ratios are unaffected.

The derivation introduces **no lookahead**: every input (the annual figure
and the three prior 10-Qs) was public at the 10-K's filing date, i.e.,
known by the time the panel row exists at all. The fix grew the labeled set
from **53,895 to 130,338** company-quarters and restored Q3/Q4/Q1 labels in
the post-2021 period.

### 5. Label definition

The target `at_risk` is built in `build_label.py`:

1. `op_margin_trend` is forward-shifted within company to give
   `next_op_margin_trend` — quarter T+1's margin change, attached to the
   quarter-T row.
2. Within each `(fiscal_quarter, SIC)` peer group of quarter-T rows,
   `at_risk = 1` if the company's `next_op_margin_trend` is **at or below
   the peer group's 25th percentile**, else 0. Because the flag is
   `≤ quantile(0.25)`, the realized base rate sits slightly above 25%
   (ties and the ≤ comparison): 0.2807 in the current labeled set.
3. A row is labelable only if it has a valid `op_margin_trend` (feature
   side), a valid `next_op_margin_trend` (label side), a non-null SIC code,
   and **at least 4 labelable companies** in its peer group (a quartile over
   fewer than four firms is not meaningful). All other rows get a missing
   label and are excluded from `labeled_panel.parquet`.

`next_op_margin_trend` and `at_risk` are label-side columns only. The
training script asserts they are absent from the feature list; no
T+1-derived quantity appears among the 13 features.

### 6. Train/test protocol and evaluation

- **Chronological split, never random.** Rows with `fiscal_quarter` before
  **2022 Q1** train; 2022 Q1 onward tests. A random split on panel data
  would place a company's quarter T+1 in train and quarter T in test,
  letting the model memorize forward information; the chronological split
  also mimics deployment, where the model scores quarters it has never
  seen.
- **Complete-case features.** Rows with any missing feature are dropped
  before splitting (no imputation at the baseline stage).
- **Metrics: PR-AUC (average precision) and precision@top-10%**, not
  accuracy or ROC-AUC alone. With a ~27% base rate and an early-warning use
  case, what matters is the quality of the highest-risk decile a screener
  would actually review. Both metrics are reported as **lift over a naive
  baseline** that assigns every test row a constant probability equal to
  the training base rate — its average precision equals the test positive
  rate, and its expected precision@top-k is the test base rate.

### 7. Baseline models

Two deliberately untuned, defaults-first baselines from scikit-learn:

- **`HistGradientBoostingClassifier`** — library defaults; re-trained with
  `random_state=42` in the feature-analysis pass for reproducibility. The
  saved model (`data/models/hgb_baseline.joblib`) and every reported HGB
  metric come from this seeded run.
- **`RandomForestClassifier(n_estimators=300, random_state=42)`**.

On the 2022 Q1+ test window, HGB reaches **PR-AUC 0.459 (≈1.6× lift)** and
**precision@top-10% 0.545 (≈1.9× lift)** over the naive baseline
(`outputs/model_metrics.csv`).

`feature_analysis.py` adds three diagnostics:

- **Multicollinearity**: training-set correlation matrix with pairs at
  |r| > 0.7 flagged, including each raw ratio against its own sector
  percentile.
- **Permutation importance** (scoring = average precision, 5 repeats,
  seed 42) on the full test set for both models. `op_margin_trend`
  dominates, at roughly 3× the next feature — consistent with the label
  being the sector-relative quartile of the same quantity one quarter
  ahead.
- **Temporal stability**: permutation importance recomputed separately on
  the two halves of the test window (2022 Q1–2023 Q4 vs 2024 Q1 onward);
  the top-6 feature ranking is stable across the two periods.

## Limitations

- **Revenue / operating-margin coverage constrains the sample.** An
  operating margin is computable on only **49.2%** of panel rows (quarterly
  revenues and operating income are jointly tagged on 51.4%; verified
  directly from the saved parquet), and the label additionally requires a
  three-quarter margin chain and a ≥ 4-firm peer group (§5) — so **130,338
  of 355,209** company-quarters (~37%) are labelable, even after the Q4 flow
  derivation of §4. The narrow debt ratio is excluded from the feature set
  for the same reason (**58.77%** coverage in the labeled set; §3).
- **Survivorship / exit bias.** EDGAR reflects companies that are still
  filing. A row is labeled only if the company files a comparable quarter
  T+1, so firms that delist, are acquired, or go bankrupt exit the panel at
  exactly the moment of greatest deterioration; the worst outcomes are
  under-represented in both training and evaluation.
- **SIC coarseness as a sector proxy.** Peer groups are raw SIC codes
  recomputed each quarter (§3). SIC is a dated classification of uneven
  granularity — some four-digit codes mix dissimilar businesses while others
  split similar ones — peer groups with fewer than 4 labelable firms are
  dropped (§5), and a misclassified company is ranked against the wrong
  peers.
- **Scope: operating companies only.** The ratios presuppose an operating
  business with meaningful revenues and margins. Financial firms, funds,
  shells, and pre-revenue companies have undefined ratios under the
  `safe_ratio` rule (§3) and fall out of the complete-case sample; results
  apply to operating companies with reportable revenue/margin data, not the
  full EDGAR filer population.
