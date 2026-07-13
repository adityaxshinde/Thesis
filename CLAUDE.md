## Project context
Thesis: sector-adjusted financial deterioration scoring using SEC EDGAR
structured data. Predicting whether a company's operating margin trend will
fall into the bottom quartile relative to its SIC-code sector peers in the
following quarter.

## Hard rules — never violate these
- Train/test splits must always be chronological. Never use a random
  train_test_split on this panel data.
- The label is forward-shifted: features come from quarter T, the label is
  based on quarter T+1's sector-relative operating margin quartile. Never let
  T+1 data leak into feature columns.
- Sector peer groups are defined by SIC code, computed fresh per quarter.
- Evaluate with PR-AUC and precision@top-k, not plain accuracy or ROC-AUC alone.
- SEC EDGAR requires a descriptive User-Agent header (name + email) on every
  automated request, or requests get blocked.

## Data
- Raw EDGAR files live in data/raw/, never edit these directly
- Processed panel goes in data/processed/panel.parquet
- Sample window: 2012 Q1 through 2026 Q1
- Quarters are downloaded through 2026 Q1, but usable feature-label pairs
  only extend through 2025 Q4, since the label needs the following
  quarter's data and 2026 Q2 hasn't been published by the SEC yet.
