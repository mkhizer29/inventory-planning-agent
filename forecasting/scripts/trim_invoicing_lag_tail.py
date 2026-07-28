"""
2026-07-26: trim the most recent days off `training_dataset_30skus.csv`
because they're still-settling / under-invoiced, not real demand.

Found while investigating why a model retrain's test-set accuracy dropped
after extending the file to 2026-07-23: `net_qty` is built from
`qty_invoiced - qty_refunded` (see training_dataset_30skus_column_sources.md),
and invoicing lags behind order placement by a few days. Compared the
invoiced/ordered ratio for the newly-added tail against a fully-settled
baseline period (mid-June 2026, which runs ~42-74% day to day -- not every
order gets invoiced same-day, that's normal):

  2026-07-20  62.1%  (normal)
  2026-07-21  50.2%  (normal)
  2026-07-22  38.0%  (below normal range -- still settling)
  2026-07-23  21.5%  (well below normal range -- still settling)
  2026-07-24  0.0%   (already-known partial/cutoff day)

Only 07-22 and 07-23 are actually still-settling -- 07-21 and earlier are
within the normal ratio range. Real cutoff for reliable net_qty is
2026-07-21, not 07-23.

No lag/rolling recompute needed: those are shift()/rolling() over each SKU's
own prior rows, so dropping rows off the END of the series doesn't change
any earlier row's feature values.
"""
import pandas as pd

MASTER = "../data/training_dataset_30skus.csv"
CUTOFF = pd.Timestamp("2026-07-21")

df = pd.read_csv(MASTER)
parsed = pd.to_datetime(df["date"], format="%d/%m/%Y")
n_before = len(df)
df = df[parsed <= CUTOFF].copy()
print(f"Trimmed to <= {CUTOFF.date()}: {n_before} -> {len(df)} rows "
      f"(dropped 07-22, 07-23 as still-settling/under-invoiced)")

df.to_csv(MASTER, index=False)
print(f"Saved -> {MASTER}")
