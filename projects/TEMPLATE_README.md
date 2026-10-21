# <Project name>

## Question
One sentence: what is predicted, for which assets, over which horizon.

## Data and timing
- Source and date range:
- Universe rule (and how survivorship is handled):
- Timing convention: row t predicts r_t, features use data up to t-1 (see docs/CONVENTIONS.md)

## Method
- Features (all lagged):
- Target:
- Split scheme (walk-forward / purged k-fold, embargo):
- Models compared: baseline, OLS, <your model>

## Results (out of sample only)
| Model | OOS R2 (benchmark: ...) | IC (t-stat) | Net Sharpe (cost: ... bp) | Notes |
| --- | --- | --- | --- | --- |
| Zero forecast |  |  |  |  |
| OLS |  |  |  |  |
| <model> |  |  |  |  |

Trials evaluated (from experiments/log.jsonl):

## What failed or looked suspicious
-

## Finance checklist
Copy the tracker's Finance Checklist column for this project here and mark each item.

## Next steps
-
