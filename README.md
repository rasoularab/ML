<<<<<<< HEAD
# quantml-starter

A fixed project layout for the 10-week ML-for-quant-finance plan. You write the logic; the stubs give the signatures, the conventions and the tests that define "done".

## Quick start (you run these; nothing here has been executed)

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # everything fails with NotImplementedError at first: that is expected
```

Work in this order every day: read the day's row in the tracker, open the file named in the "Output" column, fill the TODOs, run the tests named in "Done when", commit.

## The seven-step finance recipe

Every experiment, from Week 1 to the capstone, follows the same steps. `pipeline.run_experiment` is the skeleton of this recipe.

1. **Question and timing.** Write one sentence: what do we predict, for which assets, over which horizon. Fix the timing convention (see `docs/CONVENTIONS.md`).
2. **Data.** Load through `data.py` only. Record the universe rule and the date range.
3. **Features and target.** Features only from data up to t-1. Target from `labels.py`. Run the look-ahead tests.
4. **Split.** Walk-forward or purged CV from `cv.py`. Never shuffle a time series.
5. **Baseline first.** Fit `ZeroForecast` or `PrevailingMean`, then OLS, then your model. A model without a baseline next to it is not a result.
6. **Evaluate out of sample.** `oos_r2` (state the benchmark), `information_coefficient`, and net-of-cost Sharpe where a portfolio exists.
7. **Log and write up.** `ExperimentLog.record(...)` for every configuration you evaluate (this counts trials for the deflated Sharpe ratio in Week 10). Fill `projects/TEMPLATE_README.md`.

## Rules

- One Output file per day. If the tests named in the tracker pass, the day is done.
- Do not fit scalers, PCA or feature selectors on anything but the training fold.
- Commit every night: `Wk<N> <day>: <what you built>`.
- New modules for later weeks are added at the start of each week, so you only see what you need now.

## Layout

```
src/quantml/        data, features, labels, cv, models/, evaluate, portfolio, pipeline
tests/              the definition of done for each module
config/             one YAML per experiment
projects/           one folder per mini-project, each with a README from the template
experiments/        throwaway scripts and the trial log (experiments/log.jsonl)
notes/              your weekly notes and derivations (markdown)
docs/               CONVENTIONS.md and WEEKLY_BUILD_MAP.md
```
=======
# ML
Machine learning for quantitative finance — feature engineering, model implementation, and backtesting on financial data.
>>>>>>> 6691d370bfcc07d5aef6a9a9aca9abe4cba0ccd7
