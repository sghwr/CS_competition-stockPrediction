# Project Contract

## Root and data

Default project root:

```text
E:/USELESS/数据分析学习/数分竞赛学习/CS-competition
```

Use paths supplied by the user when they differ. Run commands from the project root.

Canonical files:

- `data/stock_data.csv`: HS300 constituent daily stock data.
- `data/index_data.csv`: HS300 benchmark data.
- `data/hs300_stock_list.csv`: current competition universe.
- `data/stock_industry.csv`: stock-industry mapping.
- `code/config/manual_policy.json`: mutable default policy; do not use it after a run-local policy is frozen.

## Existing commands

Update stock data to the latest effective trading day:

```powershell
python "get_stock_data.py"
```

Always override the index updater's static default end date:

```powershell
python "update_index_data.py" --start-date 2019-01-02 --end-date <signal-date> --output "data/index_data.csv"
```

Generate as-of scores and portfolios:

```powershell
python "code/src/predict_asof.py" --signal-date <signal-date> --base-dir "<frozen-output-dir>" --model-dir "<frozen-model-dir>" --output-dir "<run>/prediction" --policy "<run>/frozen/manual_policy.json" --selection industry_diversified --top-k 5
```

Validate competition output:

```powershell
python "code/src/validate_result.py" "<run>/prediction/result.csv"
```

Evaluate one historical signal in test mode only:

```powershell
python "code/src/evaluate_signal.py" --prediction-dir "<run>/prediction" --signal-date <signal-date> --buy-date <buy-date> --sell-date <sell-date> --policy "<run>/frozen/manual_policy.json" --output "<run>/first_reveal_evaluation.json"
```

## Checkpoint eligibility

Model artifacts normally pair these trees:

```text
output/holdout_models/<checkpoint-date>/
model/holdout_models/<checkpoint-date>/
```

Use the pair with the newest training cutoff not later than `signal_date`. Inspect metadata when available. A directory date alone is insufficient if training-cutoff semantics are ambiguous. Never retrain in this Skill. Stop when no eligible complete pair exists.

`predict_asof.py` writes:

- `base_scores.csv`
- `portfolio_base.csv`
- `portfolio_manual.csv`
- `result.csv`
- `prediction_config.json`

Keep all five files in the run directory.

## As-of integrity

`manual_policy.py` filters recent stock features to dates `<= asof_date`. Confirm any additional feature path used by the selected checkpoint obeys the same condition. News evidence may extend from the last trading-day `signal_date` to a pre-buy weekend cutoff, but it may affect only the frozen manual layer. Having later rows in `stock_data.csv` is acceptable only when every prediction feature is explicitly sliced to `signal_date`; it is not permission to inspect target-period values during research.
