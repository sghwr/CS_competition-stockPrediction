# Evaluation Contract

## Mode separation

In test mode, freeze research and policy before reading any target-window price or return. Then make one first reveal and preserve it under a non-overwritten path.

In prediction mode, do not call evaluation code, inspect target dates, or infer realized performance from partial data. Report only portfolio construction, evidence, and validation status.

## Date convention

Use the existing project implementation:

```text
stock_return = sell_date_open / buy_date_open - 1
benchmark_return = HS300_sell_open / HS300_buy_open - 1
excess_return = portfolio_return - benchmark_return
```

The requested holding range is inclusive. Verify that `buy_date` and `sell_date` are trading days and that both stock and benchmark opens exist. Do not silently substitute adjacent dates.

## Required metrics

For each base/manual portfolio report:

- portfolio return;
- HS300 benchmark return;
- excess return;
- excess win (`excess_return > 0`);
- NDCG@5 when ranking or a single-stock strategy is being assessed;
- holding-level stock return and weighted contribution.

For multiple independent test windows additionally report mean return, mean and median excess return, excess win rate, worst excess return, cumulative excess return, and mean NDCG@5. Do not describe one window's boolean excess win as a win rate.

Use `code/src/evaluate_signal.py` and `code/src/eval_metrics.py` as the authoritative implementation. If a requested metric is absent, add it to the test report without changing the frozen predictions.

## Ablation

Always retain:

- base model scores and portfolio;
- manual-adjusted portfolio;
- frozen policy hash;
- evidence manifest hash;
- eligible checkpoint identity.

Compare base versus manual on the same signal, buy, and sell dates. Never use revealed outcomes to revise the same run's policy. A revised policy is a new, clearly labeled exploratory run and must not replace the first reveal.
