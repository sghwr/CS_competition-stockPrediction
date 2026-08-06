---
name: update-stock-news-signal
description: Run the CS-competition stock workflow with strict news cutoffs, incremental stock/index updates, frozen manual-policy evidence, as-of prediction, and leakage-safe T+5 evaluation. Use when the user explicitly says "测试模式" to simulate and reveal a completed historical date range, or "预测模式" to produce a real unrevealed forecast for a specified future date range, including requests to update stock_data, research time-bounded news, inject hard rules, generate result.csv, or evaluate excess return and NDCG@5.
---

# Stock News Signal Workflow

## Resolve the request

1. Require the user to state exactly one mode: `测试模式` or `预测模式`.
2. Resolve all relative dates to `YYYY-MM-DD` in Asia/Shanghai time.
3. Establish `news_cutoff`, `signal_date`, `buy_date`, and `sell_date` before any research.
4. Treat the requested range as the inclusive holding window. Require `signal_date < buy_date <= sell_date` and `news_cutoff` before `buy_date`. The cutoff may follow `signal_date` over a weekend because model features and news evidence have separate as-of times.
5. State any date assumption before execution. Stop if a material date cannot be inferred safely.

Read [references/project-contract.md](references/project-contract.md) before invoking project scripts. Read [references/news-cutoff-policy.md](references/news-cutoff-policy.md) before web research. Read [references/evaluation-contract.md](references/evaluation-contract.md) before revealing outcomes.

## Enforce mode boundaries

### 测试模式

- Require the holding window to have fully ended.
- Do not inspect target-window prices, returns, rankings, or later news until the research manifest and manual policy are frozen.
- After freezing, update data, run as-of prediction from an eligible frozen checkpoint, and evaluate both base and manual portfolios.
- Preserve the first reveal. Never tune a rule against the revealed result and overwrite it as if it were the original test.

### 预测模式

- Require `buy_date` to be later than the current Asia/Shanghai date.
- Update market features only through `signal_date` and use news only through `news_cutoff`; never query or read target-window outcomes.
- Use an eligible frozen checkpoint. Do not train or update model weights.
- Generate base/manual portfolios and validate `result.csv`.
- Do not invoke `evaluate_signal.py`, calculate realized return, or describe the forecast as profitable.

Do not silently switch modes. If the requested dates conflict with the selected mode, stop and explain the conflict.

## Execute the workflow

1. Create a unique run directory under `output/stock_news_signal/<mode>/<signal-date>/<run-id>/`.
2. Research news using the cutoff policy. Record every query and accepted source in a JSON manifest.
3. Write the natural-language thesis first, then encode only supported views in a run-local `manual_policy.json`.
4. Freeze the evidence and policy before data reveal:

```powershell
python "<skill>/scripts/freeze_research.py" --mode <test|predict> --news-cutoff <ISO-8601> --signal-date <YYYY-MM-DD> --buy-date <YYYY-MM-DD> --sell-date <YYYY-MM-DD> --news-manifest "<manifest.json>" --policy "<manual_policy.json>" --output-dir "<run>/frozen"
```

5. Update stock and index data. In both modes, pass `signal_date` as `--index-end-date`; in historical testing, update latest stock data only after freezing.

```powershell
python "<skill>/scripts/update_market_data.py" --project-root "<project>" --index-end-date <signal-date>
python "<skill>/scripts/validate_market_data.py" --project-root "<project>" --required-through <signal-date> --output "<run>/market_validation.json"
```

6. Select the newest checkpoint whose training cutoff is not later than `signal_date`. Stop if eligibility cannot be established from metadata or directory naming.
7. Run `predict_asof.py` with the frozen policy copy, not the mutable project policy.
8. Run `validate_result.py` on `result.csv`.
9. In `测试模式` only, run `evaluate_signal.py` after prediction and label the resulting file as the first reveal.
10. Report source coverage, policy hash, checkpoint, selected stocks, and mode-appropriate outputs. Separate factual evidence from investment judgment.

## Required manifest shape

Create UTF-8 JSON with this minimum shape:

```json
{
  "queries": ["query text"],
  "items": [
    {
      "title": "source title",
      "url": "https://...",
      "source": "publisher",
      "published_at": "2026-07-26T18:00:00+08:00",
      "retrieved_at": "2026-08-01T10:00:00+08:00",
      "query": "query text",
      "applies_to": ["industry or stock code"],
      "evidence": "claim supported by this source"
    }
  ]
}
```

Use an empty `items` list rather than weakening the cutoff when no valid evidence exists. In that case, keep the manual policy disabled unless date-independent priors were explicitly declared before the test.

## Stop conditions

Stop without prediction or evaluation when:

- a relied-on source lacks a verifiable publication date or exceeds the cutoff;
- the policy or manifest changes after freezing;
- market data validation fails;
- a checkpoint may contain training data after `signal_date`;
- as-of feature slicing cannot be verified;
- target-period outcomes were inspected before a test freeze;
- a prediction request would reveal an already-started holding window.

Never provide certainty or guaranteed-return language. Treat news-based rules as hypotheses and always preserve the base-model portfolio for comparison.
