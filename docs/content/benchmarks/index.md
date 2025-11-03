# Benchmarks

TiRex is evaluated on two public leaderboards that cover a broad mix of time-series domains and scoring rules.

## GIFT Eval

- **What it is:** The [GIFT Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) provides a unified zero-shot evaluation over 23 datasets, ~144k time series and 177 M data points, spanning seven domains (from from retail, energy, traffic, up to macroeconomic sources), 10 frequencies, multivariate inputs, and short- to long-term horizons (see Sec. 4 of the [TiRex paper](https://arxiv.org/abs/2505.23719)).
- **Metrics:** Models are reported and ranked using MASE (point forecasts) and CRPS (probabilistic forecasts); the public leaderboard exposes MASE_Rank and CRPS_Rank for comparison.
- **Reproducibility:** The evaluation uses fixed splits (final 10% of each dataset as test) with non-overlapping rolling windows; the code, data, and leaderboard are publicly available to reproduce results. Have a look at the [TiRex GIFT Eval Notebook](https://github.com/SalesforceAIResearch/gift-eval/blob/main/notebooks/tirex.ipynb) to reprocude the results.

## fev-bench

- **What it is:** The [fev-bench leaderboard](https://huggingface.co/spaces/autogluon/fev-leaderboard) reports results on 100 forecasting tasks across seven domains, including 46 tasks with covariates and both univariate and multivariate settings, with varied sampling cadences and horizons (see [fev-bench paper](https://arxiv.org/abs/2509.26468)). It includes telemetry, finance, climate, and competition datasets with varying sampling cadences.
- **Metrics:** Tasks use MASE for point forecasts and SQL (Scaled Quantile Loss) for probabilistic forecasts; leaderboard summaries emphasize win rates and skill scores with bootstrap confidence intervals for statistically rigorous aggregation.
- **Reproducibility:** Submissions follow the lightweight fev library and fully specified task definitions and splits; extended tabular and interactive results are hosted on the HF Space. Have a look at the [fev-bench TiRex Example](https://github.com/autogluon/fev/tree/main/examples/tirex) to reprocude the results.

## Key Takeaways

- TiRex consistently ranks among the top zero-shot forecasters across common leaderboards without task-specific fine-tuning.
- The public evaluation scripts, datasets, and checkpoints referenced above allow anyone to reproduce the official scores end-to-end.
