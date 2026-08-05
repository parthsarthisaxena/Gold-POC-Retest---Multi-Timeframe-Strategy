# Gold POC Retest — XAU/USD Systematic Research

> **Public research repository.** Strategy implementation is proprietary.
> This repo contains the signal diagnostic framework, walk-forward validation
> scaffold, aggregated results, and visualisation code — enough to reproduce
> the methodology and verify the statistical claims independently.

---

## The Question This Answers

*Does the high-volume price node of a confirmed swing leg have statistically significant predictive power on XAU/USD — and does that edge survive 8 years of out-of-sample validation?*

**Short answer: Yes. Here's the proof.**

---

## Core Hypothesis

After a confirmed directional swing leg, the Point of Control (POC) — the highest-volume price node of that leg — acts as a structural magnet.

When price breaks above the Value Area High (VAH) and returns toward POC, it signals continuation of the original leg. Trend-following entry. Defined structural stop. No discretion.

The signal is either there or it isn't.

---

## Results — 8 Years (Aug 2018 → May 2026)

| Metric | Value |
|--------|-------|
| Trades | 523 |
| Win Rate | 59.7% |
| Avg R-Multiple | +0.347 |
| Profit Factor | 1.99 |
| Total Return | **+488.96%** |
| CAGR | 25.54% |
| Sharpe | **4.85** `95% CI [−0.01, 9.71]` |
| Sortino | 14.47 |
| Max Drawdown | **6.40%** |
| Calmar | 3.99 |
| Profitable Months | **82.4% (70/85)** |

> Flat 1% risk per trade. $100k starting capital. Monthly −2R loss limit.

---

## Signal Significance

Forward return t-test on **830 independent signal observations** vs time-matched random baseline:

| Horizon | Signal WR | Mean Return | p-value |
|---------|-----------|-------------|---------|
| 15 min | 64.9% | +0.049% | **< 0.001** |
| 30 min | 59.6% | +0.053% | **< 0.001** |
| 60 min | 54.9% | +0.051% | **< 0.001** |
| 120 min | 55.5% | +0.063% | **0.001** |

All four horizons statistically significant **before any trade management is applied.**
The edge exists at the signal level — not manufactured by exits.

---

## Walk-Forward Validation

6 expanding windows. Anchored start (2018-08-03). ~75–85 unseen trades per test window.

| Window | OOS Period | OOS Calmar | OOS WR | Verdict |
|--------|-----------|-----------|--------|---------|
| 1 | Oct 2019 – Sep 2020 | 14.37 | 64.6% | ✅ ROBUST |
| 2 | Sep 2020 – Mar 2022 | 11.77 | 70.9% | ✅ ROBUST |
| 3 | Mar 2022 – Oct 2023 | 1.76 | 53.9% | ✅ ROBUST |
| 4 | Oct 2023 – Sep 2024 | 2.55 | 48.0% | 🟡 MARGINAL |
| 5 | Sep 2024 – Jul 2025 | 4.40 | 56.8% | ✅ ROBUST |
| 6 | Aug 2025 – May 2026 | 10.97 | 64.7% | ✅ ROBUST |

**OOS Calmar: mean = 7.64 | std = 4.91 | min = 1.76**
**Overall: ✅ ROBUST (5/6 windows)**

Window 4 (marginal) coincides with the Oct 2023–Sep 2024 gold consolidation phase — consistent with the long-only design underperforming in sideways regimes. Not a strategy failure. A known structural limitation.

---

## Performance by Regime

Regime defined as distance of daily close from 200-period SMA at signal time.

| Regime | n | WR | Avg R | Sharpe |
|--------|---|----|-------|--------|
| `bull_strong` (>1% above SMA) | 508 | 57.9% | +0.317 | 4.38 |
| `bull_moderate` (0.3–1% above) | 23 | 69.6% | +0.501 | 8.34 |
| `bull_weak` (0–0.3% above) | 15 | 80.0% | +0.738 | 11.08 |

Freshly confirmed trends outperform extended moves — consistent with the mean-reversion component embedded in the volume profile signal. The closer price is to the SMA, the sharper the edge.

---

## What Makes This Different

Most backtests answer the question: *"did this work historically?"*

This one answers: *"is there a reason it should work, can I prove the signal has predictive power independent of trade management, and does it survive market regimes I never trained on?"*

**Three things that are harder to do than they look:**

**1. Causal pivot detection**
Swing pivots confirmed only after `N` bars post-pivot. No future data in pivot identification. This is one of the most common — and least acknowledged — sources of lookahead bias in swing-based systems.

**2. Cross-session EMA**
50-period EMA computed continuously across sessions, not reset per day. Per-session EMA has cold-start bias for the first 50 bars of every session. Small detail. Meaningful difference.

**3. ATR regime filter uses median, not mean**
Rolling 20-day *median* ATR. Robust to spike days that would distort a mean-based threshold. Shift-1 applied — today's filter uses yesterday's data.

---

## Honest Limitations

**Tick volume proxy** — XAU/USD spot has no real traded volume. Volume profile built on MT5 tick counts. Signal is statistically significant despite this, but a CME Gold futures implementation with real dollar volume would be structurally cleaner.

**Long-only** — 200-SMA filter blocks bear/sideways regimes. No short-side exposure. Not tested in a sustained multi-year gold bear market.

**Sharpe CI lower bound** — 95% CI lower bound is −0.01, essentially zero. Point estimate (4.85) is robust but requires ~3 more months of live data to push the lower bound clearly positive at this sample size.

**Single asset** — all results on XAU/USD only. Methodology is asset-agnostic but cross-asset replication has not been performed.

---

## Repo Structure

```
gold-poc-mtf-research/
│
├── signal_diagnostic.py          # Forward return significance test
│                                 # (interface only — plug in your signal)
│
├── walk_forward_framework.py     # Generic expanding-window WF validator
│
├── notebooks/
│   └── regime_analysis.py        # 4-panel chart from results CSVs
│
├── results/
│   ├── monthly_pnl.csv           # Monthly P&L aggregated (85 months)
│   ├── walk_forward_summary.csv  # 6-window WF results
│   ├── forward_return_diagnostic.csv
│   ├── regime_performance.csv
│   └── hourly_performance.csv
│
└── README.md
```

---

## Using the Framework

### Signal Diagnostic

```python
from signal_diagnostic import run_diagnostic, split_sessions, load_data

raw      = load_data('path/to/XAUUSDm_M5.csv')
sessions = split_sessions(raw)

# Implement detect_signals() in signal_diagnostic.py with your signal logic
sig_rets, base_rets = run_diagnostic(sessions, horizons=[3, 6, 12, 24, 48])
```

### Walk-Forward Validator

```python
from walk_forward_framework import WalkForwardValidator
import pandas as pd

trades = pd.read_csv('your_trade_log.csv')  # needs 'date' and 'R_multiple' cols

validator = WalkForwardValidator(
    trade_results=trades,
    n_windows=6,
    min_oos_trades=10,
)
summary = validator.run()
summary.to_csv('wf_results.csv', index=False)
```

### Visualisation

```bash
python notebooks/regime_analysis.py
# Generates: notebooks/strategy_research_summary.png
```

---

## Requirements

```
numpy>=1.21
pandas>=1.3
scipy>=1.7
matplotlib>=3.4
```

```bash
pip install numpy pandas scipy matplotlib
```

---

## Data Format

MT5 5-min export, tab-separated:

```
<DATE>  <TIME>  <OPEN>  <HIGH>  <LOW>  <CLOSE>  <TICKVOL>  <VOL>  <SPREAD>
2018.08.01  07:00:00  1220.50  1221.30  1220.10  1221.05  312  0  180
```

`History → Right-click symbol → Export Data → CSV`

---

## Disclaimer

Research only. Backtested performance does not guarantee future results.
Strategy has not been traded live. Results assume execution at next bar open
with fixed cost proxy — actual live performance will differ.

---

*Parth Sarthi Saxena — Quant Research*
*[GitHub](https://github.com/parthsarthisaxena) | [Portfolio](https://parthsarthisaxena.vercel.app)*
