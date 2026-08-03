"""
walk_forward_framework.py
==========================
Generic expanding-window walk-forward validation framework.

Designed to assess whether a systematic strategy's performance
is robust out-of-sample or an artefact of in-sample optimisation.

Methodology — Expanding window (anchored):
  - All windows share the same start date
  - Training window grows by one slice per iteration
  - Test window is always the next unseen slice
  - Parameters are NEVER re-optimised on test data
  - This is the most conservative WF design — no data leakage

Why expanding (not rolling)?
  With rolling windows you discard early data. For strategies that
  benefit from longer calibration (like trend-based regime filters),
  expanding windows give a fairer test. Rolling windows are better
  for detecting regime decay over time — both are valid; choose based
  on your hypothesis.

Verdict thresholds:
  ROBUST   : OOS Calmar ≥ 1.5 AND OOS win rate ≥ 50%
  MARGINAL : OOS Calmar ≥ 0.5 AND OOS win rate ≥ 45%
  WEAK     : OOS Calmar > 0   (profitable but low quality)
  BROKEN   : OOS Calmar ≤ 0   (losing money out-of-sample)



Usage:
    from walk_forward_framework import WalkForwardValidator

    validator = WalkForwardValidator(
        trade_results=df,     # DataFrame with 'date', 'R_multiple' columns
        n_windows=6,
        min_oos_trades=10,
    )
    summary = validator.run()

Requirements:
    pip install numpy pandas
"""

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Expanding-window walk-forward validator.

    Parameters
    ----------
    trade_results : pd.DataFrame
        Must contain columns: 'date' (str or datetime), 'R_multiple' (float)
    n_windows : int
        Number of expanding windows (default: 6)
    min_oos_trades : int
        Minimum trades required in OOS window to report a result (default: 10)
    starting_capital : float
        Hypothetical starting capital for equity curve (default: 100_000)
    risk_pct : float
        Fraction of capital risked per trade (default: 0.01)
    """

    def __init__(self, trade_results, n_windows=6, min_oos_trades=10,
                 starting_capital=100_000, risk_pct=0.01):
        self.res    = trade_results.copy()
        self.res['date'] = pd.to_datetime(self.res['date'])
        self.res    = self.res.sort_values('date').reset_index(drop=True)
        self.n_win  = n_windows
        self.min_oos= min_oos_trades
        self.cap    = starting_capital
        self.risk   = risk_pct

    # ── EQUITY SIMULATION ────────────────────────────────────────────────
    def _simulate(self, subset):
        """Simulate flat-risk equity curve on a subset of trades."""
        capital = self.cap
        equity  = [capital]
        for r in subset['R_multiple']:
            capital += r * capital * self.risk
            equity.append(capital)
        return equity

    # ── METRICS FROM EQUITY CURVE ─────────────────────────────────────────
    def _metrics(self, subset, equity):
        if len(subset) == 0 or len(equity) < 2:
            return {}
        eq   = pd.Series(equity)
        ret  = eq.pct_change().dropna()
        wr   = (subset['R_multiple'] > 0).mean()
        mdd  = ((eq.cummax() - eq) / eq.cummax()).max()
        if len(subset) > 1:
            first = subset['date'].iloc[0]
            last  = subset['date'].iloc[-1]
            days  = max((last - first).days, 1)
        else:
            days = 252
        years = max(days / 365.25, 0.01)
        cagr  = (equity[-1] / self.cap) ** (1 / years) - 1
        calmar= cagr / mdd if mdd > 0 else 0.0
        sharpe= (ret.mean() / ret.std() * np.sqrt(252)
                 if ret.std() > 0 else 0.0)
        return {'n': len(subset), 'wr': wr, 'calmar': calmar,
                'sharpe': sharpe, 'mdd': mdd, 'cagr': cagr}

    # ── VERDICT ───────────────────────────────────────────────────────────
    @staticmethod
    def _verdict(calmar, wr):
        if calmar >= 1.5 and wr >= 0.50: return '✅ ROBUST'
        if calmar >= 0.5 and wr >= 0.45: return '🟡 MARGINAL'
        if calmar >  0.0:                return '⚠️  WEAK'
        return '❌ BROKEN'

    # ── RUN ───────────────────────────────────────────────────────────────
    def run(self, verbose=True):
        """
        Run walk-forward validation and return a summary DataFrame.

        Returns
        -------
        pd.DataFrame with one row per window containing:
            window, is_start, is_end, oos_start, oos_end,
            is_n, is_calmar, is_sharpe,
            oos_n, oos_calmar, oos_sharpe, oos_wr, verdict
        """
        all_dates = sorted(self.res['date'].dt.date.unique())
        n         = len(all_dates)
        slice_sz  = n // (self.n_win + 1)

        if slice_sz < 5:
            raise ValueError(
                f'Too few unique trade dates ({n}) for {self.n_win} windows. '
                f'Reduce n_windows or use more data.')

        rows = []

        if verbose:
            SEP = '═' * 96
            print(f'\n{SEP}')
            print(f'  WALK-FORWARD VALIDATION  ({self.n_win} expanding windows, '
                  f'anchored start)')
            print(SEP)
            print(f'  {"Win":>4}  {"IS period":>24}  {"OOS period":>24}  '
                  f'{"IS n":>5}  {"IS Cal":>7}  '
                  f'{"OOS n":>6}  {"OOS Cal":>8}  {"OOS WR":>7}  {"Verdict":>12}')
            print(f'  {"-"*92}')

        oos_calmars = []
        verdicts    = []

        for w in range(1, self.n_win + 1):
            train_end = w * slice_sz
            test_end  = min((w + 1) * slice_sz, n)
            if test_end <= train_end:
                continue

            train_dates = all_dates[:train_end]
            test_dates  = all_dates[train_end:test_end]

            is_r  = self.res[self.res['date'].dt.date.isin(train_dates)]
            oos_r = self.res[self.res['date'].dt.date.isin(test_dates)]

            if len(is_r) < 5 or len(oos_r) < self.min_oos:
                continue

            is_eq  = self._simulate(is_r)
            oos_eq = self._simulate(oos_r)

            is_m   = self._metrics(is_r,  is_eq)
            oos_m  = self._metrics(oos_r, oos_eq)

            verdict = self._verdict(oos_m.get('calmar', 0),
                                    oos_m.get('wr', 0))
            oos_calmars.append(oos_m.get('calmar', 0))
            verdicts.append(verdict)

            row = {
                'window':    w,
                'is_start':  str(train_dates[0]),
                'is_end':    str(train_dates[-1]),
                'oos_start': str(test_dates[0]),
                'oos_end':   str(test_dates[-1]),
                'is_n':      is_m.get('n', 0),
                'is_calmar': round(is_m.get('calmar', 0), 2),
                'is_sharpe': round(is_m.get('sharpe', 0), 2),
                'oos_n':     oos_m.get('n', 0),
                'oos_calmar':round(oos_m.get('calmar', 0), 2),
                'oos_sharpe':round(oos_m.get('sharpe', 0), 2),
                'oos_wr':    round(oos_m.get('wr', 0), 3),
                'verdict':   verdict,
            }
            rows.append(row)

            if verbose:
                is_p  = f"{train_dates[0]} → {train_dates[-1]}"
                oos_p = f"{test_dates[0]}  → {test_dates[-1]}"
                print(f'  {w:>4}  {is_p:>24}  {oos_p:>24}  '
                      f'{is_m.get("n",0):>5}  {is_m.get("calmar",0):>7.2f}  '
                      f'{oos_m.get("n",0):>6}  {oos_m.get("calmar",0):>8.2f}  '
                      f'{oos_m.get("wr",0):>7.1%}  {verdict:>12}')

        summary_df = pd.DataFrame(rows)

        if verbose and oos_calmars:
            robust_n = sum(1 for v in verdicts if 'ROBUST' in v)
            overall  = ('✅ STRATEGY ROBUST'
                        if robust_n >= len(verdicts) // 2
                        else ('⚠️  REGIME DEPENDENT'
                              if np.mean(oos_calmars) > 0
                              else '❌ STRATEGY BROKEN'))
            print(f'\n  OOS Calmar  mean={np.mean(oos_calmars):.2f}  '
                  f'std={np.std(oos_calmars):.2f}  '
                  f'min={np.min(oos_calmars):.2f}  '
                  f'max={np.max(oos_calmars):.2f}')
            print(f'  Robust windows : {robust_n} / {len(verdicts)}')
            print(f'  Overall verdict: {overall}')
            print('═' * 96)

        return summary_df


# ── EXAMPLE USAGE ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    """
    Demonstration with synthetic trade data.
    Replace with your actual trade log DataFrame.
    """
    np.random.seed(42)
    n_trades = 400
    dates    = pd.date_range('2018-08-01', periods=n_trades * 3,
                             freq='B').to_series().sample(n_trades).sort_values()

    # Simulate a strategy with slight positive expectancy
    r_values = np.where(
        np.random.rand(n_trades) > 0.42,
        np.random.uniform(0.5, 2.5, n_trades),   # winners
        np.random.uniform(-1.2, -0.8, n_trades)  # losers
    )

    trades = pd.DataFrame({
        'date':       dates.values,
        'R_multiple': r_values,
    })

    print('Walk-Forward Framework — Demonstration')
    print(f'Synthetic trades: {len(trades)}  '
          f'WR={( trades["R_multiple"]>0).mean():.1%}  '
          f'Avg R={trades["R_multiple"].mean():+.3f}')

    validator = WalkForwardValidator(
        trade_results=trades,
        n_windows=6,
        min_oos_trades=15,
    )
    summary = validator.run()

    print('\nSummary DataFrame:')
    print(summary[['window','is_calmar','oos_calmar','oos_wr','verdict']].to_string(index=False))
    summary.to_csv('walk_forward_results.csv', index=False)
    print('\nSaved: walk_forward_results.csv')
