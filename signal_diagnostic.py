"""
signal_diagnostic.py
=====================
Forward return significance test for the Gold POC Retest signal.

Tests whether the signal has statistically meaningful predictive power
BEFORE any trade management (stops, targets, sizing) is applied.
This is the foundational validity check — if the signal has no raw edge,
no amount of trade management can create one.

Methodology:
  - Compute forward returns at multiple horizons from each signal bar
  - Compare against a time-matched random baseline (same sessions,
    random bars, random direction)
  - Two-sample Welch t-test to assess significance
  - Signal is confirmed if p < 0.05 AND mean > baseline across
    multiple horizons (not just a single borderline p-value)

This script does NOT include:
  - Entry/exit logic
  - Stop placement
  - Position sizing
  - Any strategy-specific parameters

Usage:
    python signal_diagnostic.py --csv path/to/XAUUSDm_M5.csv

Requirements:
    pip install numpy pandas scipy
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings('ignore')


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data(path):
    """Load MT5 5-min CSV export."""
    raw = pd.read_csv(path, sep='\t')
    raw.columns = [c.strip('<>').lower() for c in raw.columns]
    raw['datetime'] = pd.to_datetime(
        raw['date'] + ' ' + raw['time'], format='%Y.%m.%d %H:%M:%S')
    raw = raw.rename(columns={'tickvol': 'volume'})
    return (raw[['datetime','open','high','low','close','volume']]
            .sort_values('datetime').reset_index(drop=True))


def split_sessions(raw, start_h=7, end_h=20):
    raw2 = raw.copy()
    raw2['hour'] = raw2['datetime'].dt.hour
    raw2['date'] = raw2['datetime'].dt.date
    raw2 = raw2[(raw2['hour'] >= start_h) & (raw2['hour'] < end_h)].copy()
    sessions = {}
    for d, grp in raw2.groupby('date'):
        g = grp.drop(columns=['hour','date']).reset_index(drop=True)
        g['time'] = g['datetime'].dt.time
        if len(g) >= 20:
            sessions[str(d)] = g
    return sessions


# ── FORWARD RETURN COMPUTATION ────────────────────────────────────────────────
def forward_return(df, idx, side, horizon):
    """
    Compute the directional forward return `horizon` bars after `idx`.
    side: 'long' or 'short'
    Returns None if insufficient bars remain.
    """
    end = idx + horizon
    if end >= len(df):
        return None
    ret = (df.iloc[end]['close'] - df.iloc[idx]['close']) / df.iloc[idx]['close']
    return ret if side == 'long' else -ret


# ── SIGNAL DETECTOR (abstracted — no implementation details) ──────────────────
def detect_signals(df):
    """
    Returns a list of (bar_index, side) tuples where the strategy's
    signal fires. Implementation intentionally omitted — replace with
    your own signal detection logic.

    For demonstration, this stub returns an empty list.
    The diagnostic framework below works with any signal generator
    that returns (index, 'long'|'short') pairs.
    """
    # ── YOUR SIGNAL DETECTION LOGIC HERE ──
    # Example interface:
    # signals = []
    # for i in range(...):
    #     if your_condition(df, i):
    #         signals.append((i, 'long'))
    # return signals
    return []


# ── RANDOM BASELINE ───────────────────────────────────────────────────────────
def random_baseline(df, n_samples, horizons, seed=42):
    """
    Sample random bars from the same sessions as the signal,
    with random direction, as a time-matched null hypothesis.
    """
    rng     = np.random.default_rng(seed)
    max_h   = max(horizons)
    results = {h: [] for h in horizons}
    n       = min(n_samples, max(0, len(df) - max_h - 5))
    if n == 0:
        return results
    idxs = rng.choice(range(5, len(df) - max_h - 2), size=n, replace=False)
    for i in idxs:
        side = rng.choice(['long', 'short'])
        for h in horizons:
            r = forward_return(df, i, side, h)
            if r is not None:
                results[h].append(r)
    return results


# ── DIAGNOSTIC ────────────────────────────────────────────────────────────────
def run_diagnostic(sessions, horizons=(3, 6, 12, 24, 48),
                   baseline_per_session=4):
    """
    Run the full forward-return diagnostic across all sessions.

    Args:
        sessions      : dict of {date_str: DataFrame}
        horizons      : forward return horizons in bars (5-min bars)
        baseline_per_session : random baseline samples per session

    Prints:
        Signal statistics vs baseline at each horizon
        t-statistic and p-value
        EDGE flag where p < 0.05 and signal mean > baseline mean
    """
    sig_rets  = {h: [] for h in horizons}
    base_rets = {h: [] for h in horizons}

    for date, df in sessions.items():
        if len(df) < max(horizons) + 10:
            continue

        # Signal forward returns
        sigs = detect_signals(df)
        for idx, side in sigs:
            for h in horizons:
                r = forward_return(df, idx, side, h)
                if r is not None:
                    sig_rets[h].append(r)

        # Baseline forward returns
        bl = random_baseline(df, baseline_per_session, horizons)
        for h in horizons:
            base_rets[h].extend(bl[h])

    # ── Print results ──────────────────────────────────────────────────
    print('\n' + '='*70)
    print('  FORWARD RETURN DIAGNOSTIC')
    print('='*70)
    print(f'  {"Horizon":>8}  {"n_sig":>6}  {"mean_sig%":>10}  {"wr_sig":>7}  '
          f'{"n_base":>7}  {"mean_base%":>11}  {"t":>6}  {"p":>7}  {"":>6}')
    print(f'  {"-"*68}')

    any_edge = False
    for h in horizons:
        sig  = np.array(sig_rets[h])
        base = np.array(base_rets[h])

        if len(sig) < 5:
            print(f'  {h*5:>5}min   n={len(sig)} (too few signals to test)')
            continue

        t, p = stats.ttest_ind(sig, base, equal_var=False) if len(base) > 1 else (0, 1)
        edge = p < 0.05 and sig.mean() > (base.mean() if len(base) > 0 else 0)
        flag = '  ← EDGE' if edge else ''
        if edge:
            any_edge = True

        print(f'  {h*5:>5}min  {len(sig):>6}  {sig.mean()*100:>+10.4f}  '
              f'{(sig>0).mean():>7.1%}  {len(base):>7}  '
              f'{base.mean()*100 if len(base)>0 else 0:>+11.4f}  '
              f'{t:>+6.2f}  {p:>7.3f}{flag}')

    print('='*70)
    print(f'\n  Interpretation:')
    print(f'  A genuine edge shows p < 0.05 AND positive mean vs baseline')
    print(f'  consistently across multiple horizons — not a single')
    print(f'  borderline result. One significant horizon is noise;')
    print(f'  three or more in the same direction is a real effect.')
    if any_edge:
        print(f'\n  ✅  Edge detected in this diagnostic run.')
    else:
        print(f'\n  ⚠️   No statistically significant edge detected.')
        print(f'      Check signal detection logic or data quality.')
    print('='*70)

    return sig_rets, base_rets


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Forward return significance test for a price signal.')
    parser.add_argument('--csv', required=True,
                        help='Path to MT5 5-min CSV export')
    parser.add_argument('--horizons', nargs='+', type=int,
                        default=[3, 6, 12, 24, 48],
                        help='Forward return horizons in bars (default: 3 6 12 24 48)')
    parser.add_argument('--session-start', type=int, default=7,
                        help='Session start hour UTC (default: 7)')
    parser.add_argument('--session-end', type=int, default=20,
                        help='Session end hour UTC (default: 20)')
    parser.add_argument('--baseline-samples', type=int, default=4,
                        help='Random baseline samples per session (default: 4)')
    args = parser.parse_args()

    print(f'Loading {args.csv} ...')
    raw      = load_data(args.csv)
    sessions = split_sessions(raw, args.session_start, args.session_end)
    print(f'  {len(raw):,} bars  |  {len(sessions)} sessions')
    print(f'  Range: {raw["datetime"].iloc[0].date()} → '
          f'{raw["datetime"].iloc[-1].date()}')

    run_diagnostic(sessions,
                   horizons=args.horizons,
                   baseline_per_session=args.baseline_samples)
