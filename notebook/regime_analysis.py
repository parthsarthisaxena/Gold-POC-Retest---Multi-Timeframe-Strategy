"""
regime_analysis.py
===================
Visualises performance breakdowns from the Gold POC Retest strategy.
Generates four publication-quality charts from the results CSVs.

Charts produced:
  1. Monthly P&L bar chart with cumulative R overlay
  2. Walk-forward OOS Calmar by window
  3. Performance by regime (200-SMA distance)
  4. Forward return significance by horizon

Run from the repo root:
    python notebooks/regime_analysis.py

Requirements:
    pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUT_DIR     = os.path.dirname(__file__)


def load(fname):
    return pd.read_csv(os.path.join(RESULTS_DIR, fname))


def chart_monthly_pnl(df, ax):
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in df['total_R']]
    x = range(len(df))
    ax.bar(x, df['total_R'], color=colors, alpha=0.8, width=0.8)

    # Cumulative R line
    ax2 = ax.twinx()
    ax2.plot(x, df['total_R'].cumsum(), color='#2c3e50', lw=2, label='Cumulative R')
    ax2.set_ylabel('Cumulative R', fontsize=9)
    ax2.yaxis.label.set_color('#2c3e50')

    # x-axis labels — show every 6th month
    labels = [m if i % 6 == 0 else '' for i, m in enumerate(df['month'])]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('Monthly P&L (R-multiples)  —  2018–2026', fontweight='bold')
    ax.set_ylabel('Monthly R')

    profitable = (df['total_R'] > 0).sum()
    ax.text(0.02, 0.97,
            f'Profitable months: {profitable}/{len(df)} ({profitable/len(df):.1%})',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def chart_walk_forward(df, ax):
    colors = {'✅ ROBUST': '#2ecc71', '🟡 MARGINAL': '#f39c12',
              '⚠️  WEAK': '#e67e22', '❌ BROKEN': '#e74c3c'}
    bar_colors = [colors.get(v, '#95a5a6') for v in df['verdict']]

    x      = range(len(df))
    is_cal = df['is_calmar'].values
    oos_cal= df['oos_calmar'].values

    width = 0.35
    ax.bar([i - width/2 for i in x], is_cal,  width, label='IS Calmar',
           color='#3498db', alpha=0.7)
    ax.bar([i + width/2 for i in x], oos_cal, width, label='OOS Calmar',
           color=bar_colors, alpha=0.9)

    ax.axhline(1.5, color='#2ecc71', ls='--', lw=1, label='ROBUST threshold (1.5)')
    ax.axhline(0,   color='black',   ls='-',  lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'W{w}' for w in df['window']], fontsize=9)
    ax.set_title('Walk-Forward Validation — IS vs OOS Calmar', fontweight='bold')
    ax.set_ylabel('Calmar Ratio')
    ax.legend(fontsize=8)

    mean_oos = df['oos_calmar'].mean()
    ax.text(0.98, 0.97, f'OOS Calmar mean: {mean_oos:.2f}',
            transform=ax.transAxes, fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


def chart_regime(df, ax):
    colors = ['#2ecc71','#27ae60','#f39c12','#95a5a6']
    bars   = ax.barh(df['regime'], df['avg_R'], color=colors, alpha=0.85)
    ax.axvline(0, color='black', lw=0.8)

    for bar, n, wr in zip(bars, df['n_trades'], df['win_rate']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'  n={n}  WR={wr:.0%}', va='center', fontsize=8)

    ax.set_title('Avg R-multiple by Regime (200-SMA distance)', fontweight='bold')
    ax.set_xlabel('Avg R-multiple per trade')
    ax.set_xlim(-0.1, df['avg_R'].max() * 1.5)
    ax.invert_yaxis()


def chart_diagnostic(df, ax):
    x      = range(len(df))
    colors = ['#2ecc71' if s else '#e74c3c' for s in df['significant']]

    bars = ax.bar(x, df['signal_wr'] * 100, color=colors, alpha=0.85, width=0.4)
    ax.bar([i + 0.42 for i in x],
           [50] * len(df), width=0.4, color='#95a5a6', alpha=0.5,
           label='Baseline win rate (50%)')

    ax.axhline(50, color='black', ls='--', lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f'{h}min' for h in df['horizon_min']], fontsize=9)
    ax.set_title('Signal Win Rate vs 50% Baseline by Horizon', fontweight='bold')
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(40, 75)

    for i, (bar, p) in enumerate(zip(bars, df['p_value'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'p={p:.3f}', ha='center', fontsize=7.5, fontweight='bold')

    green = mpatches.Patch(color='#2ecc71', alpha=0.85, label='p < 0.05 (significant)')
    ax.legend(handles=[green], fontsize=8)


def main():
    monthly = load('monthly_pnl.csv')
    wf      = load('walk_forward_summary.csv')
    regime  = load('regime_performance.csv')
    diag    = load('forward_return_diagnostic.csv')

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Gold POC Retest MTF v4 — Research Summary  (2018–2026)',
                 fontsize=14, fontweight='bold', y=1.01)

    chart_monthly_pnl (monthly, axes[0, 0])
    chart_walk_forward(wf,      axes[0, 1])
    chart_regime      (regime,  axes[1, 0])
    chart_diagnostic  (diag,    axes[1, 1])

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, 'strategy_research_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
