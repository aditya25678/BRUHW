#!/usr/bin/env python3
"""
monte_carlo.py

Run Monte Carlo balance-sheet forecasts for multiple LOBs & Accounts,
using a bootstrap (empirical) distribution of historical returns.
Accounts with insufficient history are skipped.
"""

import argparse
import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    """
    Reads a CSV with columns ['LOB','ACCOUNT','2019Q1',…,'2023Q4'],
    melts to long, converts Quarter→Period, pivots to wide.
    Returns a DataFrame indexed by (LOB,ACCOUNT), columns=PeriodIndex.
    """
    df = pd.read_csv(path)
    df_long = df.melt(
        id_vars=['LOB','ACCOUNT'],
        var_name='Quarter',
        value_name='Value'
    )
    df_long['Quarter'] = pd.PeriodIndex(df_long['Quarter'], freq='Q')
    df_piv = (
        df_long
        .pivot_table(
            index=['LOB','ACCOUNT'],
            columns='Quarter',
            values='Value'
        )
        .sort_index(axis=1)
    )
    return df_piv

def simulate_future(
    hist: pd.Series,
    n_sims: int,
    horizon: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Given a historical Series (index=Period, values=balances),
    compute empirical quarter-over-quarter returns, then
    bootstrap (sample with replacement) those returns to
    simulate n_sims paths for 'horizon' quarters.
    Raises ValueError if there are no returns.
    Returns a DataFrame of simulated balances:
      index = future PeriodIndex,
      columns = sim_0 ... sim_{n_sims-1}.
    """
    rets = hist.pct_change().dropna().values
    if len(rets) == 0:
        raise ValueError("Not enough data to compute returns")

    rng = np.random.default_rng(seed)
    sim_rets = rng.choice(rets, size=(n_sims, horizon), replace=True)
    last_val  = hist.iloc[-1]
    sim_paths = last_val * np.cumprod(1 + sim_rets, axis=1)

    last_q          = hist.index[-1]
    future_quarters = pd.period_range(last_q + 1, periods=horizon, freq='Q')

    sim_df = pd.DataFrame(
        sim_paths.T,
        index   = future_quarters,
        columns = [f"sim_{i}" for i in range(n_sims)]
    )
    return sim_df

def summarize_simulations(
    sim_df: pd.DataFrame,
    quantiles: dict[str, float]
) -> pd.DataFrame:
    """
    From a sim_df (index=Period, cols=sims), compute each quantile,
    return a DataFrame shaped (len(quantiles) × horizon) with:
      index = quantile labels (e.g. 'p05','p50','p95'),
      columns = stringified future quarters.
    """
    pieces = []
    for label, q in quantiles.items():
        s = sim_df.quantile(q, axis=1).rename(label)
        pieces.append(s)
    df_q = pd.concat(pieces, axis=1).T
    df_q.columns = [str(p) for p in df_q.columns]
    return df_q

def run_simulations(
    input_csv: str,
    output_csv: str,
    n_sims: int,
    horizon: int,
    quantiles: dict[str, float],
    seed: int
):
    df_piv = load_data(input_csv)
    records = []

    for (lob, acct), hist_row in df_piv.iterrows():
        hist = hist_row.dropna().sort_index()
        try:
            sim_df     = simulate_future(hist, n_sims, horizon, seed)
            summary_df = summarize_simulations(sim_df, quantiles)
        except ValueError as e:
            # skip accounts without enough data
            print(f"Skipping {lob} / {acct}: {e}")
            continue

        for qlabel, row in summary_df.iterrows():
            rec = {
                'LOB': lob,
                'ACCOUNT': acct,
                'Quantile': qlabel,
                **row.to_dict()
            }
            records.append(rec)

    out = pd.DataFrame.from_records(records)
    out = out.set_index(['LOB','ACCOUNT','Quantile']).sort_index()
    out.to_csv(output_csv)
    print(f"Forecasts saved → {output_csv}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo forecasting of quarterly balances via bootstrap"
    )
    p.add_argument('-i','--input',  required=True, help="historical CSV file")
    p.add_argument('-o','--output', default='montecarlo_bootstrap.csv',
                   help="where to write forecasts")
    p.add_argument('--nsims', type=int, default=1000,
                   help="number of Monte Carlo paths per account")
    p.add_argument('--horizon', type=int, default=5,
                   help="quarters to forecast (e.g. 5 ⇒ 2024Q1–2025Q1)")
    p.add_argument('--seed',   type=int, default=42,
                   help="random seed for reproducibility")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    quantiles = {'p05':0.05, 'p50':0.50, 'p95':0.95}
    run_simulations(
        input_csv  = args.input,
        output_csv = args.output,
        n_sims     = args.nsims,
        horizon    = args.horizon,
        quantiles  = quantiles,
        seed       = args.seed
    )
