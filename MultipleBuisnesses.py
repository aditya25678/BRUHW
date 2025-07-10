#!/usr/bin/env python3
"""
monte_carlo.py

Run Monte Carlo balance-sheet forecasts for multiple LOBs & Accounts.
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
    # melt
    df_long = df.melt(
        id_vars=['LOB','ACCOUNT'],
        var_name='Quarter',
        value_name='Value'
    )
    # Quarter→Period
    df_long['Quarter'] = pd.PeriodIndex(df_long['Quarter'], freq='Q')
    # pivot
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
    fit a Normal(mu,sigma) to the q-over-q returns,
    simulate n_sims paths for 'horizon' quarters,
    and return a DataFrame of simulated balances:
      index = future PeriodIndex,
      columns = sim_0 ... sim_{n_sims-1}.
    """
    # compute returns
    rets = hist.pct_change().dropna()
    mu, sigma = rets.mean(), rets.std(ddof=1)
    # RNG
    rng = np.random.default_rng(seed)
    # simulate returns
    sim_rets = rng.normal(loc=mu, scale=sigma, size=(n_sims, horizon))
    # build paths
    last_val     = hist.iloc[-1]
    sim_paths    = last_val * np.cumprod(1 + sim_rets, axis=1)
    # future quarters
    last_q       = hist.index[-1]
    future_quarters = pd.period_range(last_q + 1, periods=horizon, freq='Q')
    # return DataFrame
    sim_df = pd.DataFrame(
        sim_paths.T,
        index=future_quarters,
        columns=[f"sim_{i}" for i in range(n_sims)]
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
        # Series indexed by Period
        s = sim_df.quantile(q, axis=1).rename(label)
        pieces.append(s)
    df_q = pd.concat(pieces, axis=1)  # cols = labels, idx = Periods
    # transpose → rows=labels, cols=Periods
    df_q = df_q.T
    # stringify column names
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
    # 1) load
    df_piv = load_data(input_csv)
    records = []

    # 2) loop each LOB/ACCOUNT
    for (lob, acct), hist_row in df_piv.iterrows():
        hist = hist_row.dropna().sort_index()
        if len(hist) < 2:
            # need at least one return to fit
            continue

        sim_df = simulate_future(hist, n_sims, horizon, seed)
        summary_df = summarize_simulations(sim_df, quantiles)

        # flatten
        for qlabel, row in summary_df.iterrows():
            rec = {
                'LOB': lob,
                'ACCOUNT': acct,
                'Quantile': qlabel,
                **row.to_dict()
            }
            records.append(rec)

    # 3) assemble & save
    out = pd.DataFrame.from_records(records)
    out = out.set_index(['LOB','ACCOUNT','Quantile']).sort_index()
    out.to_csv(output_csv)
    print(f"Forecasts saved → {output_csv}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Monte Carlo forecasting of quarterly balances"
    )
    p.add_argument('-i','--input',  required=True, help="historical CSV file")
    p.add_argument('-o','--output', default='montecarlo_forecasts.csv',
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
    # percentiles you want
    quantiles = {'p05':0.05, 'p50':0.50, 'p95':0.95}
    run_simulations(
        input_csv = args.input,
        output_csv= args.output,
        n_sims     = args.nsims,
        horizon    = args.horizon,
        quantiles  = quantiles,
        seed       = args.seed
    )
