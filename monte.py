import pandas as pd
import numpy as np

# ----------------------------
# 1. LOAD & RESHAPE YOUR DATA
# ----------------------------
# adjust the filename as needed
df = pd.read_csv('commodities_data.csv')

# assume columns: ['LOB','ACCOUNT','2019Q1','2019Q2',…,'2023Q4']
# melt into long form
df_long = df.melt(id_vars=['LOB','ACCOUNT'], 
                  var_name='Quarter', 
                  value_name='Value')

# convert '2019Q1' → Period('2019Q1')
df_long['Quarter'] = pd.PeriodIndex(df_long['Quarter'], freq='Q')

# pivot so each row is one (LOB, ACCOUNT) and columns are Periods
df_piv = df_long.pivot_table(index=['LOB','ACCOUNT'],
                             columns='Quarter',
                             values='Value').sort_index(axis=1)

# ------------------------------------
# 2. SET UP SIMULATION PARAMETERS
# ------------------------------------
n_sims  = 1000              # number of Monte Carlo paths
horizon = 5                 # number of quarters to forecast (2024Q1–2025Q1)

# define your future quarters
last_period   = df_piv.columns[-1]
future_periods = pd.period_range(last_period + 1, periods=horizon, freq='Q')

# storage for summary results
# will become a DataFrame with multi-index (LOB, ACCOUNT, Quantile) × future_periods
results = []

# ------------------------------------
# 3. SIMULATION LOOP
# ------------------------------------
for (lob, acct), row in df_piv.iterrows():
    # get historical series and compute returns
    hist = row.dropna().sort_index()
    rets = hist.pct_change().dropna()
    mu, sigma = rets.mean(), rets.std()

    # simulate future returns ~ Normal(mu, sigma)
    rnd = np.random.default_rng(seed=42)
    sim_rets = rnd.normal(loc=mu, scale=sigma, size=(n_sims, horizon))

    # build simulated balance paths
    last_val = hist.iloc[-1]
    # each path: last_val * ∏(1 + return_i)
    sim_paths = last_val * np.cumprod(1 + sim_rets, axis=1)

    # wrap into DataFrame: index=future_periods, columns=sim_1…sim_n
    sim_df = pd.DataFrame(sim_paths.T, index=future_periods,
                          columns=[f'sim_{i+1}' for i in range(n_sims)])

    # compute percentiles
    pctiles = sim_df.quantile([0.05, 0.50, 0.95], axis=1).T
    pctiles.index = ['p05','p50','p95']
    
    # flatten for combining
    for q in pctiles.index:
        results.append({
            'LOB': lob,
            'ACCOUNT': acct,
            'Quantile': q,
            **pctiles.loc[q].to_dict()
        })

# ------------------------------------
# 4. ASSEMBLE & SAVE RESULTS
# ------------------------------------
out = pd.DataFrame(results)
# columns: ['LOB','ACCOUNT','Quantile','2024Q1','2024Q2',…,'2025Q1']
out = out.set_index(['LOB','ACCOUNT','Quantile']).sort_index()

# write to CSV (or do further analysis/plotting)
out.to_csv('commodities_forecast_montecarlo.csv')

print("Forecast complete. Summary saved to 'commodities_forecast_montecarlo.csv'.")
