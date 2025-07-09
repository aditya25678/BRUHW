import pandas as pd
import numpy as np

# ----------------------------
# 1. LOAD & RESHAPE HISTORICAL DATA
# ----------------------------
df = pd.read_csv('commodities_data.csv')   # your raw quarterly history
# melt to long form
df_long = df.melt(id_vars=['LOB','ACCOUNT'],
                  var_name='Quarter',
                  value_name='Value')
# convert quarter strings to Periods
df_long['Quarter'] = pd.PeriodIndex(df_long['Quarter'], freq='Q')
# pivot back to wide: index=(LOB,ACCOUNT), columns=Period, values=Value
df_piv = (df_long
          .pivot_table(index=['LOB','ACCOUNT'],
                       columns='Quarter',
                       values='Value')
          .sort_index(axis=1))

# ----------------------------
# 2. SET UP 1-QTR MONTE CARLO
# ----------------------------
n_sims       = 5000    # number of simulated paths
horizon      = 1       # only 2024Q1
last_period  = df_piv.columns[-1]
forecast_qtr = last_period + 1  # this will be Period('2024Q1')

# we'll collect one record per account
records = []
rng = np.random.default_rng(seed=42)

for (lob, acct), hist_row in df_piv.iterrows():
    hist = hist_row.dropna().sort_index()
    # need at least two quarters to compute returns
    if len(hist) < 2:
        continue
    
    rets = hist.pct_change().dropna()
    mu, sigma = rets.mean(), rets.std()
    # if sigma is zero (flat series), we'll still simulate zero-vol
    sim_rets = rng.normal(loc=mu, scale=sigma, size=(n_sims, horizon))
    
    # build simulated balances for 2024Q1
    last_val   = hist.iloc[-1]
    sim_paths  = last_val * np.cumprod(1 + sim_rets, axis=1).flatten()
    
    # compute stats on the simulated distribution
    p50      = np.percentile(sim_paths, 50)
    variance = np.var(sim_paths, ddof=1)
    stddev   = np.std(sim_paths, ddof=1)
    
    records.append({
        'LOB': lob,
        'ACCOUNT': acct,
        'p50_forecast_2024Q1': p50,
        'var_2024Q1': variance,
        'std_2024Q1': stddev
    })

sim_stats = pd.DataFrame(records)

# ----------------------------
# 3. LOAD ACTUALS & MERGE
# ----------------------------
actuals = pd.read_csv('actuals_2024Q1.csv')  
# ensure column is named consistently
actuals = actuals.rename(columns={'2024Q1': 'actual_2024Q1'})

# merge on LOB & ACCOUNT
df_eval = sim_stats.merge(actuals, on=['LOB','ACCOUNT'], how='inner')

# ----------------------------
# 4. COMPUTE ERROR METRICS
# ----------------------------
df_eval['abs_error_2024Q1'] = df_eval['p50_forecast_2024Q1'] - df_eval['actual_2024Q1']
df_eval['pct_diff_2024Q1']  = (
    df_eval['abs_error_2024Q1'] 
    / df_eval['actual_2024Q1']
) * 100

# ----------------------------
# 5. SAVE RESULTS
# ----------------------------
df_eval.to_csv('forecast_accuracy_2024Q1.csv', index=False)
print("Done — accuracy metrics written to 'forecast_accuracy_2024Q1.csv'")
