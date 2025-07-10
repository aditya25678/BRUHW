import pandas as pd

# 1) Load your Monte Carlo forecasts
#    Expect columns: LOB, ACCOUNT, Quantile, 2019Q1…2025Q1
forecasts = pd.read_csv('forecasts.csv')

# 2) Keep only the median (p50) forecasts and rename the 2025Q1 column
p50 = (
    forecasts
    .query("Quantile == 'p50'")
    [['LOB','ACCOUNT','2025Q1']]
    .rename(columns={'2025Q1':'p50_forecast_2025Q1'})
)

# 3) Load your actuals for 2025Q1
#    File should have: LOB, ACCOUNT, 2025Q1
actuals = (
    pd.read_csv('actuals_2025Q1.csv')
    .rename(columns={'2025Q1':'actual_2025Q1'})
)

# 4) Merge forecasts with actuals
df_eval = p50.merge(actuals, on=['LOB','ACCOUNT'], how='inner')

# 5) Compute error metrics
df_eval['abs_error_2025Q1'] = df_eval['p50_forecast_2025Q1'] - df_eval['actual_2025Q1']
df_eval['pct_diff_2025Q1']  = (df_eval['abs_error_2025Q1'] 
                              / df_eval['actual_2025Q1']) * 100

# 6) Inspect or save the results
df_eval.head()
# If you want to persist:
# df_eval.to_csv('accuracy_2025Q1.csv', index=False)
