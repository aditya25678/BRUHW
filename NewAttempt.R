library(dplyr)
library(ggplot2)
library(forecast)
library(lubridate)
library(PerformanceAnalytics)
library(readxl)
library(tseries)
library(car)
library(MTS)
library(TSA)
library(urca)

# Read in Data and autoplot
#input_path <- "MBAM WP/"
#data <- read_excel(file.path(input_path,"Actuals_ccar2025.xlsx")) # for CCAR Data
#data <- read_excel(file.path(input_path,"Actuals_202025.xlsx"))   # 202025 data
data <- read_excel(file.path("MBAM_WP/Actuals_ccar2025.xlsx"))      # 102025 data
#data <- merge(data, more_features, by="date")

# Time‐Series data
autoplot(data[,"MBAM_TAP"], facets=TRUE) +
  xlab("Year") + xlab("") + ggtitle("MBAM TAP History")
dep_var <- ts(data$MBAM_TAP, frequency = 4, start = c(2007, 3))

# Linear Model
m <- lm(dep_var ~ VIX_diff_sqrtd + MMD_MUNI_30Y_AAA_YIELD_Diff, data = data)
summary(m)
vif(m)

# ARIMAX
m2 <- arima(
  dep_var,
  order = c(1, 1, 0),
  xreg  = as.matrix(data[, c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")])
)

#### Diagnostics ####
# check residuals(fit2)
t_test  <- coef(m2)
p_value <- 2 * pt(t_test, df)
r2      <- residuals(m2)

raittest(m2)
res2 <- residuals(m2)
plot(res2)

# Normality (p-value > 0.05)
shapiro.test(resid(m2))

# Stationarity and homoscedasticity
adf.test(m2$residuals) # p < .05 ⇒ stationary
bptest(res2, ~ data$VIX_diff_sqrtd + data$MMD_MUNI_30Y_AAA_YIELD_Diff)
archTest(res2, lag=10)
archTest(res2, lag=5)
archTest(res2, lag=2)

# Autocorrelation
Box.test(res2, lag=8, type = "Ljung-Box") # p > .05 ⇒ no autocorrelation

#### Projection files ####
baseline <- read_excel(file.path(input_path, "baseline_ccar.xlsx"))
baca     <- read_excel(file.path(input_path, "baca_ccar.xlsx"))
bacsa    <- read_excel(file.path(input_path, "bacsa_ccar.xlsx"))

f2base  <- predict(m2, n = 12,
                   newxreg = as.matrix(baseline[, c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2baca  <- predict(m2, n = 12,
                   newxreg = as.matrix(baca[,     c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2bacsa <- predict(m2, n = 12,
                   newxreg = as.matrix(bacsa[,    c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
back2   <- fitted(m2)

arima_frozen <- arima(
  x    = data$MBAM_TAP,
  order = c(1, 1, 0),
  xreg  = as.matrix(data[, c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")]),
  fixed = c(
    -0.580041,      # ar1
    -0.146099,      # VIX_diff_sqrtd
    -5160.969914    # MMD_MUNI_30Y_AAA_YIELD_Diff
  )
)

last_fitted <- tail(fitted(m2), 1)
last_diff   <- last_fitted - tail(fitted(m2), 2)[1]

ar1 <- coef(m2)["ar1"]
b1  <- coef(m2)["VIX_diff_sqrtd"]
b2  <- coef(m2)["MMD_MUNI_30Y_AAA_YIELD_Diff"]

x1_future <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)
h         <- length(x1_future)
dy_prev   <- last_fitted
y_prev    <- last_diff
y_forecast <- numeric(h)

for (i in 1:h) {
  dy_new <- ar1 * dy_prev + b1 * x1_future[i] + b2 * x2_future[i]
  y_forecast[i] <- y_prev + dy_new
  dy_prev <- dy_new
  y_prev  <- y_forecast[i]
}

print(y_forecast)

#baseline manual prediction
y_hist  <- as.numeric(data$MBAM_TAP)
dy_hist <- diff(y_hist)      # differenced series
y_last  <- tail(y_hist, 1)
dy_last <- tail(dy_hist, 1)

# Get future exogenous variables from baseline
x1_future <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)
h         <- length(x1_future)

# Extract coefficients (including any intercept / drift)
coef_m2 <- coef(m2)
ar1     <- coef_m2["ar1"]
b1      <- coef_m2["VIX_diff_sqrtd"]
b2      <- coef_m2["MMD_MUNI_30Y_AAA_YIELD_Diff"]

# Determine intercept or drift (if present)
const <- 0
if ("intercept" %in% names(coef_m2)) {
  const <- coef_m2["intercept"]
} else if ("drift" %in% names(coef_m2)) {
  const <- coef_m2["drift"]
}

y_forecast <- numeric(h)
dy_prev    <- dy_last
y_prev     <- y_last

for (i in 1:h) {
  # forecast the differenced value, including the constant
  dy_new <- const + ar1 * dy_prev + b1 * x1_future[i] + b2 * x2_future[i]
  # undo differencing
  y_new <- y_prev + dy_new
  y_forecast[i] <- y_new
  dy_prev <- dy_new
  y_prev  <- y_new
}

print(y_forecast)   # now matches f2base$pred exactly

# Combine into a forecast table
f2base_pred  <- as.numeric(f2base$pred)
f2baca_pred  <- as.numeric(f2baca$pred)
f2bacsa_pred <- as.numeric(f2bacsa$pred)

library(zoo)
forecast_periods <- length(f2base_pred)
start_year      <- 2025
start_quarter   <- 2
time_index <- seq(from = as.yearqtr(paste(start_year, start_quarter, sep = " Q")),
                  by   = 0.25, length.out = forecast_periods)

forecast_table <- data.frame(
  Quarter  = as.character(time_index),
  Baseline = f2base_pred,
  BACA     = f2baca_pred,
  BACSA    = f2bacsa_pred
)

print(forecast_table)

#testing
library(tseries)
adf.test(residuals(m2))    # p ~ .01, stationary
shapiro.test(residuals(m2))# p ~ .94, normal residuals

arimax_resid_df <- data.frame(
  resid                       = residuals(m2),
  VIX_diff_sqrtd              = data$VIX_diff_sqrtd,
  MMD_MUNI_30Y_AAA_YIELD_Diff = data$MMD_MUNI_30Y_AAA_YIELD_Diff
)
aux_lm <- lm(resid ~ VIX_diff_sqrtd + MMD_MUNI_30Y_AAA_YIELD_Diff,
             data = arimax_resid_df)
bptest(aux_lm)                              # Breusch-Pagan
bptest(aux_lm, ~ fitted(aux_lm) + I(fitted(aux_lm)^2))  # White

# ACF/PACF of residuals
acf(residuals(m2), main="ACF of ARIMAX Residuals")
pacf(residuals(m2), main="PACF of ARIMAX Residuals")

# CUSUM stability test
library(strucchange)
cusum_test <- efp(aux_lm, type="OLS-CUSUM")

# Ljung-Box
box_results <- Box.test(residuals(m2), lag = 8, type = "Ljung-Box")
print(box_results)

# Durbin-Watson
library(lmtest)
dwtest(aux_lm)

# In‐Sample Testing
summary(m2)
in_sample_fitted <- fitted(m2)
in_sample_actual <- dep_var[!is.na(dep_var)]
in_sample_rmse  <- sqrt(mean((in_sample_actual - in_sample_fitted)^2))
in_sample_r2    <- cor(in_sample_fitted, in_sample_actual)^2

cat("In-sample RMSE:", in_sample_rmse, "\n")
cat("In-sample R-squared:", in_sample_r2, "\n")

library(zoo)
data$DateQ <- as.yearqtr(data$date, format = "%Y Q%q")
cutoff_q   <- as.yearqtr("2021 Q4", format = "%Y Q%q")

# Out‐of‐Sample Testing
in_sample_idx <- which(data$DateQ <= cutoff_q)
data_in       <- data[in_sample_idx, ]
dep_in        <- ts(data_in$MBAM_TAP, frequency = 4, start = c(2007,3))

mdl_in <- arima(
  dep_in,
  order = c(1, 1, 0),
  xreg  = as.matrix(data_in[, c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")])
)

coefs <- coef(mdl_in)
se    <- sqrt(diag(mdl_in$var.coef))
tval  <- coefs / se
pval  <- 2 * (1 - pnorm(abs(tval)))
coef_names <- names(coefs)

oos_coef_table <- data.frame(
  Coefficient = coef_names,
  Estimate     = coefs,
  Stderr       = se,
  tValue       = tval,
  pValue       = pval,
  stringsAsFactors = FALSE
)
print(oos_coef_table)
write.csv(oos_coef_table, "oos_coef_table.csv", row.names = FALSE)

data$DateQ <- as.yearqtr(data$date, format = "%Y Q%q")
oos_idx    <- which(data$DateQ > cutoff_q)
data_oos   <- data[oos_idx, ]

oos_xreg   <- as.matrix(data_oos[, c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")])
oos_actual <- data_oos$MBAM_TAP
oos_pred   <- predict(mdl_in, n.ahead = nrow(data_oos), newxreg = oos_xreg)$pred

oos_rmse <- sqrt(mean((oos_actual - oos_pred)^2))
oos_r2   <- cor(oos_actual, oos_pred)^2
mdl_aic  <- AIC(mdl_in)
mdl_bic  <- BIC(mdl_in)

summary_table <- data.frame(
  RMSE      = oos_rmse,
  R_Squared = oos_r2,
  AIC       = mdl_aic,
  BIC       = mdl_bic
)
print(summary_table)

# Plot in‐ and out‐of‐sample
library(ggplot2)
library(zoo)

plot_df <- data.frame(
  Date  = c(data_in$date, data_oos$date, data_oos$date),
  Value = c(data_in$MBAM_TAP, data_oos$MBAM_TAP, oos_pred),
  Type  = c(
    rep("In-sample Actual", length(data_in$MBAM_TAP)),
    rep("Out-of-sample Actual", length(data_oos$MBAM_TAP)),
    rep("Out-of-sample Forecast", length(oos_pred))
  )
)

ggplot(plot_df, aes(x = as.yearqtr(Date, format = "%Y Q%q"), y = Value, color = Type, linetype = Type)) +
  geom_line(size = 1.2) +
  labs(title = "In-sample VS Out-of-sample: Actuals and Forecasts", x = "Date", y = "MBAM_TAP") +
  scale_color_manual(values = c("In-sample Actual" = "black", "Out-of-sample Actual" = "blue", "Out-of-sample Forecast" = "red")) +
  scale_linetype_manual(values = c("In-sample Actual" = "solid", "Out-of-sample Actual" = "solid", "Out-of-sample Forecast" = "dashed")) +
  theme_minimal()

# Forecast intervals and breaches
oos_predse <- predict(mdl_in, n.ahead = nrow(data_oos), newxreg = oos_xreg)$se
lower      <- oos_pred - 1.96 * oos_predse
upper      <- oos_pred + 1.96 * oos_predse
breaches   <- sum(oos_actual < lower | oos_actual > upper)
forecasted_size <- length(oos_actual)
conclusion      <- ifelse(breaches <= 0.05 * forecasted_size, "Pass", "Fail")

oos_summary <- data.frame(
  Forecasted_Size = forecasted_size,
  MAPE            = mean(abs(oos_actual - oos_pred) / oos_actual),
  RMSE            = oos_rmse,
  No_of_Breaches  = breaches,
  Conclusion      = conclusion
)

library(tidyr)
plot_long <- tidyr::pivot_longer(
  plot_df,
  cols      = c("Actual", "Predicted", "Upper95", "Lower95"),
  names_to  = "series",
  values_to = "Value"
)

ggplot(plot_df, aes(x = Date)) +
  geom_line(aes(y = Actual)) +
  geom_line(aes(y = Predicted)) +
  geom_line(aes(y = Upper95)) +
  geom_line(aes(y = Lower95)) +
  labs(title = "Out-of-Sample Forecast vs Actuals", y = "MBAM_TAP", x = "Date") +
  theme_minimal()

### 10.3 benchmarking
dep_series   <- data$MBAM_TAP[-nrow(data)]
date_series  <- as.yearqtr(data$date, format = "%Y Q%q")[-nrow(data)]
roll_9q      <- zoo::rollsum(dep_series, 9, align = "right", na.pad = FALSE)
date_9q_end  <- date_series[9:length(date_series)]
date_9q_start<- date_series[1:(length(date_series) - 8)]

most_recent_idx   <- length(roll_9q)
most_recent_sum   <- roll_9q[most_recent_idx]
most_recent_start <- date_9q_start[most_recent_idx]
most_recent_end   <- date_9q_end[most_recent_idx]

max_idx   <- which.max(roll_9q)
max_sum   <- roll_9q[max_idx]
max_start <- date_9q_start[max_idx]
max_end   <- date_9q_end[max_idx]

min_idx   <- which.min(roll_9q)
min_sum   <- roll_9q[min_idx]
min_start <- date_9q_start[min_idx]
min_end   <- date_9q_end[min_idx]

nineq_table <- data.frame(
  Period = c("Most Recent 9Q", "Highest 9Q", "Lowest 9Q"),
  Start  = as.character(c(most_recent_start, max_start, min_start)),
  End    = as.character(c(most_recent_end, max_end, min_end)),
  Sum    = c(most_recent_sum, max_sum, min_sum)
)

print(nineq_table)

# pp test for mmd muni
pp_test <- pp.test(data$MMD_MUNI_30Y_AAA_YIELD_Diff)
print(pp_test)

# KPSS test for mmd muni
kpss_test <- kpss.test(data$MMD_MUNI_30Y_AAA_YIELD_Diff)
print(kpss_test)
