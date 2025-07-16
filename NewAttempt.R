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
library(zoo)

#— 1. Setup -------------------------------------------------------------------

input_path <- "MBAM_WP"

# Read in actuals
data <- read_excel(file.path(input_path, "Actuals_ccar2025.xlsx"))

# Time‐series object
autoplot(data[,"MBAM_TAP"], facets=TRUE) +
  xlab("Year") + xlab("") + ggtitle("MBAM TAP History")
dep_var <- ts(data$MBAM_TAP, frequency = 4, start = c(2007, 3))

#— 2. In‐sample linear & ARIMAX fit -------------------------------------------

# Linear model (for VIF, etc.)
m <- lm(dep_var ~ VIX_diff_sqrtd + MMD_MUNI_30Y_AAA_YIELD_Diff, data = data)
summary(m); vif(m)

# ARIMAX(1,1,0) with two exogenous regressors
m2 <- arima(
  dep_var,
  order = c(1, 1, 0),
  xreg  = as.matrix(data[, c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")])
)

#— 3. Diagnostics -------------------------------------------------------------

res2 <- residuals(m2)
raittest(m2)
plot(res2)
shapiro.test(res2)
adf.test(res2)
bptest(res2, ~ VIX_diff_sqrtd + MMD_MUNI_30Y_AAA_YIELD_Diff, data = data)
archTest(res2, lag=10); archTest(res2, lag=5); archTest(res2, lag=2)
Box.test(res2, lag = 8, type = "Ljung-Box")

#— 4. Load projection scenarios ----------------------------------------------

baseline <- read_excel(file.path(input_path, "baseline_ccar.xlsx"))
baca     <- read_excel(file.path(input_path, "baca_ccar.xlsx"))
bacsa    <- read_excel(file.path(input_path, "bacsa_ccar.xlsx"))

# forecast lengths
h_base  <- nrow(baseline)
h_baca  <- nrow(baca)
h_bacsa <- nrow(bacsa)

#— 5. ARIMA built‑in forecasts -----------------------------------------------

f2base  <- predict(m2,
                   n.ahead = h_base,
                   newxreg  = as.matrix(baseline[, c("VIX_diff_sqrtd",
                                                     "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2baca  <- predict(m2,
                   n.ahead = h_baca,
                   newxreg  = as.matrix(baca[,     c("VIX_diff_sqrtd",
                                                     "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2bacsa <- predict(m2,
                   n.ahead = h_bacsa,
                   newxreg  = as.matrix(bacsa[,    c("VIX_diff_sqrtd",
                                                     "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
back2   <- fitted(m2)

#— 6. Manual “seeded” forecast (fixed) ----------------------------------------

# a) extract last two one‑step fitted values
fv      <- as.numeric(fitted(m2))
dy_prev <- diff( tail(fv, 2) )   # length 1
dy_prev <- dy_prev[1]
y_prev  <- fv[length(fv)]

# b) coefficients
cm2  <- coef(m2)
ar1  <- cm2["ar1"]
b1   <- cm2["VIX_diff_sqrtd"]
b2   <- cm2["MMD_MUNI_30Y_AAA_YIELD_Diff"]

# c) future exogenous (baseline)
x1f <- as.numeric(baseline$VIX_diff_sqrtd)
x2f <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)

# d) loop
manual_seeded <- numeric(h_base)
for (j in seq_along(x1f)) {
  # differenced forecast
  dy_new <- ar1 * dy_prev + b1 * x1f[j] + b2 * x2f[j]
  # invert differencing
  y_new  <- y_prev + dy_new
  manual_seeded[j] <- y_new
  # update for next step
  dy_prev <- dy_new
  y_prev  <- y_new
}

# e) sanity check
print(manual_seeded)

#— 7. Compare to predict() ---------------------------------------------------

comparison <- data.frame(
  Horizon       = seq_len(h_base),
  Manual_seeded = manual_seeded,
  ARIMA_Pred    = as.numeric(f2base$pred),
  Diff          = manual_seeded - as.numeric(f2base$pred)
)
print(comparison)   # all Diff should be 0 (or extremely small)

#— 8. Build combined forecast table -----------------------------------------

time_index <- seq(from = as.yearqtr("2025 Q2"),
                  by   = 0.25,
                  length.out = h_base)

forecast_table <- data.frame(
  Quarter  = as.character(time_index),
  Baseline = as.numeric(f2base$pred),
  BACA     = as.numeric(f2baca$pred),
  BACSA    = as.numeric(f2bacsa$pred)
)
print(forecast_table)

#— 9. (rest of in‑sample / out‑of‑sample tests & plotting unchanged) ---------
