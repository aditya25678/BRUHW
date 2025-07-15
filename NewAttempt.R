# Load libraries
library(dplyr)
library(ggplot2)
library(forecast)       # for predict.Arima()
library(lubridate)
library(PerformanceAnalytics)
library(readxl)
library(tseries)
library(car)
library(MTS)
library(TSA)
library(urca)
library(zoo)

# Read in Data and autoplot
input_path <- "MBAM_WP/"
data <- read_excel(file.path(input_path, "Actuals_ccar2025.xlsx"))
# (Your Actuals_ccar2025.xlsx must include VIX_diff_sqrtd & MMD_MUNI_30Y_AAA_YIELD_Diff)

autoplot(data[ , "MBAM_TAP"], facets = TRUE) +
  xlab("Year") + xlab("") + ggtitle("MBAM TAP History")

# Time-Series data
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
raittest(m2)
res2 <- residuals(m2)
plot(res2)

# Normality
shapiro.test(resid(m2))

# Stationarity
adf.test(m2$residuals)

# Heteroskedasticity
bptest(res2, ~ data$VIX_diff_sqrtd + data$MMD_MUNI_30Y_AAA_YIELD_Diff)
archTest(res2, lag = 10)

# Autocorrelation
Box.test(res2, lag = 8, type = "Ljung-Box")

# Projection files
baseline <- read_excel(file.path(input_path, "baseline_ccar.xlsx"))
baca     <- read_excel(file.path(input_path, "baca_ccar.xlsx"))
bacsa    <- read_excel(file.path(input_path, "bacsa_ccar.xlsx"))

# ---------------------------------------------
# Option A: manual recursion seeded with *fitted* state
# ---------------------------------------------

# 1. Built‑in ARIMAX forecasts for each scenario
h <- nrow(baseline)

f_base  <- predict(m2, n.ahead = h,
                   newxreg = as.matrix(baseline[, c("VIX_diff_sqrtd",
                                                    "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f_baca  <- predict(m2, n.ahead = h,
                   newxreg = as.matrix(baca[,     c("VIX_diff_sqrtd",
                                                    "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f_bacsa <- predict(m2, n.ahead = h,
                   newxreg = as.matrix(bacsa[,    c("VIX_diff_sqrtd",
                                                    "MMD_MUNI_30Y_AAA_YIELD_Diff")]))

# 2. Extract point forecasts
pred_base  <- as.numeric(f_base$pred)
pred_baca  <- as.numeric(f_baca$pred)
pred_bacsa <- as.numeric(f_bacsa$pred)

# 3. Seed manual loop from *fitted* values
fitted_vals <- fitted(m2)
y_prev      <- tail(fitted_vals, 1)
dy_prev     <- y_prev - tail(fitted_vals, 2)[1]

# 4. Get coefficients and future X
phi1        <- coef(m2)["ar1"]
b1          <- coef(m2)["VIX_diff_sqrtd"]
b2          <- coef(m2)["MMD_MUNI_30Y_AAA_YIELD_Diff"]
x1_future   <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future   <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)

# 5. Recursive manual forecast (baseline)
manual_base <- numeric(h)
for (i in seq_len(h)) {
  dy_new         <- phi1 * dy_prev + b1 * x1_future[i] + b2 * x2_future[i]
  y_new          <- y_prev + dy_new
  manual_base[i] <- y_new
  dy_prev        <- dy_new
  y_prev         <- y_new
}

# 6. Build forecast quarter index
start_year    <- 2025
start_quarter <- 2
forecast_quarters <- seq(
  from   = as.yearqtr(paste(start_year, start_quarter, sep = " Q")),
  by     = 0.25,
  length.out = h
)

# 7. Print comparison for baseline
comparison <- data.frame(
  Quarter        = as.character(forecast_quarters),
  ARIMA_Predict  = pred_base,
  Manual_Seeded  = manual_base
)
print(comparison)

# 8. (Optional) Print full scenario forecasts
all_scenarios <- data.frame(
  Quarter  = as.character(forecast_quarters),
  Baseline = pred_base,
  BACA     = pred_baca,
  BACSA    = pred_bacsa
)
print(all_scenarios)
