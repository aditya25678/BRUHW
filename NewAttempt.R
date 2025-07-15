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
data <- read_excel(file.path("MBAM_WP/Actuals_ccar2025.xlsx"))      #102025 data

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

# Baseline manual prediction
y_hist  <- as.numeric(data$MBAM_TAP)
dy_hist <- diff(y_hist)      # differenced series
y_last  <- tail(y_hist, 1)
dy_last <- tail(dy_hist, 1)

# Future exogenous variables from baseline
x1_future <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)
h         <- length(x1_future)

# Extract coefficients (including any intercept/drift)
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

# Compare manual vs predict() outputs
comparison <- data.frame(
  Manual     = y_forecast,
  ARIMA_Pred = as.numeric(f2base$pred),
  Diff       = y_forecast - as.numeric(f2base$pred)
)
print(comparison)

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

# … the remainder of your diagnostics, in‐sample/out‐of‐sample tests, plots, etc. unchanged …
