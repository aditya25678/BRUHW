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

# Define input path
input_path <- "MBAM_WP"

# Read in Data and autoplot
# data <- read_excel(file.path(input_path,"Actuals_ccar2025.xlsx")) # for CCAR Data
# data <- read_excel(file.path(input_path,"Actuals_202025.xlsx"))   # for 202025 data
data <- read_excel(file.path(input_path, "Actuals_ccar2025.xlsx"))

# Time‑Series data
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
raittest(m2)
res2 <- residuals(m2)
plot(res2)
shapiro.test(res2)
adf.test(res2)
bptest(res2, ~ data$VIX_diff_sqrtd + data$MMD_MUNI_30Y_AAA_YIELD_Diff)
archTest(res2, lag=10); archTest(res2, lag=5); archTest(res2, lag=2)
Box.test(res2, lag = 8, type = "Ljung-Box")

#### Projection files ####
baseline <- read_excel(file.path(input_path, "baseline_ccar.xlsx"))
baca     <- read_excel(file.path(input_path, "baca_ccar.xlsx"))
bacsa    <- read_excel(file.path(input_path, "bacsa_ccar.xlsx"))

# Determine each forecast horizon from your sheets
h_base  <- nrow(baseline)
h_baca  <- nrow(baca)
h_bacsa <- nrow(bacsa)

# ARIMA predictions
f2base  <- predict(m2,
                   n.ahead = h_base,
                   newxreg  = as.matrix(baseline[,  c("VIX_diff_sqrtd",
                                                       "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2baca  <- predict(m2,
                   n.ahead = h_baca,
                   newxreg  = as.matrix(baca[,      c("VIX_diff_sqrtd",
                                                       "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2bacsa <- predict(m2,
                   n.ahead = h_bacsa,
                   newxreg  = as.matrix(bacsa[,     c("VIX_diff_sqrtd",
                                                       "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
back2   <- fitted(m2)

### Manual “seeded” forecast — replicate predict()’s methodology exactly ###

# 1. Grab the last two *fitted* (one‑step) values of the original series
fitted_vals   <- fitted(m2)                       # length = length(dep_var)
last_fit      <- tail(fitted_vals, 1)             # ŷₙ|ₙ₋₁
prev_fit      <- tail(fitted_vals, 2)[1]          # ŷₙ₋₁|ₙ₋₂
dy_prev       <- last_fit - prev_fit              # Δŷₙ
y_prev        <- last_fit                         # ŷₙ

# 2. Future exogenous variables for baseline
x1_future <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)

# 3. Coefficients from the fitted model
coef_m2 <- coef(m2)
ar1     <- coef_m2["ar1"]
b1      <- coef_m2["VIX_diff_sqrtd"]
b2      <- coef_m2["MMD_MUNI_30Y_AAA_YIELD_Diff"]
# No intercept/drift (include.mean = FALSE when d=1 in stats::arima)

# 4. Iterate the ARIMA(1,1,0) forecasting equation
manual_seeded <- numeric(h_base)
for (i in seq_len(h_base)) {
  # forecast the *difference* Δŷₙ₊ᵢ
  dy_new <- ar1 * dy_prev + b1 * x1_future[i] + b2 * x2_future[i]
  # invert differencing to get ŷₙ₊ᵢ
  y_new <- y_prev + dy_new
  # store
  manual_seeded[i] <- y_new
  # update for next
  dy_prev <- dy_new
  y_prev  <- y_new
}

# 5. Compare manual vs predict() outputs
comparison <- data.frame(
  Horizon       = 1:h_base,
  Manual_seeded = manual_seeded,
  ARIMA_Pred    = as.numeric(f2base$pred),
  Diff          = manual_seeded - as.numeric(f2base$pred)
)
print(comparison)   # should all be zero (or very near zero)

#### Combine into a forecast table ####
library(zoo)
time_index <- seq(
  from          = as.yearqtr("2025 Q2"),
  by            = 0.25,
  length.out    = h_base
)

forecast_table <- data.frame(
  Quarter  = as.character(time_index),
  Baseline = as.numeric(f2base$pred),
  BACA     = as.numeric(f2baca$pred),
  BACSA    = as.numeric(f2bacsa$pred)
)
print(forecast_table)

# … the remainder of your diagnostics, in‑sample/out‑of‑sample tests, plotting, etc. remains unchanged …
