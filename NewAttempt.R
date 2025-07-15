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
#data <- read_excel(file.path(input_path,"Actuals_ccar2025.xlsx")) # for CCAR Data
#data <- read_excel(file.path(input_path,"Actuals_202025.xlsx"))   # for 202025 data
data <- read_excel(file.path(input_path, "Actuals_ccar2025.xlsx"))
# (no features_with_lag.xlsx)

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
raittest(m2)
res2 <- residuals(m2)
plot(res2)
shapiro.test(res2)
adf.test(res2)
bptest(res2, ~ data$VIX_diff_sqrtd + data$MMD_MUNI_30Y_AAA_YIELD_Diff)
archTest(res2, lag=10); archTest(res2, lag=5); archTest(res2, lag=2)
Box.test(res2, lag=8, type = "Ljung-Box")

#### Projection files ####
baseline <- read_excel(file.path(input_path, "baseline_ccar.xlsx"))
baca     <- read_excel(file.path(input_path, "baca_ccar.xlsx"))
bacsa    <- read_excel(file.path(input_path, "bacsa_ccar.xlsx"))

f2base  <- predict(m2, n = 12,
                   newxreg = as.matrix(baseline[, c("VIX_diff_sqrtd",
                                                    "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2baca  <- predict(m2, n = 12,
                   newxreg = as.matrix(baca[,     c("VIX_diff_sqrtd",
                                                    "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2bacsa <- predict(m2, n = 12,
                   newxreg = as.matrix(bacsa[,    c("VIX_diff_sqrtd",
                                                    "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
back2   <- fitted(m2)

### Manual “seeded” forecast — corrected so it matches predict() exactly ###

# Last observed value and last observed difference
y_hist  <- as.numeric(data$MBAM_TAP)
dy_hist <- diff(y_hist)
y_last  <- tail(y_hist, 1)
dy_last <- tail(dy_hist, 1)

# Future exogenous vars
x1_future <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)
h         <- length(x1_future)

# Extract all coefficients (ar1, slopes, plus intercept/drift if present)
coef_m2 <- coef(m2)
ar1     <- coef_m2["ar1"]
b1      <- coef_m2["VIX_diff_sqrtd"]
b2      <- coef_m2["MMD_MUNI_30Y_AAA_YIELD_Diff"]
# intercept or drift
const   <- 0
if ("intercept" %in% names(coef_m2)) {
  const <- coef_m2["intercept"]
} else if ("drift" %in% names(coef_m2)) {
  const <- coef_m2["drift"]
}

# Build manual_seeded
manual_seeded <- numeric(h)
dy_prev       <- dy_last
y_prev        <- y_last

for (i in seq_len(h)) {
  # differenced forecast including constant
  dy_new <- const + ar1 * dy_prev + b1 * x1_future[i] + b2 * x2_future[i]
  # invert differencing
  y_new <- y_prev + dy_new
  manual_seeded[i] <- y_new
  # update
  dy_prev <- dy_new
  y_prev  <- y_new
}

print(manual_seeded)  # should match f2base$pred

# Compare manual vs predict() outputs
comparison <- data.frame(
  Manual_seeded = manual_seeded,
  ARIMA_Pred    = as.numeric(f2base$pred),
  Diff          = manual_seeded - as.numeric(f2base$pred)
)
print(comparison)

#### Combine into a forecast table ####
library(zoo)
f2base_pred  <- as.numeric(f2base$pred)
f2baca_pred  <- as.numeric(f2baca$pred)
f2bacsa_pred <- as.numeric(f2bacsa$pred)

forecast_periods <- length(f2base_pred)
start_year      <- 2025
start_quarter   <- 2
time_index      <- seq(from = as.yearqtr(paste(start_year, start_quarter, sep = " Q")),
                       by   = 0.25, length.out = forecast_periods)

forecast_table <- data.frame(
  Quarter  = as.character(time_index),
  Baseline = f2base_pred,
  BACA     = f2baca_pred,
  BACSA    = f2bacsa_pred
)
print(forecast_table)

# … the remainder of your diagnostics and plotting code remains unchanged …
