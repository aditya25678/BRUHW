#— 0. libraries
library(dplyr)
library(ggplot2)
library(forecast)
library(readxl)
library(zoo)

#— 1. load your actuals (must include the two X variables)
input_path <- "MBAM_WP/"
data       <- read_excel(file.path(input_path, "Actuals_ccar2025.xlsx"))

#— 2. dependent quarterly ts
dep_var <- ts(data$MBAM_TAP, frequency = 4, start = c(2007, 3))

#— 3. fit ARIMA(1,1,0) with exogenous regressors
m2 <- arima(
  x     = dep_var,
  order = c(1, 1, 0),
  xreg  = as.matrix(data[, c("VIX_diff_sqrtd",
                             "MMD_MUNI_30Y_AAA_YIELD_Diff")])
)

#— 4. read your five CCAR projection files
baseline <- read_excel(file.path(input_path, "baseline_ccar.xlsx"))
baca     <- read_excel(file.path(input_path, "baca_ccar.xlsx"))
usda     <- read_excel(file.path(input_path, "usda_ccar.xlsx"))
supbl    <- read_excel(file.path(input_path, "supbl_ccar.xlsx"))
supsa    <- read_excel(file.path(input_path, "supsa_ccar.xlsx"))

#— 5. horizon = number of quarters in baseline
h <- nrow(baseline)

#— 6. built‑in forecasts via forecast()
fc_base  <- forecast(m2, h = h,
                     xreg = as.matrix(baseline[, c("VIX_diff_sqrtd",
                                                   "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
fc_baca  <- forecast(m2, h = h,
                     xreg = as.matrix(baca[,     c("VIX_diff_sqrtd",
                                                   "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
fc_usda  <- forecast(m2, h = h,
                     xreg = as.matrix(usda[,     c("VIX_diff_sqrtd",
                                                   "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
fc_supbl <- forecast(m2, h = h,
                     xreg = as.matrix(supbl[,    c("VIX_diff_sqrtd",
                                                   "MMD_MUNI_30Y_AAA_YIELD_Diff")]))
fc_supsa <- forecast(m2, h = h,
                     xreg = as.matrix(supsa[,    c("VIX_diff_sqrtd",
                                                   "MMD_MUNI_30Y_AAA_YIELD_Diff")]))

# extract point forecasts
base_pred  <- as.numeric(fc_base$mean)
baca_pred  <- as.numeric(fc_baca$mean)
usda_pred  <- as.numeric(fc_usda$mean)
supbl_pred <- as.numeric(fc_supbl$mean)
supsa_pred <- as.numeric(fc_supsa$mean)

#— 7. manual replication for *baseline* starting from the *observed* last Δy
y_hist     <- as.numeric(data$MBAM_TAP)
dy_hist    <- diff(y_hist)
y_prev     <- tail(y_hist,  1)    # last actual level
dy_prev    <- tail(dy_hist, 1)    # last actual difference
phi1       <- coef(m2)["ar1"]
b1         <- coef(m2)["VIX_diff_sqrtd"]
b2         <- coef(m2)["MMD_MUNI_30Y_AAA_YIELD_Diff"]
x1_future  <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future  <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff]

manual_base <- numeric(h)
for(i in seq_len(h)) {
  dy_new         <- phi1 * dy_prev +
                    b1   * x1_future[i] +
                    b2   * x2_future[i]
  y_new          <- y_prev + dy_new
  manual_base[i] <- y_new
  # update
  y_prev    <- y_new
  dy_prev   <- dy_new
}

#— 8. build a quarterly index for the forecast
time_index <- as.character(as.yearqtr(time(fc_base$mean)))

#— 9. compare built‑in vs manual (should match exactly)
comparison_table <- data.frame(
  Quarter     = time_index,
  Built_In    = base_pred,
  Manual_Test = manual_base
)
print(comparison_table)

#— 10. optional: full scenario table
all_scenarios <- data.frame(
  Quarter  = time_index,
  Baseline = base_pred,
  BACA     = baca_pred,
  USDA     = usda_pred,
  SUPBL    = supbl_pred,
  SUPSA    = supsa_pred
)
print(all_scenarios)
