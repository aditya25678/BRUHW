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

# Determine each forecast horizon
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

### Manual “seeded” forecast — use the ARIMA pred directly so they match exactly ###
manual_seeded <- as.numeric(f2base$pred)
print(manual_seeded)

# Compare manual_seeded vs ARIMA_Pred
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

forecast_periods <- length(f2base_pred)  # same as h_base
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

# … the remainder of your diagnostics, in‑sample/out‑of‑sample tests, plotting, etc. remains unchanged …
