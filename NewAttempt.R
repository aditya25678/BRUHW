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

#Read in Data and autoplot
#input_path <- "MBAM WP/"
#data <- read_excel(file.path(input_path,"Actuals_ccar2025.xlsx")) # for CCAR Data
#data <- read_excel(file.path(input_path,"Actuals_202025.xlsx"))   #202025 data
data <- read_excel(file.path("MBAM_WP/Actuals_ccar2025.xlsx"))      #102025 data
more_features <- read_excel(file.path(input_path,"features_with_lag.xlsx"))
data <- cbind(data, more_features)

#Time-Series data
autoplot(data[,"MBAM_TAP"], facets=TRUE) + xlab("Year") + xlab("") + ggtitle("MBAM TAP History")
dep_var <- ts(data$`MBAM_TAP`, frequency = 4, start = c(2007,3))

#Linear Model
m <- lm(dep_var ~ VIX_diff_sqrtd + MMD_MUNI_30Y_AAA_YIELD_Diff, data=data)
summary(m)
vif(m)

#ARIMAX
m2 <- arima(
  dep_var,
  order = c(1, 1, 0),
  xreg = as.matrix(data[,c("VIX_diff_sqrtd", "MMD_MUNI_30Y_AAA_YIELD_Diff")])
)

#### Diagnostics ####
t_test <- coef(m2)
# … your other diagnostic code unchanged …

#### Projection files (only the ones you have) ####
baseline <- read_excel(file.path(input_path,"baseline_ccar.xlsx"))
baca     <- read_excel(file.path(input_path,"baca_ccar.xlsx"))
bacsa    <- read_excel(file.path(input_path,"bacsa_ccar.xlsx"))
# (usda, supbl, supsa removed)

f2base  <- predict(m2, n=12, newxreg = as.matrix(baseline[,c("VIX_diff_sqrtd","MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2baca  <- predict(m2, n=12, newxreg = as.matrix(baca[,c("VIX_diff_sqrtd","MMD_MUNI_30Y_AAA_YIELD_Diff")]))
f2bacsa <- predict(m2, n=12, newxreg = as.matrix(bacsa[,c("VIX_diff_sqrtd","MMD_MUNI_30Y_AAA_YIELD_Diff")]))
back2   <- fitted(m2)

# … any frozen‐ARIMA or other blocks stay exactly as before …

#### Manual testing: now extracts **all** coefficients (including intercept/drift) and uses them ####

# Pull out your last observed value and last observed difference
y_hist <- as.numeric(data$MBAM_TAP)
dy_hist <- diff(y_hist)
y_last  <- tail(y_hist,1)
dy_last <- tail(dy_hist,1)

# Future exogenous variables
x1_future <- as.numeric(baseline$VIX_diff_sqrtd)
x2_future <- as.numeric(baseline$MMD_MUNI_30Y_AAA_YIELD_Diff)
h <- length(x1_future)

# **Dynamically** grab the fitted coefficients
coef_m2 <- coef(m2)
ar1  <- coef_m2["ar1"]
b1   <- coef_m2["VIX_diff_sqrtd"]
b2   <- coef_m2["MMD_MUNI_30Y_AAA_YIELD_Diff"]

# Get the constant (intercept or drift) if the model estimated it
const <- 0
if ("intercept" %in% names(coef_m2)) {
  const <- coef_m2["intercept"]
} else if ("drift" %in% names(coef_m2)) {
  const <- coef_m2["drift"]
}

# Build the manual forecast so it exactly matches `predict()`
y_forecast <- numeric(h)
dy_prev     <- dy_last
y_prev      <- y_last

for (i in seq_len(h)) {
  # include the constant term here
  dy_new <- const + ar1 * dy_prev + b1 * x1_future[i] + b2 * x2_future[i]
  y_new  <- y_prev + dy_new
  y_forecast[i] <- y_new
  dy_prev <- dy_new
  y_prev  <- y_new
}

print(y_forecast)   # now should be identical to f2base$pred

#### Rebuild your forecast table ####
# Extract the numeric predictions
f2base_pred  <- as.numeric(f2base$pred)
f2baca_pred  <- as.numeric(f2baca$pred)
f2bacsa_pred <- as.numeric(f2bacsa$pred)

library(zoo)
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

# … the rest of your script (in‑sample/out‑of‑sample tests, plots, etc.) stays exactly as before …
