import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Auto ARIMA Forecast", page_icon="📈", layout="wide")
st.title("📈 Automatic ARIMA Model with KPSS, Detrending & Diagnostics")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
days = st.sidebar.slider("Days to Analyze", 90, 730, 365)
forecast_steps = st.sidebar.slider("Forecast Steps", 1, 30, 5)

# ---------------- Download Data ----------------
st.subheader(f"1️⃣ Downloading Data for {ticker}")
end_date = datetime.now()
start_date = end_date - timedelta(days=days)

data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

series = data["High"].dropna()
st.line_chart(series, use_container_width=True)
st.success(f"Downloaded {len(series)} points.")

# ---------------- KPSS Test ----------------
st.subheader("2️⃣ KPSS Stationarity Test")

def kpss_test(ts):
    stat, p, lags, crit = kpss(ts, regression="c", nlags="auto")
    return stat, p, crit

stat, pval, crit = kpss_test(series)
st.write(f"**KPSS Statistic:** {stat:.4f}, **p-value:** {pval:.4f}")

if pval > 0.05:
    st.success("✅ Fail to reject H₀ → Series is trend stationary. Proceeding with detrending.")
    stationary_type = "trend_stationary"
else:
    st.warning("⚠️ Reject H₀ → Series is difference stationary. Proceeding with differencing.")
    stationary_type = "difference_stationary"

# ---------------- Detrending or Differencing ----------------
if stationary_type == "trend_stationary":
    st.subheader("3️⃣ Detrending using Polynomial Regression (<10)")

    x = np.arange(len(series)).reshape(-1, 1)
    best_deg, best_r2 = 1, -np.inf
    best_model, best_poly = None, None

    for deg in range(1, 10):
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(x)
        model = LinearRegression().fit(X_poly, series)
        r2 = model.score(X_poly, series)
        if r2 > best_r2:
            best_deg, best_r2 = deg, r2
            best_model, best_poly = model, poly

    trend = best_model.predict(best_poly.transform(x))
    detrended = series - trend
    st.write(f"**Best Polynomial Degree:** {best_deg} (R²={best_r2:.4f})")

    fig, ax = plt.subplots()
    ax.plot(series.index, series, label="Original", color="blue")
    ax.plot(series.index, trend, label=f"Trend (deg={best_deg})", color="red")
    ax.legend(); ax.set_title("Trend Fit")
    st.pyplot(fig)

    processed_series = detrended
    d = 0  # no differencing needed for detrended

else:
    st.subheader("3️⃣ Differencing (First Order)")
    processed_series = series.diff().dropna()
    st.line_chart(processed_series)
    st.info("Performed first-order differencing.")
    d = 1  # differencing needed

# ---------------- Automatic ARIMA Grid Search ----------------
st.subheader("4️⃣ Automatic ARIMA Model Selection (p, q ∈ [0,5])")

best_aic = np.inf
best_order = None
best_model = None

for p in range(0, 6):
    for q in range(0, 6):
        try:
            model = ARIMA(processed_series, order=(p, d, q))
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
                best_model = fitted
        except:
            continue

if best_model is None:
    st.error("ARIMA fitting failed for all combinations.")
    st.stop()

st.success(f"✅ Best ARIMA Order: {best_order}, AIC={best_aic:.2f}")

# ---------------- Residual Analysis ----------------
st.subheader("5️⃣ Residual Analysis")
residuals = best_model.resid

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(residuals)
ax[0].set_title("Residuals Over Time")
ax[1].hist(residuals, bins=25, color='gray', edgecolor='black')
ax[1].set_title("Residual Histogram")
plt.tight_layout()
st.pyplot(fig)

# ---------------- Diagnostic Tests ----------------
jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
ljung = acorr_ljungbox(residuals, lags=[10], return_df=True)
lb_p = ljung['lb_pvalue'].iloc[0]

st.write(f"**Jarque–Bera p-value:** {jb_p:.4f}")
st.write(f"**Ljung–Box p-value:** {lb_p:.4f}")

if jb_p > 0.05:
    st.success("✅ Residuals appear normally distributed.")
else:
    st.warning("⚠️ Residuals may not be normal.")

if lb_p > 0.05:
    st.success("✅ No autocorrelation detected in residuals.")
else:
    st.warning("⚠️ Residuals show autocorrelation.")

# ---------------- Forecasting ----------------
st.subheader("6️⃣ Forecasting")

forecast = best_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

forecast_index = pd.date_range(series.index[-1] + timedelta(days=1), periods=forecast_steps, freq='B')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(series.index, series, label="Original", color="blue")
ax.plot(forecast_index, forecast_mean, label="Forecast", color="red")
ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="pink", alpha=0.3)
ax.legend()
ax.set_title(f"{ticker} Forecast (ARIMA{best_order}) - 95% CI")
st.pyplot(fig)

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("Built with ❤️ Streamlit | Auto ARIMA • KPSS • Detrending • Diagnostics")
