import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from statsmodels.tsa.stattools import kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ------------------- Streamlit Configuration -------------------
st.set_page_config(page_title="ARIMA Stock Forecast with KPSS", page_icon="📈", layout="wide")
st.title("📈 ARIMA Forecast with KPSS, Detrending & Diagnostics")
st.markdown("Automatically checks stationarity, detrends or differences data, fits ARIMA, and tests residuals.")

# ------------------- Sidebar Inputs -------------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
days = st.sidebar.slider("Days to Analyze", 90, 730, 365)

# ------------------- Data Download -------------------
end_date = datetime.now()
start_date = end_date - timedelta(days=days)

st.subheader(f"1️⃣ Downloading Data for {ticker}")
data = yf.download(ticker, start=start_date, end=end_date, progress=False)

if data.empty:
    st.error("No data found for this ticker. Try another symbol.")
    st.stop()

series = data["High"].dropna()
st.line_chart(series, height=300, use_container_width=True)
st.success(f"Downloaded {len(series)} data points.")

# ------------------- KPSS Test -------------------
st.subheader("2️⃣ KPSS Stationarity Test")

def kpss_test(timeseries):
    statistic, p_value, _, crit = kpss(timeseries, regression='c', nlags='auto')
    return statistic, p_value, crit

stat, pval, crit = kpss_test(series)
st.write(f"**KPSS Statistic:** {stat:.4f}, **p-value:** {pval:.4f}")
if pval > 0.05:
    st.success("✅ Fail to reject H₀ → Series is *trend stationary*. Proceeding with detrending.")
    stationary_type = "trend_stationary"
else:
    st.warning("⚠️ Reject H₀ → Series is *difference stationary*. Proceeding with differencing.")
    stationary_type = "difference_stationary"

# ------------------- Detrending or Differencing -------------------
if stationary_type == "trend_stationary":
    st.subheader("3️⃣ Detrending using Polynomial Regression (degree < 10)")

    best_deg, best_r2 = 1, -np.inf
    x = np.arange(len(series)).reshape(-1, 1)

    for deg in range(1, 10):
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(x)
        model = LinearRegression().fit(X_poly, series)
        r2 = model.score(X_poly, series)
        if r2 > best_r2:
            best_r2, best_deg = r2, deg
            best_model = model
            best_poly = poly

    trend = best_model.predict(best_poly.transform(x))
    detrended = series - trend

    st.write(f"**Best Polynomial Degree:** {best_deg} (R²={best_r2:.4f})")
    fig, ax = plt.subplots()
    ax.plot(series.index, series, label="Original Series", color="blue")
    ax.plot(series.index, trend, label=f"Trend (deg={best_deg})", color="red")
    ax.set_title("Trend Fit")
    ax.legend()
    st.pyplot(fig)
    processed_series = detrended

else:
    st.subheader("3️⃣ Differencing (First Order)")
    processed_series = series.diff().dropna()
    st.line_chart(processed_series, height=300)
    st.info("Performed first-order differencing to achieve stationarity.")

# ------------------- ACF & PACF -------------------
st.subheader("4️⃣ ACF & PACF Plots")

lags = min(40, len(processed_series)//2)
acf_vals = acf(processed_series, nlags=lags)
pacf_vals = pacf(processed_series, nlags=lags)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].stem(range(len(acf_vals)), acf_vals, use_line_collection=True)
ax[0].set_title("ACF")
ax[1].stem(range(len(pacf_vals)), pacf_vals, use_line_collection=True)
ax[1].set_title("PACF")
plt.tight_layout()
st.pyplot(fig)

st.info("Use ACF and PACF plots to choose p and q values for ARIMA.")

# ------------------- Sidebar ARIMA Parameters -------------------
st.sidebar.subheader("ARIMA(p, d, q) Parameters")
p = st.sidebar.slider("AR order (p)", 0, 10, 1)
d = st.sidebar.slider("Differencing (d)", 0, 2, 0)
q = st.sidebar.slider("MA order (q)", 0, 10, 1)

# ------------------- ARIMA Model Fitting -------------------
st.subheader("5️⃣ ARIMA Model Fitting")

try:
    model = ARIMA(processed_series, order=(p, d, q))
    fitted = model.fit()
    st.success(f"Fitted ARIMA({p},{d},{q}) successfully.")

    # Residuals
    residuals = fitted.resid

    # Residual line plot
    st.subheader("6️⃣ Residual Analysis")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(residuals)
    ax[0].set_title("Residuals over Time")
    ax[1].hist(residuals, bins=25, color='gray', edgecolor='black')
    ax[1].set_title("Residual Histogram")
    plt.tight_layout()
    st.pyplot(fig)

    # Diagnostic Tests
    jb_stat, jb_p = jarque_bera(residuals)[:2]
    ljung = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_p = ljung['lb_pvalue'].iloc[0]

    st.write(f"**Jarque–Bera Test p-value:** {jb_p:.4f}")
    st.write(f"**Ljung–Box Test p-value:** {lb_p:.4f}")

    if jb_p > 0.05:
        st.success("✅ Residuals appear normally distributed.")
    else:
        st.warning("⚠️ Residuals may not be normal.")

    if lb_p > 0.05:
        st.success("✅ No significant autocorrelation in residuals.")
    else:
        st.warning("⚠️ Autocorrelation detected in residuals.")

    # ------------------- Forecast -------------------
    st.subheader("7️⃣ Forecasting")
    forecast_steps = st.slider("Forecast Days", 1, 30, 5)
    forecast_res = fitted.get_forecast(steps=forecast_steps)
    forecast_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series, label="Original")
    forecast_index = pd.date_range(series.index[-1] + timedelta(days=1), periods=forecast_steps, freq='B')
    ax.plot(forecast_index, forecast_mean, label="Forecast", color='red')
    ax.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
    ax.legend()
    ax.set_title(f"{ticker} Forecast (95% CI)")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Model fitting failed: {str(e)}")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data Source: Yahoo Finance")
