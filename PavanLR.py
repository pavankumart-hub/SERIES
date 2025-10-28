import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Auto ARIMA with KPSS", layout="wide")
st.title("📈 ARIMA Model Selection via KPSS, ACF & PACF")

# Step 1: Download data
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
days = st.sidebar.slider("Days of Data", 100, 2000, 365)

start = datetime.now() - timedelta(days=days)
end = datetime.now()

if st.button("Run Analysis"):
    with st.spinner("Downloading data..."):
        data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        st.error("No data found. Check ticker symbol.")
        st.stop()

    series = data["High"].dropna()
    st.line_chart(series)

    # Step 2: KPSS Test
    st.subheader("KPSS Test for Stationarity")

    def kpss_test(ts, regression='ct'):
        try:
            stat, p, lags, crit = kpss(ts, regression=regression, nlags='auto')
            return stat, p, crit
        except Exception as e:
            return None, None, str(e)

    stat, p_value, crit = kpss_test(series)
    if isinstance(crit, str):
        st.error(f"KPSS Error: {crit}")
        st.stop()

    st.write(f"KPSS Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    st.write("Critical Values:", crit)

    if p_value < 0.05:
        st.warning("❌ Series is non-stationary → differencing once (d=1)")
        series_diff = series.diff().dropna()
        d = 1
    else:
        st.success("✅ Series is stationary → no differencing (d=0)")
        series_diff = series
        d = 0

    # Step 3: ACF & PACF
    st.subheader("ACF and PACF Plots")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(series_diff, ax=ax[0], lags=40)
    plot_pacf(series_diff, ax=ax[1], lags=40, method="ywm")
    ax[0].set_title("ACF")
    ax[1].set_title("PACF")
    st.pyplot(fig)

    # Step 4: Auto ARIMA(p,q) suggestion
    st.subheader("Suggested ARIMA Parameters from ACF/PACF")

    # Basic heuristic based on first significant lag
    acf_vals = acf(series_diff, nlags=20)
    pacf_vals = pacf(series_diff, nlags=20)

    # ignore lag 0
    acf_peaks = np.where(np.abs(acf_vals[1:]) > 0.2)[0] + 1
    pacf_peaks = np.where(np.abs(pacf_vals[1:]) > 0.2)[0] + 1

    p = pacf_peaks[0] if len(pacf_peaks) > 0 else 0
    q = acf_peaks[0] if len(acf_peaks) > 0 else 0

    st.write(f"Based on ACF/PACF, suggested ARIMA order: **p={p}, d={d}, q={q}**")

    # Step 5: Fit ARIMA model
    st.subheader(f"Fitting ARIMA({p},{d},{q}) model")
    try:
        model = ARIMA(series, order=(p, d, q))
        result = model.fit()
        st.text(result.summary())
    except Exception as e:
        st.error(f"ARIMA fitting error: {e}")
        st.stop()

    # Step 6: Residual Plot
    residuals = result.resid.dropna()
    st.subheader("Residual Analysis")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(residuals, color='purple', linewidth=1)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("Residuals over Time")
    st.pyplot(fig)

    # Step 7: Diagnostic Tests
    st.subheader("Diagnostic Tests")

    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
    lb_stat, lb_p = acorr_ljungbox(residuals, lags=[10], return_df=False)
    lb_stat, lb_p = lb_stat[0], lb_p[0]

    st.write(f"**Jarque–Bera Test:** Statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")
    st.write(f"**Ljung–Box Test (lag=10):** Statistic = {lb_stat:.4f}, p-value = {lb_p:.4f}")

    if jb_p > 0.05:
        st.success("✅ Residuals are normally distributed (Jarque–Bera).")
    else:
        st.warning("⚠️ Residuals deviate from normality.")

    if lb_p > 0.05:
        st.success("✅ No autocorrelation in residuals (Ljung–Box).")
    else:
        st.warning("⚠️ Autocorrelation detected in residuals.")

    # Step 8: Forecast
    st.subheader("Forecast (Next 10 Steps)")
    forecast = result.get_forecast(steps=10)
    mean_forecast = forecast.predicted_mean
    conf_int = forecast.conf_int()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(series, label="Observed")
    ax2.plot(mean_forecast.index, mean_forecast.values, color='red', label="Forecast")
    ax2.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                     color='pink', alpha=0.3, label="95% CI")
    ax2.legend()
    st.pyplot(fig2)
