import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA with KPSS and Trend Handling", layout="wide")
st.title("📈 ARIMA Modelling with KPSS, Detrending & Diagnostic Tests")

# Step 1: Download data
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
days = st.sidebar.slider("Days of Data", 200, 2000, 365)

start = datetime.now() - timedelta(days=days)
end = datetime.now()

if st.button("Run Full Analysis"):
    with st.spinner("Downloading data..."):
        data = yf.download(ticker, start=start, end=end, progress=False)

    if data.empty:
        st.error("No data found. Check ticker symbol.")
        st.stop()

    st.subheader("Step 1️⃣: High Price Data")
    st.line_chart(data["High"])
    series = data["High"].dropna()

    # Step 2: KPSS Test
    st.subheader("Step 2️⃣: KPSS Test for Stationarity")

    def kpss_test(ts, regression='ct'):
        stat, p_value, lags, crit = kpss(ts, regression=regression, nlags='auto')
        return stat, p_value, crit

    stat, p_value, crit = kpss_test(series)
    st.write(f"**KPSS Statistic:** {stat:.4f}, **p-value:** {p_value:.4f}")
    st.write("Critical Values:", crit)

    # If H0 accepted => trend stationary → detrend using best polynomial (<10)
    if p_value > 0.05:
        st.success("✅ KPSS H₀ accepted: Series is Trend-Stationary → Performing Detrending")

        best_deg = 1
        best_r2 = -np.inf
        t = np.arange(len(series)).reshape(-1, 1)

        for deg in range(1, 10):
            poly = PolynomialFeatures(degree=deg)
            X_poly = poly.fit_transform(t)
            model = LinearRegression().fit(X_poly, series)
            r2 = model.score(X_poly, series)
            if r2 > best_r2:
                best_r2 = r2
                best_deg = deg
                best_model = model
                best_poly = poly

        trend = best_model.predict(best_poly.fit_transform(t))
        detrended = series - trend

        st.write(f"Best Polynomial Degree: {best_deg}, R² = {best_r2:.4f}")
        fig, ax = plt.subplots()
        ax.plot(series, label="Original Series")
        ax.plot(trend, label=f"Polynomial Trend (deg={best_deg})", color='red')
        ax.legend()
        st.pyplot(fig)

        series_final = detrended
        d = 0

    else:
        st.warning("❌ KPSS H₁ accepted: Series is Non-Stationary → Differencing Applied")
        series_final = series.diff().dropna()
        d = 1

    # Step 3: ACF and PACF
    st.subheader("Step 3️⃣: ACF & PACF Plots and ARIMA Fitting")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(series_final, ax=ax[0], lags=40)
    plot_pacf(series_final, ax=ax[1], lags=40, method="ywm")
    ax[0].set_title("ACF")
    ax[1].set_title("PACF")
    st.pyplot(fig)

    acf_vals = acf(series_final, nlags=20)
    pacf_vals = pacf(series_final, nlags=20)
    acf_peaks = np.where(np.abs(acf_vals[1:]) > 0.2)[0] + 1
    pacf_peaks = np.where(np.abs(pacf_vals[1:]) > 0.2)[0] + 1

    p = pacf_peaks[0] if len(pacf_peaks) > 0 else 0
    q = acf_peaks[0] if len(acf_peaks) > 0 else 0

    st.write(f"Suggested ARIMA Order: (p={p}, d={d}, q={q})")

    try:
        model = ARIMA(series, order=(p, d, q))
        result = model.fit()
        st.text(result.summary())
    except Exception as e:
        st.error(f"ARIMA fitting error: {e}")
        st.stop()

    # Step 4: Residuals
    st.subheader("Step 4️⃣: Residual Analysis")
    residuals = result.resid.dropna()

    col1, col2 = st.columns(2)
    with col1:
        st.write("Residual Line Plot")
        fig1, ax1 = plt.subplots()
        ax1.plot(residuals, color="purple")
        ax1.axhline(0, color='black', linestyle='--')
        st.pyplot(fig1)

    with col2:
        st.write("Residual Histogram")
        fig2, ax2 = plt.subplots()
        ax2.hist(residuals, bins=25, color='teal', alpha=0.7)
        st.pyplot(fig2)

    # Step 5: Tests
    st.subheader("Step 5️⃣: Diagnostic Tests")
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
