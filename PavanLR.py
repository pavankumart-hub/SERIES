import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime, timedelta

st.set_page_config(page_title="ARIMA + KPSS Analysis", layout="wide")
st.title("ARIMA Modeling with KPSS, Detrending/Differencing, and Diagnostic Tests")

# --- Step 1: Download data ---
ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
days = st.sidebar.slider("Days of Data", 100, 2000, 365)
start = (datetime.now() - timedelta(days=days)).date()
end = datetime.now().date()
p = st.sidebar.slider("p (AR order)", 0, 5, 1)
d = st.sidebar.slider("d (diff order)", 0, 2, 0)
q = st.sidebar.slider("q (MA order)", 0, 5, 1)

if st.button("Run Analysis"):
    with st.spinner("Downloading data..."):
        df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        st.error("No data found for the ticker.")
        st.stop()

    series = df["High"].dropna()
    st.write(f"Downloaded {len(series)} points for {ticker}")
    st.line_chart(series)

    # --- Step 2: KPSS Test ---
    def run_kpss(series, regression='ct'):
        try:
            stat, p, lags, crit = kpss(series, regression=regression, nlags='auto')
            return stat, p, crit
        except Exception as e:
            return None, None, str(e)

    st.subheader("KPSS Test Results")
    stat, p_value, crit = run_kpss(series.values, regression='ct')

    if isinstance(crit, str):
        st.error(f"KPSS error: {crit}")
        st.stop()

    st.write(f"KPSS Statistic: {stat:.4f}, p-value: {p_value:.4f}")
    st.write("Critical Values:", crit)

    # --- Step 3: Differencing or Detrending ---
    processed_series = None
    label = ""

    if p_value > 0.05:
        st.info("Series is **Trend Stationary** → detrending using polynomial fit below degree 10")
        X = np.arange(len(series)).reshape(-1, 1)
        best_aic, best_deg = np.inf, 1
        for deg in range(1, 10):
            poly = PolynomialFeatures(degree=deg)
            X_poly = poly.fit_transform(X)
            model = sm.OLS(series.values, X_poly).fit()
            if model.aic < best_aic:
                best_aic, best_deg, best_model, best_poly = model.aic, deg, model, poly
        trend = best_model.predict(best_poly.fit_transform(X))
        detrended = series - trend
        processed_series = detrended
        label = f"Detrended (degree {best_deg})"
    else:
        st.info("Series is **Not Trend Stationary** → checking differenced series")
        diff = series.diff().dropna()
        stat2, p2, crit2 = run_kpss(diff.values, regression='c')
        st.write(f"KPSS on 1st Difference: Statistic={stat2:.4f}, p-value={p2:.4f}")
        if p2 > 0.05:
            st.success("Series is **Difference Stationary** → differencing applied")
            processed_series = diff
            label = "First Difference"
        else:
            st.warning("Still non-stationary → using original for ARIMA anyway")
            processed_series = series
            label = "Original (Non-stationary)"

    # --- Step 4: ACF and PACF ---
    st.subheader("ACF and PACF of Stationary Series")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_acf(processed_series.dropna(), ax=axes[0], lags=40)
    plot_pacf(processed_series.dropna(), ax=axes[1], lags=40, method='ywm')
    axes[0].set_title("ACF Plot")
    axes[1].set_title("PACF Plot")
    st.pyplot(fig)

    # --- Step 5: Fit ARIMA ---
    st.subheader(f"ARIMA({p},{d},{q}) Model Fit")
    try:
        model = ARIMA(processed_series, order=(p, d, q))
        result = model.fit()
        st.text(result.summary())
    except Exception as e:
        st.error(f"ARIMA model failed: {e}")
        st.stop()

    # --- Step 6: Residual Diagnostics ---
    residuals = result.resid.dropna()
    st.subheader("Residuals Plot")
    st.line_chart(residuals)

    # --- Step 7: Jarque–Bera and Ljung–Box Tests ---
    jb_stat, jb_p, skew, kurt = jarque_bera(residuals)
    lb_stat, lb_p = acorr_ljungbox(residuals, lags=[10], return_df=False)
    lb_stat, lb_p = lb_stat[0], lb_p[0]

    st.subheader("Diagnostic Tests on Residuals")
    st.write(f"**Jarque–Bera Test:** Statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")
    st.write(f"**Ljung–Box Test (lag 10):** Statistic = {lb_stat:.4f}, p-value = {lb_p:.4f}")

    if jb_p > 0.05:
        st.success("✅ Residuals appear normally distributed (Jarque–Bera).")
    else:
        st.warning("⚠️ Residuals deviate from normality (Jarque–Bera).")

    if lb_p > 0.05:
        st.success("✅ No autocorrelation in residuals (Ljung–Box).")
    else:
        st.warning("⚠️ Autocorrelation present (Ljung–Box).")

    # Forecast plot
    st.subheader("Forecast (Next 10 Steps)")
    forecast = result.get_forecast(steps=10)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(series, label="Original Series")
    ax2.plot(forecast_mean.index, forecast_mean.values, label="Forecast", color="red")
    ax2.fill_between(forecast_mean.index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color='pink', alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)
