# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# Modeling / stats
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from statsmodels.tsa.stattools import kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ARIMA with KPSS → Detrend/Diff → ACF/PACF → Diagnostics", layout="wide")
st.title("ARIMA workflow: KPSS → detrend/difference → ACF/PACF → ARIMA → Residual Tests")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
days = st.sidebar.slider("Days to download", min_value=60, max_value=2000, value=365)
sig_threshold = st.sidebar.slider("Significance threshold for ACF/PACF peaks", 0.15, 0.40, 0.20, step=0.01)
forecast_steps = st.sidebar.number_input("Forecast steps", min_value=1, max_value=60, value=10)

run = st.sidebar.button("Run Full Analysis")

def safe_jarque_bera(resids):
    """Return (stat, pvalue) in a backward/forward compatible way."""
    try:
        res = jarque_bera(resids)
        if isinstance(res, (tuple, list)):
            # older SciPy returns (stat, pvalue, skew, kurtosis)
            stat = float(res[0])
            pval = float(res[1])
        else:
            # newer SciPy returns object with .statistic and .pvalue
            stat = float(res.statistic)
            pval = float(res.pvalue)
        return stat, pval, None
    except Exception as e:
        return np.nan, np.nan, str(e)

def best_polynomial_by_aic(y_vals, max_deg=9):
    """Fit polynomial degrees 1..max_deg with OLS and select degree with lowest AIC.
       Returns: best_deg, trend_values (array), model_obj, poly_obj, results_list
    """
    t = np.arange(len(y_vals)).reshape(-1, 1).astype(float)
    best_aic = np.inf
    best_deg = 1
    best_model = None
    best_poly = None
    results_list = []
    for deg in range(1, max_deg + 1):
        poly = PolynomialFeatures(degree=deg, include_bias=True)  # include constant
        X_poly = poly.fit_transform(t)
        try:
            ols = sm.OLS(y_vals, X_poly).fit()
            results_list.append((deg, float(ols.aic)))
            if ols.aic < best_aic:
                best_aic = ols.aic
                best_deg = deg
                best_model = ols
                best_poly = poly
        except Exception:
            results_list.append((deg, np.nan))
    if best_model is None:
        raise RuntimeError("Failed to fit polynomial trend models.")
    trend_vals = best_model.predict(best_poly.transform(t))
    return best_deg, trend_vals, best_model, best_poly, results_list

if run:
    st.header(f"1) Downloading data for {ticker}")
    start_date = (datetime.now() - timedelta(days=days)).date().isoformat()
    end_date = datetime.now().date().isoformat()
    st.write(f"Date range: {start_date} → {end_date}")
    with st.spinner("Downloading from Yahoo Finance..."):
        df = yf.download(ticker, start=(datetime.now() - timedelta(days=days)), end=datetime.now(), progress=False)

    if df is None or df.empty or "High" not in df.columns:
        st.error("No data found or 'High' column missing. Check ticker or range.")
        st.stop()

    series = df["High"].dropna()
    st.write(f"Downloaded {len(series)} data points.")
    st.line_chart(series)

    # 2) KPSS Test (trend test — regression='ct')
    st.header("2) KPSS Test (trend stationarity)")
    try:
        k_stat, k_pvalue, k_lags, k_crit = kpss(series.values, regression='ct', nlags='auto')
        st.write(f"KPSS statistic = {k_stat:.4f}, p-value = {k_pvalue:.4f}")
        st.write("Critical values:", k_crit)
    except Exception as ex:
        st.error(f"KPSS test failed: {ex}")
        st.stop()

    # Decision: if p > 0.05 => fail to reject null => trend-stationary => detrend
    processed_series = None
    mode = None  # 'detrended' or 'differenced'
    chosen_degree = None
    if k_pvalue is not None and k_pvalue > 0.05:
        st.success("KPSS H₀ accepted: series appears TREND-STATIONARY. Proceeding to polynomial detrending (<10).")
        yvals = series.values.astype(float)
        try:
            best_deg, trend_vals, best_model, best_poly, deg_results = best_polynomial_by_aic(yvals, max_deg=9)
            chosen_degree = best_deg
            detrended = series - pd.Series(trend_vals, index=series.index)
            processed_series = detrended.dropna()
            mode = "detrended"
            st.write(f"Selected polynomial degree = {best_deg} (lowest AIC).")
            if st.checkbox("Show polynomial AICs and OLS summary"):
                st.write(pd.DataFrame(deg_results, columns=["degree", "AIC"]))
                st.text(best_model.summary().as_text())
            # Plot original + trend
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series.index, series.values, label="Original")
            ax.plot(series.index, trend_vals, label=f"Polynomial trend (deg={best_deg})", color="red")
            ax.legend()
            ax.set_title("Original series and fitted polynomial trend")
            st.pyplot(fig)
        except Exception as ex:
            st.error(f"Polynomial detrending failed: {ex}")
            st.stop()
    else:
        st.info("KPSS H₀ rejected: series appears NON-STATIONARY. Applying first difference.")
        differenced = series.diff().dropna()
        if len(differenced) < 10:
            st.warning("Too few points after differencing — results may be unreliable.")
        processed_series = differenced
        mode = "differenced"
        chosen_degree = None
        # plot differenced
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(processed_series.index, processed_series.values)
        ax.set_title("First-differenced series (used for modeling)")
        st.pyplot(fig)

    st.write(f"Mode selected: **{mode}**")
    st.write("Preview of processed series (first/last 5):")
    st.write(processed_series.head().to_frame("processed").append(processed_series.tail().to_frame("processed")))

    # 3) Plot ACF & PACF of processed series
    st.header("3) ACF & PACF (processed series)")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(processed_series.dropna(), ax=axes[0], lags=40, zero=False)
    plot_pacf(processed_series.dropna(), ax=axes[1], lags=40, zero=False, method='ywm')
    axes[0].set_title("ACF of processed series")
    axes[1].set_title("PACF of processed series")
    st.pyplot(fig)

    # Auto-select p and q using simple threshold on first significant lag
    st.header("Select AR/MA orders from ACF/PACF (automatic suggestion)")
    acf_vals = acf(processed_series.dropna(), nlags=20, fft=False)
    pacf_vals = pacf(processed_series.dropna(), nlags=20, method='ywm')
    # detect first lag where abs(value) > threshold
    thr = float(sig_threshold)
    acf_peaks = np.where(np.abs(acf_vals[1:]) > thr)[0] + 1
    pacf_peaks = np.where(np.abs(pacf_vals[1:]) > thr)[0] + 1
    suggested_p = int(pacf_peaks[0]) if len(pacf_peaks) > 0 else 0
    suggested_q = int(acf_peaks[0]) if len(acf_peaks) > 0 else 0
    # d for ARIMA when we fit on original vs processed:
    # we will fit ARIMA on the processed_series directly (stationary), so arima_d = 0
    arima_d = 0
    st.write(f"Suggested (p, d, q) on processed series (d = {arima_d}): (p={suggested_p}, d={arima_d}, q={suggested_q})")

    # Allow user to override if they want (but defaults set to suggested)
    st.write("You can override suggested p and q below if you prefer.")
    user_p = st.number_input("p (AR order)", min_value=0, max_value=10, value=suggested_p, step=1)
    user_q = st.number_input("q (MA order)", min_value=0, max_value=10, value=suggested_q, step=1)
    user_d = arima_d  # keep as 0 because processed_series is stationary (either detrended or differenced)

    # 4) Fit ARIMA on processed_series (since it's stationary)
    st.header("4) Fit ARIMA on processed (stationary) series")
    try:
        # Fit ARIMA with order (p, d, q) where d=0 because processed_series is already stationary
        model = ARIMA(processed_series, order=(int(user_p), int(user_d), int(user_q)),
                      enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit()
        st.success(f"ARIMA({int(user_p)},{int(user_d)},{int(user_q)}) fitted to processed series.")
        st.text(fitted.summary().as_text())
    except Exception as ex:
        st.error(f"ARIMA fitting failed: {ex}")
        st.stop()

    # 5) Residuals: line chart + histogram
    st.header("5) Residuals: line plot & histogram")
    resid = fitted.resid.dropna()
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(resid.index, resid.values, linewidth=1)
        ax.axhline(0, color='k', linestyle='--')
        ax.set_title("Residuals over time")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(resid.values, bins=30, alpha=0.8)
        ax.set_title("Residuals histogram")
        st.pyplot(fig)

    # 6) Jarque-Bera and Ljung-Box tests (robust handling)
    st.header("6) Residual diagnostics: Jarque–Bera & Ljung–Box")
    jb_stat, jb_p, jb_err = safe_jarque_bera(resid.values)
    if jb_err:
        st.warning(f"Jarque–Bera failed: {jb_err}")
    else:
        st.write(f"Jarque–Bera: statistic = {jb_stat:.4f}, p-value = {jb_p:.4f}")
        if jb_p > 0.05:
            st.success("Residuals: appear normally distributed (fail to reject normality).")
        else:
            st.warning("Residuals: deviate from normality (reject normality).")

    try:
        lb_df = acorr_ljungbox(resid, lags=[10], return_df=True)
        lb_stat = float(lb_df['lb_stat'].iloc[0])
        lb_p = float(lb_df['lb_pvalue'].iloc[0])
        st.write(f"Ljung–Box (lag=10): statistic = {lb_stat:.4f}, p-value = {lb_p:.4f}")
        if lb_p > 0.05:
            st.success("No significant autocorrelation in residuals (Ljung–Box).")
        else:
            st.warning("Residuals show autocorrelation (Ljung–Box).")
    except Exception as ex:
        st.warning(f"Ljung–Box test failed: {ex}")

    # 7) Forecast 10 steps ahead of the processed series (with 95% CI)
    st.header("7) Forecast (on processed series) — 95% CI")
    try:
        fc = fitted.get_forecast(steps=int(forecast_steps))
        fc_mean = fc.predicted_mean
        fc_ci = fc.conf_int()

        # Build future index: if datetime index, advance by business days
        last_index = processed_series.index
        if isinstance(last_index, pd.DatetimeIndex):
            freq = last_index.inferred_freq
            if freq is None:
                # fallback to business day
                future_index = pd.bdate_range(start=last_index[-1] + pd.Timedelta(days=1), periods=int(forecast_steps))
            else:
                future_index = pd.date_range(start=last_index[-1] + pd.Timedelta(1, unit='D'), periods=int(forecast_steps), freq=freq)
        else:
            future_index = np.arange(len(last_index), len(last_index) + int(forecast_steps))

        fc_series = pd.Series(fc_mean.values, index=future_index)
        lower = pd.Series(fc_ci.iloc[:, 0].values, index=future_index)
        upper = pd.Series(fc_ci.iloc[:, 1].values, index=future_index)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(processed_series.index, processed_series.values, label="Processed series (used for modeling)")
        ax.plot(fc_series.index, fc_series.values, linestyle="--", marker="o", label="Forecast")
        ax.fill_between(fc_series.index, lower.values, upper.values, color='lightgray', alpha=0.6, label="95% CI")
        ax.legend()
        ax.set_title("Forecast on processed series (stationary)")
        st.pyplot(fig)
    except Exception as ex:
        st.warning(f"Forecast failed: {ex}")

    st.markdown("---")
    st.info("Notes: \n• If series was detrended, the forecast is for the detrended residual series (not the original level). If you want forecasted levels, you must re-add the trend (inverse transform). \n• Processed series is stationary (detrended or differenced), so ARIMA was fit with d=0 on that series. You can alternatively fit ARIMA on original series using order (p,d,q) if you want level forecasts directly.")
