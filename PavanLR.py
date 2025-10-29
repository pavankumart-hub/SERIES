# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import shapiro
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 ARIMA + Polynomial Forecast", layout="wide")
st.title("ARIMA + Polynomial Forecast App (stable polynomial section integrated)")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL or RELIANCE.NS)", "AAPL").upper()
price_type = st.sidebar.selectbox("Price Type", ["Open", "High", "Low", "Close"])
start_date = st.sidebar.date_input("Start date", datetime.now().date() - timedelta(days=365))
degree = st.sidebar.slider("Polynomial Degree (for polynomial fit / detrend)", 1, 10, 3)
forecast_steps = st.sidebar.slider("Forecast steps (days)", 1, 30, 7)
run_btn = st.sidebar.button("Run Analysis")

# safe Ljung-box wrapper
def safe_ljungbox(resids, max_lag=10):
    n = len(resids)
    lag = min(max_lag, max(1, n - 1))
    try:
        result = acorr_ljungbox(resids, lags=[lag], return_df=True)
        pval = float(result["lb_pvalue"].iloc[0])
        stat = float(result["lb_stat"].iloc[0]) if "lb_stat" in result.columns else float(result["lb_value"].iloc[0])
        return stat, pval, None
    except Exception as ex:
        return None, None, str(ex)

if run_btn:
    st.header(f"Analysis for {ticker} — {price_type}")
    # 1) Download data
    try:
        with st.spinner(f"Downloading {ticker} from Yahoo Finance..."):
            df = yf.download(ticker, start=start_date, end=datetime.now(), progress=False, threads=False)
    except Exception as e:
        st.error(f"Download error: {e}")
        st.stop()

    if df is None or df.empty or price_type not in df.columns:
        st.error("No data found or selected price column missing. Try another ticker or adjust date range.")
        st.stop()

    # Ensure DatetimeIndex
    df.index = pd.to_datetime(df.index)
    series = df[price_type].dropna()
    n = len(series)
    if n < 10:
        st.error(f"Not enough data points ({n}). Use longer history or different ticker.")
        st.stop()

    # Basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${float(series.iloc[-1]):.2f}")
    with col2:
        st.metric("Data Points", n)
    with col3:
        st.metric("Start Date", f"{series.index[0].date()}")

    st.subheader("Price Series")
    st.line_chart(series, use_container_width=True)

    # Prepare centered/scaled ordinal dates for polynomial fits (stable)
    dates = np.array([d.toordinal() for d in series.index]).reshape(-1, 1).astype(float)
    dates_mean = dates.mean(axis=0)
    dates_range = (dates.max(axis=0) - dates.min(axis=0))[0]
    if dates_range == 0:
        st.error("All dates identical (unexpected).")
        st.stop()
    X = (dates - dates_mean) / dates_range  # scaled dates ~[-0.5,0.5]
    y = series.values.astype(float)

    # Fit polynomial of user-selected degree (this is the user-visible polynomial section)
    poly_user = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_user = poly_user.fit_transform(X)
    model_user = LinearRegression().fit(X_poly_user, y)
    y_pred_user = model_user.predict(X_poly_user)

    rmse_user = np.sqrt(mean_squared_error(y, y_pred_user))
    r2_user = r2_score(y, y_pred_user)

    st.subheader("Polynomial Fit (user degree)")
    c1, c2 = st.columns(2)
    c1.metric("RMSE", f"${rmse_user:.4f}")
    c2.metric("R²", f"{r2_user:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, y, label="Actual", linewidth=2)
    ax.plot(series.index, y_pred_user, label=f"Poly Pred (deg={degree})", linestyle="--", linewidth=2)
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Residuals of user polynomial (display)
    residuals_user = y - y_pred_user
    st.subheader("Residuals of user polynomial")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(series.index, residuals_user, label="Residuals")
    ax.axhline(0, linestyle="--", color="k")
    ax.set_title("Residuals (user polynomial)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Residual histogram
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(residuals_user, bins=30, alpha=0.9)
    ax.set_title("Residual Histogram (user polynomial)")
    st.pyplot(fig)

    # Next day polynomial forecast (user-degree)
    last_date = series.index[-1]
    next_day_num = last_date.toordinal() + 1
    next_X = (np.array([[next_day_num]], dtype=float) - dates_mean) / dates_range
    next_X_poly = poly_user.transform(next_X)
    try:
        next_pred_user = float(model_user.predict(next_X_poly)[0])
    except Exception:
        next_pred_user = None

    if next_pred_user is not None:
        st.subheader("Next Day Prediction (user polynomial)")
        prev = float(y[-1])
        change = next_pred_user - prev
        change_pct = (change / prev) * 100.0
        colA, colB = st.columns(2)
        colA.metric("Next Day (poly)", f"${next_pred_user:.2f}", delta=f"{change:.2f} ({change_pct:.2f}%)")
        colB.metric("Current", f"${prev:.2f}")

    # --- Now do KPSS to decide detrend vs difference ---
    st.subheader("Stationarity check & preprocessing (KPSS + detrend/diff)")

    try:
        k_stat, k_p, _, _ = kpss(series.values, regression="c", nlags="auto")
    except Exception as e:
        st.warning(f"KPSS failed: {e}; proceeding to difference by default.")
        k_p = 0.0

    st.write(f"KPSS p-value = {k_p:.4f}")
    if k_p > 0.05:
        st.success("Series is trend-stationary (KPSS fail to reject) → performing polynomial detrending (AIC selection degree 1..9).")
        # Choose best polynomial degree 1..9 by AIC on OLS
        t = np.arange(len(series)).reshape(-1, 1).astype(float)
        best_aic = np.inf
        best_deg = 1
        best_trend = None
        best_trend_poly = None
        best_trend_model = None
        for deg in range(1, 10):
            poly = PolynomialFeatures(degree=deg, include_bias=True)
            Xp = poly.fit_transform(t)
            try:
                ols = sm.OLS(y, Xp).fit()
                if ols.aic < best_aic:
                    best_aic = ols.aic
                    best_deg = deg
                    best_trend = ols.predict(Xp)
                    best_trend_poly = poly
                    best_trend_model = ols
            except Exception:
                continue
        if best_trend is None:
            st.error("Polynomial detrending failed for all degrees.")
            st.stop()
        st.write(f"Selected trend degree = {best_deg} (AIC = {best_aic:.2f})")
        trend_series = pd.Series(best_trend, index=series.index)
        detrended_series = series - trend_series
        st.line_chart(pd.concat([series.rename("original"), trend_series.rename("trend")], axis=1))
        # ADF on detrended
        adf_stat, adf_p, _, _, _, _ = adfuller(detrended_series.dropna())
        st.write(f"ADF on detrended: stat={adf_stat:.4f}, p={adf_p:.4f}")
        if adf_p < 0.05:
            st.success("Detrended series stationary → use detrended for ARIMA (d=0).")
            processed = detrended_series.dropna()
            post_d = 0
            trend_for_forecast = (best_trend_poly, t, best_trend_model)  # keep for future trend preds
        else:
            st.warning("Detrended still non-stationary → apply first differencing to detrended.")
            processed = detrended_series.diff().dropna()
            post_d = 1
            trend_for_forecast = (best_trend_poly, t, best_trend_model)
    else:
        st.warning("Series is non-stationary → apply first difference.")
        processed = series.diff().dropna()
        post_d = 1
        trend_for_forecast = None

    st.write(f"Processed series length: {len(processed)} (d for ARIMA on processed series = {post_d})")
    st.line_chart(processed)

    # ARIMA grid search p,q in 0..5 on processed (stationary) series
    st.subheader("ARIMA grid search (p,q in 0..5) — choose model with lowest AIC")
    best_aic = np.inf
    best_cfg = None
    best_fit = None
    for p_ in range(0, 6):
        for q_ in range(0, 6):
            try:
                model = ARIMA(processed, order=(p_, 0, q_))  # d=0 because processed is considered stationary series
                fit = model.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_cfg = (p_, post_d, q_)  # store full order with post_d for clarity
                    best_fit = fit
            except Exception:
                continue

    if best_fit is None:
        st.error("Failed to fit ARIMA models on processed series.")
        st.stop()

    st.success(f"Selected ARIMA order (on processed series) p={best_cfg[0]}, d={best_cfg[1]}, q={best_cfg[2]} with AIC={best_aic:.2f}")
    st.text(best_fit.summary().as_text())

    # Residuals and diagnostics
    resid = best_fit.resid.dropna()
    st.subheader("Residuals and diagnostics")
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(resid.index, resid.values, linewidth=1)
    axs[0].axhline(0, color='k', linestyle='--')
    axs[0].set_title("Residuals (time)")
    axs[1].hist(resid.values, bins=30, edgecolor='k')
    axs[1].set_title("Residual Histogram")
    plt.tight_layout()
    st.pyplot(fig)

    # Shapiro & Ljung
    try:
        sh_stat, sh_p = shapiro(resid)
    except Exception as e:
        sh_stat, sh_p = np.nan, np.nan
        st.warning(f"Shapiro test failed: {e}")
    lb_stat, lb_p, lb_err = safe_ljungbox(resid, max_lag=10)
    st.write(f"Shapiro–Wilk p-value = {sh_p}")
    st.write(f"Ljung–Box p-value = {lb_p}")
    if not np.isnan(sh_p):
        if sh_p > 0.05:
            st.success("Residuals look normal (Shapiro).")
        else:
            st.warning("Residuals NOT normal (Shapiro).")
    if lb_err:
        st.warning(f"Ljung–Box error: {lb_err}")
    else:
        if lb_p > 0.05:
            st.success("No autocorrelation in residuals (Ljung–Box).")
        else:
            st.warning("Autocorrelation present in residuals (Ljung–Box).")

    # Forecast on processed series and convert back to original scale
    st.subheader("Forecast (converted to original scale)")

    fc = best_fit.get_forecast(steps=forecast_steps)
    fc_mean = np.array(fc.predicted_mean).flatten()
    fc_ci = np.array(fc.conf_int())

    if trend_for_forecast is not None:
        # trend_for_forecast = (best_trend_poly, t, best_trend_model)
        poly_for_trend, t_full, ols_model = trend_for_forecast
        # predict trend for future x
        future_x = np.arange(len(series), len(series) + forecast_steps).reshape(-1, 1).astype(float)
        X_future_trend = poly_for_trend.transform(future_x)
        try:
            trend_future_vals = ols_model.predict(X_future_trend)
        except Exception:
            # fallback using user polynomial model if ols didn't predict
            trend_future_vals = model_user.predict(poly_user.transform(((future_x - dates_mean)/dates_range)))
        # processed was detrended or detrended-differenced
        if post_d == 0:
            # processed = detrended; forecast of processed + trend_future = forecast of original scale
            forecast_orig = fc_mean + trend_future_vals
            ci_lower = fc_ci[:, 0] + trend_future_vals
            ci_upper = fc_ci[:, 1] + trend_future_vals
        else:
            # processed = diff(detrended) -> need to cumsum forecast and add last detrended value, then add trend
            last_detrended = detrended.dropna().iloc[-1]
            cumul = np.cumsum(fc_mean) + last_detrended
            forecast_orig = cumul + trend_future_vals
            ci_lower = np.cumsum(fc_ci[:, 0]) + last_detrended + trend_future_vals
            ci_upper = np.cumsum(fc_ci[:, 1]) + last_detrended + trend_future_vals
    else:
        # no trend (we used differenced original), processed is first difference of original
        if post_d == 0:
            # processed is original? unlikely in this branch, but handle
            forecast_orig = fc_mean
            ci_lower = fc_ci[:, 0]
            ci_upper = fc_ci[:, 1]
        else:
            # processed = diff(original). Invert by cumsum + last observed original
            last_orig = series.iloc[-1]
            cumul = np.cumsum(fc_mean) + last_orig
            forecast_orig = cumul
            ci_lower = np.cumsum(fc_ci[:, 0]) + last_orig
            ci_upper = np.cumsum(fc_ci[:, 1]) + last_orig

    # ensure lengths align
    forecast_index = pd.bdate_range(start=series.index[-1] + timedelta(days=1), periods=len(forecast_orig))
    # convert arrays to float
    forecast_orig = np.array(forecast_orig, dtype=float)
    ci_lower = np.array(ci_lower, dtype=float)
    ci_upper = np.array(ci_upper, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series.values, label="Original", color="blue")
    ax.plot(forecast_index, forecast_orig, label="Forecast (original scale)", color="red", marker="o")
    ax.fill_between(forecast_index, ci_lower, ci_upper, color="pink", alpha=0.3)
    ax.legend()
    ax.set_title(f"{ticker} {price_type} Forecast (original scale)")
    st.pyplot(fig)

    # display forecast table
    forecast_df = pd.DataFrame({
        "date": forecast_index,
        f"forecast_{price_type}": forecast_orig,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }).set_index("date")
    st.subheader("Forecast values (original scale)")
    st.dataframe(forecast_df)

    st.success("Done — analysis completed.")
