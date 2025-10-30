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
from scipy.stats import jarque_bera, skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Polynomial Stock Forecast (stable)", layout="wide")
st.title("Polynomial Regression Stock Forecast — Fixed for higher degrees")
st.markdown("Center & scale date before polynomial transform to avoid numerical instability.")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()
days = st.sidebar.slider("Days to Analyze", 30, 365, 180)
degree = st.sidebar.slider("Polynomial Degree", 1, 10, 3)
run_btn = st.sidebar.button("Run Analysis")

end_date = datetime.now()
start_date = end_date - timedelta(days=days)

def safe_ljungbox(resids, max_lag=10):
    n = len(resids)
    # choose a safe lag: at most n-1 and at most requested max_lag
    lag = min(max_lag, max(1, n - 1))
    try:
        result = acorr_ljungbox(resids, lags=[lag], return_df=True)
        pval = float(result["lb_pvalue"].iloc[0])
        stat = float(result["lb_stat"].iloc[0]) if "lb_stat" in result.columns else float(result["lb_value"].iloc[0])
        return stat, pval, None
    except Exception as ex:
        return None, None, str(ex)

def safe_jarque_bera(resids):
    try:
        jb_stat, jb_p = jarque_bera(resids)
        return float(jb_stat), float(jb_p), None
    except Exception as ex:
        return None, None, str(ex)

if run_btn:
    st.header(f"Analysis for {ticker}")

    # download data
    try:
        with st.spinner(f"Downloading {ticker}..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data is None or data.empty or "High" not in data.columns:
            st.error("No data or 'High' column not found. Check ticker or date range.")
            st.stop()

        high = data["High"].copy()
        high = high.dropna()
        n = len(high)
        if n < 10:
            st.error(f"Not enough data points ({n}). Increase days or pick another ticker.")
            st.stop()

        # show basics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current High Price", f"${float(high.iloc[-1]):.2f}")
        with col2:
            st.metric("Data Points", n)
        with col3:
            st.metric("Analysis Period (days)", days)

        # Prepare X: center + scale ordinal dates to range ~ [-0.5, 0.5]
        dates = np.array([d.toordinal() for d in high.index]).reshape(-1, 1).astype(float)
        # center - extract scalar values from arrays
        dates_mean = float(dates.mean(axis=0)[0])  # Convert to scalar
        dates_max = float(dates.max(axis=0)[0])
        dates_min = float(dates.min(axis=0)[0])
        dates_range = dates_max - dates_min
        if dates_range == 0:
            st.error("All dates identical (unexpected).")
            st.stop()
        X = (dates - dates_mean) / dates_range  # now roughly in [-0.5, 0.5]

        y = high.values.astype(float)

        # Build polynomial features (use include_bias=False so intercept comes from LinearRegression)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Fit model
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        st.subheader("Model Performance")
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"${rmse:.4f}")
        c2.metric("R²", f"{r2:.4f}")

        # Plot actual vs predicted (sorted by date to avoid line crossing)
        st.subheader(f"Actual vs Predicted (Degree = {degree})")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(high.index, y, label="Actual High", linewidth=2)
        ax.plot(high.index, y_pred, label="Predicted", linestyle="--", linewidth=2)
        ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Residuals
        residuals = y - y_pred
        st.subheader("Residual Analysis")

        # Residual time plot
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(high.index, residuals, label="Residuals")
        ax.axhline(0, linestyle="--", color="k")
        ax.set_xlabel("Date"); ax.set_ylabel("Residual ($)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Residual histogram only (QQ plot removed)
        st.subheader("Residual Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title("Residual Histogram")
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # Calculate skewness and kurtosis
        residual_skew = float(skew(residuals))  # Ensure it's a float
        residual_kurtosis = float(kurtosis(residuals, fisher=True))  # Ensure it's a float
        
        # Display skewness and kurtosis
        st.subheader("Residual Distribution Statistics")
        col_skew, col_kurt = st.columns(2)
        with col_skew:
            st.write(f"**Skewness:** {residual_skew:.6f}")
            st.write(f"**Interpretation:**")
            if abs(residual_skew) < 0.5:
                st.write("✓ Near symmetric (good)")
            elif abs(residual_skew) < 1.0:
                st.write("∼ Moderately skewed")
            else:
                st.write("✗ Highly skewed")
                
        with col_kurt:
            st.write(f"**Kurtosis:** {residual_kurtosis:.6f}")
            st.write(f"**Interpretation:**")
            if abs(residual_kurtosis) < 0.5:
                st.write("✓ Near normal kurtosis (good)")
            elif abs(residual_kurtosis) < 1.0:
                st.write("∼ Moderate deviation from normal")
            else:
                st.write("✗ Heavy-tailed or light-tailed")

        # Diagnostics: Ljung-Box and Jarque-Bera (safe wrappers)
        st.subheader("Residual Diagnostics (statistical tests)")

        lb_stat, lb_p, lb_err = safe_ljungbox(residuals, max_lag=10)
        jb_stat, jb_p, jb_err = safe_jarque_bera(residuals)

        st.write("**Statistical Test Results:**")
        
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if lb_err:
                st.error(f"Ljung–Box test error: {lb_err}")
            else:
                st.write(f"**Ljung–Box Test:**")
                st.write(f"Statistic: {lb_stat:.6f}")
                st.write(f"p-value: {lb_p:.6f}")
                if lb_p > 0.05:
                    st.success("✓ No significant autocorrelation (p > 0.05)")
                else:
                    st.error("✗ Significant autocorrelation detected (p ≤ 0.05)")
                    
        with dcol2:
            if jb_err:
                st.error(f"Jarque–Bera test error: {jb_err}")
            else:
                st.write(f"**Jarque–Bera Test:**")
                st.write(f"Statistic: {jb_stat:.6f}")
                st.write(f"p-value: {jb_p:.6f}")
                if jb_p > 0.05:
                    st.success("✓ Residuals appear normal (p > 0.05)")
                else:
                    st.error("✗ Residuals not normal (p ≤ 0.05)")

        # Additional helpful debug info (only if user wants)
        if st.checkbox("Show model debug info (coefficients & poly powers)"):
            st.code(f"Polynomial degree: {degree}")
            st.write("Polynomial feature powers (each column):")
            st.write(poly.powers_)
            st.write("Model coefficients (len = number of poly terms):")
            st.write(model.coef_)
            st.write("Model intercept:")
            st.write(model.intercept_)

        # Next day forecast using same centering & scaling
        st.subheader("Next Day Forecast")
        last_date = high.index[-1]
        next_date_ord = np.array([[last_date.toordinal() + 1]], dtype=float)
        next_X = (next_date_ord - dates_mean) / dates_range
        next_X_poly = poly.transform(next_X)
        try:
            next_pred = float(model.predict(next_X_poly)[0])
            change = next_pred - float(y[-1])
            change_pct = (change / float(y[-1])) * 100.0
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Next Day Prediction", f"${next_pred:.2f}", delta=f"{change:.2f} ({change_pct:.2f}%)")
            pcol2.metric("Current High", f"${float(y[-1]):.2f}")
        except Exception as ex:
            st.error(f"Next-day prediction failed: {ex}")

    except Exception as main_ex:
        st.error(f"Main pipeline error: {main_ex}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        st.info("Try a smaller degree, shorter date range, or different ticker.")
