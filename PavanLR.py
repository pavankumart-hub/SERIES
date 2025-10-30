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
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Polynomial Stock Forecast (stable)", layout="wide")
st.title("Polynomial Regression Stock Forecast — Fixed for higher degrees")
st.markdown("Center & scale date before polynomial transform to avoid numerical instability.")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()

# Calendar date selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2008, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.now())

# Price type selection
price_type = st.sidebar.selectbox("Select Price Type", ["High", "Low", "Open", "Close", "Adj Close"])
degree = st.sidebar.slider("Polynomial Degree", 1, 10, 3)
run_btn = st.sidebar.button("Run Analysis")

# ARIMA parameters
st.sidebar.header("ARIMA Parameters")
p_range = st.sidebar.slider("P (AR) Range", 0, 5, (0, 2))
q_range = st.sidebar.slider("Q (MA) Range", 0, 5, (0, 2))
d_range = st.sidebar.slider("D (Differencing) Range", 0, 2, (0, 1))
run_arima_btn = st.sidebar.button("Run ARIMA Analysis")

# Function to detect currency based on ticker
def detect_currency(ticker):
    # Indian stock indicators
    indian_indicators = ['.NS', '.BO', '.NSE', '.BSE', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'HDFCBANK', 
                         'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'ITC', 'LT', 'BHARTIARTL']
    
    # Check if ticker contains Indian indicators
    if any(indicator in ticker.upper() for indicator in indian_indicators):
        return "₹"
    else:
        return "$"

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

def safe_kpss(data):
    try:
        # KPSS test with different regression types
        kpss_stat, p_value, lags, critical_values = kpss(data, regression='ct', nlags='auto')
        
        # Manual p-value calculation based on critical values
        # KPSS critical values at 1%, 5%, 10% significance levels
        cv_1pct = critical_values['1%']
        cv_5pct = critical_values['5%'] 
        cv_10pct = critical_values['10%']
        
        # Determine p-value based on test statistic and critical values
        if kpss_stat > cv_1pct:
            manual_pvalue = 0.01  # Reject null at 1% level - Strong evidence of non-stationarity
        elif kpss_stat > cv_5pct:
            manual_pvalue = 0.05  # Reject null at 5% level - Evidence of non-stationarity
        elif kpss_stat > cv_10pct:
            manual_pvalue = 0.10  # Reject null at 10% level - Weak evidence of non-stationarity
        else:
            manual_pvalue = 0.50  # Fail to reject null - Evidence of stationarity
            
        return float(kpss_stat), float(manual_pvalue), critical_values, None
    except Exception as ex:
        return None, None, None, str(ex)

def safe_adfuller(data):
    try:
        adf_result = adfuller(data)
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
        return adf_stat, adf_p, None
    except Exception as ex:
        return None, None, str(ex)

def fit_arima_model(data, p, d, q):
    try:
        model = SARIMAX(data, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
        fitted_model = model.fit(disp=False)
        return fitted_model, None
    except Exception as ex:
        return None, str(ex)

# Global variables to store data for ARIMA analysis
if 'price_data' not in st.session_state:
    st.session_state.price_data = None
if 'residuals' not in st.session_state:
    st.session_state.residuals = None

if run_btn:
    st.header(f"Analysis for {ticker}")
    
    # Auto-detect currency
    currency_symbol = detect_currency(ticker)
    st.sidebar.info(f"Detected Currency: {currency_symbol}")

    # download data
    try:
        with st.spinner(f"Downloading {ticker}..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data is None or data.empty or price_type not in data.columns:
            st.error(f"No data or '{price_type}' column not found. Check ticker or date range.")
            st.stop()

        price_data = data[price_type].copy()
        price_data = price_data.dropna()
        n = len(price_data)
        if n < 10:
            st.error(f"Not enough data points ({n}). Increase date range or pick another ticker.")
            st.stop()

        # Store data for ARIMA analysis
        st.session_state.price_data = price_data

        # KPSS Test on original data (before modeling)
        st.subheader("KPSS Test - Stationarity Check (Original Data)")
        kpss_stat, kpss_p, kpss_critical_values, kpss_err = safe_kpss(price_data)
        
        if kpss_err:
            st.error(f"KPSS test error: {kpss_err}")
        else:
            st.write(f"**KPSS Test Statistic:** {kpss_stat:.6f}")
            st.write(f"**KPSS p-value:** {kpss_p:.4f}")
            
            # Display critical values
            st.write("**Critical Values:**")
            st.write(f"5% Critical Value: {kpss_critical_values['5%']:.4f}")

            # Interpretation using only 5% critical value
            if kpss_stat > kpss_critical_values['5%']:
                st.error("✗ Data is Difference-stationary (test statistic > 5% critical value)")
            else:
                st.success("✓ Data appears Trend-stationary (test statistic < 5% critical value)")
                    
        st.info("""
        **KPSS Test Interpretation:**
        - Null Hypothesis: Data is Trend-stationary
        - Reject null if test statistic > critical value (data is Difference-stationary)
        - Fail to reject null if test statistic < critical value (data is Trend-stationary)
        """)

        # show basics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Current {price_type} Price", f"{currency_symbol}{float(price_data.iloc[-1]):.2f}")
        with col2:
            st.metric("Data Points", n)
        with col3:
            actual_days = (price_data.index[-1] - price_data.index[0]).days
            st.metric("Analysis Period (days)", actual_days)

        # Prepare X: center + scale ordinal dates to range ~ [-0.5, 0.5]
        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        # center - extract scalar values from arrays
        dates_mean = float(dates.mean(axis=0)[0])  # Convert to scalar
        dates_max = float(dates.max(axis=0)[0])
        dates_min = float(dates.min(axis=0)[0])
        dates_range = dates_max - dates_min
        if dates_range == 0:
            st.error("All dates identical (unexpected).")
            st.stop()
        X = (dates - dates_mean) / dates_range  # now roughly in [-0.5, 0.5]

        y = price_data.values.astype(float)

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
        c1.metric("RMSE", f"{currency_symbol}{rmse:.4f}")
        c2.metric("R²", f"{r2:.4f}")

        # Plot actual vs predicted (sorted by date to avoid line crossing)
        st.subheader(f"Actual vs Predicted (Degree = {degree})")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(price_data.index, y, label=f"Actual {price_type}", linewidth=2)
        ax.plot(price_data.index, y_pred, label="Predicted", linestyle="--", linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Price ({currency_symbol})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Residuals
        residuals = y - y_pred
        st.session_state.residuals = residuals
        
        st.subheader("Residual Analysis")

        # Residual time plot
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(price_data.index, residuals, label="Residuals")
        ax.axhline(0, linestyle="--", color="k")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Residual ({currency_symbol})")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Residual histogram only (QQ plot removed)
        st.subheader("Residual Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title("Residual Histogram")
        ax.set_xlabel(f"Residual Value ({currency_symbol})")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        # ACF and PACF plots for residuals
        st.subheader("ACF and PACF Plots for Residuals")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(residuals, ax=ax1, lags=20)
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(residuals, ax=ax2, lags=20)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        plt.tight_layout()
        st.pyplot(fig)

        # Calculate skewness and kurtosis
        residual_skew = float(skew(residuals))  # Ensure it's a float
        residual_kurtosis = float(kurtosis(residuals, fisher=False))  # Fisher=False gives actual kurtosis (normal=3)
        
        # Display skewness and kurtosis with normal distribution reference
        st.subheader("Residual Distribution Statistics")
        st.info("**Normal Distribution Reference:** Skewness = 0, Kurtosis = 3")
        
        col_skew, col_kurt = st.columns(2)
        with col_skew:
            st.write(f"**Skewness:** {residual_skew:.6f}")
            st.write(f"**Normal Reference:** 0")
            st.write(f"**Interpretation:**")
            if abs(residual_skew) < 0.5:
                st.success("✓ Near symmetric (good - close to normal)")
            elif abs(residual_skew) < 1.0:
                st.warning("∼ Moderately skewed")
            else:
                st.error("✗ Highly skewed (far from normal)")
                
        with col_kurt:
            st.write(f"**Kurtosis:** {residual_kurtosis:.6f}")
            st.write(f"**Normal Reference:** 3")
            st.write(f"**Interpretation:**")
            if abs(residual_kurtosis - 3) < 0.5:
                st.success("✓ Near normal kurtosis (good)")
            elif abs(residual_kurtosis - 3) < 1.0:
                st.warning("∼ Moderate deviation from normal")
            else:
                st.error("✗ Heavy-tailed or light-tailed (far from normal)")

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

        # ADF Test on residuals (at the end)
        st.subheader("ADF Test - Stationarity Check (Residuals)")
        adf_stat, adf_p, adf_err = safe_adfuller(residuals)
        
        if adf_err:
            st.error(f"ADF test error: {adf_err}")
        else:
            st.write(f"**ADF Test Statistic:** {adf_stat:.6f}")
            st.write(f"**ADF p-value:** {adf_p:.6f}")
            if adf_p < 0.05:
                st.success("✓ Residuals are stationary (p < 0.05) - Good!")
            else:
                st.error("✗ Residuals are non-stationary (p ≥ 0.05) - Model may not be adequate")
        
        st.info("""
        **ADF Test Interpretation:**
        - Null Hypothesis: Data has a unit root (non-stationary)
        - Reject null if p-value < 0.05 (data is stationary)
        - Fail to reject null if p-value ≥ 0.05 (data is non-stationary)
        """)

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
        last_date = price_data.index[-1]
        next_date_ord = np.array([[last_date.toordinal() + 1]], dtype=float)
        next_X = (next_date_ord - dates_mean) / dates_range
        next_X_poly = poly.transform(next_X)
        try:
            next_pred = float(model.predict(next_X_poly)[0])
            change = next_pred - float(y[-1])
            change_pct = (change / float(y[-1])) * 100.0
            pcol1, pcol2 = st.columns(2)
            pcol1.metric("Next Day Prediction", f"{currency_symbol}{next_pred:.2f}", 
                        delta=f"{change:.2f} ({change_pct:.2f}%)")
            pcol2.metric(f"Current {price_type}", f"{currency_symbol}{float(y[-1]):.2f}")
        except Exception as ex:
            st.error(f"Next-day prediction failed: {ex}")

    except Exception as main_ex:
        st.error(f"Main pipeline error: {main_ex}")
        st.info("Try a smaller degree, shorter date range, or different ticker.")

# ARIMA Analysis
if run_arima_btn:
    if st.session_state.price_data is None:
        st.error("Please run the polynomial regression analysis first to load the data.")
    else:
        st.header("ARIMA Model Analysis")
        price_data = st.session_state.price_data
        
        # Auto-detect currency
        currency_symbol = detect_currency(ticker)
        
        st.write(f"**ARIMA Parameters Range:**")
        st.write(f"- P (AR): {p_range[0]} to {p_range[1]}")
        st.write(f"- D (Differencing): {d_range[0]} to {d_range[1]}")
        st.write(f"- Q (MA): {q_range[0]} to {q_range[1]}")
        
        results = []
        with st.spinner("Fitting ARIMA models..."):
            for p in range(p_range[0], p_range[1] + 1):
                for d in range(d_range[0], d_range[1] + 1):
                    for q in range(q_range[0], q_range[1] + 1):
                        try:
                            model, error = fit_arima_model(price_data, p, d, q)
                            if model is not None:
                                aic = model.aic
                                bic = model.bic
                                results.append({
                                    'p': p,
                                    'd': d, 
                                    'q': q,
                                    'AIC': aic,
                                    'BIC': bic
                                })
                        except:
                            continue
        
        if results:
            # Convert to DataFrame and sort by AIC
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('AIC')
            
            st.subheader("ARIMA Model Comparison (Sorted by AIC)")
            st.dataframe(results_df)
            
            # Best model
            best_model = results_df.iloc[0]
            st.subheader("Best ARIMA Model")
            st.write(f"**ARIMA({best_model['p']},{best_model['d']},{best_model['q']})**")
            st.write(f"**AIC:** {best_model['AIC']:.2f}")
            st.write(f"**BIC:** {best_model['BIC']:.2f}")
            
            # Fit and display the best model
            best_model_fit, error = fit_arima_model(price_data, int(best_model['p']), int(best_model['d']), int(best_model['q']))
            if best_model_fit:
                st.subheader("Best Model Summary")
                st.text(str(best_model_fit.summary()))
                
                # Forecast
                st.subheader("ARIMA Forecast")
                forecast_steps = st.slider("Forecast Steps", 1, 100, 30)
                forecast = best_model_fit.get_forecast(steps=forecast_steps)
                forecast_mean = forecast.predicted_mean
                forecast_ci = forecast.conf_int()
                
                # Plot forecast
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(price_data.index, price_data.values, label='Historical Data')
                last_date = price_data.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_steps, freq='D')
                ax.plot(future_dates, forecast_mean, label='Forecast', color='red')
                ax.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
                ax.set_xlabel('Date')
                ax.set_ylabel(f'Price ({currency_symbol})')
                ax.set_title(f'ARIMA({best_model["p"]},{best_model["d"]},{best_model["q"]}) Forecast')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.error("No ARIMA models could be fitted with the selected parameters.")
