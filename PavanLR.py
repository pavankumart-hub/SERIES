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
st.title("Polynomial Regression + ARIMA Stock Forecast")
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

# ARIMA parameters
st.sidebar.header("ARIMA Parameters")
p_range = st.sidebar.slider("P (AR) Range", 0, 5, (0, 2))
q_range = st.sidebar.slider("Q (MA) Range", 0, 5, (0, 2))
d_range = st.sidebar.slider("D (Differencing) Range", 0, 2, (0, 1))

run_analysis_btn = st.sidebar.button("Run Complete Analysis")

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
        cv_1pct = critical_values['1%']
        cv_5pct = critical_values['5%'] 
        cv_10pct = critical_values['10%']
        
        # Determine p-value based on test statistic and critical values
        if kpss_stat > cv_1pct:
            manual_pvalue = 0.01
        elif kpss_stat > cv_5pct:
            manual_pvalue = 0.05
        elif kpss_stat > cv_10pct:
            manual_pvalue = 0.10
        else:
            manual_pvalue = 0.50
            
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

if run_analysis_btn:
    st.header(f"Complete Analysis for {ticker}")
    
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

        # KPSS Test on original data
        st.subheader("KPSS Test - Stationarity Check (Original Data)")
        kpss_stat, kpss_p, kpss_critical_values, kpss_err = safe_kpss(price_data)
        
        if kpss_err:
            st.error(f"KPSS test error: {kpss_err}")
        else:
            st.write(f"**KPSS Test Statistic:** {kpss_stat:.6f}")
            st.write(f"**5% Critical Value:** {kpss_critical_values['5%']:.4f}")

            if kpss_stat > kpss_critical_values['5%']:
                st.error("✗ Data is Difference-stationary (test statistic > 5% critical value)")
            else:
                st.success("✓ Data appears Trend-stationary (test statistic < 5% critical value)")

        # show basics - FIXED: Extract scalar values for metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_price = float(price_data.iloc[-1])
            st.metric(f"Current {price_type} Price", f"{currency_symbol}{current_price:.2f}")
        with col2:
            st.metric("Data Points", n)
        with col3:
            actual_days = (price_data.index[-1] - price_data.index[0]).days
            st.metric("Analysis Period (days)", actual_days)

        # Prepare X: center + scale ordinal dates
        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        dates_mean = float(dates.mean(axis=0)[0])
        dates_max = float(dates.max(axis=0)[0])
        dates_min = float(dates.min(axis=0)[0])
        dates_range = dates_max - dates_min
        if dates_range == 0:
            st.error("All dates identical (unexpected).")
            st.stop()
        X = (dates - dates_mean) / dates_range

        y = price_data.values.astype(float)

        # Build polynomial features and fit model
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Metrics - FIXED: Ensure we're using scalar values
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        r2 = float(r2_score(y, y_pred))

        st.subheader("Polynomial Model Performance")
        c1, c2 = st.columns(2)
        c1.metric("RMSE", f"{currency_symbol}{rmse:.4f}")
        c2.metric("R²", f"{r2:.4f}")

        # Plot actual vs predicted
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

        # ACF and PACF plots for residuals
        st.subheader("ACF and PACF Plots for Residuals")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(residuals, ax=ax1, lags=20)
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(residuals, ax=ax2, lags=20)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        plt.tight_layout()
        st.pyplot(fig)

        # NEW: Residual Statistical Tests before ARIMA
        st.header("Residual Statistical Tests")
        
        # ADF Test for Residuals
        st.subheader("ADF Test - Stationarity Check (Residuals)")
        adf_stat, adf_p, adf_err = safe_adfuller(residuals)
        
        if adf_err:
            st.error(f"ADF test error: {adf_err}")
        else:
            st.write(f"**ADF Test Statistic:** {adf_stat:.6f}")
            st.write(f"**ADF p-value:** {adf_p:.6f}")
            
            if adf_p <= 0.05:
                st.success("✓ Residuals are Stationary (p-value ≤ 0.05)")
            else:
                st.error("✗ Residuals are Non-Stationary (p-value > 0.05)")
        
        # Residual Histogram with Skewness and Kurtosis
        st.subheader("Residual Distribution Analysis")
        
        # Calculate statistics - FIXED: Ensure scalar values
        residual_skew = float(skew(residuals))
        residual_kurtosis = float(kurtosis(residuals))
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{residual_mean:.6f}")
        with col2:
            st.metric("Std Dev", f"{residual_std:.6f}")
        with col3:
            st.metric("Skewness", f"{residual_skew:.4f}")
        with col4:
            st.metric("Kurtosis", f"{residual_kurtosis:.4f}")
        
        # Plot histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        n_bins, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add normal distribution curve for comparison
        from scipy.stats import norm
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, residual_mean, residual_std)
        ax.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
        
        ax.axvline(residual_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {residual_mean:.4f}')
        ax.set_xlabel(f'Residual Value ({currency_symbol})')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution Histogram')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interpret skewness and kurtosis
        st.write("**Distribution Interpretation:**")
        if abs(residual_skew) < 0.5:
            st.write("✓ Skewness: Approximately symmetric (close to 0)")
        elif residual_skew > 0.5:
            st.write("↗️ Skewness: Right-skewed (positive skew)")
        else:
            st.write("↙️ Skewness: Left-skewed (negative skew)")
            
        if abs(residual_kurtosis) < 1:
            st.write("✓ Kurtosis: Approximately normal (close to 0)")
        elif residual_kurtosis > 1:
            st.write("📈 Kurtosis: Leptokurtic (heavy-tailed)")
        else:
            st.write("📉 Kurtosis: Platykurtic (light-tailed)")

        # Normality Test
        st.subheader("Normality Test (Jarque-Bera)")
        jb_stat, jb_p, jb_err = safe_jarque_bera(residuals)
        
        if jb_err:
            st.error(f"Jarque-Bera test error: {jb_err}")
        else:
            st.write(f"**Jarque-Bera Statistic:** {jb_stat:.4f}")
            st.write(f"**Jarque-Bera p-value:** {jb_p:.4f}")
            
            if jb_p > 0.05:
                st.success("✓ Residuals are Normally Distributed (p-value > 0.05)")
            else:
                st.error("✗ Residuals are NOT Normally Distributed (p-value ≤ 0.05)")

        # Autocorrelation Test (Ljung-Box)
        st.subheader("Autocorrelation Test (Ljung-Box)")
        lb_stat, lb_p, lb_err = safe_ljungbox(residuals, max_lag=10)
        
        if lb_err:
            st.error(f"Ljung-Box test error: {lb_err}")
        else:
            st.write(f"**Ljung-Box Statistic:** {lb_stat:.4f}")
            st.write(f"**Ljung-Box p-value:** {lb_p:.4f}")
            
            if lb_p > 0.05:
                st.success("✓ No Significant Autocorrelation (p-value > 0.05)")
            else:
                st.error("✗ Significant Autocorrelation Present (p-value ≤ 0.05)")

        # ARIMA Analysis on Residuals
        st.header("ARIMA Analysis on Residuals")
        
        st.write(f"**ARIMA Parameters Range:**")
        st.write(f"- P (AR): {p_range[0]} to {p_range[1]}")
        st.write(f"- D (Differencing): {d_range[0]} to {d_range[1]}")
        st.write(f"- Q (MA): {q_range[0]} to {q_range[1]}")
        
        results = []
        with st.spinner("Fitting ARIMA models on residuals..."):
            for p in range(p_range[0], p_range[1] + 1):
                for d in range(d_range[0], d_range[1] + 1):
                    for q in range(q_range[0], q_range[1] + 1):
                        try:
                            model_arima, error = fit_arima_model(residuals, p, d, q)
                            if model_arima is not None:
                                aic = float(model_arima.aic)
                                bic = float(model_arima.bic)
                                # Get fitted values from ARIMA
                                fitted_residuals = model_arima.fittedvalues
                                results.append({
                                    'p': p,
                                    'd': d, 
                                    'q': q,
                                    'AIC': aic,
                                    'BIC': bic,
                                    'model': model_arima,
                                    'fitted_residuals': fitted_residuals
                                })
                        except:
                            continue
        
        if results:
            # Convert to DataFrame and sort by AIC
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('AIC')
            
            st.subheader("ARIMA Model Comparison (Sorted by AIC)")
            # Display without model and fitted_residuals columns
            display_df = results_df[['p', 'd', 'q', 'AIC', 'BIC']].copy()
            st.dataframe(display_df)
            
            # Best model
            best_model_info = results_df.iloc[0]
            best_arima_model = best_model_info['model']
            fitted_residuals = best_model_info['fitted_residuals']
            
            st.subheader("Best ARIMA Model for Residuals")
            st.write(f"**ARIMA({best_model_info['p']},{best_model_info['d']},{best_model_info['q']})**")
            st.write(f"**AIC:** {best_model_info['AIC']:.2f}")
            st.write(f"**BIC:** {best_model_info['BIC']:.2f}")
            
            # Plot Fitted vs Actual Residuals
            st.subheader("ARIMA: Fitted vs Actual Residuals")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot actual residuals
            ax.plot(price_data.index, residuals, label='Actual Residuals', linewidth=2, alpha=0.7)
            
            # Plot fitted residuals (ARIMA predictions)
            # Note: fitted_residuals might be shorter due to differencing
            start_idx = len(residuals) - len(fitted_residuals)
            ax.plot(price_data.index[start_idx:], fitted_residuals, 
                   label='ARIMA Fitted Residuals', linewidth=2, linestyle='--')
            
            ax.axhline(0, linestyle='-', color='k', alpha=0.3)
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Residual Value ({currency_symbol})')
            ax.set_title(f'ARIMA({best_model_info["p"]},{best_model_info["d"]},{best_model_info["q"]}): Fitted vs Actual Residuals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # ARIMA Forecast for next 5 days
            st.subheader("ARIMA Forecast for Next 5 Days (Residuals)")
            
            # Forecast next 5 days
            forecast_steps = 5
            arima_forecast = best_arima_model.get_forecast(steps=forecast_steps)
            residual_forecast = arima_forecast.predicted_mean
            residual_ci = arima_forecast.conf_int()
            
            # FIX: Properly handle confidence intervals and ensure scalar values
            if hasattr(residual_ci, 'iloc'):
                # It's a pandas DataFrame
                residual_ci_lower = residual_ci.iloc[:, 0].values
                residual_ci_upper = residual_ci.iloc[:, 1].values
            else:
                # It's already a numpy array
                residual_ci_lower = residual_ci[:, 0]
                residual_ci_upper = residual_ci[:, 1]
            
            # Generate future dates
            last_date = price_data.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]
            
            # PRINT THE 5 FORECASTED VALUES CLEARLY
            st.subheader("🎯 5 Forecasted Residual Values")
            
            # Method 1: Simple list - FIXED: Extract scalar values
            st.write("**Forecasted Values:**")
            for i in range(forecast_steps):
                forecast_value = float(residual_forecast[i])
                st.write(f"Day {i+1} ({future_dates[i].strftime('%Y-%m-%d')}): {currency_symbol}{forecast_value:.6f}")
            
            # Method 2: Table - FIXED: Extract scalar values
            st.subheader("📋 Forecast Table")
            forecast_data = []
            for i in range(forecast_steps):
                forecast_value = float(residual_forecast[i])
                ci_lower_val = float(residual_ci_lower[i])
                ci_upper_val = float(residual_ci_upper[i])
                forecast_data.append({
                    'Day': i + 1,
                    'Date': future_dates[i].strftime('%Y-%m-%d'),
                    'Forecasted_Residual': f"{forecast_value:.6f}",
                    'CI_Lower': f"{ci_lower_val:.6f}",
                    'CI_Upper': f"{ci_upper_val:.6f}"
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            st.dataframe(forecast_df)
            
            # Method 3: Raw values for verification
            st.subheader("🔢 Raw Forecast Values (Verification)")
            st.write(f"**Forecast array:** {residual_forecast}")
            
            # Plot ARIMA forecast (optional)
            st.subheader("📈 Forecast Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical residuals (last 30 points for clarity)
            plot_points = min(30, len(residuals))
            ax.plot(price_data.index[-plot_points:], residuals[-plot_points:], 
                   label='Historical Residuals', linewidth=2, color='blue')
            
            # Plot forecast - FIXED: Ensure we're plotting scalar values
            forecast_scalar = [float(x) for x in residual_forecast]
            ci_lower_scalar = [float(x) for x in residual_ci_lower]
            ci_upper_scalar = [float(x) for x in residual_ci_upper]
            
            ax.plot(future_dates, forecast_scalar, label='ARIMA Forecast', 
                   linewidth=3, color='red', marker='o', markersize=8)
            ax.fill_between(future_dates, ci_lower_scalar, ci_upper_scalar, 
                          color='pink', alpha=0.3, label='95% Confidence Interval')
            
            ax.axhline(0, linestyle='-', color='k', alpha=0.3)
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Residual Value ({currency_symbol})')
            ax.set_title(f'ARIMA({best_model_info["p"]},{best_model_info["d"]},{best_model_info["q"]}): 5-Day Residual Forecast')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        # ARIMA Analysis on Original Stock Data
        st.header("ARIMA Analysis on Original Stock Data")
        
        st.subheader("ARIMA Model Selection for Original Data")
        st.write(f"**ARIMA Parameters Range:**")
        st.write(f"- P (AR): 0 to 5")
        st.write(f"- D (Differencing): 0 to 2") 
        st.write(f"- Q (MA): 0 to 5")
        
        original_results = []
        with st.spinner("Fitting ARIMA models to original stock data..."):
            for p in range(0, 6):  # 0 to 5
                for d in range(0, 3):  # 0 to 2
                    for q in range(0, 6):  # 0 to 5
                        try:
                            model_arima, error = fit_arima_model(price_data.values, p, d, q)
                            if model_arima is not None:
                                aic = float(model_arima.aic)
                                bic = float(model_arima.bic)
                                # Get fitted values from ARIMA
                                fitted_values = model_arima.fittedvalues
                                original_results.append({
                                    'p': p,
                                    'd': d, 
                                    'q': q,
                                    'AIC': aic,
                                    'BIC': bic,
                                    'model': model_arima,
                                    'fitted_values': fitted_values
                                })
                        except Exception as e:
                            continue
        
        if original_results:
            # Convert to DataFrame and sort by AIC
            original_results_df = pd.DataFrame(original_results)
            original_results_df = original_results_df.sort_values('AIC')
            
            st.subheader("ARIMA Model Comparison for Original Data (Sorted by AIC)")
            # Display top 10 models
            display_original_df = original_results_df[['p', 'd', 'q', 'AIC', 'BIC']].head(10).copy()
            st.dataframe(display_original_df)
            
            # Best model
            best_original_model_info = original_results_df.iloc[0]
            best_original_arima_model = best_original_model_info['model']
            fitted_original_values = best_original_model_info['fitted_values']
            
            st.subheader("Best ARIMA Model for Original Stock Data")
            st.write(f"**ARIMA({best_original_model_info['p']},{best_original_model_info['d']},{best_original_model_info['q']})**")
            st.write(f"**AIC:** {best_original_model_info['AIC']:.2f}")
            st.write(f"**BIC:** {best_original_model_info['BIC']:.2f}")
            
            # Plot Fitted vs Actual Original Data
            st.subheader("ARIMA: Fitted vs Actual Stock Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot actual prices
            ax.plot(price_data.index, price_data.values, label='Actual Prices', linewidth=2, alpha=0.7)
            
            # Plot fitted values (ARIMA predictions)
            # Note: fitted_values might be shorter due to differencing
            start_idx = len(price_data) - len(fitted_original_values)
            ax.plot(price_data.index[start_idx:], fitted_original_values, 
                   label='ARIMA Fitted Values', linewidth=2, linestyle='--')
            
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Price ({currency_symbol})')
            ax.set_title(f'ARIMA({best_original_model_info["p"]},{best_original_model_info["d"]},{best_original_model_info["q"]}): Fitted vs Actual Prices')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # ARIMA Forecast for next 5 days on Original Data
            st.subheader("ARIMA Forecast for Next 5 Days (Stock Prices)")
            
            # Forecast next 5 days
            forecast_steps = 5
            arima_forecast_original = best_original_arima_model.get_forecast(steps=forecast_steps)
            price_forecast = arima_forecast_original.predicted_mean
            price_ci = arima_forecast_original.conf_int()
            
            # Handle confidence intervals
            if hasattr(price_ci, 'iloc'):
                price_ci_lower = price_ci.iloc[:, 0].values
                price_ci_upper = price_ci.iloc[:, 1].values
            else:
                price_ci_lower = price_ci[:, 0]
                price_ci_upper = price_ci[:, 1]
            
            # Generate future dates
            last_date = price_data.index[-1]
            future_dates_original = [last_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]
            
            # Display forecasted values
            st.subheader("🎯 5-Day Stock Price Forecast")
            
            # Forecast table
            st.write("**Forecasted Stock Prices:**")
            forecast_original_data = []
            for i in range(forecast_steps):
                forecast_value = float(price_forecast[i])
                ci_lower_val = float(price_ci_lower[i])
                ci_upper_val = float(price_ci_upper[i])
                forecast_original_data.append({
                    'Day': i + 1,
                    'Date': future_dates_original[i].strftime('%Y-%m-%d'),
                    f'Forecasted_Price ({currency_symbol})': f"{forecast_value:.2f}",
                    'CI_Lower': f"{ci_lower_val:.2f}",
                    'CI_Upper': f"{ci_upper_val:.2f}"
                })
            
            forecast_original_df = pd.DataFrame(forecast_original_data)
            st.dataframe(forecast_original_df)
            
            # Plot ARIMA forecast for original data
            st.subheader("📈 Stock Price Forecast Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical prices (last 60 points for clarity)
            plot_points = min(60, len(price_data))
            ax.plot(price_data.index[-plot_points:], price_data.values[-plot_points:], 
                   label='Historical Prices', linewidth=2, color='blue')
            
            # Plot forecast
            forecast_scalar = [float(x) for x in price_forecast]
            ci_lower_scalar = [float(x) for x in price_ci_lower]
            ci_upper_scalar = [float(x) for x in price_ci_upper]
            
            ax.plot(future_dates_original, forecast_scalar, label='ARIMA Forecast', 
                   linewidth=3, color='red', marker='o', markersize=8)
            ax.fill_between(future_dates_original, ci_lower_scalar, ci_upper_scalar, 
                          color='pink', alpha=0.3, label='95% Confidence Interval')
            
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Price ({currency_symbol})')
            ax.set_title(f'ARIMA({best_original_model_info["p"]},{best_original_model_info["d"]},{best_original_model_info["q"]}): 5-Day Stock Price Forecast')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # ARIMA Residuals Analysis for Original Data
            st.header("ARIMA Residuals Analysis for Original Stock Data")
            
            # Get residuals from the best ARIMA model - FIXED: Handle numpy array properly
            arima_residuals = best_original_arima_model.resid
            
            # Convert to pandas Series if it's a numpy array and remove NaN values
            if isinstance(arima_residuals, np.ndarray):
                arima_residuals_clean = pd.Series(arima_residuals).dropna()
            else:
                arima_residuals_clean = arima_residuals.dropna()
            
            # Residuals time plot
            st.subheader("ARIMA Residuals Time Series")
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Get the corresponding dates for the residuals
            residual_dates = price_data.index[len(price_data) - len(arima_residuals_clean):]
            ax.plot(residual_dates, arima_residuals_clean, label='ARIMA Residuals', linewidth=1)
            ax.axhline(0, linestyle='--', color='k')
            ax.set_xlabel('Date')
            ax.set_ylabel(f'Residual ({currency_symbol})')
            ax.set_title('ARIMA Model Residuals Over Time')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # ACF and PACF of ARIMA residuals
            st.subheader("ACF and PACF of ARIMA Residuals")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            plot_acf(arima_residuals_clean, ax=ax1, lags=20)
            ax1.set_title("Autocorrelation Function (ACF) - ARIMA Residuals")
            plot_pacf(arima_residuals_clean, ax=ax2, lags=20)
            ax2.set_title("Partial Autocorrelation Function (PACF) - ARIMA Residuals")
            plt.tight_layout()
            st.pyplot(fig)
            
            # Statistical Tests on ARIMA Residuals
            st.subheader("Statistical Tests on ARIMA Residuals")
            
            # ADF Test for ARIMA Residuals
            adf_stat_arima, adf_p_arima, adf_err_arima = safe_adfuller(arima_residuals_clean)
            
            if adf_err_arima:
                st.error(f"ADF test error: {adf_err_arima}")
            else:
                st.write(f"**ADF Test Statistic:** {adf_stat_arima:.6f}")
                st.write(f"**ADF p-value:** {adf_p_arima:.6f}")
                
                if adf_p_arima <= 0.05:
                    st.success("✓ ARIMA Residuals are Stationary (p-value ≤ 0.05)")
                else:
                    st.error("✗ ARIMA Residuals are Non-Stationary (p-value > 0.05)")
            
            # ARIMA Residuals Histogram
            st.subheader("ARIMA Residuals Distribution")
            
            # Calculate statistics
            arima_residual_skew = float(skew(arima_residuals_clean))
            arima_residual_kurtosis = float(kurtosis(arima_residuals_clean))
            arima_residual_mean = float(np.mean(arima_residuals_clean))
            arima_residual_std = float(np.std(arima_residuals_clean))
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{arima_residual_mean:.6f}")
            with col2:
                st.metric("Std Dev", f"{arima_residual_std:.6f}")
            with col3:
                st.metric("Skewness", f"{arima_residual_skew:.4f}")
            with col4:
                st.metric("Kurtosis", f"{arima_residual_kurtosis:.4f}")
            
            # Plot histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            n_bins_arima, bins_arima, patches_arima = ax.hist(arima_residuals_clean, bins=30, density=True, 
                                                             alpha=0.7, color='lightgreen', edgecolor='black')
            
            # Add normal distribution curve for comparison
            from scipy.stats import norm
            xmin_arima, xmax_arima = ax.get_xlim()
            x_arima = np.linspace(xmin_arima, xmax_arima, 100)
            p_arima = norm.pdf(x_arima, arima_residual_mean, arima_residual_std)
            ax.plot(x_arima, p_arima, 'k', linewidth=2, label='Normal Distribution')
            
            ax.axvline(arima_residual_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {arima_residual_mean:.4f}')
            ax.set_xlabel(f'Residual Value ({currency_symbol})')
            ax.set_ylabel('Density')
            ax.set_title('ARIMA Residuals Distribution Histogram')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Normality Test for ARIMA Residuals
            st.subheader("Normality Test for ARIMA Residuals (Jarque-Bera)")
            jb_stat_arima, jb_p_arima, jb_err_arima = safe_jarque_bera(arima_residuals_clean)
            
            if jb_err_arima:
                st.error(f"Jarque-Bera test error: {jb_err_arima}")
            else:
                st.write(f"**Jarque-Bera Statistic:** {jb_stat_arima:.4f}")
                st.write(f"**Jarque-Bera p-value:** {jb_p_arima:.4f}")
                
                if jb_p_arima > 0.05:
                    st.success("✓ ARIMA Residuals are Normally Distributed (p-value > 0.05)")
                else:
                    st.error("✗ ARIMA Residuals are NOT Normally Distributed (p-value ≤ 0.05)")

            # Autocorrelation Test for ARIMA Residuals (Ljung-Box)
            st.subheader("Autocorrelation Test for ARIMA Residuals (Ljung-Box)")
            lb_stat_arima, lb_p_arima, lb_err_arima = safe_ljungbox(arima_residuals_clean, max_lag=10)
            
            if lb_err_arima:
                st.error(f"Ljung-Box test error: {lb_err_arima}")
            else:
                st.write(f"**Ljung-Box Statistic:** {lb_stat_arima:.4f}")
                st.write(f"**Ljung-Box p-value:** {lb_p_arima:.4f}")
                
                if lb_p_arima > 0.05:
                    st.success("✓ No Significant Autocorrelation in ARIMA Residuals (p-value > 0.05)")
                else:
                    st.error("✗ Significant Autocorrelation Present in ARIMA Residuals (p-value ≤ 0.05)")
            
            # Model Summary
            st.subheader("ARIMA Model Summary")
            st.write("A well-fitting ARIMA model should have:")
            st.write("✓ Stationary residuals (ADF test p-value ≤ 0.05)")
            st.write("✓ No significant autocorrelation in residuals (Ljung-Box p-value > 0.05)") 
            st.write("✓ Normally distributed residuals (Jarque-Bera p-value > 0.05)")
            
            # Check model quality
            quality_checks = []
            if adf_p_arima <= 0.05:
                quality_checks.append("✓ Residuals are stationary")
            else:
                quality_checks.append("✗ Residuals are not stationary")
                
            if lb_p_arima > 0.05:
                quality_checks.append("✓ No significant autocorrelation")
            else:
                quality_checks.append("✗ Significant autocorrelation present")
                
            if jb_p_arima > 0.05:
                quality_checks.append("✓ Residuals are normally distributed")
            else:
                quality_checks.append("✗ Residuals are not normally distributed")
            
            st.write("**Model Quality Assessment:**")
            for check in quality_checks:
                st.write(check)

            # FINAL ARIMA FORECAST VALUES - NEW SECTION
            st.header("🎯 Final ARIMA Forecast Values")
            
            # Create a clean display of the 5-day forecast
            st.subheader("5-Day Stock Price Forecast Summary")
            
            forecast_summary = []
            for i in range(forecast_steps):
                forecast_value = float(price_forecast[i])
                ci_lower_val = float(price_ci_lower[i])
                ci_upper_val = float(price_ci_upper[i])
                
                # Calculate percentage change from current price
                current_price = float(price_data.iloc[-1])
                price_change = forecast_value - current_price
                percent_change = (price_change / current_price) * 100
                
                forecast_summary.append({
                    'Day': f"Day {i+1}",
                    'Date': future_dates_original[i].strftime('%Y-%m-%d'),
                    'Forecasted Price': f"{currency_symbol}{forecast_value:.2f}",
                    'Change': f"{currency_symbol}{price_change:+.2f}",
                    'Change %': f"{percent_change:+.2f}%",
                    'Confidence Interval': f"[{currency_symbol}{ci_lower_val:.2f}, {currency_symbol}{ci_upper_val:.2f}]"
                })
            
            # Display as a nice table
            final_forecast_df = pd.DataFrame(forecast_summary)
            st.dataframe(final_forecast_df, use_container_width=True)
            
            # Also show as individual metrics for Day 1 forecast
            st.subheader("Tomorrow's Forecast (Day 1)")
            col1, col2, col3 = st.columns(3)
            
            day1_forecast = float(price_forecast[0])
            day1_change = day1_forecast - current_price
            day1_percent = (day1_change / current_price) * 100
            
            with col1:
                st.metric(
                    "Forecasted Price", 
                    f"{currency_symbol}{day1_forecast:.2f}",
                    f"{day1_change:+.2f} ({day1_percent:+.2f}%)"
                )
            
            with col2:
                st.metric("Confidence Lower", f"{currency_symbol}{float(price_ci_lower[0]):.2f}")
            
            with col3:
                st.metric("Confidence Upper", f"{currency_symbol}{float(price_ci_upper[0]):.2f}")
            
            # Overall trend analysis
            st.subheader("📊 Forecast Trend Analysis")
            
            # Calculate overall trend
            first_forecast = float(price_forecast[0])
            last_forecast = float(price_forecast[-1])
            overall_trend = last_forecast - first_forecast
            overall_trend_percent = (overall_trend / first_forecast) * 100
            
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                if overall_trend > 0:
                    st.success(f"📈 Bullish Trend: +{currency_symbol}{overall_trend:.2f} (+{overall_trend_percent:.2f}%) over 5 days")
                elif overall_trend < 0:
                    st.error(f"📉 Bearish Trend: {currency_symbol}{overall_trend:.2f} ({overall_trend_percent:.2f}%) over 5 days")
                else:
                    st.info("➡️ Neutral Trend: No change over 5 days")
            
            with trend_col2:
                avg_daily_change = overall_trend / (forecast_steps - 1) if forecast_steps > 1 else 0
                st.metric("Average Daily Change", f"{currency_symbol}{avg_daily_change:+.2f}")
                
            # Risk Assessment
            st.subheader("⚠️ Risk Assessment")
            
            # Calculate confidence interval width as a measure of uncertainty
            avg_ci_width = np.mean([float(price_ci_upper[i]) - float(price_ci_lower[i]) for i in range(forecast_steps)])
            uncertainty_percent = (avg_ci_width / current_price) * 100
            
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                if uncertainty_percent < 2:
                    st.success(f"Low Uncertainty: {uncertainty_percent:.1f}%")
                elif uncertainty_percent < 5:
                    st.warning(f"Medium Uncertainty: {uncertainty_percent:.1f}%")
                else:
                    st.error(f"High Uncertainty: {uncertainty_percent:.1f}%")
            
            with risk_col2:
                # Check if current price is within confidence intervals
                within_ci = any(float(price_ci_lower[i]) <= current_price <= float(price_ci_upper[i]) for i in range(min(3, forecast_steps)))
                if within_ci:
                    st.success("Current price within forecast range")
                else:
                    st.warning("Current price outside forecast range")
            
            with risk_col3:
                # Volatility indicator based on historical data
                historical_volatility = price_data.pct_change().std() * np.sqrt(252) * 100  # Annualized
                st.metric("Historical Volatility", f"{historical_volatility:.1f}%")
                
        else:
            st.error("No valid ARIMA models could be fitted to the original data. Try different parameter ranges.")

    except Exception as main_ex:
        st.error(f"Main pipeline error: {main_ex}")
        st.info("Try a smaller degree, shorter date range, or different ticker.")
