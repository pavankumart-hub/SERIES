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

st.set_page_config(page_title="📈 PAVAN-HYBRID ARIMA Stock Forecast (stable)", layout="wide")
st.title("Polynomial Regression + ARIMA Stock Forecast")
st.markdown("Live your Life as an Exclamation rather than an Explanation-SIR ISSAC NEWTON")
st.markdown("True perspective of God's creation lies in the Art of understanding Mathematics-PAVAN KUMAR THOTA")
st.markdown("Earning in the face of Risk-STOCK MARKET")
st.markdown("Tests: ADF, KPSS, PP, Jarque-Bera, L-jung Box")

# Sidebar inputs
st.sidebar.header("INPUT-ARIMA ORIGINAL")
ticker = st.sidebar.text_input("Stock Ticker", "TATASTEEL.NS").upper()

# Calendar date selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", 
                              value=datetime(2008, 1, 1).date(),
                              min_value=datetime(2000, 1, 1).date(),
                              max_value=datetime.now().date())
with col2:
    end_date = st.date_input("End Date", 
                            value=datetime.now().date(),
                            min_value=datetime(2000, 1, 1).date(),
                            max_value=datetime.now().date())

# Price type selection
price_type = st.sidebar.selectbox("Select Dependent Price Type (Y)", ["High", "Low", "Open", "Close", "Adj Close"])
price_type1 = st.sidebar.selectbox("Select Independent Price Type (X)", ["High", "Low", "Open", "Close", "Adj Close"])
st.sidebar.header("Forecast Input")
today_open_input = st.sidebar.number_input(f"Enter Today's Open Price", 
                                       value=100.0, 
                                       min_value=0.0, 
                                       step=0.1,
                                       key="today_open_input")
degree = st.sidebar.slider("Polynomial Degree", 1, 20, 3)

# ARIMA parameters
st.sidebar.header("ARIMA Parameters")
p_range = st.sidebar.slider("P (AR) Range", 0, 5, (0, 2))
q_range = st.sidebar.slider("Q (MA) Range", 0, 5, (0, 2))
d_range = st.sidebar.slider("D (Differencing) Range", 0, 2, (0, 1))
run_analysis_btn = st.sidebar.button("Run Complete Analysis", type="primary")

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

def adf_test(data):
    try:
        adf_result = adfuller(data)
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
        return adf_stat, adf_p, None
    except Exception as ex:
        return None, None, f"ADF test failed: {str(ex)}"

        
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

        with st.spinner(f"Downloading {ticker}..."):
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data is None or data.empty:
                st.error("No data found. Check ticker symbol or date range.")
                st.stop()

        # Check if both price columns exist
        if price_type not in data.columns:
                st.error(f"Price type '{price_type}' not found in data. Available columns: {list(data.columns)}")
                st.stop()
        
        if price_type1 not in data.columns:
                st.error(f"Price type '{price_type1}' not found in data. Available columns: {list(data.columns)}")
                st.stop()

        # Extract and clean both price data series
        price_data = data[price_type].copy()
        price_data = price_data.dropna()
        
        price_data1 = data[price_type1].copy()
        price_data1 = price_data1.dropna()
        
        # Ensure both series have the same length after dropping NaN values
        # Align both series to have the same dates
        common_dates = price_data.index.intersection(price_data1.index)
        price_data = price_data.loc[common_dates]
        price_data1 = price_data1.loc[common_dates]
        
        n = len(price_data)
        if n < 10:
                st.error(f"Not enough common data points after cleaning ({n}). Increase date range or pick another ticker.")
                st.stop()
                
        st.success(f"Successfully loaded {n} data points for both price series")
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
        # Prepare X: center + scale ordinal dates


        # Date preprocessing and normalization
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
        # Polynomial regression model
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Forecast next day value
        last_normalized_date = X[-1][0]  # Get last normalized date
        next_normalized_date = last_normalized_date + (1 / dates_range)  # Add one day in normalized scale

        # Create next day's features and apply polynomial transformation
        next_day_features = np.array([[next_normalized_date]])
        next_day_poly = poly.transform(next_day_features)

        # Predict next day value
        next_day_prediction = model.predict(next_day_poly)

        # Convert to scalar values to avoid numpy array formatting issues
        current_price = float(y[-1]) if hasattr(y[-1], '__iter__') else y[-1]
        predicted_price = float(next_day_prediction[0])
        price_change = predicted_price - current_price
        percent_change = (price_change / current_price) * 100

        # Streamlit display
        st.subheader("📈 Next Day Forecast")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Current Price",
                value=f"{currency_symbol}{current_price:.2f}"
            )
        with col2:
            st.metric(
                label="Predicted Price",
                value=f"{currency_symbol}{predicted_price:.2f}",
                delta=f"{price_change:.2f}"
            )

        # Additional forecast details
        with st.expander("Forecast Details"):
            st.write(f"**Model:** Polynomial Regression (Degree {degree})")
            st.write(f"**Date Range:** {dates_range:.0f} days")
            st.write(f"**Last available date:** {price_data.index[-1].strftime('%Y-%m-%d')}")
            
            # Calculate next actual date
            next_actual_date = price_data.index[-1] + pd.Timedelta(days=1)
            st.write(f"**Forecast date:** {next_actual_date.strftime('%Y-%m-%d')}")
            
            st.write(f"**Price change:** {currency_symbol}{price_change:.2f}")
            st.write(f"**Percent change:** {percent_change:.2f}%")

        # Optional: Show prediction confidence or model performance
        st.info(f"*Forecast based on polynomial regression model with degree {degree}*")

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
        ax.plot(price_data.index, residuals, label="Residuals",color="red")
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
        plot_acf(residuals, ax=ax1, lags=20, color="green")
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(residuals, ax=ax2, lags=20,color="orange")
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
        n_bins, bins, patches = ax.hist(residuals, bins=30, density=True, alpha=0.7, color='red', edgecolor='black')
        
        # Add normal distribution curve for comparison
        from scipy.stats import norm
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, residual_mean, residual_std)
        ax.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
        
        ax.axvline(residual_mean, color='green', linestyle='--', linewidth=2, label=f'Mean: {residual_mean:.4f}')
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
            ax.plot(price_data.index, residuals, label='Actual Residuals', linewidth=2, alpha=0.7,color="red")
            
            # Plot fitted residuals (ARIMA predictions)
            # Note: fitted_residuals might be shorter due to differencing
            start_idx = len(residuals) - len(fitted_residuals)
            ax.plot(price_data.index[start_idx:], fitted_residuals, 
                   label='ARIMA Fitted Residuals', linewidth=2,color="green")
            
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
            
        #Polynomial Regression with Multiple Features
        st.header("📊 Polynomial Regression with Multiple Features") 
             
        # Prepare features (dates as ordinal)
        dates = np.array([d.toordinal() for d in price_data.index]).reshape(-1, 1).astype(float)
        dates_mean = float(dates.mean(axis=0)[0])
        dates_max = float(dates.max(axis=0)[0])
        dates_min = float(dates.min(axis=0)[0])
        dates_range = dates_max - dates_min
        
        if dates_range == 0:
                st.error("All dates identical (unexpected).")
                st.stop()
        
        # Normalize dates
        X_dates = (dates - dates_mean) / dates_range
        
        # Prepare target variable
        y = price_data.values.astype(float)
        open_prices = price_data1.values.reshape(-1, 1)
        X = np.column_stack([X_dates, open_prices])
        
        # Polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
               # Ensure 1-D arrays for elementwise comparison
        y_pred_1d = np.asarray(y_pred).ravel()
        open_prices_1d = np.asarray(open_prices).ravel()

        # Debug shapes (optional)
        st.write("Shapes — y_pred:", y_pred_1d.shape, "open_prices:", open_prices_1d.shape)

        # Ensure same length
        if y_pred_1d.shape[0] != open_prices_1d.shape[0]:
            st.error(f"Length mismatch: y_pred ({y_pred_1d.shape[0]}) vs open_prices ({open_prices_1d.shape[0]})")
        else:
            # --- Element-wise comparison (counts) ---
            pred_higher = int(np.sum(y_pred_1d > open_prices_1d))
            pred_equal  = int(np.sum(y_pred_1d == open_prices_1d))
            pred_lower  = int(np.sum(y_pred_1d < open_prices_1d))

            # --- Create summary with three categories ---
            comparison_data = {
                'Category': ['Predicted > Open', 'Predicted = Open', 'Predicted < Open'],
                'Count': [pred_higher, pred_equal, pred_lower]
            }

            # --- Display counts ---
            st.subheader("📈 Prediction Comparison")
            st.write(f"**Predicted > Open:** {pred_higher}")
            st.write(f"**Predicted = Open:** {pred_equal}")
            st.write(f"**Predicted < Open:** {pred_lower}")

            # --- Convert to DataFrame for plotting ---
            comparison_df = pd.DataFrame(comparison_data)

            # --- Plot bar chart ---
            fig, ax = plt.subplots()
            ax.bar(comparison_df['Category'], comparison_df['Count'])
            ax.set_title("Predicted vs Open Price Comparison")
            ax.set_ylabel("Count")
            ax.set_xlabel("Category")

            st.pyplot(fig)



        # --- Line Chart: Actual Open Prices vs Predicted Prices ---
        st.subheader("📈 Actual vs Predicted Prices")

        # Flatten both arrays to 1D
        y_pred_1d = np.asarray(y_pred).ravel()
        open_prices_1d = np.asarray(open_prices).ravel()

        # Create a date index for plotting
        dates_list = price_data.index

        # Plot line chart
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(dates_list, open_prices_1d, label='Actual Open Price', linewidth=2)
        ax2.plot(dates_list, y_pred_1d, label='Predicted Price', linestyle='--', linewidth=2)

        ax2.set_title("Actual vs Predicted Prices Over Time")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig2)

# Coefficients section
        st.header("📊 Model Coefficients Analysis")

# Get coefficients - ensure they are 1-dimensional
        coefficients = model.coef_
        intercept = model.intercept_

        # Ensure coefficients is 1D array
        coefficients = np.ravel(coefficients)  # This flattens any multi-dimensional array
        
        # Ensure intercept is a scalar value
        if hasattr(intercept, '__len__'):
            intercept = intercept[0] if len(intercept) > 0 else 0.0
        else:
            intercept = float(intercept)

# Create two columns
        col1, col2 = st.columns(2)

        with col1:
                st.metric("Intercept", f"{intercept:.4f}")

        with col2:
                st.metric("Number of Features", len(coefficients))

# Coefficients table
        st.subheader("Feature Coefficients")

        # Get proper feature names for polynomial features
        try:
            # If using PolynomialFeatures, get the feature names
            feature_names = poly.get_feature_names_out()
            # Ensure feature_names has the same length as coefficients
            if len(feature_names) != len(coefficients):
                feature_names = feature_names[:len(coefficients)]
        except:
            # Fallback to default names
            if hasattr(X_poly, 'columns') and len(X_poly.columns) == len(coefficients):
                feature_names = X_poly.columns.tolist()
            else:
                feature_names = [f'Feature_{i+1}' for i in range(len(coefficients))]

        # Debug information (you can remove this later)
        st.write(f"Number of features: {len(feature_names)}")
        st.write(f"Number of coefficients: {len(coefficients)}")

        # Create DataFrame with proper dimensions
        coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)

        st.dataframe(coef_df.style.format({'Coefficient': '{:.6f}', 'Abs_Coefficient': '{:.6f}'}),
                         use_container_width=True)

# Coefficient visualization
        st.subheader("Coefficient Magnitude Visualization")

        # Only create visualization if we have reasonable number of features
        if len(coef_df) <= 20:  # Limit for better visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Bar plot of coefficients
            colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
            ax1.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
            ax1.set_xlabel('Coefficient Value')
            ax1.set_title('Feature Coefficients')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            # Absolute value bar plot
            ax2.barh(coef_df['Feature'], coef_df['Abs_Coefficient'], color='green', alpha=0.7)
            ax2.set_xlabel('Absolute Coefficient Value')
            ax2.set_title('Feature Importance (Absolute Values)')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"Too many features ({len(coef_df)}) to display visualization clearly. Showing top 10 features only.")
            
            # Show top 10 features
            top_10_df = coef_df.head(10)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            colors = ['red' if x < 0 else 'blue' for x in top_10_df['Coefficient']]
            ax1.barh(top_10_df['Feature'], top_10_df['Coefficient'], color=colors)
            ax1.set_xlabel('Coefficient Value')
            ax1.set_title('Top 10 Feature Coefficients')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)

            ax2.barh(top_10_df['Feature'], top_10_df['Abs_Coefficient'], color='green', alpha=0.7)
            ax2.set_xlabel('Absolute Coefficient Value')
            ax2.set_title('Top 10 Feature Importance')

            plt.tight_layout()
            st.pyplot(fig)

        # Calculate residuals
        residuals = y.flatten() - y_pred.flatten()
        
        # Model performance
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        
        # Display model performance
        st.subheader("📈 Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
                st.metric("R² Score", f"{r2:.4f}")
        with col2:
                st.metric("MSE", f"{mse:.4f}")

        # Forecast next day value
        st.subheader("🔮 Next Day Forecast")

        today_open = today_open_input

        # Get last date and prepare for forecasting
        last_date = X_dates[-1][0]
        next_date = last_date + (1 / dates_range)  # Forecast only one day ahead
       
        # Print next date for debugging
        st.write(f"Last normalized date: {last_date}")
        st.write(f"Next normalized date: {next_date}")
        
        # Calculate actual next date
        next_actual_date = price_data.index[-1] + pd.Timedelta(days=1)
        st.write(f"Next actual date: {next_actual_date.strftime('%Y-%m-%d')}")
        # Create next day's features with user-provided open price
        next_features = np.array([[next_date, today_open]])
        next_poly = poly.transform(next_features)
        next_pred = model.predict(next_poly)
        forecast_value = float(next_pred[0])

        # Calculate actual next date
        next_actual_date = price_data.index[-1] + pd.Timedelta(days=1)

        # Display forecast
        col1, col2, col3 = st.columns(3)
        with col1:
                st.metric(
                        "Today's Open",
                        f"{currency_symbol}{today_open:.2f}"
                )
        with col2:
                st.metric(
                        "Predicted Price",
                        f"{currency_symbol}{forecast_value:.2f}",
                        delta=f"{forecast_value - today_open:.2f}"
                )
        with col3:
                percent_change = ((forecast_value - today_open) / today_open) * 100
                st.metric(
                        "Expected Change",
                        f"{percent_change:+.2f}%"
                )

        # Additional forecast details
        with st.expander("Forecast Details"):
                st.write(f"**Model:** Polynomial Regression (Degree {degree})")
                st.write(f"**Forecast Date:** {next_actual_date.strftime('%Y-%m-%d')}")
                st.write(f"**Input Open Price:** {currency_symbol}{today_open:.2f}")
                st.write(f"**Predicted Price:** {currency_symbol}{forecast_value:.2f}")
                st.write(f"**Expected Gain/Loss:** {currency_symbol}{forecast_value - today_open:+.2f}")
                st.write(f"**Percentage Change:** {percent_change:+.2f}%")
        # Plotting section
        st.subheader("📊 Diagnostic Plots")
        
        # Residuals Line Chart
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(price_data.index, residuals, color='red', linewidth=1, label="Residuals")
        ax1.axhline(0, linestyle='--', color='black', alpha=0.7)
        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"Residuals ({currency_symbol})")
        ax1.set_title("Residuals Over Time")
        ax1.legend()
        ax1.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # Residuals Histogram
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(residuals, bins=30, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel(f"Residuals ({currency_symbol})")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Residuals")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Statistical Tests
        st.subheader("📋 Statistical Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
                # Normality Test (Jarque-Bera)
                st.write("**Normality Test (Jarque-Bera):**")
                jb_stat, jb_p = jarque_bera(residuals)
                st.write(f"Test Statistic: {jb_stat:.4f}")
                st.write(f"P-value: {jb_p:.4f}")
                if jb_p > 0.05:
                        st.success("Residuals appear normal (p > 0.05)")
                else:
                        st.warning("Residuals not normal (p ≤ 0.05)")
        
        with col2:
                # Autocorrelation Test (Ljung-Box)
                st.write("**Autocorrelation Test (Ljung-Box):**")
                lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
                lb_stat = float(lb_result['lb_stat'].iloc[-1])
                lb_p = float(lb_result['lb_pvalue'].iloc[-1])
                st.write(f"Test Statistic: {lb_stat:.4f}")
                st.write(f"P-value: {lb_p:.4f}")
                if lb_p > 0.05:
                        st.success("No significant autocorrelation (p > 0.05)")
                else:
                        st.warning("Significant autocorrelation present (p ≤ 0.05)")
        
        # ACF Plot
        st.write("**Autocorrelation Function (ACF):**")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        plot_acf(residuals, ax=ax3, lags=20, alpha=0.05)
        ax3.set_title("Autocorrelation Function of Residuals")
        ax3.set_ylabel("Correlation")
        ax3.set_xlabel("Lag")
        plt.tight_layout()
        st.pyplot(fig3)

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
 
#------------------------------------
            # --- Prepare fitted values and align with Open prices ---
            fitted_values_1d = np.asarray(fitted_original_values).ravel()
            open_prices_1d = np.asarray(price_data1).ravel()
            price_dates = price_data.index

            # Always remove first two values
            if len(fitted_values_1d) > 2 and len(open_prices_1d) > 2:
                fitted_values_1d = fitted_values_1d[2:]
                open_prices_1d = open_prices_1d[2:]
                price_dates = price_dates[2:]

            # Ensure same length
            min_len = min(len(fitted_values_1d), len(open_prices_1d), len(price_dates))
            fitted_values_1d = fitted_values_1d[:min_len]
            open_prices_1d = open_prices_1d[:min_len]
            price_dates = price_dates[:min_len]

            # --- Element-wise comparison ---
            pred_higher = int(np.sum(fitted_values_1d > open_prices_1d))
            pred_equal  = int(np.sum(fitted_values_1d == open_prices_1d))
            pred_lower  = int(np.sum(fitted_values_1d < open_prices_1d))

            # --- Compute percentages ---
            total = pred_higher + pred_equal + pred_lower
            pct_higher = (pred_higher / total) * 100 if total > 0 else 0
            pct_equal  = (pred_equal / total) * 100 if total > 0 else 0
            pct_lower  = (pred_lower / total) * 100 if total > 0 else 0

            # --- Create summary ---
            comparison_data = {
                'Category': ['Fitted > Open', 'Fitted = Open', 'Fitted < Open'],
                'Count': [pred_higher, pred_equal, pred_lower],
                'Percentage': [pct_higher, pct_equal, pct_lower]
            }

            # --- Display counts and percentages ---
            st.subheader("📊 ARIMA Fitted vs Open Price Comparison")
            st.write(f"**Fitted > Open:** {pred_higher} ({pct_higher:.2f}%)")
            st.write(f"**Fitted = Open:** {pred_equal} ({pct_equal:.2f}%)")
            st.write(f"**Fitted < Open:** {pred_lower} ({pct_lower:.2f}%)")

            # --- Bar Chart with % Labels ---
            comparison_df = pd.DataFrame(comparison_data)
            fig, ax = plt.subplots()
            bars = ax.bar(comparison_df['Category'], comparison_df['Count'], color=['green', 'orange', 'red'])
            ax.set_title("ARIMA Fitted vs Open Price Comparison")
            ax.set_ylabel("Count")
            ax.set_xlabel("Category")

            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f"{comparison_df['Percentage'][i]:.2f}%",
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )

            st.pyplot(fig)

            # --- Line Chart ---
            st.subheader("📈 ARIMA Fitted vs Open Price Over Time")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(price_dates, open_prices_1d, label="Actual Open", linewidth=2)
            ax2.plot(price_dates, fitted_values_1d, label="ARIMA Fitted", linestyle='--', linewidth=2)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)

#--------

            # Plot Fitted vs Actual Original Data
            st.subheader("ARIMA: Fitted vs Actual Stock Prices")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot actual prices
            ax.plot(price_data.index, price_data.values, label='Actual Prices', linewidth=2, alpha=0.7)
            
            # Plot fitted values (ARIMA predictions)
            # Remove initial values based on differencing order (d)
            start_idx = len(price_data) - len(fitted_original_values)
            d = best_original_model_info['d']
            
            # Remove first 'd' data points to avoid NaN values due to differencing
            if len(fitted_original_values) > d:
                ax.plot(price_data.index[start_idx+d:], fitted_original_values[d:], 
                       label='ARIMA Fitted Values', linewidth=2, linestyle='--')
            else:
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
            
            # Get residuals from the best ARIMA model
            arima_residuals = best_original_arima_model.resid
            
            # Remove initial NaN values based on differencing order (d)
            d = best_original_model_info['d']
            if isinstance(arima_residuals, pd.Series):
                arima_residuals_clean = arima_residuals.dropna()
                # Remove first 'd' observations due to differencing
                if len(arima_residuals_clean) > d:
                    arima_residuals_clean = arima_residuals_clean.iloc[d:]
            else:
                # For numpy arrays
                arima_residuals_clean = pd.Series(arima_residuals).dropna()
                if len(arima_residuals_clean) > d:
                    arima_residuals_clean = arima_residuals_clean.iloc[d:]
            
            # Residuals time plot
            st.subheader("ARIMA Residuals Time Series")
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Get the corresponding dates for the residuals (adjust for removed values)
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
                                                             alpha=0.7, color='red', edgecolor='black')
            
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
            
            # HIGH-OPEN PERCENTAGE ANALYSIS - NEW SECTION
            st.header("📊 High-Open Percentage Analysis")
            st.markdown("This analysis shows the daily price movement as percentage: `(High - Open) / Open * 100`")
            
            # Calculate high-open percentage
            if 'High' in data.columns and 'Open' in data.columns:
                high_open_data = data[['High', 'Open']].copy()
                high_open_data = high_open_data.dropna()
                
                # Calculate percentage: (High - Open) / Open * 100
                high_open_data['High_Open_Pct'] = ((high_open_data['High'] - high_open_data['Open']) / high_open_data['Open']) * 100
                
                # Basic statistics
                st.subheader("Basic Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_pct = float(high_open_data['High_Open_Pct'].mean())
                max_pct = float(high_open_data['High_Open_Pct'].max())
                min_pct = float(high_open_data['High_Open_Pct'].min())
                std_pct = float(high_open_data['High_Open_Pct'].std())
                
                with col1:
                    st.metric("Average %", f"{avg_pct:.2f}%")
                with col2:
                    st.metric("Maximum %", f"{max_pct:.2f}%")
                with col3:
                    st.metric("Minimum %", f"{min_pct:.2f}%")
                with col4:
                    st.metric("Std Dev %", f"{std_pct:.2f}%")
                
                # Plot 1: Time series of High-Open percentage
                st.subheader("Daily High-Open Percentage Over Time")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(high_open_data.index, high_open_data['High_Open_Pct'], 
                        linewidth=1, alpha=0.7, color='blue', label='Daily %')
                
                # Add rolling average
                rolling_avg = high_open_data['High_Open_Pct'].rolling(window=20).mean()
                ax1.plot(high_open_data.index, rolling_avg, 
                        linewidth=2, color='red', label='20-Day Moving Avg')
                
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax1.axhline(y=avg_pct, color='green', linestyle='--', alpha=0.7, label=f'Overall Avg: {avg_pct:.2f}%')
                
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Percentage (%)')
                ax1.set_title(f'{ticker} Daily (High-Open)/Open Percentage')
                ax1.legend()
                ax1.grid(alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)
                
                # Plot 2: Histogram of High-Open percentage
                st.subheader("Distribution of High-Open Percentage")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                n, bins, patches = ax2.hist(high_open_data['High_Open_Pct'], bins=50, 
                                          alpha=0.7, color='skyblue', edgecolor='black', 
                                          density=True)
                
                # Add normal distribution curve for comparison
                from scipy.stats import norm
                xmin, xmax = ax2.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, avg_pct, std_pct)
                ax2.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
                
                ax2.axvline(avg_pct, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {avg_pct:.2f}%')
                ax2.axvline(avg_pct + std_pct, color='orange', linestyle=':', alpha=0.7,
                          label=f'±1 Std Dev: {std_pct:.2f}%')
                ax2.axvline(avg_pct - std_pct, color='orange', linestyle=':', alpha=0.7)
                
                ax2.set_xlabel('(High - Open) / Open (%)')
                ax2.set_ylabel('Density')
                ax2.set_title('Distribution of Daily High-Open Percentage')
                ax2.legend()
                ax2.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Plot 3: Box plot by year
                st.subheader("High-Open Percentage by Year")
                high_open_data['Year'] = high_open_data.index.year
                
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                yearly_data = [high_open_data[high_open_data['Year'] == year]['High_Open_Pct'] 
                             for year in sorted(high_open_data['Year'].unique())]
                
                box_plot = ax3.boxplot(yearly_data, labels=sorted(high_open_data['Year'].unique()),
                                     patch_artist=True)
                
                # Color the boxes
                for patch in box_plot['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.axhline(y=avg_pct, color='red', linestyle='--', alpha=0.7, 
                          label=f'Overall Avg: {avg_pct:.2f}%')
                
                ax3.set_xlabel('Year')
                ax3.set_ylabel('(High - Open) / Open (%)')
                ax3.set_title('Yearly Distribution of High-Open Percentage')
                ax3.legend()
                ax3.grid(alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig3)
                
                # Statistical insights
                st.subheader("📈 Statistical Insights")
                
                # Positive vs Negative days
                positive_days = (high_open_data['High_Open_Pct'] > 0).sum()
                negative_days = (high_open_data['High_Open_Pct'] < 0).sum()
                zero_days = (high_open_data['High_Open_Pct'] == 0).sum()
                total_days = len(high_open_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive Days", f"{positive_days} ({positive_days/total_days*100:.1f}%)")
                with col2:
                    st.metric("Negative Days", f"{negative_days} ({negative_days/total_days*100:.1f}%)")
                with col3:
                    st.metric("Zero Days", f"{zero_days} ({zero_days/total_days*100:.1f}%)")
                
                # Extreme moves analysis
                extreme_positive = (high_open_data['High_Open_Pct'] > 5).sum()
                extreme_negative = (high_open_data['High_Open_Pct'] < -5).sum()
                
                st.write("**Extreme Moves (> ±5%):**")
                st.write(f"- Days with > +5%: {extreme_positive} ({extreme_positive/total_days*100:.1f}%)")
                st.write(f"- Days with < -5%: {extreme_negative} ({extreme_negative/total_days*100:.1f}%)")
                
                # Recent performance
                st.subheader("Recent Performance (Last 30 Days)")
                recent_data = high_open_data.tail(30)
                recent_avg = recent_data['High_Open_Pct'].mean()
                recent_positive = (recent_data['High_Open_Pct'] > 0).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recent Average %", f"{recent_avg:.2f}%", 
                             f"{(recent_avg - avg_pct):+.2f}% vs overall")
                with col2:
                    st.metric("Recent Positive Days", f"{recent_positive}/30 ({recent_positive/30*100:.1f}%)")
                
                # Trading insights
                st.subheader("💡 Trading Insights")
                st.write(f"**Average daily upward potential:** {avg_pct:.2f}%")
                st.write(f"**Typical daily range (1 std dev):** ±{std_pct:.2f}%")
                st.write(f"**Consistency:** {positive_days/total_days*100:.1f}% of days see highs above opening price")
                
                if avg_pct > 1.0:
                    st.success("🔍 **Observation:** This stock shows strong upward intraday momentum on average")
                elif avg_pct < 0.5:
                    st.info("🔍 **Observation:** This stock shows modest upward intraday momentum")
                else:
                    st.warning("🔍 **Observation:** This stock shows moderate upward intraday momentum")
                    
            else:
                st.warning("High and Open price data not available for analysis")
            
            # LOW-OPEN PERCENTAGE ANALYSIS - NEW SECTION
            st.header("📊 Low-Open Percentage Analysis")
            st.markdown("This analysis shows the daily price movement as percentage: `(Open - Low) / Open * 100`")
            
            # Calculate low-open percentage
            if 'Low' in data.columns and 'Open' in data.columns:
                low_open_data = data[['Low', 'Open']].copy()
                low_open_data = low_open_data.dropna()
                
                # Calculate percentage: (Open - Low) / Open * 100
                low_open_data['Low_Open_Pct'] = ((low_open_data['Open'] - low_open_data['Low']) / low_open_data['Open']) * 100
                
                # Basic statistics
                st.subheader("Basic Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_pct = float(low_open_data['Low_Open_Pct'].mean())
                max_pct = float(low_open_data['Low_Open_Pct'].max())
                min_pct = float(low_open_data['Low_Open_Pct'].min())
                std_pct = float(low_open_data['Low_Open_Pct'].std())
                
                with col1:
                    st.metric("Average %", f"{avg_pct:.2f}%")
                with col2:
                    st.metric("Maximum %", f"{max_pct:.2f}%")
                with col3:
                    st.metric("Minimum %", f"{min_pct:.2f}%")
                with col4:
                    st.metric("Std Dev %", f"{std_pct:.2f}%")
                
                # Plot 1: Time series of Low-Open percentage
                st.subheader("Daily Low-Open Percentage Over Time")
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(low_open_data.index, low_open_data['Low_Open_Pct'], 
                        linewidth=1, alpha=0.7, color='blue', label='Daily %')
                
                # Add rolling average
                rolling_avg = low_open_data['Low_Open_Pct'].rolling(window=20).mean()
                ax1.plot(low_open_data.index, rolling_avg, 
                        linewidth=2, color='red', label='20-Day Moving Avg')
                
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax1.axhline(y=avg_pct, color='green', linestyle='--', alpha=0.7, label=f'Overall Avg: {avg_pct:.2f}%')
                
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Percentage (%)')
                ax1.set_title(f'{ticker} Daily (Open-Low)/Open Percentage')
                ax1.legend()
                ax1.grid(alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)
                
                # Plot 2: Histogram of Low-Open percentage
                st.subheader("Distribution of Low-Open Percentage")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                n, bins, patches = ax2.hist(low_open_data['Low_Open_Pct'], bins=50, 
                                          alpha=0.7, color='skyblue', edgecolor='black', 
                                          density=True)
                
                # Add normal distribution curve for comparison
                from scipy.stats import norm
                xmin, xmax = ax2.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, avg_pct, std_pct)
                ax2.plot(x, p, 'k', linewidth=2, label='Normal Distribution')
                
                ax2.axvline(avg_pct, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {avg_pct:.2f}%')
                ax2.axvline(avg_pct + std_pct, color='orange', linestyle=':', alpha=0.7,
                          label=f'±1 Std Dev: {std_pct:.2f}%')
                ax2.axvline(avg_pct - std_pct, color='orange', linestyle=':', alpha=0.7)
                
                ax2.set_xlabel('(Open - Low) / Open (%)')
                ax2.set_ylabel('Density')
                ax2.set_title('Distribution of Daily Low-Open Percentage')
                ax2.legend()
                ax2.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Plot 3: Box plot by year
                st.subheader("Low-Open Percentage by Year")
                low_open_data['Year'] = low_open_data.index.year
                
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                yearly_data = [low_open_data[low_open_data['Year'] == year]['Low_Open_Pct'] 
                             for year in sorted(low_open_data['Year'].unique())]
                
                box_plot = ax3.boxplot(yearly_data, labels=sorted(low_open_data['Year'].unique()),
                                     patch_artist=True)
                
                # Color the boxes
                for patch in box_plot['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax3.axhline(y=avg_pct, color='red', linestyle='--', alpha=0.7, 
                          label=f'Overall Avg: {avg_pct:.2f}%')
                
                ax3.set_xlabel('Year')
                ax3.set_ylabel('(Open - Low) / Open (%)')
                ax3.set_title('Yearly Distribution of Low-Open Percentage')
                ax3.legend()
                ax3.grid(alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig3)
                
                # Statistical insights
                st.subheader("📈 Statistical Insights")
                
                # Positive vs Negative days (always positive since Open >= Low)
                positive_days = (low_open_data['Low_Open_Pct'] > 0).sum()
                zero_days = (low_open_data['Low_Open_Pct'] == 0).sum()
                total_days = len(low_open_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Positive Days", f"{positive_days} ({positive_days/total_days*100:.1f}%)")
                with col2:
                    st.metric("Zero Days", f"{zero_days} ({zero_days/total_days*100:.1f}%)")
                
                # Extreme moves analysis
                extreme_moves = (low_open_data['Low_Open_Pct'] > 5).sum()
                
                st.write("**Extreme Moves (> 5%):**")
                st.write(f"- Days with > 5% downward move: {extreme_moves} ({extreme_moves/total_days*100:.1f}%)")
                
                # Recent performance
                st.subheader("Recent Performance (Last 30 Days)")
                recent_data = low_open_data.tail(30)
                recent_avg = recent_data['Low_Open_Pct'].mean()
                recent_positive = (recent_data['Low_Open_Pct'] > 0).sum()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recent Average %", f"{recent_avg:.2f}%", 
                             f"{(recent_avg - avg_pct):+.2f}% vs overall")
                with col2:
                    st.metric("Recent Days > 0", f"{recent_positive}/30 ({recent_positive/30*100:.1f}%)")
                
                # Trading insights
                st.subheader("💡 Trading Insights")
                st.write(f"**Average daily downward potential:** {avg_pct:.2f}%")
                st.write(f"**Typical daily downward range (1 std dev):** ±{std_pct:.2f}%")
                st.write(f"**Consistency:** {positive_days/total_days*100:.1f}% of days see lows below opening price")
                
                if avg_pct > 2.0:
                    st.success("🔍 **Observation:** This stock shows significant downward intraday volatility on average")
                elif avg_pct < 0.5:
                    st.info("🔍 **Observation:** This stock shows modest downward intraday volatility")
                else:
                    st.warning("🔍 **Observation:** This stock shows moderate downward intraday volatility")
                    
            else:
                st.warning("Low and Open price data not available for analysis")
#Second Analysis
st.set_page_config(page_title="High-Open ARIMA Forecast", layout="wide")
st.title("📈 High-Open Percentage ARIMA Forecast")
st.markdown("Model and forecast the `(High - Open) / Open * 100` percentage using ARIMA")

# Sidebar inputs
st.sidebar.header("INPUT-ARIMA PERCENTAGE")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime(2008, 1, 1))
with col2:
    end_date = st.date_input("End Date", datetime.now())
# ARIMA parameters
st.sidebar.header("ARIMA Parameters")
p = st.sidebar.slider("AR Order (p)", 0, 5, 1)
d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
q = st.sidebar.slider("MA Order (q)", 0, 5, 1)

forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 5)

run_forecast_btn = st.sidebar.button("Run High-Open ARIMA Forecast", type="primary")

def fit_arima_model(data, p, d, q):
    try:
        model = SARIMAX(data, order=(p, d, q), seasonal_order=(0, 0, 0, 0))
        fitted_model = model.fit(disp=False)
        return fitted_model, None
    except Exception as ex:
        return None, str(ex)

def safe_adfuller(data):
    try:
        adf_result = adfuller(data)
        adf_stat = float(adf_result[0])
        adf_p = float(adf_result[1])
        return adf_stat, adf_p, None
    except Exception as ex:
        return None, None, str(ex)

def safe_jarque_bera(data):
    try:
        jb_stat, jb_p = jarque_bera(data)
        return float(jb_stat), float(jb_p), None
    except Exception as ex:
        return None, None, str(ex)

def safe_ljungbox(data, max_lag=10):
    try:
        result = acorr_ljungbox(data, lags=[max_lag], return_df=True)
        lb_stat = float(result['lb_stat'].iloc[0])
        lb_p = float(result['lb_pvalue'].iloc[0])
        return lb_stat, lb_p, None
    except Exception as ex:
        return None, None, str(ex)

# =============================================================================
# HIGH-OPEN ARIMA FORECAST SECTION
# =============================================================================

if run_forecast_btn:
    try:
        with st.spinner(f"Downloading {ticker} data..."):
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data is None or data.empty or 'High' not in data.columns or 'Open' not in data.columns:
            st.error("High/Open price data not available. Check ticker symbol.")
            st.stop()

        # Calculate High-Open percentage
        high_open_data = data[['High', 'Open']].copy()
        high_open_data = high_open_data.dropna()

        # Calculate percentage: (High - Open) / Open * 100
        high_open_data['High_Open_Pct'] = ((high_open_data['High'] - high_open_data['Open']) / high_open_data['Open']) * 100

        # Remove any infinite values
        high_open_data = high_open_data[np.isfinite(high_open_data['High_Open_Pct'])]

        if len(high_open_data) < 30:
            st.error(f"Not enough data points ({len(high_open_data)}). Need at least 30 days.")
            st.stop()

        st.header(f"High-Open Analysis for {ticker}")

        # Display basic statistics
        st.subheader("📊 Basic Statistics")
        col1, col2, col3, col4 = st.columns(4)

        current_pct = float(high_open_data['High_Open_Pct'].iloc[-1])
        avg_pct = float(high_open_data['High_Open_Pct'].mean())
        max_pct = float(high_open_data['High_Open_Pct'].max())
        min_pct = float(high_open_data['High_Open_Pct'].min())
        std_pct = float(high_open_data['High_Open_Pct'].std())

        with col1:
            st.metric("Current %", f"{current_pct:.2f}%")
        with col2:
            st.metric("Average %", f"{avg_pct:.2f}%")
        with col3:
            st.metric("Maximum %", f"{max_pct:.2f}%")
        with col4:
            st.metric("Std Dev %", f"{std_pct:.2f}%")

        # Plot historical High-Open percentage
        st.subheader("📈 Historical High-Open Percentage")
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(high_open_data.index, high_open_data['High_Open_Pct'],
                linewidth=1, alpha=0.7, color='blue', label='Daily %')

        # Add rolling average
        rolling_avg = high_open_data['High_Open_Pct'].rolling(window=20).mean()
        ax1.plot(high_open_data.index, rolling_avg,
                linewidth=2, color='red', label='20-Day Moving Avg')

        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axhline(y=avg_pct, color='green', linestyle='--', alpha=0.7,
                   label=f'Overall Avg: {avg_pct:.2f}%')

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title(f'{ticker} Historical (High-Open)/Open Percentage')
        ax1.legend()
        ax1.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)

        # Stationarity test
        st.subheader("📊 Stationarity Analysis")
        adf_stat, adf_p, adf_err = safe_adfuller(high_open_data['High_Open_Pct'])

        if adf_err:
            st.error(f"ADF test error: {adf_err}")
        else:
            st.write(f"**ADF Test Statistic:** {adf_stat:.6f}")
            st.write(f"**ADF p-value:** {adf_p:.6f}")

            if adf_p <= 0.05:
                st.success("✓ Data is Stationary (p-value ≤ 0.05)")
            else:
                st.warning("⚠ Data is Non-Stationary (p-value > 0.05)")
                st.info("Consider using differencing (d > 0) in ARIMA model")

        # ACF and PACF plots
        st.subheader("📊 ACF and PACF Plots")
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(high_open_data['High_Open_Pct'], ax=ax1, lags=20)
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(high_open_data['High_Open_Pct'], ax=ax2, lags=20)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        plt.tight_layout()
        st.pyplot(fig2)

        # =============================================================================
        # ARIMA MODELING SECTION
        # =============================================================================
        
        st.header("🎯 ARIMA Forecasting")

        with st.spinner("Fitting ARIMA model..."):
            model, error = fit_arima_model(high_open_data['High_Open_Pct'], p, d, q)

            if error:
                st.error(f"ARIMA model fitting failed: {error}")
                st.stop()

            # Get fitted values and remove initial values based on differencing order
            fitted_values = model.fittedvalues
            
            # Remove first 'd' values due to differencing
            if len(fitted_values) > d:
                fitted_values_clean = fitted_values.iloc[d:]
                fitted_dates_clean = high_open_data.index[len(high_open_data) - len(fitted_values) + d:]
            else:
                fitted_values_clean = fitted_values
                fitted_dates_clean = high_open_data.index[len(high_open_data) - len(fitted_values):]

            # Forecast future values
            forecast = model.get_forecast(steps=forecast_days)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

            # Generate future dates
            last_date = high_open_data.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]

        # Display model summary
        st.subheader("Model Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AR Order (p)", p)
        with col2:
            st.metric("Differencing (d)", d)
        with col3:
            st.metric("MA Order (q)", q)

        st.write(f"**AIC:** {model.aic:.2f}")
        st.write(f"**BIC:** {model.bic:.2f}")

        # Plot fitted vs actual
        st.subheader("🔄 Model Fit: Actual vs Fitted")
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        # Plot last 100 days for clarity
        plot_days = min(100, len(high_open_data))
        ax3.plot(high_open_data.index[-plot_days:],
                high_open_data['High_Open_Pct'].iloc[-plot_days:],
                label='Actual', linewidth=2, color='blue')

        # Plot cleaned fitted values
        ax3.plot(fitted_dates_clean, fitted_values_clean,
                label='ARIMA Fitted', linewidth=2, linestyle='--', color='red')

        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title(f'ARIMA({p},{d},{q}): Actual vs Fitted Values')
        ax3.legend()
        ax3.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

        # Forecast results
        st.subheader("🎯 Forecast Results")

        # Display forecast table
        forecast_data = []
        for i in range(forecast_days):
            forecast_value = float(forecast_mean.iloc[i])
            ci_lower = float(forecast_ci.iloc[i, 0])
            ci_upper = float(forecast_ci.iloc[i, 1])

            forecast_data.append({
                'Day': i + 1,
                'Date': future_dates[i].strftime('%Y-%m-%d'),
                'Forecasted %': f"{forecast_value:.2f}%",
                'Confidence Interval': f"[{ci_lower:.2f}%, {ci_upper:.2f}%]"
            })

        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True)

        # Plot forecast
        st.subheader("📈 Forecast Visualization")
        fig4, ax4 = plt.subplots(figsize=(12, 6))

        # Plot historical data (last 100 days)
        hist_days = min(100, len(high_open_data))
        ax4.plot(high_open_data.index[-hist_days:],
                high_open_data['High_Open_Pct'].iloc[-hist_days:],
                label='Historical', linewidth=2, color='blue')

        # Plot forecast
        forecast_values = [float(x) for x in forecast_mean]
        ci_lower_values = [float(x) for x in forecast_ci.iloc[:, 0]]
        ci_upper_values = [float(x) for x in forecast_ci.iloc[:, 1]]

        ax4.plot(future_dates, forecast_values,
                label='Forecast', linewidth=3, color='red', marker='o', markersize=6)
        ax4.fill_between(future_dates, ci_lower_values, ci_upper_values,
                        color='pink', alpha=0.3, label='95% Confidence Interval')

        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=avg_pct, color='green', linestyle='--', alpha=0.5,
                   label=f'Historical Avg: {avg_pct:.2f}%')

        ax4.set_xlabel('Date')
        ax4.set_ylabel('Percentage (%)')
        ax4.set_title(f'ARIMA({p},{d},{q}): {forecast_days}-Day High-Open Percentage Forecast')
        ax4.legend()
        ax4.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)

        # Trading insights
        st.header("💡 Trading Insights")

        # Analyze forecast trend
        first_forecast = float(forecast_mean.iloc[0])
        last_forecast = float(forecast_mean.iloc[-1])
        forecast_trend = last_forecast - first_forecast

        col1, col2, col3 = st.columns(3)

        with col1:
            if forecast_trend > 0.5:
                st.success("📈 Bullish Forecast Trend")
                st.write("Expected increasing intraday highs")
            elif forecast_trend < -0.5:
                st.error("📉 Bearish Forecast Trend")
                st.write("Expected decreasing intraday highs")
            else:
                st.info("➡️ Neutral Forecast Trend")
                st.write("Stable intraday high patterns expected")

        with col2:
            avg_forecast = float(forecast_mean.mean())
            if avg_forecast > avg_pct + 0.5:
                st.success("Above Average Potential")
            elif avg_forecast < avg_pct - 0.5:
                st.warning("Below Average Potential")
            else:
                st.info("Normal Range Expected")

        with col3:
            uncertainty = np.mean([ci_upper_values[i] - ci_lower_values[i] for i in range(forecast_days)])
            if uncertainty > 3:
                st.error(f"High Uncertainty: {uncertainty:.1f}%")
            elif uncertainty > 1.5:
                st.warning(f"Medium Uncertainty: {uncertainty:.1f}%")
            else:
                st.success(f"Low Uncertainty: {uncertainty:.1f}%")

        # Risk assessment
        st.subheader("⚠️ Risk Assessment")

        # Check for extreme forecasts
        extreme_forecasts = sum(1 for x in forecast_values if abs(x) > 5)
        if extreme_forecasts > 0:
            st.warning(f"**Extreme moves forecast:** {extreme_forecasts} day(s) with > ±5% expected")

        # Volatility assessment
        forecast_volatility = np.std(forecast_values)
        if forecast_volatility > 2:
            st.warning(f"**High forecast volatility:** {forecast_volatility:.2f}% std dev")
        else:
            st.success(f"**Moderate forecast volatility:** {forecast_volatility:.2f}% std dev")

        # =============================================================================
        # MODEL DIAGNOSTICS SECTION
        # =============================================================================
        
        st.header("🔍 Model Diagnostics")

        # Residuals analysis - remove initial values based on differencing
        residuals = model.resid
        if len(residuals) > d:
            residuals_clean = residuals.iloc[d:]
        else:
            residuals_clean = residuals

        st.subheader("Residuals Analysis")
        col1, col2, col3, col4 = st.columns(4)

        residual_mean = float(residuals_clean.mean())
        residual_std = float(residuals_clean.std())
        residual_skew = float(skew(residuals_clean.dropna()))

        with col1:
            st.metric("Residual Mean", f"{residual_mean:.4f}")
        with col2:
            st.metric("Residual Std", f"{residual_std:.4f}")
        with col3:
            st.metric("Residual Skew", f"{residual_skew:.4f}")
        with col4:
            # Check if residuals are white noise
            lb_stat, lb_p, lb_err = safe_ljungbox(residuals_clean.dropna(), max_lag=10)
            if lb_err:
                st.error("LB Test Error")
            elif lb_p > 0.05:
                st.success("White Noise ✓")
            else:
                st.error("Not White Noise ✗")

        # Normality test
        st.subheader("Normality Test")
        jb_stat, jb_p, jb_err = safe_jarque_bera(residuals_clean.dropna())
        
        if jb_err:
            st.error(f"Jarque-Bera test error: {jb_err}")
        else:
            st.write(f"**Jarque-Bera Statistic:** {jb_stat:.4f}")
            st.write(f"**Jarque-Bera p-value:** {jb_p:.4f}")
            if jb_p > 0.05:
                st.success("✓ Residuals are normally distributed")
            else:
                st.warning("✗ Residuals are not normally distributed")

        # Residuals plot
        st.subheader("Residuals Plots")
        fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Time series plot of residuals
        residual_dates = high_open_data.index[len(high_open_data) - len(residuals_clean):]
        ax1.plot(residual_dates, residuals_clean, label='Residuals', color='blue')
        ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Model Residuals Over Time')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Histogram of residuals
        ax2.hist(residuals_clean.dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Add normal distribution curve
        from scipy.stats import norm
        x = np.linspace(residuals_clean.min(), residuals_clean.max(), 100)
        ax2.plot(x, norm.pdf(x, residual_mean, residual_std), 'r-', label='Normal Distribution')
        
        ax2.set_title('Residuals Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig5)

        # Model quality assessment
        st.subheader("Model Quality Assessment")
        quality_checks = []
        
        # Stationarity check
        if adf_p <= 0.05:
            quality_checks.append("✓ Original data is stationary")
        else:
            quality_checks.append("✗ Original data is non-stationary")
            
        # White noise check
        if lb_p > 0.05:
            quality_checks.append("✓ Residuals are white noise")
        else:
            quality_checks.append("✗ Residuals show autocorrelation")
            
        # Normality check
        if jb_p > 0.05:
            quality_checks.append("✓ Residuals are normally distributed")
        else:
            quality_checks.append("✗ Residuals are not normally distributed")
            
        for check in quality_checks:
            st.write(check)

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.info("Try adjusting ARIMA parameters or using a different ticker")

# Add some help text
with st.sidebar:
    st.markdown("---")
    st.markdown("""**"One of the funny things about the stock market is that every time one person buys, another sells, and both think they are astute"-William Feather**""")
