# app.py
import streamlit as st
import traceback

st.set_page_config(page_title="Stock Forecast", layout="wide")

def main():
    st.title("📈 Stock Price Forecast")
    
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        
        st.success("✅ All imports successful!")
        
        # Simple test
        ticker = st.text_input("Enter Ticker:", "AAPL")
        
        if st.button("Test Download"):
            data = yf.download(ticker, period="1mo")
            st.write(f"Data shape: {data.shape}")
            st.dataframe(data.head())
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
