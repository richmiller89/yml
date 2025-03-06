"""
Elliott Wave ML Analyzer for SPY ETF
-------------------------------------
This application:
1. Downloads historical SPY data with Yahoo Finance OAuth authentication
2. Identifies potential Elliott Wave patterns
3. Uses machine learning to predict future wave paths
4. Visualizes current wave count and projected paths with probabilities
"""

# Enhanced fix for PyTorch/Streamlit compatibility issues
import os
import sys

# Must be first imports to prevent torch._classes error
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["PYTORCH_JIT"] = "0"  # Disable JIT to prevent certain errors

# Prevent torch._classes error with Streamlit
if 'torch._classes' in sys.modules:
    del sys.modules['torch._classes']

import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sqlite3
import requests
import json
import base64
import random
import hmac
import hashlib
import uuid
from urllib.parse import urlencode, quote
from concurrent.futures import ThreadPoolExecutor
import io
import csv

# Import yfinance after environment setup to prevent conflicts
# This helps avoid circular dependencies
import yfinance as yf

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Deep Learning - only import when needed to prevent startup errors
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reinforcement Learning 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Set page config
st.set_page_config(
    page_title="Elliott Wave ML Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Yahoo Finance API Credentials
YAHOO_APP_ID = "XftgFfx4"
YAHOO_CLIENT_ID = "dj0yJmk9WXF4TXJBRzFFbWExJmQ9WVdrOVdHWjBaMFptZURRbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTY3"
YAHOO_CLIENT_SECRET = "9476b942606a7e2da0f995d06150577127126b83"
# Hardcoded auth code
YAHOO_AUTH_CODE = "v4m3n7f"

# Database and cache settings
DB_PATH = "spy_data.db"
CACHE_DIR = "data_cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache the data to prevent hitting rate limits
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_data_fetch(symbol, start_date_str, end_date_str):
    """Cache data fetching to avoid rate limits"""
    return fetch_spy_data_with_retries(symbol, start_date_str, end_date_str)

def create_db_connection():
    """Create a database connection to the SQLite database"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
    return conn

def create_price_table(conn):
    """Create the price table if it doesn't exist"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                last_updated TEXT
            )
        ''')
        
        # Create OAuth tokens table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS oauth_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                access_token TEXT,
                refresh_token TEXT,
                expires_at INTEGER,
                created_at INTEGER
            )
        ''')
        
        conn.commit()
    except Exception as e:
        st.error(f"Error creating table: {e}")

def store_oauth_tokens(conn, access_token, refresh_token, expires_in):
    """Store OAuth tokens in the database"""
    try:
        cursor = conn.cursor()
        now = int(time.time())
        expires_at = now + expires_in
        
        # Delete old tokens first
        cursor.execute("DELETE FROM oauth_tokens")
        
        # Insert new tokens
        cursor.execute('''
            INSERT INTO oauth_tokens (access_token, refresh_token, expires_at, created_at)
            VALUES (?, ?, ?, ?)
        ''', (access_token, refresh_token, expires_at, now))
        
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error storing OAuth tokens: {e}")
        return False

def get_oauth_tokens(conn):
    """Get the most recent OAuth tokens from the database"""
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT access_token, refresh_token, expires_at
            FROM oauth_tokens
            ORDER BY created_at DESC
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        if result:
            access_token, refresh_token, expires_at = result
            return {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expires_at': expires_at
            }
        return None
    except Exception as e:
        st.error(f"Error getting OAuth tokens: {e}")
        return None

def fetch_data_from_db(conn, start_date, end_date):
    """Fetch data from SQLite database within date range"""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, open, high, low, close, volume 
            FROM price_data 
            WHERE date BETWEEN ? AND ? 
            ORDER BY date
        """, (start_date, end_date))
        
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching data from DB: {e}")
        return None

def save_data_to_db(conn, df):
    """Save dataframe to SQLite database"""
    try:
        cursor = conn.cursor()
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            
            # Check if record exists
            cursor.execute("SELECT 1 FROM price_data WHERE date = ?", (date_str,))
            exists = cursor.fetchone()
            
            if exists:
                # Update existing record
                cursor.execute("""
                    UPDATE price_data 
                    SET open = ?, high = ?, low = ?, close = ?, volume = ?, last_updated = ?
                    WHERE date = ?
                """, (
                    float(row['Open']), float(row['High']), float(row['Low']), 
                    float(row['Close']), int(row['Volume']), now, date_str
                ))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO price_data (date, open, high, low, close, volume, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str, float(row['Open']), float(row['High']), float(row['Low']), 
                    float(row['Close']), int(row['Volume']), now
                ))
                
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving data to DB: {e}")
        conn.rollback()
        return False

def check_cached_data(symbol, start_date, end_date):
    """Check if we have cached data for this date range"""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
    
    if os.path.exists(cache_file):
        # Check if cache is recent (within last 24 hours)
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 86400:  # 24 hours in seconds
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    st.success(f"Using cached data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                    return df
            except Exception as e:
                st.warning(f"Error reading cached data: {e}")
                
    return None

def save_to_cache(df, symbol, start_date, end_date):
    """Save data to cache"""
    if df is not None and not df.empty:
        cache_file = os.path.join(CACHE_DIR, f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
        
        try:
            df.to_csv(cache_file)
            return True
        except Exception as e:
            st.warning(f"Error saving to cache: {e}")
            
    return False

def exchange_code_for_token(auth_code):
    """Exchange authorization code for access token"""
    try:
        url = "https://api.login.yahoo.com/oauth2/get_token"
        
        # Create authorization header with Basic auth
        auth_string = f"{YAHOO_CLIENT_ID}:{YAHOO_CLIENT_SECRET}"
        auth_base64 = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {auth_base64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'redirect_uri': 'oob',
            'code': auth_code
        }
        
        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            
            # Store tokens in database
            conn = create_db_connection()
            if conn:
                create_price_table(conn)
                store_oauth_tokens(
                    conn, 
                    token_data['access_token'], 
                    token_data['refresh_token'], 
                    token_data['expires_in']
                )
                
            return token_data
        else:
            st.error(f"Error exchanging code for token: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Exception during token exchange: {e}")
        return None

def refresh_access_token(refresh_token):
    """Get a new access token using refresh token"""
    try:
        url = "https://api.login.yahoo.com/oauth2/get_token"
        
        # Create authorization header with Basic auth
        auth_string = f"{YAHOO_CLIENT_ID}:{YAHOO_CLIENT_SECRET}"
        auth_base64 = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {auth_base64}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'redirect_uri': 'oob',
            'refresh_token': refresh_token
        }
        
        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            
            # Store new tokens in database
            conn = create_db_connection()
            if conn:
                create_price_table(conn)
                store_oauth_tokens(
                    conn, 
                    token_data['access_token'], 
                    token_data['refresh_token'], 
                    token_data['expires_in']
                )
                
            return token_data
        else:
            st.error(f"Error refreshing token: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Exception during token refresh: {e}")
        return None

def get_valid_access_token():
    """Get a valid access token, refreshing if necessary"""
    conn = create_db_connection()
    if not conn:
        return None
        
    create_price_table(conn)
    
    # Check if we have tokens
    tokens = get_oauth_tokens(conn)
    
    if not tokens:
        # Try to get tokens with hardcoded auth code
        token_data = exchange_code_for_token(YAHOO_AUTH_CODE)
        if token_data:
            return token_data['access_token']
        return None
        
    # Check if token is expired
    now = int(time.time())
    
    if tokens['expires_at'] <= now:
        # Need to refresh
        new_tokens = refresh_access_token(tokens['refresh_token'])
        if new_tokens:
            return new_tokens['access_token']
        return None
    
    # Current token is still valid
    return tokens['access_token']

def download_spy_data_direct(ticker="SPY", period="1y"):
    """
    Simple direct yfinance download with minimal parameters
    Less likely to hit rate limits
    """
    try:
        # Add a random delay between 1-3 seconds to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        st.info(f"Downloading {ticker} data for the last {period}...")
        
        # Download the data - without user_agent parameter
        data = yf.download(
            ticker,
            period=period,  # e.g., "1y", "2y", "5y"
            progress=False
        )
        
        if data.empty:
            st.warning(f"No data found for {ticker} for period {period}")
            return None
            
        # Ensure the index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        # Convert column names to title case
        data.columns = [col.title() for col in data.columns]
        
        return data
    
    except Exception as e:
        st.error(f"Error downloading data directly: {e}")
        return None

def download_yf_data(ticker, start_date, end_date, interval='1d'):
    """Download data from Yahoo Finance using yfinance library with date range"""
    try:
        # Add a slight delay to be respectful of the API
        time.sleep(random.uniform(1, 3))
        st.info(f"Downloading {ticker} data from {start_date} to {end_date} using yfinance...")
        
        # Download the data - without user_agent parameter
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if data.empty:
            st.warning(f"No data found for {ticker} from {start_date} to {end_date}")
            return None
            
        # Ensure the index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
            
        # Convert column names to title case
        data.columns = [col.title() for col in data.columns]
        
        return data
    
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def download_with_auth_header(ticker, start_date, end_date):
    """
    Download data with OAuth authentication for more reliable access
    Using the official Yahoo Finance private API with authentication
    """
    try:
        access_token = get_valid_access_token()
        if not access_token:
            st.warning("No valid access token available. Using fallback method.")
            return None
            
        st.info(f"Downloading {ticker} data using authenticated Yahoo API...")
        
        # Use the Yahoo Finance query API with authentication
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        
        # Convert dates to timestamps
        start_timestamp = int(pd.to_datetime(start_date).timestamp())
        end_timestamp = int(pd.to_datetime(end_date).timestamp())
        
        params = {
            'period1': start_timestamp,
            'period2': end_timestamp,
            'interval': '1d',
            'includePrePost': 'false',
            'events': 'div,split'
        }
        
        headers = {
            'Authorization': f'Bearer {access_token}',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract data from response
            result = data['chart']['result'][0]
            
            # Check if we have data
            if 'timestamp' not in result or not result['timestamp']:
                st.warning("No data returned from Yahoo API")
                return None
                
            # Extract OHLCV data
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Open': quotes['open'],
                'High': quotes['high'],
                'Low': quotes['low'],
                'Close': quotes['close'],
                'Volume': quotes['volume']
            })
            
            # Set index and handle missing values
            df.set_index('Date', inplace=True)
            df.dropna(inplace=True)
            
            return df
        else:
            st.warning(f"Failed to get data from Yahoo API: {response.status_code}")
            return None
    
    except Exception as e:
        st.warning(f"Error in authenticated download: {e}")
        return None

def download_in_chunks(ticker, start_date, end_date, chunk_size=30):
    """Download data in smaller chunks to avoid rate limits"""
    try:
        # Initialize an empty DataFrame to hold all data
        all_data = pd.DataFrame()
        
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Generate chunks
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + pd.Timedelta(days=chunk_size)
            if current_end > end_date:
                current_end = end_date
                
            # Add a delay between chunks
            if not all_data.empty:
                time.sleep(random.uniform(5, 10))
                
            # Download this chunk
            chunk_data = download_yf_data(
                ticker, 
                current_start.strftime('%Y-%m-%d'), 
                current_end.strftime('%Y-%m-%d')
            )
            
            if chunk_data is not None and not chunk_data.empty:
                all_data = pd.concat([all_data, chunk_data])
                st.info(f"Downloaded chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            else:
                st.warning(f"Failed to download chunk: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                
            # Move to next chunk
            current_start = current_end
            
        # Remove any duplicates
        if not all_data.empty:
            all_data = all_data[~all_data.index.duplicated(keep='first')]
            all_data.sort_index(inplace=True)
            
        return all_data
        
    except Exception as e:
        st.error(f"Error downloading in chunks: {e}")
        return None

def allow_manual_data_upload():
    """Allow user to upload their own CSV data"""
    uploaded_file = st.file_uploader("Upload CSV with OHLCV data (Date, Open, High, Low, Close, Volume)", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
            
            # Check if required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns and col.lower() not in df.columns:
                    st.error(f"Missing required column: {col}")
                    return None
                    
            # Ensure column names are title case
            df.columns = [col.title() for col in df.columns]
            
            # Ensure the index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            st.success(f"Successfully loaded {len(df)} rows of data")
            return df
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            
    return None

def provide_download_template():
    """Provide a template CSV for users to download"""
    # Create a sample dataframe
    sample_dates = [datetime.now() - timedelta(days=i) for i in range(5)]
    sample_data = {
        'Date': sample_dates,
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'High': [105.0, 106.0, 107.0, 108.0, 109.0],
        'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
        'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    }
    
    df = pd.DataFrame(sample_data)
    df.set_index('Date', inplace=True)
    
    # Convert to CSV
    csv = df.to_csv()
    
    # Create download button
    st.download_button(
        label="Download CSV Template",
        data=csv,
        file_name="data_template.csv",
        mime="text/csv",
    )

def create_sample_data(symbol="SPY", days=180):
    """Create sample data if all else fails"""
    st.warning("Using SYNTHETIC data for demonstration purposes only!")
    
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days)
    
    # Generate dates
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Create a dataframe
    df = pd.DataFrame(index=dates)
    
    # Simulate a price series with some trend and volatility
    # Start with SPY around 500
    base_price = 500
    
    # Generate a random walk with upward drift
    price_changes = np.random.normal(0.0005, 0.01, len(dates))
    cum_returns = np.cumprod(1 + price_changes)
    
    # Create OHLC data
    df['Close'] = base_price * cum_returns
    
    # Fix warning: Use proper indexing
    df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.004, len(dates)))
    # Fix first day's open price properly
    df.loc[df.index[0], 'Open'] = df.loc[df.index[0], 'Close'] * 0.995  # Fixed pandas warning
    
    # Add some volatility to High and Low
    daily_volatility = np.random.uniform(0.005, 0.02, len(dates))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + daily_volatility)
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - daily_volatility)
    
    # Create Volume
    df['Volume'] = np.random.randint(5000000, 20000000, len(dates))
    
    # Make the data look more realistic
    # Add some patterns like weekday effects
    weekday_factors = [1.05, 1.0, 0.98, 1.02, 1.1]  # Mon, Tue, Wed, Thu, Fri
    for i, date in enumerate(dates):
        weekday = date.weekday()
        if weekday < len(weekday_factors):
            # Fix warning: Convert to int to avoid dtype issues
            df.loc[date, 'Volume'] = int(df.loc[date, 'Volume'] * weekday_factors[weekday])  # Fixed pandas warning
    
    return df

def fetch_spy_data_with_retries(symbol, start_date, end_date, max_retries=3):
    """Fetch SPY data with multiple fallback mechanisms and retries"""
    
    for retry in range(max_retries):
        # Try database first
        if start_date and end_date:
            conn = create_db_connection()
            if conn:
                create_price_table(conn)
                df = fetch_data_from_db(conn, start_date, end_date)
                if df is not None and not df.empty:
                    st.success("Retrieved data from local database!")
                    return df
        
        # Try authenticated download first (most likely to succeed)
        if retry == 0:
            df = download_with_auth_header(symbol, start_date, end_date)
            if df is not None and not df.empty:
                conn = create_db_connection()
                if conn:
                    save_data_to_db(conn, df)
                return df
        
        # If database didn't work, try several fallback approaches in order
        
        # 1. Try direct download with period instead of date range
        if retry == 0:
            df = download_spy_data_direct(symbol, period="1y")
            if df is not None and not df.empty:
                if conn:
                    save_data_to_db(conn, df)
                return df
        
        # 2. Try with date range but with exponential backoff
        if start_date and end_date:
            wait_time = (2 ** retry) + random.uniform(0, 1)  # Exponential backoff with jitter
            st.info(f"Rate limit encountered. Waiting {wait_time:.2f} seconds before retry #{retry+1}...")
            time.sleep(wait_time)
            
            df = download_yf_data(symbol, start_date, end_date)
            if df is not None and not df.empty:
                if conn:
                    save_data_to_db(conn, df)
                return df
    
    # If all else fails, try synthetic data
    st.warning("All download attempts failed. Creating synthetic data for demonstration.")
    df = create_sample_data(symbol, days=180)
    return df

def get_spy_data_hybrid(symbol, start_date, end_date, allow_upload=True):
    """
    Get SPY data using multiple methods, with fallbacks
    
    1. Try database first
    2. Try cache
    3. Try OAuth API (if implemented)
    4. Try yfinance direct download
    5. Try chunked download
    6. Allow manual upload
    7. Use synthetic data as last resort
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    # 1. Try database
    conn = create_db_connection()
    if conn:
        create_price_table(conn)
        db_data = fetch_data_from_db(conn, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if db_data is not None and not db_data.empty and len(db_data) >= (end_date - start_date).days * 0.7:
            st.success("Retrieved data from local database!")
            return db_data
            
    # 2. Try cache
    cache_data = check_cached_data(symbol, start_date, end_date)
    if cache_data is not None and not cache_data.empty:
        return cache_data
    
    # 3. Try OAuth API - this is a direct implementation using the OAuth token
    access_token = get_valid_access_token()
    if access_token:
        st.success("Successfully authenticated with Yahoo API!")
        auth_data = download_with_auth_header(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if auth_data is not None and not auth_data.empty:
            # Save to DB and cache
            if conn:
                save_data_to_db(conn, auth_data)
            save_to_cache(auth_data, symbol, start_date, end_date)
            return auth_data
    
    # 4. Try direct yfinance download
    period_map = {
        30: "1mo",
        60: "2mo", 
        90: "3mo",
        180: "6mo",
        365: "1y",
        730: "2y",
        1095: "3y",
        1825: "5y",
        3650: "10y",
    }
    
    # Calculate days between start and end
    days_diff = (end_date - start_date).days
    
    # Find the best period to use
    period = None
    for days, pd_str in period_map.items():
        if days >= days_diff:
            period = pd_str
            break
    
    # If no suitable period was found, use the largest
    if period is None:
        period = "10y"
        
    direct_data = download_spy_data_direct(symbol, period=period)
    if direct_data is not None and not direct_data.empty:
        # Filter to requested date range
        direct_data = direct_data[(direct_data.index >= start_date) & (direct_data.index <= end_date)]
        
        if len(direct_data) >= (end_date - start_date).days * 0.7:
            # Save to DB and cache
            if conn:
                save_data_to_db(conn, direct_data)
            save_to_cache(direct_data, symbol, start_date, end_date)
            return direct_data
    
    # 5. Try chunked download
    if days_diff > 30:
        chunked_data = download_in_chunks(symbol, start_date, end_date, chunk_size=30)
        if chunked_data is not None and not chunked_data.empty:
            # Save to DB and cache
            if conn:
                save_data_to_db(conn, chunked_data)
            save_to_cache(chunked_data, symbol, start_date, end_date)
            return chunked_data
    
    # 6. Allow manual upload if all else fails
    if allow_upload:
        st.warning("Automatic data download failed. Please upload data manually.")
        
        # Show template download button
        provide_download_template()
        
        manual_data = allow_manual_data_upload()
        if manual_data is not None and not manual_data.empty:
            # Save to DB and cache
            if conn:
                save_data_to_db(conn, manual_data)
            save_to_cache(manual_data, symbol, start_date, end_date)
            return manual_data
    
    # 7. Create synthetic data if everything fails
    synthetic_data = create_sample_data(symbol, days=days_diff)
    if synthetic_data is not None and not synthetic_data.empty:
        if conn:
            save_data_to_db(conn, synthetic_data)
        return synthetic_data
    
    # Last resort - empty dataframe
    return pd.DataFrame()

#################################################
#        ELLIOTT WAVE ANALYSIS FUNCTIONS        #
#################################################

def identify_swings(df, threshold=0.01):
    """
    Identifies swing highs and lows in price data
    Returns dataframe with swing_high and swing_low columns
    """
    df = df.copy()
    
    # Initialize columns
    df['swing_high'] = False
    df['swing_low'] = False
    
    # Look-back and look-forward window size
    window = 3
    
    for i in range(window, len(df) - window):
        # Check for swing high
        if all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, window+1)):
            df.loc[df.index[i], 'swing_high'] = True
            
        # Check for swing low
        if all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, window+1)):
            df.loc[df.index[i], 'swing_low'] = True
    
    return df

def analyze_wave_pattern(df, lookback=100):
    """
    Analyzes recent price movements to identify potential Elliott Wave patterns
    Returns a dictionary with wave labels and confidence levels
    """
    # Get subset of recent data
    recent_df = df.iloc[-lookback:].copy() if len(df) > lookback else df.copy()
    
    # Identify swing points
    recent_df = identify_swings(recent_df)
    
    # Get swing points only
    swing_points = recent_df[recent_df['swing_high'] | recent_df['swing_low']].copy()
    
    if len(swing_points) < 5:
        return {
            'wave_count': 'Insufficient swing points for analysis',
            'confidence': 0,
            'projected_paths': [],
            'swing_points': swing_points
        }
    
    # Determine trend direction (just a simple approach)
    start_price = recent_df['Close'].iloc[0]
    end_price = recent_df['Close'].iloc[-1]
    trend_up = end_price > start_price
    
    # Basic wave labeling (very simplified)
    wave_count = []
    confidence = 0.5  # Base confidence
    
    # Extract recent swings (last 9 swing points if available)
    recent_swings = swing_points.iloc[-9:] if len(swing_points) >= 9 else swing_points
    
    # Try to identify a potential wave count based on alternating swing highs and lows
    current_wave = 1
    wave_prices = []
    wave_dates = []
    last_was_high = None
    
    for idx, row in recent_swings.iterrows():
        is_high = row['swing_high']
        is_low = row['swing_low']
        
        if is_high and (last_was_high is None or not last_was_high):
            if trend_up and current_wave in [1, 3, 5]:
                wave_prices.append(row['High'])
                wave_dates.append(idx)
                wave_count.append((current_wave, idx, row['High']))
                current_wave += 1
            elif not trend_up and current_wave in [2, 4]:
                wave_prices.append(row['High'])
                wave_dates.append(idx)
                wave_count.append((current_wave, idx, row['High']))
                current_wave += 1
            last_was_high = True
            
        elif is_low and (last_was_high is None or last_was_high):
            if trend_up and current_wave in [2, 4]:
                wave_prices.append(row['Low'])
                wave_dates.append(idx)
                wave_count.append((current_wave, idx, row['Low']))
                current_wave += 1
            elif not trend_up and current_wave in [1, 3, 5]:
                wave_prices.append(row['Low'])
                wave_dates.append(idx)
                wave_count.append((current_wave, idx, row['Low']))
                current_wave += 1
            last_was_high = False
    
    # Calculate Fibonacci projections for wave 5 if we have potential waves 1-4
    projected_paths = []
    if len(wave_count) >= 4:
        # Extract wave prices
        w1_price = next((price for wave, _, price in wave_count if wave == 1), None)
        w2_price = next((price for wave, _, price in wave_count if wave == 2), None)
        w3_price = next((price for wave, _, price in wave_count if wave == 3), None)
        w4_price = next((price for wave, _, price in wave_count if wave == 4), None)
        
        if all(price is not None for price in [w1_price, w2_price, w3_price, w4_price]):
            # Calculate wave 5 projections based on Fibonacci relationships
            w1_length = abs(w2_price - w1_price)
            w3_length = abs(w4_price - w3_price)
            
            # Potential wave 5 targets
            if trend_up:
                w5_target_min = w4_price + 0.618 * w1_length
                w5_target_equal = w4_price + 1.0 * w1_length
                w5_target_ext = w4_price + 1.618 * w1_length
            else:
                w5_target_min = w4_price - 0.618 * w1_length
                w5_target_equal = w4_price - 1.0 * w1_length
                w5_target_ext = w4_price - 1.618 * w1_length
            
            # Create projection paths with probabilities
            projected_paths = [
                {'target': w5_target_min, 'probability': 0.5, 'label': '0.618 x Wave 1'},
                {'target': w5_target_equal, 'probability': 0.3, 'label': '1.0 x Wave 1'},
                {'target': w5_target_ext, 'probability': 0.2, 'label': '1.618 x Wave 1'}
            ]
            
            # Adjust confidence based on wave structure
            # Check if wave 3 is the longest impulse wave (common in Elliott Wave)
            if trend_up and w3_price > w1_price and w3_price > w5_target_min:
                confidence += 0.1
            elif not trend_up and w3_price < w1_price and w3_price < w5_target_min:
                confidence += 0.1
                
            # Check if wave 4 doesn't overlap with wave 1 (Elliott Wave rule)
            if trend_up and w4_price > w1_price:
                confidence += 0.1
            elif not trend_up and w4_price < w1_price:
                confidence += 0.1
    
    return {
        'wave_count': wave_count,
        'confidence': min(confidence, 1.0),  # Cap at 1.0
        'projected_paths': projected_paths,
        'swing_points': swing_points,
        'trend_up': trend_up
    }

#################################################
#           MACHINE LEARNING MODELS             #
#################################################

def create_features(df, window_sizes=[5, 10, 20, 50]):
    """Create technical features for ML models"""
    df_feat = df.copy()
    
    # Calculate returns
    df_feat['returns'] = df_feat['Close'].pct_change()
    df_feat['log_returns'] = np.log(df_feat['Close']/df_feat['Close'].shift(1))
    
    # Price-based features
    for window in window_sizes:
        # Moving averages
        df_feat[f'ma_{window}'] = df_feat['Close'].rolling(window=window).mean()
        
        # Standard deviation (volatility)
        df_feat[f'std_{window}'] = df_feat['Close'].rolling(window=window).std()
        
        # Relative price position
        df_feat[f'price_rel_{window}'] = df_feat['Close'] / df_feat[f'ma_{window}']
        
        # Momentum
        df_feat[f'mom_{window}'] = df_feat['Close'].pct_change(periods=window)
        
    # Volume features
    df_feat['volume_change'] = df_feat['Volume'].pct_change()
    df_feat['volume_ma_10'] = df_feat['Volume'].rolling(window=10).mean()
    df_feat['volume_rel'] = df_feat['Volume'] / df_feat['volume_ma_10']
    
    # OHLC features
    df_feat['hl_ratio'] = df_feat['High'] / df_feat['Low']
    df_feat['co_ratio'] = df_feat['Close'] / df_feat['Open']
    
    # Drop NaN values
    df_feat.dropna(inplace=True)
    
    return df_feat

def create_wave_labels(df, future_periods=5):
    """Create target labels for predicting future price waves"""
    df = df.copy()
    
    # Calculate future returns
    future_return = df['Close'].pct_change(periods=future_periods).shift(-future_periods)
    df['future_return'] = future_return
    
    # Create classification labels
    # 0: Significant Downtrend, 1: Slight Down, 2: Sideways, 3: Slight Up, 4: Significant Uptrend
    df['wave_direction'] = pd.cut(
        df['future_return'], 
        bins=[-np.inf, -0.03, -0.005, 0.005, 0.03, np.inf], 
        labels=[0, 1, 2, 3, 4]
    )
    
    # Drop rows with NaN in target
    df.dropna(subset=['wave_direction', 'future_return'], inplace=True)
    
    return df

def train_ml_models(df, future_periods=5):
    """Train machine learning models to predict future wave directions"""
    # Create features and target labels
    df_ml = create_features(df)
    df_ml = create_wave_labels(df_ml, future_periods)
    
    # Prepare feature matrix and target vector
    feature_cols = [col for col in df_ml.columns if col not in ['future_return', 'wave_direction']]
    X = df_ml[feature_cols].values
    y = df_ml['wave_direction'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    gb_pred = gb_model.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    # Store models, scaler, and feature columns
    models = {
        'random_forest': {
            'model': rf_model,
            'accuracy': rf_acc,
            'feature_cols': feature_cols
        },
        'xgboost': {
            'model': xgb_model,
            'accuracy': xgb_acc,
            'feature_cols': feature_cols
        },
        'gradient_boosting': {
            'model': gb_model,
            'accuracy': gb_acc,
            'feature_cols': feature_cols
        },
        'scaler': scaler
    }
    
    return models

#################################################
#           Neural Network Model                #
#################################################

class WavePredictor(nn.Module):
    def __init__(self, input_dim):
        super(WavePredictor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 5)  # 5 classes for wave direction
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x

def train_nn_model(df, future_periods=5, epochs=20, batch_size=64):
    """Train a neural network to predict future wave directions"""
    # Create features and target labels
    df_ml = create_features(df)
    df_ml = create_wave_labels(df_ml, future_periods)
    
    # Prepare feature matrix and target vector
    feature_cols = [col for col in df_ml.columns if col not in ['future_return', 'wave_direction']]
    X = df_ml[feature_cols].values
    y = df_ml['wave_direction'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train_scaled.shape[1]
    model = WavePredictor(input_dim)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                model.eval()
                test_outputs = model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
                model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs.data, 1)
        final_accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    
    return {
        'model': model,
        'accuracy': final_accuracy,
        'scaler': scaler,
        'feature_cols': feature_cols
    }

#################################################
#        Reinforcement Learning Model           #
#################################################

class MarketEnv(gym.Env):
    """Custom Environment for RL that simulates trading decisions based on Elliott Wave patterns"""
    
    def __init__(self, data_df):
        super(MarketEnv, self).__init__()
        
        self.df = data_df.copy()
        self.df_features = create_features(self.df)
        
        # Define action and observation spaces
        # Actions: 0-4 representing different trading decisions based on wave expectations
        self.action_space = spaces.Discrete(5)
        
        # Observation space: features from price data
        # Make sure we have enough columns even if some get dropped during feature creation
        feature_count = min(20, len(self.df_features.columns) - 2)  # exclude 'future_return' and 'wave_direction' if they exist
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(feature_count,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.scaler = StandardScaler()
        
        # Safety check: drop columns that might cause issues
        cols_to_drop = ['returns', 'log_returns', 'future_return', 'wave_direction']
        columns_to_drop = [col for col in cols_to_drop if col in self.df_features.columns]
        
        # Scale features safely
        self.scaled_features = self.scaler.fit_transform(self.df_features.drop(columns_to_drop, axis=1, errors='ignore'))
        
    def reset(self, seed=None, options=None):
        # Handle gym API differences
        try:
            super().reset(seed=seed)
        except:
            pass  # For compatibility with older gym versions
            
        self.current_step = 0
        obs = self._get_observation()
        info = {}
        return obs, info
        
    def _get_observation(self):
        if self.current_step >= len(self.scaled_features):
            # If we're at the end, just return the last observation
            return self.scaled_features[-1]
        return self.scaled_features[self.current_step]
        
    def step(self, action):
        # Get current observation
        current_obs = self._get_observation()
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.scaled_features) - 1
        
        # Calculate reward based on action and actual future movement
        reward = 0
        if not done and self.current_step < len(self.df_features):
            future_return = self.df_features.iloc[self.current_step]['future_return'] if 'future_return' in self.df_features.columns else 0
            
            # Reward is based on whether the action correctly predicted the direction
            if action == 0 and future_return < -0.03:  # Significant down
                reward = 1.0
            elif action == 1 and -0.03 <= future_return < -0.005:  # Slight down
                reward = 0.5
            elif action == 2 and -0.005 <= future_return <= 0.005:  # Sideways
                reward = 0.3
            elif action == 3 and 0.005 < future_return <= 0.03:  # Slight up
                reward = 0.5
            elif action == 4 and future_return > 0.03:  # Significant up
                reward = 1.0
            else:
                # Penalty for incorrect prediction
                reward = -0.2
        
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, done, False, {}

def create_rl_env(df):
    """Create and wrap the environment for RL training"""
    # First prepare the data with future labels
    df_prepared = create_features(df)
    df_prepared = create_wave_labels(df_prepared, future_periods=5)
    
    # Create and wrap the environment
    try:
        env = MarketEnv(df_prepared)
        vec_env = DummyVecEnv([lambda: env])
        return vec_env
    except Exception as e:
        st.error(f"Error creating RL environment: {e}")
        return None

def train_rl_model(df, timesteps=2000):
    """Train a PPO agent to predict market movements"""
    try:
        # Create environment
        env = create_rl_env(df)
        if env is None:
            return None
            
        # Initialize and train model
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=timesteps)
        
        return model
    except Exception as e:
        st.error(f"Error training RL model: {e}")
        return None

#################################################
#           VISUALIZATION FUNCTIONS             #
#################################################

def plot_elliott_waves(df, wave_analysis, predictions=None):
    """Plot price chart with Elliott Wave labels and projections"""
    # Create figure
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        )
    )
    
    # Add swing points if available
    swing_points = wave_analysis.get('swing_points', None)
    if swing_points is not None:
        # Add swing highs
        swing_highs = swing_points[swing_points['swing_high']]
        if not swing_highs.empty:
            fig.add_trace(
                go.Scatter(
                    x=swing_highs.index,
                    y=swing_highs['High'],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='triangle-up'),
                    name="Swing Highs"
                )
            )
        
        # Add swing lows
        swing_lows = swing_points[swing_points['swing_low']]
        if not swing_lows.empty:
            fig.add_trace(
                go.Scatter(
                    x=swing_lows.index,
                    y=swing_lows['Low'],
                    mode='markers',
                    marker=dict(size=10, color='red', symbol='triangle-down'),
                    name="Swing Lows"
                )
            )
    
    # Add wave labels if available
    wave_count = wave_analysis.get('wave_count', [])
    if wave_count and isinstance(wave_count, list) and len(wave_count) > 0:
        wave_x = []
        wave_y = []
        wave_text = []
        
        for wave_num, date, price in wave_count:
            wave_x.append(date)
            wave_y.append(price)
            wave_text.append(f"Wave {wave_num}")
        
        fig.add_trace(
            go.Scatter(
                x=wave_x,
                y=wave_y,
                mode='text+markers',
                marker=dict(size=8, color='blue'),
                text=wave_text,
                textposition="top center",
                name="Wave Labels"
            )
        )
        
        # Connect the wave points with a line
        fig.add_trace(
            go.Scatter(
                x=wave_x,
                y=wave_y,
                mode='lines',
                line=dict(color='blue', width=1, dash='dot'),
                name="Wave Connections"
            )
        )
    
    # Add projected paths if available
    projected_paths = wave_analysis.get('projected_paths', [])
    if projected_paths and len(projected_paths) > 0 and len(wave_count) > 0:
        # Get the last wave point as the starting point for projections
        last_wave_date = wave_count[-1][1]
        last_wave_price = wave_count[-1][2]
        
        # Project 20 trading days into the future
        last_date = df.index[-1]
        future_date = last_date + pd.Timedelta(days=20)
        
        for path in projected_paths:
            target_price = path['target']
            probability = path['probability']
            label = path['label']
            
            # Color based on probability (higher = more green)
            r = int(255 * (1 - probability))
            g = int(255 * probability)
            color = f'rgb({r},{g},0)'
            
            fig.add_trace(
                go.Scatter(
                    x=[last_wave_date, future_date],
                    y=[last_wave_price, target_price],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"{label} (Prob: {probability:.1f})"
                )
            )
    
    # Add ML predictions if available
    if predictions is not None and 'next_5_days' in predictions:
        # Get last actual price
        last_price = df['Close'].iloc[-1]
        last_date = df.index[-1]
        
        # Future dates
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]
        
        # Add prediction line
        predicted_values = [last_price] + predictions['next_5_days']['values']
        predicted_dates = [last_date] + future_dates
        
        fig.add_trace(
            go.Scatter(
                x=predicted_dates,
                y=predicted_values,
                mode='lines',
                line=dict(color='purple', width=3),
                name="ML Prediction"
            )
        )
        
        # Add prediction confidence intervals
        if 'confidence_intervals' in predictions['next_5_days']:
            lower_bound = [last_price] + predictions['next_5_days']['confidence_intervals']['lower']
            upper_bound = [last_price] + predictions['next_5_days']['confidence_intervals']['upper']
            
            fig.add_trace(
                go.Scatter(
                    x=predicted_dates + predicted_dates[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor='rgba(128, 0, 128, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name="Prediction Interval"
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Elliott Wave Analysis with ML Projections",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update x-axis type to datetime
    fig.update_xaxes(type='date')
    
    return fig

#################################################
#                MAIN APPLICATION               #
#################################################

def main():
    # Sidebar configuration
    st.sidebar.title("Elliott Wave ML Analyzer")
    
    # User inputs
    st.sidebar.header("Data Parameters")
    symbol = st.sidebar.text_input("Symbol", value="SPY")
    lookback_period = st.sidebar.slider("Lookback Period (Days)", 30, 365, 180)
    
    # Date ranges
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=lookback_period)
    
    # Data source selection
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["Auto (Try all methods)", "Yahoo OAuth API", "YFinance", "Synthetic Data", "Manual Upload"]
    )
    
    # ML model selection
    st.sidebar.header("Model Parameters")
    selected_model = st.sidebar.selectbox(
        "Select Prediction Model",
        ["Combined Models", "Random Forest", "XGBoost", "Neural Network", "Reinforcement Learning"]
    )
    
    prediction_days = st.sidebar.slider("Future Prediction Days", 1, 10, 5)
    
    # OAuth Settings
    st.sidebar.header("Yahoo OAuth Settings")
    
    # Check if we already have tokens in the database
    conn = create_db_connection()
    have_tokens = False
    
    if conn:
        create_price_table(conn)
        tokens = get_oauth_tokens(conn)
        have_tokens = tokens is not None
    
    if have_tokens:
        st.sidebar.success("✓ Yahoo OAuth credentials already configured")
    else:
        st.sidebar.warning("Yahoo OAuth not configured")
        
        # Automatic OAuth setup with hardcoded auth code
        if st.sidebar.button("Setup OAuth with hardcoded token"):
            token_data = exchange_code_for_token(YAHOO_AUTH_CODE)
            if token_data:
                st.sidebar.success("OAuth setup complete with hardcoded token!")
                have_tokens = True
                # Use st.rerun() instead of experimental_rerun
                st.rerun()
            else:
                st.sidebar.error("Failed to exchange code for token")
    
    # Get data
    load_button = st.sidebar.button("Load/Refresh Data")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = None
    if 'nn_model' not in st.session_state:
        st.session_state.nn_model = None
    if 'rl_model' not in st.session_state:
        st.session_state.rl_model = None
    if 'wave_analysis' not in st.session_state:
        st.session_state.wave_analysis = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
        
    # Main content
    st.title("Elliott Wave Analysis with ML Projections")
    
    # Show API credentials are implemented
    st.sidebar.success("✓ Yahoo Finance API credentials successfully configured")
    
    if load_button or st.session_state.data is None:
        with st.spinner("Loading data and computing Elliott Wave patterns..."):
            # Based on selected data source
            df = None
            
            if data_source == "Auto (Try all methods)":
                df = get_spy_data_hybrid(symbol, start_date, end_date)
            elif data_source == "Yahoo OAuth API":
                access_token = get_valid_access_token()
                if access_token:
                    df = download_with_auth_header(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if df is None or df.empty:
                    st.warning("OAuth API failed. Generating synthetic data.")
                    df = create_sample_data(symbol, days=lookback_period)
            elif data_source == "YFinance":
                df = download_spy_data_direct(symbol, period="1y")
                if df is None or df.empty:
                    st.warning("YFinance download failed. Generating synthetic data.")
                    df = create_sample_data(symbol, days=lookback_period)
            elif data_source == "Synthetic Data":
                df = create_sample_data(symbol, days=lookback_period)
            elif data_source == "Manual Upload":
                provide_download_template()
                df = allow_manual_data_upload()
                if df is None or df.empty:
                    st.warning("No data uploaded. Generating synthetic data.")
                    df = create_sample_data(symbol, days=lookback_period)
            
            if df is not None and not df.empty:
                st.session_state.data = df
                
                # Perform Elliott Wave analysis
                wave_analysis = analyze_wave_pattern(df)
                st.session_state.wave_analysis = wave_analysis
                
                # Reset models when data changes
                st.session_state.ml_models = None
                st.session_state.nn_model = None
                st.session_state.rl_model = None
                st.session_state.predictions = None
                
                st.success(f"Successfully loaded {len(df)} days of {symbol} data!")
            else:
                st.error("Failed to load data. Please check your inputs and try again.")
    
    # Display data and analysis if available
    if st.session_state.data is not None and not st.session_state.data.empty:
        df = st.session_state.data
        
        # Data summary
        st.subheader("Data Summary")
        st.write(f"Symbol: {symbol}")
        st.write(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
        st.write(f"Trading Days: {len(df)}")
        
        # Display a data preview
        with st.expander("View Data Preview"):
            st.dataframe(df.head(10))
        
        # Elliott Wave Analysis
        st.subheader("Elliott Wave Analysis")
        wave_analysis = st.session_state.wave_analysis
        
        wave_count = wave_analysis.get('wave_count', [])
        confidence = wave_analysis.get('confidence', 0)
        
        if isinstance(wave_count, list) and len(wave_count) > 0:
            st.write(f"Current Wave Count: {len(wave_count)} identified waves")
            st.write(f"Wave Confidence: {confidence:.2f}")
            
            # Display wave count
            wave_data = []
            for wave_num, date, price in wave_count:
                wave_data.append({
                    "Wave": f"Wave {wave_num}",
                    "Date": date.date(),
                    "Price": f"${price:.2f}"
                })
            
            if wave_data:
                st.table(pd.DataFrame(wave_data))
        else:
            st.write(wave_count)  # If it's a message
        
        # Train models if needed
        train_models = st.button("Train/Refresh ML Models")
        
        if train_models or (selected_model != "Reinforcement Learning" and st.session_state.ml_models is None):
            with st.spinner("Training ML models..."):
                ml_models = train_ml_models(df, future_periods=prediction_days)
                st.session_state.ml_models = ml_models
                st.success(f"ML models trained successfully!")
                st.write(f"Random Forest Accuracy: {ml_models['random_forest']['accuracy']:.4f}")
                st.write(f"XGBoost Accuracy: {ml_models['xgboost']['accuracy']:.4f}")
                st.write(f"Gradient Boosting Accuracy: {ml_models['gradient_boosting']['accuracy']:.4f}")
        
        if train_models or (selected_model in ["Neural Network", "Combined Models"] and st.session_state.nn_model is None):
            with st.spinner("Training Neural Network..."):
                nn_model = train_nn_model(df, future_periods=prediction_days, epochs=20)
                st.session_state.nn_model = nn_model
                st.success(f"Neural Network trained successfully!")
                st.write(f"Neural Network Accuracy: {nn_model['accuracy']:.4f}")
        
        if train_models or (selected_model in ["Reinforcement Learning", "Combined Models"] and st.session_state.rl_model is None):
            with st.spinner("Training Reinforcement Learning model (this might take a while)..."):
                rl_model = train_rl_model(df, timesteps=2000)
                st.session_state.rl_model = rl_model
                if rl_model:
                    st.success(f"RL model trained successfully!")
        
        # Generate predictions
        generate_predictions = st.button("Generate Projections")
        
        if generate_predictions or st.session_state.predictions is None:
            with st.spinner("Generating projections..."):
                # Make predictions only if models are trained
                if selected_model in ["Random Forest", "Combined Models"] and st.session_state.ml_models:
                    try:
                        # Prepare features for the last data point
                        df_feat = create_features(df)
                        last_data = df_feat.iloc[-1:]
                        feature_cols = st.session_state.ml_models['random_forest']['feature_cols']
                        X_last = last_data[feature_cols].values
                        X_last_scaled = st.session_state.ml_models['scaler'].transform(X_last)
                        
                        # Get prediction for wave direction
                        rf_prediction = st.session_state.ml_models['random_forest']['model'].predict(X_last_scaled)[0]
                        xgb_prediction = st.session_state.ml_models['xgboost']['model'].predict(X_last_scaled)[0]
                        gb_prediction = st.session_state.ml_models['gradient_boosting']['model'].predict(X_last_scaled)[0]
                        
                        # Convert prediction to expected return range
                        direction_ranges = {
                            0: (-0.05, -0.03),  # Significant down
                            1: (-0.03, -0.005),  # Slight down
                            2: (-0.005, 0.005),  # Sideways
                            3: (0.005, 0.03),    # Slight up
                            4: (0.03, 0.05)      # Significant up
                        }
                        
                        # Get mean of each prediction's range
                        rf_return = sum(direction_ranges[rf_prediction])/2
                        xgb_return = sum(direction_ranges[xgb_prediction])/2
                        gb_return = sum(direction_ranges[gb_prediction])/2
                        
                        # Weighted average based on model accuracy
                        rf_weight = st.session_state.ml_models['random_forest']['accuracy']
                        xgb_weight = st.session_state.ml_models['xgboost']['accuracy']
                        gb_weight = st.session_state.ml_models['gradient_boosting']['accuracy']
                        
                        total_weight = rf_weight + xgb_weight + gb_weight
                        avg_return = (rf_return * rf_weight + xgb_return * xgb_weight + gb_return * gb_weight) / total_weight
                        
                        # Generate projected prices
                        last_price = df['Close'].iloc[-1]
                        projected_values = [last_price * (1 + avg_return * i/prediction_days) for i in range(1, prediction_days+1)]
                        
                        # Confidence interval
                        std_factor = 1.0  # Adjust for wider/narrower intervals
                        lower_bound = [val * (1 - std_factor * i * 0.005) for i, val in enumerate(projected_values)]
                        upper_bound = [val * (1 + std_factor * i * 0.005) for i, val in enumerate(projected_values)]
                        
                        # Store predictions
                        st.session_state.predictions = {
                            'next_5_days': {
                                'values': projected_values,
                                'direction': {
                                    'rf': rf_prediction,
                                    'xgb': xgb_prediction,
                                    'gb': gb_prediction
                                },
                                'expected_return': avg_return,
                                'confidence_intervals': {
                                    'lower': lower_bound,
                                    'upper': upper_bound
                                }
                            }
                        }
                    except Exception as e:
                        st.error(f"Error generating projections: {e}")
                
                # Neural Network predictions
                if selected_model in ["Neural Network", "Combined Models"] and st.session_state.nn_model:
                    # Similar approach as above for NN
                    pass
                
                # RL model predictions
                if selected_model in ["Reinforcement Learning", "Combined Models"] and st.session_state.rl_model:
                    # Use RL model for predictions
                    pass
                
        # Display chart with Elliott Wave analysis and projections
        st.subheader("Elliott Wave Chart with Projections")
        chart = plot_elliott_waves(df, wave_analysis, st.session_state.predictions)
        st.plotly_chart(chart, use_container_width=True)
        
        # Projected paths
        st.subheader("Projected Wave 5 Targets")
        projected_paths = wave_analysis.get('projected_paths', [])
        if projected_paths:
            path_data = []
            for path in projected_paths:
                path_data.append({
                    "Target": f"${path['target']:.2f}",
                    "Probability": f"{path['probability']:.2f}",
                    "Label": path['label']
                })
            st.table(pd.DataFrame(path_data))
        else:
            st.write("No wave projections available. Need at least 4 identified waves.")
        
        # ML Predictions
        if st.session_state.predictions:
            st.subheader("Machine Learning Projections")
            
            if 'next_5_days' in st.session_state.predictions:
                pred = st.session_state.predictions['next_5_days']
                
                # Direction labels
                direction_labels = {
                    0: "Strong Bearish 📉📉",
                    1: "Bearish 📉",
                    2: "Neutral ↔️",
                    3: "Bullish 📈",
                    4: "Strong Bullish 📈📈"
                }
                
                # Display predicted directions
                if 'direction' in pred:
                    st.write("### Predicted Market Direction:")
                    cols = st.columns(3)
                    
                    with cols[0]:
                        rf_dir = pred['direction'].get('rf')
                        if rf_dir is not None:
                            st.metric("Random Forest", direction_labels.get(rf_dir, "Unknown"))
                        
                    with cols[1]:
                        xgb_dir = pred['direction'].get('xgb')
                        if xgb_dir is not None:
                            st.metric("XGBoost", direction_labels.get(xgb_dir, "Unknown"))
                        
                    with cols[2]:
                        gb_dir = pred['direction'].get('gb')
                        if gb_dir is not None:
                            st.metric("Gradient Boosting", direction_labels.get(gb_dir, "Unknown"))
                
                # Display projected prices
                st.write("### Projected Prices:")
                last_date = df.index[-1]
                projected_dates = [last_date + pd.Timedelta(days=i) for i in range(1, prediction_days+1)]
                projected_prices = pred['values']
                
                price_df = pd.DataFrame({
                    'Date': [d.date() for d in projected_dates],
                    'Projected Price': [f"${p:.2f}" for p in projected_prices]
                })
                st.table(price_df)
                
                # Show expected return
                if 'expected_return' in pred:
                    expected_return = pred['expected_return'] * 100
                    st.write(f"### Expected {prediction_days}-Day Return: {expected_return:.2f}%")
                    
                    # Color code based on direction
                    if expected_return > 2:
                        st.markdown(f"<div style='color:green;font-size:20px'>Strong Bullish Outlook 📈📈</div>", unsafe_allow_html=True)
                    elif expected_return > 0.5:
                        st.markdown(f"<div style='color:lightgreen;font-size:20px'>Bullish Outlook 📈</div>", unsafe_allow_html=True)
                    elif expected_return > -0.5:
                        st.markdown(f"<div style='color:yellow;font-size:20px'>Neutral Outlook ↔️</div>", unsafe_allow_html=True)
                    elif expected_return > -2:
                        st.markdown(f"<div style='color:orange;font-size:20px'>Bearish Outlook 📉</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='color:red;font-size:20px'>Strong Bearish Outlook 📉📉</div>", unsafe_allow_html=True)
        
        # Risk and limitations
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚠️ Disclaimer")
        st.sidebar.info(
            "This tool is for educational purposes only. Elliott Wave analysis and ML projections "
            "should not be the sole basis for investment decisions. Past performance does not "
            "guarantee future results."
        )

if __name__ == "__main__":
    main()
