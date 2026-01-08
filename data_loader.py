# data_loader.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
from utils import get_secret
from config import TICKER, START_DATE, HMM_PARAMS

@st.cache_data(ttl=3600)  # Cache dei dati per 1 ora
def download_data():
    """Scarica i dati OHLCV da EODHD."""
    api_key = get_secret('EODHD_API_KEY')
    
    if not api_key:
        raise ValueError("EODHD_API_KEY non trovata nei secrets o environment variables.")

    url = f"https://eodhd.com/api/eod/{TICKER}"
    params = {
        'api_token': api_key,
        'from': START_DATE,
        'fmt': 'json'
    }

    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Errore API EODHD: {response.status_code}")

    data = response.json()
    if not data:
        raise Exception("Nessun dato ricevuto dall'API")

    df = pd.DataFrame(data)
    df = df.rename(columns={
        'date': 'Date', 'open': 'Open', 'high': 'High', 
        'low': 'Low', 'close': 'Close', 'adjusted_close': 'Adj_Close', 'volume': 'Volume'
    })
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Conversione numerica sicura
    cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    return df

def calculate_features(df):
    """Calcola le features di volatilità (Garman-Klass, Returns, ecc)."""
    df = df.copy()
    
    # Rendimenti Logaritmici
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Garman-Klass Volatility (La feature principale per HMM)
    # Formula: 0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    df['Garman_Klass'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    
    # Volatilità realizzata rolling (per confronto e grafici)
    window = HMM_PARAMS['vol_window']
    # GK Rolling Annualizzata
    gk_clean = df['Garman_Klass'].clip(lower=0) # Gestione rari negativi
    df['GK_Vol'] = np.sqrt(gk_clean.rolling(window=window).mean() * 252)
    
    # Rimuoviamo NaN iniziali generati dalle finestre mobili
    df.dropna(inplace=True)
    
    return df
