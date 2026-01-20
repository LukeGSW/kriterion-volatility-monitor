# data_loader.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yfinance as yf  # Aggiunto yfinance
from datetime import datetime, timedelta

from utils import get_secret
from config import TICKER, START_DATE, HMM_PARAMS

@st.cache_data(ttl=3600)  # Cache dei dati per 1 ora
def download_data():
    """
    Scarica i dati OHLCV. 
    Usa Yahoo Finance per il VIX (per evitare limiti abbonamento EODHD) 
    e EODHD per tutto il resto.
    """
    
    # --- LOGICA SPECIFICA PER IL VIX (YAHOO FINANCE) ---
    # Se il ticker è VIX o ^VIX, usiamo Yahoo Finance per avere i dati aggiornati
    if 'VIX' in TICKER.upper():
        print(f"⚠️ Ticker '{TICKER}' rilevato: switch forzato a Yahoo Finance (EODHD non include VIX).")
        
        # Gestione simbolo Yahoo (vuole ^VIX)
        yf_ticker = TICKER if '^' in TICKER else f"^{TICKER}"
        
        try:
            # FIX DATE: Usiamo period="10y" invece di start/end manuali.
            # Questo risolve il problema del "venerdì scorso": Yahoo scarica automaticamente
            # fino all'ultima chiusura disponibile, saltando correttamente le feste.
            ticker_obj = yf.Ticker(yf_ticker)
            df = ticker_obj.history(period="10y", auto_adjust=False)
            
            if df.empty:
                raise Exception("Yahoo Finance non ha restituito dati.")

            # Pulizia e standardizzazione colonne per compatibilità con il resto del codice
            # Yahoo restituisce: Open, High, Low, Close, Adj Close, Volume
            df = df.rename(columns={
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                'Close': 'Close', 'Adj Close': 'Adj_Close', 'Volume': 'Volume'
            })
            
            # Se manca Adj_Close (a volte succede con gli indici), lo creiamo uguale a Close
            if 'Adj_Close' not in df.columns:
                df['Adj_Close'] = df['Close']
            
            # Assicuriamoci che l'indice sia datetime e abbia il nome giusto
            df.index.name = 'Date'
            df.index = pd.to_datetime(df.index).tz_localize(None) # Rimuove timezone se presente
            
            # Filtriamo dalla data di inizio configurata
            mask = df.index >= pd.to_datetime(START_DATE)
            df = df.loc[mask]
            
            print(f"✅ Dati scaricati da Yahoo: {len(df)} righe (Ultima: {df.index[-1].strftime('%Y-%m-%d')})")
            return df

        except Exception as e:
            raise Exception(f"Errore download Yahoo Finance: {str(e)}")

    # --- LOGICA STANDARD (EODHD) PER ALTRI TICKER ---
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
