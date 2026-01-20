# data_loader.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yfinance as yf
from datetime import datetime, time
import pytz # Necessario per gestire il fuso orario di NY

from utils import get_secret
from config import TICKER, START_DATE, HMM_PARAMS

# NOTA: Riduciamo il TTL della cache per evitare di vedere dati vecchi in fasi critiche
@st.cache_data(ttl=600) 
def download_data():
    """
    Scarica i dati OHLCV. 
    Usa Yahoo Finance per il VIX (o se forzato) e EODHD per tutto il resto.
    Applica la logica di 'Ultima Chiusura Giornaliera' per garantire dati consolidati.
    """
    df = pd.DataFrame()

    # --- 1. SELEZIONE FONTE DATI ---
    # Se il ticker è VIX, usiamo Yahoo Finance
    if 'VIX' in TICKER.upper():
        print(f"⚠️ Ticker '{TICKER}' rilevato: switch forzato a Yahoo Finance.")
        df = _download_from_yahoo()
    else:
        # Tenta EODHD, se fallisce o ticker strano, potresti voler gestire fallback
        try:
            df = _download_from_eodhd()
        except Exception as e:
            print(f"❌ Errore EODHD: {e}. Tento fallback su Yahoo...")
            df = _download_from_yahoo()

    # --- 2. VALIDAZIONE CHIUSURA GIORNALIERA ---
    # Questa è la parte cruciale per risolvere il tuo problema
    df = _validate_market_close(df)

    return df

def _download_from_yahoo():
    """Scarica dati da Yahoo Finance (helper interno)."""
    # Gestione simbolo Yahoo (vuole ^VIX per l'indice)
    yf_ticker = TICKER
    if 'VIX' in TICKER.upper() and '^' not in TICKER:
        yf_ticker = f"^{TICKER}"
    
    try:
        # Scarichiamo un periodo ampio per essere sicuri di avere tutto
        ticker_obj = yf.Ticker(yf_ticker)
        df = ticker_obj.history(period="5y", auto_adjust=False)
        
        if df.empty:
            raise Exception("Yahoo Finance non ha restituito dati.")

        # Rinomina e standardizza
        df = df.rename(columns={
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 
            'Close': 'Close', 'Adj Close': 'Adj_Close', 'Volume': 'Volume'
        })
        
        # Fix colonne mancanti
        if 'Adj_Close' not in df.columns:
            df['Adj_Close'] = df['Close']
        
        # Pulizia indice
        df.index.name = 'Date'
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Filtro data inizio
        df = df[df.index >= pd.to_datetime(START_DATE)]
        
        print(f"✅ Dati scaricati da Yahoo: {len(df)} righe")
        return df

    except Exception as e:
        raise Exception(f"Errore download Yahoo Finance: {str(e)}")

def _download_from_eodhd():
    """Scarica dati da EODHD (helper interno)."""
    api_key = get_secret('EODHD_API_KEY')
    if not api_key:
        raise ValueError("EODHD_API_KEY non trovata.")

    clean_ticker = TICKER.replace('^', '').strip()
    url = f"https://eodhd.com/api/eod/{clean_ticker}"
    params = {'api_token': api_key, 'from': START_DATE, 'fmt': 'json'}

    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code != 200:
        raise Exception(f"API EODHD errore {response.status_code}")

    data = response.json()
    if not data:
        # Fallback specifico per VIX su EODHD
        if clean_ticker == 'VIX':
            print("⚠️ EODHD 'VIX' vuoto, tento 'VIX.INDX'...")
            response = requests.get(f"https://eodhd.com/api/eod/VIX.INDX", params=params)
            data = response.json()
    
    if not data:
        raise Exception("Nessun dato da EODHD.")

    df = pd.DataFrame(data)
    df = df.rename(columns={
        'date': 'Date', 'open': 'Open', 'high': 'High', 
        'low': 'Low', 'close': 'Close', 'adjusted_close': 'Adj_Close', 'volume': 'Volume'
    })
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Conversione numerica
    for c in ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    return df

def _validate_market_close(df):
    """
    Logica 'Smart':
    - Se l'ultima data è < di Oggi: OK (è una chiusura passata).
    - Se l'ultima data è == Oggi:
        - Se ora < 16:15 NY (mercato aperto): SCARTA l'ultima riga (è incompleta).
        - Se ora >= 16:15 NY (mercato chiuso): TIENI l'ultima riga (è la chiusura di oggi).
    """
    if df.empty:
        return df

    # Configurazione Timezone NY
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    today_date_ny = now_ny.date()
    
    last_date_in_df = df.index[-1].date()

    # Se il dato più recente nel DF è di oggi
    if last_date_in_df == today_date_ny:
        # Orario di chiusura mercato USA (16:00) + 15 min buffer
        market_close_time = time(16, 15)
        current_time = now_ny.time()
        
        if current_time < market_close_time:
            print(f"🕒 Mercato NY ancora aperto ({current_time.strftime('%H:%M')}).")
            print(f"⚠️ Rimuovo la candela di oggi ({last_date_in_df}) perché incompleta.")
            print(f"   L'analisi verrà fatta sulla chiusura di IERI.")
            df = df.iloc[:-1]
        else:
            print(f"🌑 Mercato NY chiuso ({current_time.strftime('%H:%M')}).")
            print(f"✅ La candela di oggi ({last_date_in_df}) è confermata e verrà usata.")
            
    return df

def calculate_features(df):
    """Calcola le features di volatilità."""
    if df.empty:
        raise ValueError("DataFrame vuoto in calculate_features")

    df = df.copy()
    
    # Rendimenti Logaritmici
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Garman-Klass Volatility
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    df['Garman_Klass'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    
    # Volatilità realizzata rolling
    window = HMM_PARAMS['vol_window']
    gk_clean = df['Garman_Klass'].clip(lower=0)
    df['GK_Vol'] = np.sqrt(gk_clean.rolling(window=window).mean() * 252)
    
    # Rimuoviamo NaN iniziali
    df.dropna(inplace=True)
    
    # Check finale
    if df.empty:
        raise ValueError("Storico insufficiente dopo il calcolo delle features.")

    return df
