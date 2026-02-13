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
    # Questa è la parte cruciale per risolvere il problema dei dati parziali
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
    """
    Calcola le features di volatilità con filtro "Circuit Breaker".
    Lascia passare i crash di mercato reali (es. Covid, Flash Crash) ma blocca
    gli errori matematici o i bad ticks dell'API (es. Low=0).
    """
    if df.empty:
        raise ValueError("DataFrame vuoto in calculate_features")

    df = df.copy()
    
    # 1. Rendimenti Logaritmici
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
    
    # --- CIRCUIT BREAKER FILTER (PHYSICS BASED) ---
    # Calcola range percentuale intraday
    df['Intraday_Range'] = (df['High'] - df['Low']) / df['Open']
    
    # Soglia di "impossibilità": 25% di escursione in un giorno
    IMPOSSIBLE_THRESHOLD = 0.25 
    
    # Identifica le righe con dati corrotti
    bad_ticks = df['Intraday_Range'].abs() > IMPOSSIBLE_THRESHOLD
    
    if bad_ticks.sum() > 0:
        print(f"⚠️ Rilevati {bad_ticks.sum()} tick con range > {IMPOSSIBLE_THRESHOLD*100}%. Correzione in corso...")
        # Correzione: Impostiamo High e Low basandoci su Open e Close reali
        avg_range = df['Intraday_Range'].rolling(5).median().fillna(0.01)
        
        # Sovrascriviamo solo le righe corrotte
        df.loc[bad_ticks, 'High'] = df.loc[bad_ticks, 'Open'] * (1 + avg_range[bad_ticks]/2)
        df.loc[bad_ticks, 'Low']  = df.loc[bad_ticks, 'Open'] * (1 - avg_range[bad_ticks]/2)
    
    # ---------------------------------------------

    # 2. Garman-Klass Volatility (Raw Calculation)
    epsilon = 1e-8 
    log_hl = np.log(df['High'] / (df['Low'] + epsilon)) ** 2
    log_co = np.log(df['Close'] / (df['Open'] + epsilon)) ** 2
    
    df['Garman_Klass'] = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    
    # Pulizia residui matematici
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Garman_Klass'] = df['Garman_Klass'].interpolate(method='linear').fillna(0)
    
    # Safety Cap Rilassato (Varianza 0.05 = ~22% movimento daily)
    df['Garman_Klass'] = df['Garman_Klass'].clip(upper=0.05)
    
    # =========================================================================
    # MODIFICA KRITERION QUANT: Feature Engineering per Reattività
    # =========================================================================
    
    # 1. Volatilità Istantanea Annualizzata
    # Invece di mediare prima (che crea lag), annualizziamo subito il dato daily.
    df['GK_Daily_Ann'] = np.sqrt(df['Garman_Klass'] * 252)

    # 2. Smoothing "Fast" (Media Esponenziale)
    # Sostituisce la vecchia rolling(20). span=5 reagisce in ~2-3 giorni invece di 20.
    # Questo permette di catturare gli spike improvvisi.
    df['GK_Vol'] = df['GK_Daily_Ann'].ewm(span=5, adjust=False).mean()
    
    # 3. Log-Volatility per HMM
    # L'HMM assume distribuzioni Gaussiane. La volatilità è Log-Normale.
    # Usare il logaritmo migliora drasticamente la capacità dell'HMM di fittare i regimi.
    df['Log_Vol'] = np.log(df['GK_Vol'] + 1e-6)

    # Pulizia NaN iniziali
    df.dropna(inplace=True)
    
    if df.empty:
        raise ValueError("Storico insufficiente dopo il calcolo delle features.")

    return df
