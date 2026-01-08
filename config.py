# config.py

import datetime

# Parametri Generali
TICKER = 'SPY.US'
START_DATE = '2005-01-01'

# HMM Configuration
HMM_PARAMS = {
    'n_states': 3,
    'covariance_type': 'diag',
    'n_iter': 100,
    'random_state': 42,
    'vol_window': 20  # Finestra rolling per volatilità realizzata
}

# GARCH Configuration
GARCH_PARAMS = {
    'p': 1,
    'q': 1,
    'dist': 'normal',
    'window_size': 1000  # Finestra per il training rolling
}

# Soglie Segnali (Risk Management)
THRESHOLDS = {
    'high_vol': 0.60,       # Soglia probabilità HMM per RISK-OFF
    'low_vol': 0.60,        # Soglia probabilità HMM per RISK-ON
    'trend_window': 5,      # Giorni per calcolo trend probabilità
    'alert_change': 0.15,   # Variazione % per ALERT
    'garch_percentile': 0.75 # Percentile per definire "Alta Vol" su GARCH
}

# Etichette Regimi
REGIME_LABELS = {
    0: 'Low Volatility',
    1: 'Medium Volatility',
    2: 'High Volatility'
}

# Colori per i grafici
REGIME_COLORS = {
    0: '#28a745',  # Verde
    1: '#ffc107',  # Giallo/Ambra
    2: '#dc3545'   # Rosso
}

# Configurazione Segnali e Azioni
SIGNAL_CONFIG = {
    'STRONG_RISK_OFF': {'icon': '🔴🔴', 'color': '#8b0000', 'action': 'Copertura Aggressiva (Long VIX Futures / Put Spread)'},
    'RISK_OFF': {'icon': '🔴', 'color': '#dc3545', 'action': 'Ridurre esposizione, Hedging tattico'},
    'ALERT': {'icon': '🟠', 'color': '#fd7e14', 'action': 'Monitorare, preparare ordini copertura'},
    'NEUTRAL': {'icon': '🟡', 'color': '#ffc107', 'action': 'Allocazione standard'},
    'RISK_ON': {'icon': '🟢', 'color': '#28a745', 'action': 'Esposizione piena, strategie direzionali'},
    'WATCH': {'icon': '⚠️', 'color': '#6c757d', 'action': 'Regime instabile, cautela'}
}
