# config.py - Configurazione Kriterion Volatility Monitor v2.0

import datetime

# ============================================================================
# PARAMETRI GENERALI
# ============================================================================

TICKER = 'SPY.US'
START_DATE = '2005-01-01'

# ============================================================================
# HMM CONFIGURATION
# ============================================================================

HMM_PARAMS = {
    'n_states': 3,              # Numero di stati nascosti (Low, Medium, High)
    'covariance_type': 'diag',  # Tipo di matrice covarianza
    'n_iter': 100,              # Iterazioni massime EM
    'random_state': 42,         # Seed per riproducibilità
    'vol_window': 20            # Finestra rolling per volatilità realizzata
}

# ============================================================================
# GARCH CONFIGURATION
# ============================================================================

GARCH_PARAMS = {
    'p': 1,                     # Ordine GARCH
    'q': 1,                     # Ordine ARCH
    'dist': 'normal',           # Distribuzione errori
    'window_size': 1000         # Finestra per il training rolling
}

# ============================================================================
# SOGLIE SEGNALI (Risk Management)
# ============================================================================

THRESHOLDS = {
    'high_vol': 0.60,           # Soglia P(High Vol) per RISK-OFF
    'low_vol': 0.60,            # Soglia P(Low Vol) per RISK-ON
    'trend_window': 5,          # Giorni per calcolo trend probabilità
    'alert_change': 0.15,       # Variazione % per ALERT
    'garch_percentile': 0.75,   # Percentile per definire "Alta Vol" su GARCH
    'confidence_min': 0.70      # Confidenza minima per segnale affidabile
}

# ============================================================================
# ETICHETTE E COLORI REGIMI
# ============================================================================

REGIME_LABELS = {
    0: 'Low Volatility',
    1: 'Medium Volatility',
    2: 'High Volatility'
}

REGIME_COLORS = {
    0: '#28a745',   # Verde
    1: '#ffc107',   # Giallo/Ambra
    2: '#dc3545'    # Rosso
}

REGIME_DESCRIPTIONS = {
    0: 'Mercato tranquillo, volatilità contenuta. Condizioni favorevoli per strategie direzionali.',
    1: 'Volatilità nella norma. Mercato in fase di transizione o equilibrio.',
    2: 'Alta volatilità, stress di mercato. Aumentato rischio di movimenti bruschi.'
}

# ============================================================================
# CONFIGURAZIONE SEGNALI E AZIONI
# ============================================================================

SIGNAL_CONFIG = {
    'STRONG_RISK_OFF': {
        'icon': '🔴🔴',
        'color': '#8b0000',
        'action': 'Copertura Aggressiva: Long VIX Futures, Put Spread, riduzione drastica esposizione',
        'description': 'Consensus HMM + GARCH indica alta volatilità imminente. Massima cautela.',
        'vix_strategy': 'Long VIX Futures o Call VIX'
    },
    'RISK_OFF': {
        'icon': '🔴',
        'color': '#dc3545',
        'action': 'Ridurre esposizione equity, hedging tattico con Put o VIX',
        'description': 'HMM indica alta probabilità di regime High Volatility.',
        'vix_strategy': 'Long VIX Call Spread'
    },
    'ALERT': {
        'icon': '🟠',
        'color': '#fd7e14',
        'action': 'Monitorare attentamente, preparare ordini di copertura',
        'description': 'Probabilità High Vol in rapido aumento. Potenziale cambio regime.',
        'vix_strategy': 'Preparare ordini Long VIX'
    },
    'NEUTRAL': {
        'icon': '🟡',
        'color': '#ffc107',
        'action': 'Mantenere allocazione standard, nessuna azione particolare',
        'description': 'Nessun regime dominante. Mercato in equilibrio.',
        'vix_strategy': 'Nessuna posizione VIX'
    },
    'RISK_ON': {
        'icon': '🟢',
        'color': '#28a745',
        'action': 'Esposizione piena, strategie direzionali long, vendita volatilità',
        'description': 'Alta probabilità di regime Low Volatility. Condizioni favorevoli.',
        'vix_strategy': 'Short VIX Futures o Put Spread su VIX'
    },
    'WATCH': {
        'icon': '⚠️',
        'color': '#17a2b8',
        'action': 'Regime instabile, alta incertezza. Ridurre size posizioni.',
        'description': 'HMM con bassa confidenza. Difficile classificare il regime corrente.',
        'vix_strategy': 'Straddle/Strangle su VIX'
    }
}

# ============================================================================
# CONFIGURAZIONE DISPLAY
# ============================================================================

DISPLAY_CONFIG = {
    'default_chart_period': 252,    # Giorni default per grafici
    'max_chart_period': 1260,       # Massimo 5 anni
    'table_rows': 50,               # Righe tabella dati
    'decimal_places': 4             # Decimali per display
}

# ============================================================================
# MESSAGGI E TESTI
# ============================================================================

TEXTS = {
    'disclaimer': """
        ⚠️ **Disclaimer:** Questo strumento è fornito a scopo educativo e di ricerca. 
        I segnali generati non costituiscono consulenza finanziaria né raccomandazioni di investimento. 
        Le performance passate non sono indicative di risultati futuri. 
        L'utente è responsabile delle proprie decisioni di investimento.
    """,
    
    'hmm_explanation': """
        L'Hidden Markov Model (HMM) identifica stati nascosti di mercato basandosi sulla volatilità osservata.
        Il modello stima la probabilità di trovarsi in ciascuno dei 3 regimi (Low, Medium, High Volatility)
        e calcola la matrice di transizione tra gli stati.
    """,
    
    'garch_explanation': """
        Il modello GARCH(1,1) cattura la "memoria" della volatilità: gli shock recenti influenzano
        la previsione della volatilità futura. La somma α+β (persistenza) indica quanto velocemente
        la volatilità ritorna alla media.
    """,
    
    'signal_explanation': """
        I segnali sono generati combinando le probabilità HMM con le previsioni GARCH.
        Un segnale STRONG_RISK_OFF richiede consensus da entrambi i modelli.
    """
}
