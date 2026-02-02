# models.py
import numpy as np
import pandas as pd
from hmmlearn import hmm
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from config import HMM_PARAMS, GARCH_PARAMS, REGIME_LABELS

# models.py

def train_hmm(df):
    """Addestra il modello HMM sui dati forniti."""
    
    X = df[['GK_Vol']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = hmm.GaussianHMM(
        n_components=HMM_PARAMS['n_states'],
        covariance_type=HMM_PARAMS['covariance_type'],
        n_iter=HMM_PARAMS['n_iter'],
        random_state=HMM_PARAMS['random_state'],
        tol=1e-4, # Tolleranza aggiunta come consigliato
        init_params='stmc' # Inizializza tutto
    )
    
    model.fit(X_scaled)
    
    # =========================================================================
    # FIX ROBUSTO "CHECK_SUM_1"
    # =========================================================================
    
    # 1. Normalizza la matrice di transizione (Transmat)
    # Divide ogni riga per la sua somma
    row_sums = model.transmat_.sum(axis=1)
    model.transmat_ /= row_sums[:, np.newaxis]
    
    # FORZATURA MATEMATICA: Aggiunge la differenza (epsilon) all'elemento max
    # per garantire che np.sum sia ESATTAMENTE 1.0
    for i in range(model.transmat_.shape[0]):
        diff = 1.0 - np.sum(model.transmat_[i])
        max_idx = np.argmax(model.transmat_[i])
        model.transmat_[i, max_idx] += diff

    # 2. Normalizza le probabilità iniziali (Startprob)
    model.startprob_ /= np.sum(model.startprob_)
    # Forzatura anche qui
    diff_start = 1.0 - np.sum(model.startprob_)
    max_start_idx = np.argmax(model.startprob_)
    model.startprob_[max_start_idx] += diff_start

    # =========================================================================
    
    # Riordina gli stati per avere coerenza (0=Low, 1=Med, 2=High)
    means = model.means_.flatten()
    sorted_idx = np.argsort(means)
    mapping = {original: new for new, original in enumerate(sorted_idx)}
    
    return model, scaler, mapping

def get_hmm_states(df, model, scaler, mapping):
    """Inferenza degli stati HMM."""
    X = df[['GK_Vol']].values
    X_scaled = scaler.transform(X)
    
    hidden_states = model.predict(X_scaled)
    posteriors = model.predict_proba(X_scaled)
    
    # Rimappa gli stati
    mapped_states = np.array([mapping[s] for s in hidden_states])
    
    # Rimappa le probabilità posteriori
    mapped_posteriors = np.zeros_like(posteriors)
    reverse_mapping = {v: k for k, v in mapping.items()}
    for new_idx in range(len(mapping)):
        orig_idx = reverse_mapping[new_idx]
        mapped_posteriors[:, new_idx] = posteriors[:, orig_idx]
        
    return mapped_states, mapped_posteriors

def train_garch(df):
    """Addestra GARCH(1,1) e fa previsione 1-step ahead."""
    # GARCH vuole i ritorni in percentuale (es. 1.5 invece di 0.015) per convergere meglio
    returns_pct = df['Returns'] * 100
    
    # Usa una finestra recente per il fit (più reattivo) o tutto lo storico
    window = GARCH_PARAMS['window_size']
    train_data = returns_pct.iloc[-window:]
    
    model = arch_model(
        train_data,
        p=GARCH_PARAMS['p'],
        q=GARCH_PARAMS['q'],
        dist=GARCH_PARAMS['dist'],
        vol='Garch'
    )
    
    res = model.fit(disp='off')
    
    # Forecast 1 step
    forecast = res.forecast(horizon=1)
    # Variance dell'ultimo step
    var_forecast = forecast.variance.values[-1, 0]
    # Volatilità annualizzata stimata (ritorniamo a decimali)
    vol_forecast_ann = np.sqrt(var_forecast) / 100 * np.sqrt(252)
    
    return vol_forecast_ann, res
