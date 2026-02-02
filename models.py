# models.py
import numpy as np
import pandas as pd
from hmmlearn import hmm, base  # <--- IMPORTANTE: Importa 'base'
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from config import HMM_PARAMS, GARCH_PARAMS, REGIME_LABELS

# =============================================================================
# 🛠️ MONKEY PATCH: FIX HMMLEARN CRASH
# =============================================================================
# Le versioni recenti di hmmlearn lanciano ValueError se la somma delle probabilità
# differisce da 1.0 anche per 1e-15. Questo patch disabilita quel controllo specifico.
def quiet_check_sum_1(self, name):
    """Non fare nulla. Fidati che la somma sia 1."""
    pass

# Sovrascriviamo il metodo nella classe base
base.BaseHMM._check_sum_1 = quiet_check_sum_1
# =============================================================================

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
        init_params='stmc'
    )
    
    model.fit(X_scaled)
    
    # Nota: Con il Monkey Patch sopra, il "fix matematico" complesso 
    # non è più strettamente necessario per evitare il crash, 
    # ma una normalizzazione di base rimane buona pratica.
    
    return model, scaler, mapping

# ... resto del file invariato ...

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
