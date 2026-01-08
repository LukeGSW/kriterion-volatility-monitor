# run_daily_check.py
import pandas as pd
from data_loader import download_data, calculate_features
from models import train_hmm, get_hmm_states, train_garch
from notifications import send_telegram_alert, format_message
from config import THRESHOLDS, REGIME_LABELS, SIGNAL_CONFIG

def job():
    print("🚀 Avvio Job Giornaliero Kriterion...")
    
    # 1. Carica e Prepara Dati
    try:
        df_raw = download_data()
        df = calculate_features(df_raw)
        print(f"✅ Dati scaricati. Ultima data: {df.index[-1]}")
    except Exception as e:
        print(f"❌ Errore critico download dati: {e}")
        return

    # 2. Esegui Modelli
    print("⚙️ Training HMM...")
    model_hmm, scaler_hmm, state_mapping = train_hmm(df)
    states, posteriors = get_hmm_states(df, model_hmm, scaler_hmm, state_mapping)
    
    print("⚙️ Training GARCH...")
    garch_vol_ann, _ = train_garch(df)

    # 3. Analisi Ultimo Giorno
    last_row = df.iloc[-1]
    curr_probs = posteriors[-1] # [Low, Med, High]
    p_high = curr_probs[2]
    p_low = curr_probs[0]
    
    # Calcolo Trend
    # Nota: serve ricalcolare la colonna P_High su tutto il df per fare diff corretta
    # Per semplicità qui prendiamo la differenza tra l'ultima prob e quella di 5gg fa
    if len(posteriors) > 5:
        prev_prob = posteriors[-5][2]
        trend_p_high = p_high - prev_prob
    else:
        trend_p_high = 0.0

    # 4. Generazione Segnale
    signal_type = "NEUTRAL"
    
    # Logica identica ad app.py
    if p_high > THRESHOLDS['high_vol']:
        signal_type = "RISK_OFF"
        # Check GARCH percentile (calcolato al volo qui)
        garch_threshold = df['GK_Vol'].quantile(THRESHOLDS['garch_percentile'])
        if garch_vol_ann > garch_threshold:
            signal_type = "STRONG_RISK_OFF"
            
    elif trend_p_high > THRESHOLDS['alert_change']:
        signal_type = "ALERT"
    elif p_low > THRESHOLDS['low_vol']:
        signal_type = "RISK_ON"

    print(f"🎯 Segnale Generato: {signal_type}")

    # 5. Invio Notifica (SEMPRE o solo se cambia? Qui mandiamo sempre report giornaliero)
    msg = format_message(
        date=last_row.name.strftime('%Y-%m-%d'),
        price=last_row['Close'],
        hmm_probs=curr_probs,
        garch_vol=garch_vol_ann,
        regime_label=REGIME_LABELS[states[-1]],
        signal_type=signal_type,
        trend_prob=trend_p_high
    )
    
    send_telegram_alert(msg)
    print("🏁 Job completato.")

if __name__ == "__main__":
    job()
