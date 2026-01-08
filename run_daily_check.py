# run_daily_check.py - Job Giornaliero Kriterion Volatility Monitor v2.0
# Eseguito da GitHub Actions dopo la chiusura del mercato

import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Import moduli locali
from data_loader import download_data, calculate_features
from models import train_hmm, get_hmm_states, train_garch
from notifications import send_telegram_alert, format_daily_report, send_error_alert
from config import THRESHOLDS, REGIME_LABELS, SIGNAL_CONFIG

def job():
    """
    Job principale eseguito giornalmente.
    Scarica dati, esegue modelli, genera segnale e invia notifica.
    """
    
    print("=" * 60)
    print("🚀 KRITERION DAILY VOLATILITY CHECK")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    # =========================================================================
    # 1. CARICAMENTO DATI
    # =========================================================================
    print("\n📊 [1/5] Caricamento dati da EODHD...")
    
    try:
        df_raw = download_data()
        df = calculate_features(df_raw)
        
        last_date = df.index[-1].strftime('%Y-%m-%d')
        print(f"   ✅ Dati scaricati: {len(df):,} osservazioni")
        print(f"   📅 Ultima data disponibile: {last_date}")
        
    except Exception as e:
        error_msg = f"Errore critico download dati: {str(e)}"
        print(f"   ❌ {error_msg}")
        send_error_alert(error_msg, context="Download Dati EODHD")
        sys.exit(1)
    
    # =========================================================================
    # 2. TRAINING HMM
    # =========================================================================
    print("\n🤖 [2/5] Training Hidden Markov Model...")
    
    try:
        model_hmm, scaler_hmm, state_mapping = train_hmm(df)
        states, posteriors = get_hmm_states(df, model_hmm, scaler_hmm, state_mapping)
        
        print(f"   ✅ HMM addestrato su {len(df)} osservazioni")
        print(f"   📊 Stati: {len(set(states))} regimi identificati")
        
    except Exception as e:
        error_msg = f"Errore training HMM: {str(e)}"
        print(f"   ❌ {error_msg}")
        send_error_alert(error_msg, context="Training HMM")
        sys.exit(1)
    
    # =========================================================================
    # 3. TRAINING GARCH
    # =========================================================================
    print("\n📉 [3/5] Training GARCH(1,1)...")
    
    try:
        garch_vol_ann, garch_result = train_garch(df)
        
        print(f"   ✅ GARCH addestrato")
        print(f"   📈 Forecast volatilità: {garch_vol_ann*100:.2f}%")
        
    except Exception as e:
        error_msg = f"Errore training GARCH: {str(e)}"
        print(f"   ❌ {error_msg}")
        # GARCH non è critico, continua con warning
        garch_vol_ann = df['GK_Vol'].iloc[-1]
        print(f"   ⚠️ Usando volatilità realizzata come fallback: {garch_vol_ann*100:.2f}%")
    
    # =========================================================================
    # 4. ANALISI E GENERAZIONE SEGNALE
    # =========================================================================
    print("\n🎯 [4/5] Generazione segnale operativo...")
    
    # Estrai dati ultimo giorno
    last_row = df.iloc[-1]
    last_state = states[-1]
    curr_probs = posteriors[-1]  # [Low, Medium, High]
    
    p_low = curr_probs[0]
    p_medium = curr_probs[1]
    p_high = curr_probs[2]
    
    # Calcolo trend P(High Vol)
    if len(posteriors) > THRESHOLDS['trend_window']:
        prev_prob = posteriors[-THRESHOLDS['trend_window']][2]
        trend_p_high = p_high - prev_prob
    else:
        trend_p_high = 0.0
    
    # Confidenza (probabilità massima)
    confidence = max(p_low, p_medium, p_high)
    
    # Logica generazione segnale
    signal_type = "NEUTRAL"
    
    if p_high > THRESHOLDS['high_vol']:
        signal_type = "RISK_OFF"
        
        # Check GARCH percentile per STRONG_RISK_OFF
        garch_threshold = df['GK_Vol'].quantile(THRESHOLDS['garch_percentile'])
        if garch_vol_ann > garch_threshold:
            signal_type = "STRONG_RISK_OFF"
            
    elif trend_p_high > THRESHOLDS['alert_change']:
        signal_type = "ALERT"
        
    elif p_low > THRESHOLDS['low_vol']:
        signal_type = "RISK_ON"
    
    # Se confidenza bassa, segnala WATCH
    if confidence < THRESHOLDS.get('confidence_min', 0.70) and signal_type == "NEUTRAL":
        signal_type = "WATCH"
    
    # Report segnale
    sig_info = SIGNAL_CONFIG.get(signal_type, SIGNAL_CONFIG['NEUTRAL'])
    
    print(f"\n   {'='*50}")
    print(f"   {sig_info['icon']} SEGNALE: {signal_type}")
    print(f"   {'='*50}")
    print(f"   📊 Regime HMM: {REGIME_LABELS[last_state]}")
    print(f"   🎯 Confidenza: {confidence*100:.1f}%")
    print(f"   📈 P(Low Vol):    {p_low*100:.1f}%")
    print(f"   📊 P(Medium Vol): {p_medium*100:.1f}%")
    print(f"   📉 P(High Vol):   {p_high*100:.1f}%")
    print(f"   📈 Trend (5gg):   {trend_p_high*100:+.1f}%")
    print(f"   📉 GARCH Vol:     {garch_vol_ann*100:.2f}%")
    print(f"   💰 SPY Close:     ${last_row['Close']:.2f}")
    print(f"   📌 Azione: {sig_info['action']}")
    
    # =========================================================================
    # 5. INVIO NOTIFICA TELEGRAM
    # =========================================================================
    print("\n📱 [5/5] Invio notifica Telegram...")
    
    try:
        # Calcola rendimento giornaliero
        daily_return = last_row['Returns'] if 'Returns' in last_row else None
        
        # Usa il formato report giornaliero (più completo)
        message = format_daily_report(
            date=last_row.name.strftime('%Y-%m-%d'),
            price=last_row['Close'],
            hmm_probs=curr_probs.tolist(),
            garch_vol=garch_vol_ann,
            regime_label=REGIME_LABELS[last_state],
            signal_type=signal_type,
            trend_prob=trend_p_high,
            daily_return=daily_return
        )
        
        success = send_telegram_alert(message)
        
        if success:
            print("   ✅ Notifica inviata con successo!")
        else:
            print("   ⚠️ Notifica non inviata (controlla credenziali)")
            
    except Exception as e:
        print(f"   ❌ Errore invio notifica: {e}")
    
    # =========================================================================
    # COMPLETAMENTO
    # =========================================================================
    print("\n" + "=" * 60)
    print("🏁 JOB COMPLETATO")
    print("=" * 60)
    
    # Return del segnale per eventuali test
    return {
        'date': last_row.name.strftime('%Y-%m-%d'),
        'signal': signal_type,
        'confidence': confidence,
        'p_high': p_high,
        'garch_vol': garch_vol_ann,
        'regime': REGIME_LABELS[last_state]
    }


def test_run():
    """
    Funzione di test per verificare il funzionamento senza invio Telegram.
    """
    print("\n⚠️ MODALITÀ TEST - Telegram disabilitato\n")
    
    # Temporaneamente disabilita Telegram
    import notifications
    original_send = notifications.send_telegram_alert
    notifications.send_telegram_alert = lambda msg: print(f"[TEST] Messaggio:\n{msg[:500]}...")
    
    result = job()
    
    # Ripristina
    notifications.send_telegram_alert = original_send
    
    return result


if __name__ == "__main__":
    # Check argomenti
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_run()
    else:
        job()
