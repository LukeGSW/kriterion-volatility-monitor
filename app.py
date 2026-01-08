# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import moduli locali
from data_loader import download_data, calculate_features
from models import train_hmm, get_hmm_states, train_garch
from config import TICKER, HMM_PARAMS, REGIME_COLORS, REGIME_LABELS, SIGNAL_CONFIG, THRESHOLDS
from notifications import send_telegram_alert, format_message

# Configurazione Pagina
st.set_page_config(
    page_title="Kriterion Volatility Monitor",
    page_icon="📉",
    layout="wide"
)

# --- CSS Custom per stile "Kriterion" ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;}
    .signal-banner {padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📉 Kriterion Quant - Volatility Regime Monitor")
    st.markdown(f"**Ticker:** {TICKER} | **Modello:** HMM ({HMM_PARAMS['n_states']} Stati) + GARCH(1,1)")

    # 1. Caricamento Dati
    with st.spinner('Scaricamento dati da EODHD e calcolo features...'):
        try:
            df_raw = download_data()
            df = calculate_features(df_raw)
        except Exception as e:
            st.error(f"Errore nel caricamento dati: {e}")
            st.stop()

    # 2. Training Modelli (Cached functions would be better here for speed, but fit is fast enough)
    model_hmm, scaler_hmm, state_mapping = train_hmm(df)
    states, posteriors = get_hmm_states(df, model_hmm, scaler_hmm, state_mapping)
    
    # Aggiungi risultati al DF per visualizzazione
    df['HMM_State'] = states
    df['P_Low'] = posteriors[:, 0]
    df['P_Medium'] = posteriors[:, 1]
    df['P_High'] = posteriors[:, 2]
    
    # GARCH Forecast
    garch_vol_ann, garch_res = train_garch(df)

    # 3. Logica Segnali (Consensus)
    last_row = df.iloc[-1]
    p_high = last_row['P_High']
    p_low = last_row['P_Low']
    
    # Trend P(High) ultimi 5 giorni
    trend_p_high = df['P_High'].diff(THRESHOLDS['trend_window']).iloc[-1]
    
    # Definizione Segnale
    signal_type = "NEUTRAL"
    if p_high > THRESHOLDS['high_vol']:
        signal_type = "RISK_OFF"
        # Se anche GARCH è alto (sopra media storica o soglia fissa), rafforza il segnale
        if garch_vol_ann > df['GK_Vol'].quantile(THRESHOLDS['garch_percentile']):
            signal_type = "STRONG_RISK_OFF"
    elif trend_p_high > THRESHOLDS['alert_change']:
        signal_type = "ALERT"
    elif p_low > THRESHOLDS['low_vol']:
        signal_type = "RISK_ON"

    sig_conf = SIGNAL_CONFIG.get(signal_type, SIGNAL_CONFIG['NEUTRAL'])

    # --- UI: KPI BANNER ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("SPY Price", f"${last_row['Close']:.2f}", f"{last_row['Returns']*100:.2f}%")
    with col2:
        st.metric("Current Vol (GK Annual)", f"{last_row['GK_Vol']*100:.2f}%")
    with col3:
        st.metric("GARCH Forecast (Next Day)", f"{garch_vol_ann*100:.2f}%")
    with col4:
        st.metric("HMM Regime", REGIME_LABELS[last_row['HMM_State']])

    # --- UI: SIGNAL BOX ---
    st.markdown(f"""
    <div class="signal-banner" style="background-color: {sig_conf['color']}30; border: 2px solid {sig_conf['color']};">
        <h2 style="color: {sig_conf['color']}; margin:0;">{sig_conf['icon']} {signal_type}</h2>
        <p style="margin:0;"><strong>Azione:</strong> {sig_conf['action']}</p>
        <p style="font-size: 0.8em; margin-top:5px;">P(High Vol): {p_high*100:.1f}% | Trend: {trend_p_high*100:+.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Grafica", "🛠 Debug & Test", "📋 Dati"])

    with tab1:
        # GRAFICO 1: Prezzo e Regimi
        fig_price = go.Figure()
        
        # Aggiunge punti colorati per regime
        for state in range(3):
            mask = df['HMM_State'] == state
            fig_price.add_trace(go.Scatter(
                x=df.index[mask], y=df.loc[mask, 'Close'],
                mode='markers', name=REGIME_LABELS[state],
                marker=dict(color=REGIME_COLORS[state], size=4)
            ))
            
        fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', line=dict(color='gray', width=1), name='SPY'))
        fig_price.update_layout(title="SPY Price & HMM Volatility Regimes", height=500, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)

        # GRAFICO 2: Probabilità
        fig_probs = go.Figure()
        fig_probs.add_trace(go.Scatter(x=df.index, y=df['P_Low'], stackgroup='one', name='Low Vol', line=dict(color=REGIME_COLORS[0], width=0)))
        fig_probs.add_trace(go.Scatter(x=df.index, y=df['P_Medium'], stackgroup='one', name='Medium Vol', line=dict(color=REGIME_COLORS[1], width=0)))
        fig_probs.add_trace(go.Scatter(x=df.index, y=df['P_High'], stackgroup='one', name='High Vol', line=dict(color=REGIME_COLORS[2], width=0)))
        fig_probs.update_layout(title="Regime Probabilities (Stacked)", height=350, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_probs, use_container_width=True)

    with tab2:
        st.subheader("Test Notifiche")
        st.info("Usa questo pulsante per verificare che il bot Telegram funzioni correttamente.")
        
        if st.button("Invia Segnale Test a Telegram"):
            msg = format_message(
                date=last_row.name.strftime('%Y-%m-%d'),
                price=last_row['Close'],
                hmm_probs=[p_low, last_row['P_Medium'], p_high],
                garch_vol=garch_vol_ann,
                regime_label=REGIME_LABELS[last_row['HMM_State']],
                signal_type=signal_type,
                trend_prob=trend_p_high
            )
            success = send_telegram_alert(msg)
            if success:
                st.success("Messaggio Inviato!")
            else:
                st.error("Errore invio. Controlla i logs o le chiavi.")

    with tab3:
        st.dataframe(df.tail(100).sort_index(ascending=False))

if __name__ == "__main__":
    main()
