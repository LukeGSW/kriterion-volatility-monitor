# app.py - Kriterion Volatility Monitor v2.1
# Dashboard Professionale per Analisi Volatilità SPY con HMM + GARCH (VIX Compatible)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import moduli locali
from data_loader import download_data, calculate_features
from models import train_hmm, get_hmm_states, train_garch
from config import TICKER, HMM_PARAMS, REGIME_COLORS, REGIME_LABELS, SIGNAL_CONFIG, THRESHOLDS
from notifications import send_telegram_alert, format_message

# ============================================================================
# CONFIGURAZIONE PAGINA
# ============================================================================

st.set_page_config(
    page_title="Kriterion Volatility Monitor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rilevamento Modalità (VIX o Equity Standard)
IS_VIX = 'VIX' in TICKER.upper()

# ============================================================================
# CSS CUSTOM PROFESSIONALE
# ============================================================================

st.markdown("""
<style>
    /* Font e colori base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principale */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white !important; /* Forza titolo bianco su sfondo scuro */
    }
    
    .main-header p {
        margin: 8px 0 0 0;
        opacity: 0.85;
        font-size: 0.95rem;
        color: #e0e0e0 !important; /* Forza sottotitolo chiaro */
    }
    
    /* Signal Banner */
    .signal-banner {
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        color: #ffffff !important; /* CORREZIONE: Testo nero su sfondo colorato chiaro */
    }
    
    .signal-banner h2 {
        margin: 0 0 10px 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
        /* Il colore h2 viene impostato inline dinamicamente, quindi ok */
    }
    
    .signal-banner .action {
        font-size: 1.1rem;
        margin: 10px 0;
        color: #ffffff !important; /* CORREZIONE: Forza testo nero */
    }
    
    .signal-banner .metrics {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 15px;
        color: #ffffff !important; /* CORREZIONE: Forza testo nero */
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #007bff;
        color: #212529 !important; /* CORREZIONE: Testo scuro su sfondo bianco */
    }
    
    .metric-card h4 {
        color: #6c757d !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        margin: 0 0 8px 0;
        font-weight: 500;
    }
    
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #212529 !important;
        margin: 0;
    }
    
    .metric-card .sub {
        font-size: 0.85rem;
        color: #6c757d !important;
        margin-top: 5px;
    }
    
    /* Info Boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        color: #212529 !important; /* CORREZIONE: Testo scuro su box chiaro */
    }
    
    .info-box.warning {
        border-color: #ffc107;
        background: #fff9e6;
        color: #555555 !important;
    }
    
    .info-box.danger {
        border-color: #dc3545;
        background: #fff5f5;
        color: #555555 !important;
    }
    
    .info-box.success {
        border-color: #28a745;
        background: #f0fff4;
        color: #555555 !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #212529;
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Probability Bars */
    .prob-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        color: #212529 !important; /* CORREZIONE: Testo scuro su sfondo bianco */
    }
    
    .prob-bar {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        color: #212529 !important;
    }
    
    .prob-bar .label {
        width: 130px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .prob-bar .bar-bg {
        flex: 1;
        height: 28px;
        background: #e9ecef;
        border-radius: 14px;
        overflow: hidden;
        margin: 0 12px;
    }
    
    .prob-bar .bar-fill {
        height: 100%;
        border-radius: 14px;
        transition: width 0.5s ease;
    }
    
    .prob-bar .percentage {
        width: 60px;
        text-align: right;
        font-weight: 600;
        font-size: 1rem;
        color: #212529 !important;
    }
    
    /* Model Comparison Cards */
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 100%;
        color: #212529 !important; /* CORREZIONE: Testo scuro su sfondo bianco */
    }
    
    .model-card h4 {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 3px solid;
    }
    
    .model-card.hmm h4 { border-color: #dc3545; color: #dc3545; }
    .model-card.garch h4 { border-color: #007bff; color: #007bff; }
    
    .model-stat {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #e9ecef;
        color: #212529 !important;
    }
    
    .model-stat:last-child { border-bottom: none; }
    
    /* Footer */
    .footer {
        background: #212529;
        color: white !important; /* Footer resta testo bianco su fondo scuro */
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        font-size: 0.85rem;
    }
    
    .footer a { color: #69b3ff; }
    
    /* Disclaimer */
    .disclaimer {
        background: rgba(255, 193, 7, 0.15);
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-top: 20px;
        font-size: 0.85rem;
        color: #212529 !important; /* CORREZIONE */
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
    ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #a1a1a1; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNZIONI HELPER PER GRAFICI
# ============================================================================

def create_price_regime_chart(df, n_days=252):
    """Grafico prezzo SPY (o Livello VIX) con overlay regimi di volatilità."""
    df_plot = df.tail(n_days).copy()
    
    fig = go.Figure()
    
    # Adattamento Etichette VIX vs Equity
    y_col = 'Close'
    y_label = "Livello VIX" if IS_VIX else "Prezzo ($)"
    title_text = f"📈 {y_label} con Regimi di Volatilità"
    
    # Punti colorati per regime
    for state in range(3):
        mask = df_plot['HMM_State'] == state
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df_plot.index[mask],
                y=df_plot.loc[mask, y_col],
                mode='markers',
                name=REGIME_LABELS[state],
                marker=dict(color=REGIME_COLORS[state], size=5, opacity=0.8),
                hovertemplate='%{x}<br>' + f'{y_label}: ' + '%{y:.2f}<br>' + REGIME_LABELS[state] + '<extra></extra>'
            ))
    
    # Linea prezzo/livello
    fig.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot[y_col],
        mode='lines',
        name=TICKER,
        line=dict(color='#1f77b4', width=1.5),
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        xaxis_title="Data",
        yaxis_title=y_label,
        hovermode='x unified',
        height=450,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=80, b=50)
    )
    
    return fig


def create_probability_chart(df, n_days=252):
    """Grafico stacked area delle probabilità dei regimi."""
    df_plot = df.tail(n_days).copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['P_Low'],
        mode='lines', name='P(Low Vol)', stackgroup='one',
        fillcolor='rgba(40, 167, 69, 0.7)',
        line=dict(color='#28a745', width=0.5),
        hovertemplate='Low: %{y:.1%}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['P_Medium'],
        mode='lines', name='P(Medium Vol)', stackgroup='one',
        fillcolor='rgba(255, 193, 7, 0.7)',
        line=dict(color='#ffc107', width=0.5),
        hovertemplate='Medium: %{y:.1%}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['P_High'],
        mode='lines', name='P(High Vol)', stackgroup='one',
        fillcolor='rgba(220, 53, 69, 0.7)',
        line=dict(color='#dc3545', width=0.5),
        hovertemplate='High: %{y:.1%}<extra></extra>'
    ))
    
    # Linea soglia
    fig.add_hline(y=THRESHOLDS['high_vol'], line_dash="dash", line_color="gray",
                  annotation_text=f"Soglia Risk-Off ({THRESHOLDS['high_vol']*100:.0f}%)")
    
    fig.update_layout(
        title=dict(text="📊 Probabilità Regimi HMM nel Tempo", font=dict(size=16)),
        xaxis_title="Data",
        yaxis_title="Probabilità",
        yaxis=dict(tickformat='.0%', range=[0, 1]),
        hovermode='x unified',
        height=400,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=80, b=50)
    )
    
    return fig


def create_volatility_comparison_chart(df, garch_vol, garch_res, n_days=120):
    """
    Grafico confronto volatilità.
    ADATTATO PER VIX: Se IS_VIX è True, mostriamo il Livello VIX vs la sua Media Mobile
    invece di confrontare "VIX Level" (GK_Vol) con "VVIX" (GARCH), che hanno scale diverse.
    """
    df_plot = df.tail(n_days).copy()
    fig = go.Figure()
    
    if IS_VIX:
        # --- MODALITA' VIX ---
        # Mostriamo il livello VIX (chiusura) e una media mobile per contesto
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['Close'],
            mode='lines',
            name='VIX Level',
            line=dict(color='#212529', width=2),
            hovertemplate='VIX: %{y:.2f}<extra></extra>'
        ))
        
        # Aggiungiamo una SMA 20 per dare contesto al trend
        sma = df_plot['Close'].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=sma,
            mode='lines',
            name='Media Mobile 20gg',
            line=dict(color='#6c757d', width=1, dash='dash'),
            hovertemplate='SMA: %{y:.2f}<extra></extra>'
        ))
        
        title_text = "📉 Livello VIX vs Trend (SMA 20)"
        y_title = "VIX Index"
        
    else:
        # --- MODALITA' EQUITY (SPY) ---
        # 1. Volatilità realizzata (Garman-Klass)
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=df_plot['GK_Vol'] * 100,
            mode='lines',
            name='Realized Vol (GK)',
            line=dict(color='#212529', width=2),
            hovertemplate='Realized: %{y:.2f}%<extra></extra>'
        ))
        
        # 2. GARCH Conditional Volatility
        if garch_res is not None:
            cond_vol = garch_res.conditional_volatility * np.sqrt(252)
            common_idx = df_plot.index.intersection(cond_vol.index)
            garch_plot_series = cond_vol.loc[common_idx]
            
            fig.add_trace(go.Scatter(
                x=garch_plot_series.index,
                y=garch_plot_series.values,
                mode='lines',
                name='GARCH Conditional',
                line=dict(color='#007bff', width=2, dash='solid'),
                hovertemplate='GARCH: %{y:.2f}%<extra></extra>'
            ))

        # 3. Forecast Futuro
        last_date = df_plot.index[-1]
        fig.add_trace(go.Scatter(
            x=[last_date + timedelta(days=1)],
            y=[garch_vol * 100],
            mode='markers+text',
            name='Forecast (1-step)',
            marker=dict(color='#dc3545', size=10, symbol='diamond'),
            text=[f"{garch_vol*100:.1f}%"],
            textposition="top center",
            showlegend=False
        ))
        
        # Media storica
        avg_vol = df_plot['GK_Vol'].mean() * 100
        fig.add_hline(y=avg_vol, line_dash="dash", line_color="#6c757d",
                    annotation_text=f"Media: {avg_vol:.2f}%")
        
        title_text = "📉 Volatilità Realizzata vs GARCH Dinamico"
        y_title = "Volatilità Annualizzata (%)"
    
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=16)),
        xaxis_title="Data",
        yaxis_title=y_title,
        hovermode='x unified',
        height=400,
        template='plotly_white',
        margin=dict(l=50, r=30, t=80, b=50),
        legend=dict(orientation='h', y=1.02)
    )
    
    return fig


def create_regime_distribution_chart(df):
    """Grafico distribuzione volatilità per regime (violin plot o histogram)."""
    fig = go.Figure()
    
    # Adattamento etichette
    x_label = "Livello VIX" if IS_VIX else "Volatilità Annualizzata (%)"
    
    for state in range(3):
        mask = df['HMM_State'] == state
        
        # Se IS_VIX è True, GK_Vol nel data_loader è stato impostato come Close/100.
        # Moltiplicando per 100 riotteniamo il livello del VIX.
        # Se IS_VIX è False, GK_Vol è la volatilità decimale. Moltiplicando per 100 otteniamo la %.
        vol_data = df.loc[mask, 'GK_Vol'] * 100
        
        fig.add_trace(go.Histogram(
            x=vol_data,
            name=REGIME_LABELS[state],
            marker_color=REGIME_COLORS[state],
            opacity=0.7,
            nbinsx=40
        ))
    
    fig.update_layout(
        title=dict(text=f"📊 Distribuzione {x_label} per Regime", font=dict(size=16)),
        xaxis_title=x_label,
        yaxis_title="Frequenza",
        barmode='overlay',
        height=350,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=80, b=50)
    )
    
    return fig


def create_combined_dashboard_chart(df, garch_vol, garch_res, n_days=90):
    """Grafico combinato con prezzo, volatilità dinamica e probabilità."""
    df_plot = df.tail(n_days).copy()
    
    # Titoli dinamici
    t1 = "Livello VIX & Regimi" if IS_VIX else "Prezzo SPY & Regimi"
    t2 = "VIX Trend (Level)" if IS_VIX else "Volatilità: Realizzata vs GARCH"
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(t1, t2, "Probabilità P(High Vol)")
    )
    
    # --- ROW 1: Asset Principale (Prezzo o Livello VIX) ---
    y_col = 'Close'
    for state in range(3):
        mask = df_plot['HMM_State'] == state
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=df_plot.index[mask],
                y=df_plot.loc[mask, y_col],
                mode='markers',
                name=REGIME_LABELS[state],
                marker=dict(color=REGIME_COLORS[state], size=5),
                showlegend=True
            ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot[y_col],
        mode='lines', line=dict(color='#1f77b4', width=1),
        showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    
    # --- ROW 2: Volatilità ---
    if IS_VIX:
        # Se siamo in modalità VIX, nel secondo pannello mostriamo il livello VIX pulito
        # per enfatizzare i picchi, senza confonderlo con il GARCH (che sarebbe VVIX)
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['Close'],
            mode='lines', name='VIX Level',
            line=dict(color='#212529', width=1.5),
            showlegend=False
        ), row=2, col=1)
    else:
        # Modalità Equity Standard
        # A. Volatilità Realizzata
        fig.add_trace(go.Scatter(
            x=df_plot.index, y=df_plot['GK_Vol'] * 100,
            mode='lines', name='Realized Vol',
            line=dict(color='#212529', width=1.5),
            showlegend=False
        ), row=2, col=1)
        
        # B. GARCH Dinamico
        if garch_res is not None:
            cond_vol = garch_res.conditional_volatility * np.sqrt(252)
            common_idx = df_plot.index.intersection(cond_vol.index)
            garch_plot = cond_vol.loc[common_idx]
            
            fig.add_trace(go.Scatter(
                x=garch_plot.index,
                y=garch_plot.values,
                mode='lines',
                name='GARCH Dynamic',
                line=dict(color='#007bff', width=1.5), # Linea blu continua
                showlegend=False
            ), row=2, col=1)

        # C. Forecast puntuale
        last_date = df_plot.index[-1]
        fig.add_trace(go.Scatter(
            x=[last_date + timedelta(days=1)],
            y=[garch_vol * 100],
            mode='markers',
            marker=dict(color='#dc3545', size=6, symbol='diamond'),
            name='Forecast',
            showlegend=False
        ), row=2, col=1)
    
    # --- ROW 3: Probabilità HMM ---
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['P_High'],
        mode='lines', name='P(High)',
        line=dict(color='#dc3545', width=2),
        fill='tozeroy', fillcolor='rgba(220, 53, 69, 0.2)',
        showlegend=False
    ), row=3, col=1)
    
    fig.add_hline(y=THRESHOLDS['high_vol'], line_dash="dash", line_color="gray", row=3, col=1)
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=50, b=50)
    )
    
    y1_title = "Livello" if IS_VIX else "Prezzo ($)"
    y2_title = "Livello" if IS_VIX else "Vol (%)"
    
    fig.update_yaxes(title_text=y1_title, row=1, col=1)
    fig.update_yaxes(title_text=y2_title, row=2, col=1)
    fig.update_yaxes(title_text="Prob", tickformat='.0%', range=[0, 1.1], row=3, col=1)
    
    return fig


def calculate_regime_stats(df):
    """Calcola statistiche per ogni regime."""
    stats = []
    
    for state in range(3):
        mask = df['HMM_State'] == state
        regime_data = df[mask]
        
        if len(regime_data) == 0:
            continue
        
        # Calcola durate
        state_changes = np.diff(np.concatenate([[0], mask.astype(int).values, [0]]))
        starts = np.where(state_changes == 1)[0]
        ends = np.where(state_changes == -1)[0]
        durations = ends - starts
        
        # Adattamento etichette per VIX
        if IS_VIX:
            # Per il VIX, GK_Vol è il livello/100. Quindi *100 da il livello.
            vol_mean = f"{regime_data['GK_Vol'].mean()*100:.2f}"
            vol_std = f"{regime_data['GK_Vol'].std()*100:.2f}"
            vol_label = "Livello VIX Medio"
        else:
            vol_mean = f"{regime_data['GK_Vol'].mean()*100:.2f}%"
            vol_std = f"{regime_data['GK_Vol'].std()*100:.2f}%"
            vol_label = "Vol Media"
            
        stats.append({
            'Regime': REGIME_LABELS[state],
            'Giorni': len(regime_data),
            'Frequenza': f"{len(regime_data)/len(df)*100:.1f}%",
            vol_label: vol_mean,
            'Dev Std': vol_std,
            'Durata Media': f"{durations.mean():.1f} gg" if len(durations) > 0 else "N/A",
            'Num Periodi': len(durations)
        })
    
    return pd.DataFrame(stats)


# ============================================================================
# FUNZIONE PRINCIPALE
# ============================================================================

def main():
    # --- HEADER ---
    st.markdown("""
    <div class="main-header">
        <h1>📉 Kriterion Quant - Volatility Regime Monitor</h1>
        <p>Sistema di analisi e previsione della volatilità basato su Hidden Markov Model e GARCH</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- SIDEBAR ---
    with st.sidebar:
        # Placeholder immagine, puoi rimuoverlo o metterne uno vero
        # st.image("https://raw.githubusercontent.com/your-repo/logo.png", width=150)
        st.markdown("### ⚙️ Configurazione")
        
        st.markdown(f"""
        **Ticker:** {TICKER} ({'Modo VIX' if IS_VIX else 'Modo Equity'})  
        **Modello HMM:** {HMM_PARAMS['n_states']} Stati  
        **GARCH:** (1,1)  
        """)
        
        st.markdown("---")
        
        # Selezione periodo visualizzazione
        chart_period = st.selectbox(
            "📅 Periodo Grafici",
            options=[90, 180, 252, 504],
            format_func=lambda x: f"{x} giorni (~{x//21} mesi)",
            index=2
        )
        
        st.markdown("---")
        
        st.markdown("### 📚 Legenda Segnali")
        for sig_name, sig_conf in SIGNAL_CONFIG.items():
            st.markdown(f"{sig_conf['icon']} **{sig_name}**")
        
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[📊 Kriterion Quant](https://kriterionquant.com)")
        
    
    # --- CARICAMENTO DATI ---
    with st.spinner('🔄 Caricamento dati e training modelli...'):
        try:
            df_raw = download_data()
            df = calculate_features(df_raw)
        except Exception as e:
            st.error(f"❌ Errore nel caricamento dati: {e}")
            st.stop()
    
    # --- TRAINING MODELLI ---
    model_hmm, scaler_hmm, state_mapping = train_hmm(df)
    states, posteriors = get_hmm_states(df, model_hmm, scaler_hmm, state_mapping)
    
    df['HMM_State'] = states
    df['P_Low'] = posteriors[:, 0]
    df['P_Medium'] = posteriors[:, 1]
    df['P_High'] = posteriors[:, 2]
    
    # GARCH: Calcoliamo sempre, ma interpretiamo diversamente
    garch_vol_ann, garch_res = train_garch(df)
    
    # --- CALCOLO SEGNALE ---
    last_row = df.iloc[-1]
    p_high = last_row['P_High']
    p_low = last_row['P_Low']
    p_medium = last_row['P_Medium']
    
    # Trend P(High) ultimi 5 giorni
    trend_p_high = df['P_High'].diff(THRESHOLDS['trend_window']).iloc[-1]
    
    # Confidenza (probabilità massima)
    confidence = max(p_high, p_medium, p_low)
    
    # Logica generazione segnale
    signal_type = "NEUTRAL"
    
    if p_high > THRESHOLDS['high_vol']:
        signal_type = "RISK_OFF"
        
        # Logica STRONG_RISK_OFF differenziata
        if IS_VIX:
            # Se siamo sul VIX, STRONG se siamo nel top 15% dei livelli storici
            # (Il GARCH qui sarebbe la VVIX, utile ma diversa)
            if last_row['Close'] > df['Close'].quantile(0.85):
                signal_type = "STRONG_RISK_OFF"
        else:
            # Logica classica Equity: GARCH forecast alto
            if garch_vol_ann > df['GK_Vol'].quantile(THRESHOLDS['garch_percentile']):
                signal_type = "STRONG_RISK_OFF"
            
    elif trend_p_high > THRESHOLDS['alert_change']:
        signal_type = "ALERT"
    elif p_low > THRESHOLDS['low_vol']:
        signal_type = "RISK_ON"
    
    # Check confidenza bassa -> WATCH (allineato con run_daily_check.py)
    if confidence < THRESHOLDS.get('confidence_min', 0.70) and signal_type == "NEUTRAL":
        signal_type = "WATCH"
    
    sig_conf = SIGNAL_CONFIG.get(signal_type, SIGNAL_CONFIG['NEUTRAL'])
    
    # --- SIGNAL BANNER ---
    st.markdown(f"""
    <div class="signal-banner" style="background-color: {sig_conf['color']}20; border: 3px solid {sig_conf['color']};">
        <h2 style="color: {sig_conf['color']};">{sig_conf['icon']} {signal_type}</h2>
        <p class="action"><strong>Azione Consigliata:</strong> {sig_conf['action']}</p>
        <p class="metrics">
            📊 P(High Vol): <strong>{p_high*100:.1f}%</strong> | 
            📈 Trend (5gg): <strong>{trend_p_high*100:+.1f}%</strong> | 
            🎯 Confidenza: <strong>{confidence*100:.1f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- KPI CARDS ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Card 1: Prezzo
    with col1:
        daily_return = last_row['Returns'] * 100
        st.metric(
            label=f"💰 {TICKER}",
            value=f"${last_row['Close']:.2f}",
            delta=f"{daily_return:.2f}%"
        )
    
    # Card 2: Volatilità / Livello
    with col2:
        if IS_VIX:
            st.metric(
                label="📊 Livello VIX",
                value=f"{last_row['Close']:.2f}",
                delta=None,
                help="Livello di chiusura indice VIX"
            )
        else:
            st.metric(
                label="📊 Volatilità Attuale",
                value=f"{last_row['GK_Vol']*100:.2f}%",
                delta=None,
                help="Volatilità Garman-Klass annualizzata"
            )
    
    # Card 3: GARCH / VVIX
    with col3:
        if IS_VIX:
            # Il GARCH sul VIX è la "Volatilità del VIX" (proxy VVIX)
            st.metric(
                label="🔮 GARCH (VVIX Proxy)",
                value=f"{garch_vol_ann*100:.1f}%",
                delta=None,
                help="Volatilità della Volatilità stimata dal GARCH"
            )
        else:
            garch_delta = (garch_vol_ann - last_row['GK_Vol']) * 100
            st.metric(
                label="🔮 GARCH Forecast",
                value=f"{garch_vol_ann*100:.2f}%",
                delta=f"{garch_delta:+.2f}%",
                help="Previsione volatilità 1-step ahead GARCH(1,1)"
            )
    
    with col4:
        st.metric(
            label="🤖 Regime HMM",
            value=REGIME_LABELS[last_row['HMM_State']],
            delta=None
        )
    
    with col5:
        st.metric(
            label="📅 Ultima Data",
            value=last_row.name.strftime('%Y-%m-%d'),
            delta=None
        )
    
    # --- PROBABILITY BARS ---
    st.markdown("### 📊 Probabilità Regimi HMM")
    
    col_prob1, col_prob2 = st.columns([2, 1])
    
    with col_prob1:
        st.markdown(f"""
        <div class="prob-container">
            <div class="prob-bar">
                <span class="label">🟢 Low Vol</span>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: {p_low*100}%; background: #28a745;"></div>
                </div>
                <span class="percentage">{p_low*100:.1f}%</span>
            </div>
            <div class="prob-bar">
                <span class="label">🟡 Medium Vol</span>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: {p_medium*100}%; background: #ffc107;"></div>
                </div>
                <span class="percentage">{p_medium*100:.1f}%</span>
            </div>
            <div class="prob-bar">
                <span class="label">🔴 High Vol</span>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: {p_high*100}%; background: #dc3545;"></div>
                </div>
                <span class="percentage">{p_high*100:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_prob2:
        # Info box interpretazione
        if p_high > 0.5:
            st.markdown("""
            <div class="info-box danger">
                <strong>⚠️ Attenzione</strong><br>
                Alta probabilità di regime High Volatility. 
                Considerare strategie di copertura o riduzione esposizione.
            </div>
            """, unsafe_allow_html=True)
        elif p_low > 0.5:
            st.markdown("""
            <div class="info-box success">
                <strong>✅ Mercato Tranquillo</strong><br>
                Alta probabilità di regime Low Volatility. 
                Condizioni favorevoli per strategie direzionali.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box warning">
                <strong>⏳ Regime di Transizione</strong><br>
                Nessun regime dominante. Il mercato potrebbe essere 
                in fase di transizione. Monitorare attentamente.
            </div>
            """, unsafe_allow_html=True)
    
    # --- TABS PRINCIPALI ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Dashboard", 
        "📊 Analisi Regimi", 
        "📉 Volatilità", 
        "📚 Metodologia",
        "🛠 Test & Debug"
    ])
    
    # =========================================================================
    # TAB 1: DASHBOARD
    # =========================================================================
    with tab1:
        st.markdown("#### 🎯 Analisi Combinata")
        
        st.markdown("""
        <div class="info-box">
            <strong>📖 Come leggere questo grafico:</strong><br>
            Il grafico mostra tre pannelli sincronizzati: (1) Asset (SPY o VIX) con punti colorati in base al regime identificato,
            (2) Volatilità/Livello storico, (3) Probabilità di essere in regime High Volatility.
            Quando P(High Vol) supera la soglia del 60%, il sistema genera un segnale RISK-OFF.
        </div>
        """, unsafe_allow_html=True)
        
        fig_combined = create_combined_dashboard_chart(df, garch_vol_ann, garch_res, n_days=chart_period)
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Grafico prezzo con regimi
        st.markdown(f"#### 📈 Storico {'Livello' if IS_VIX else 'Prezzo'} e Regimi")
        fig_price = create_price_regime_chart(df, n_days=chart_period)
        st.plotly_chart(fig_price, use_container_width=True)
    
    # =========================================================================
    # TAB 2: ANALISI REGIMI
    # =========================================================================
    with tab2:
        st.markdown("#### 📊 Probabilità Regimi nel Tempo")
        
        st.markdown("""
        <div class="info-box">
            <strong>📖 Interpretazione:</strong><br>
            Questo grafico mostra l'evoluzione delle probabilità dei tre regimi nel tempo.
            Le aree colorate rappresentano la probabilità stimata dall'HMM di trovarsi in ciascun regime.
            La linea tratteggiata indica la soglia del 60% per il segnale RISK-OFF.
        </div>
        """, unsafe_allow_html=True)
        
        fig_probs = create_probability_chart(df, n_days=chart_period)
        st.plotly_chart(fig_probs, use_container_width=True)
        
        # Statistiche regimi
        st.markdown("#### 📋 Statistiche Regimi")
        
        regime_stats = calculate_regime_stats(df)
        
        col_stats1, col_stats2 = st.columns([2, 1])
        
        with col_stats1:
            st.dataframe(
                regime_stats,
                use_container_width=True,
                hide_index=True
            )
        
        with col_stats2:
            st.markdown("""
            <div class="info-box">
                <strong>📖 Note:</strong><br>
                • <strong>Durata Media:</strong> giorni medi di permanenza nel regime<br>
                • <strong>Vol/Livello Medio:</strong> valore medio nel regime<br>
                • <strong>Frequenza:</strong> percentuale di tempo trascorso nel regime
            </div>
            """, unsafe_allow_html=True)
        
        # Distribuzione volatilità per regime
        st.markdown("#### 📊 Distribuzione Valori per Regime")
        fig_dist = create_regime_distribution_chart(df)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # =========================================================================
    # TAB 3: VOLATILITÀ
    # =========================================================================
    with tab3:
        st.markdown(f"#### 📉 {'Analisi Livello VIX' if IS_VIX else 'Confronto Volatilità Realizzata vs GARCH'}")
        
        st.markdown("""
        <div class="info-box">
            <strong>📖 Analisi Volatilità:</strong><br>
            In modalità VIX, questo grafico mostra l'andamento dell'indice di paura rispetto alla sua media mobile.
            In modalità Equity, mostra la volatilità realizzata (storica) confrontata con la previsione GARCH (dinamica).
        </div>
        """, unsafe_allow_html=True)
        
        fig_vol = create_volatility_comparison_chart(df, garch_vol_ann, garch_res, n_days=chart_period)
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # Metriche GARCH
        col_garch1, col_garch2, col_garch3 = st.columns(3)
        
        with col_garch1:
            st.metric(
                "GARCH Forecast (1-step)" if not IS_VIX else "VVIX Est (GARCH)",
                f"{garch_vol_ann*100:.2f}%",
                help="Previsione volatilità per domani"
            )
        
        with col_garch2:
            avg_vol = df['GK_Vol'].mean() * 100
            st.metric(
                "Media Storica",
                f"{avg_vol:.2f}{'%' if not IS_VIX else ''}",
                help="Valore medio nel periodo"
            )
        
        with col_garch3:
            percentile_rank = (df['GK_Vol'] < last_row['GK_Vol']).mean() * 100
            st.metric(
                "Percentile Attuale",
                f"{percentile_rank:.0f}°",
                help="Posizione del valore corrente rispetto allo storico"
            )
        
        # Statistiche volatilità
        st.markdown("#### 📊 Statistiche Dettagliate")
        
        vol_stats = df['GK_Vol'].describe() * 100
        vol_df = pd.DataFrame({
            'Statistica': ['Media', 'Std Dev', 'Min', '25%', '50%', '75%', 'Max'],
            'Valore': [f"{vol_stats['mean']:.2f}", f"{vol_stats['std']:.2f}",
                      f"{vol_stats['min']:.2f}", f"{vol_stats['25%']:.2f}",
                      f"{vol_stats['50%']:.2f}", f"{vol_stats['75%']:.2f}",
                      f"{vol_stats['max']:.2f}"]
        })
        
        st.dataframe(vol_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # TAB 4: METODOLOGIA
    # =========================================================================
    with tab4:
        st.markdown("## 📚 Metodologia")
        
        st.markdown("""
        ### 🤖 Hidden Markov Model (HMM)
        
        L'HMM è un modello probabilistico che assume l'esistenza di **stati nascosti** (non osservabili direttamente)
        che governano il comportamento delle variabili osservate. Nel nostro caso:
        
        - **Stati nascosti:** 3 regimi di volatilità (Low, Medium, High)
        - **Variabile osservata:** Volatilità Garman-Klass (o Log-VIX)
        - **Output:** Probabilità di trovarsi in ciascun regime
        
        **Perché HMM per la volatilità?**
        - La volatilità presenta **clustering**: periodi di alta volatilità tendono a raggrupparsi
        - Esistono **cambi di regime** strutturali (crisi, normalità, euforia)
        - L'HMM cattura la **persistenza** dei regimi tramite la matrice di transizione
        
        ---
        
        ### 📉 GARCH(1,1)
        
        Il modello GARCH (Generalized Autoregressive Conditional Heteroskedasticity) modella la varianza condizionale:
        
        ```
        σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁
        ```
        
        Dove:
        - **ω (omega):** costante base
        - **α (alpha):** impatto degli shock recenti
        - **β (beta):** persistenza della volatilità
        - **α + β:** persistenza totale (tipicamente ~0.95 per equity)
        
        ---
        
        ### 🎯 Logica dei Segnali
        
        | Segnale | Condizione | Azione |
        |---------|------------|--------|
        | 🟢 RISK_ON | P(Low Vol) > 60% | Esposizione piena |
        | 🟡 NEUTRAL | Nessuna condizione estrema | Allocazione standard |
        | 🟠 ALERT | P(High Vol) in aumento >15% in 5gg | Preparare coperture |
        | 🔴 RISK_OFF | P(High Vol) > 60% | Ridurre esposizione |
        | 🔴🔴 STRONG_RISK_OFF | P(High Vol) > 60% AND (GARCH alto o VIX > 85° pct) | Copertura aggressiva |
        
        ---
        
        ### 📊 Volatilità Garman-Klass
        
        Stimatore della volatilità basato su prezzi OHLC, più efficiente del semplice range:
        
        ```
        GK = 0.5·ln(H/L)² - (2·ln(2)-1)·ln(C/O)²
        ```
        
        **Vantaggi:**
        - Usa tutta l'informazione OHLC
        - Più efficiente dello stimatore Close-to-Close
        - Robusto per dati giornalieri
        
        ---
        
        ### ⚠️ Limitazioni
        
        1. **HMM identifica regimi in modo contemporaneo**, non predittivo
        2. **GARCH assume stazionarietà** che può non valere durante crisi
        3. **I segnali non sono raccomandazioni di investimento**
        4. **Le performance passate non garantiscono risultati futuri**
        """)
        
        st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ Disclaimer:</strong><br>
            Questo strumento è fornito a scopo educativo e di ricerca. 
            I segnali generati non costituiscono consulenza finanziaria né raccomandazioni di investimento. 
            Le performance passate non sono indicative di risultati futuri. 
            L'utente è responsabile delle proprie decisioni di investimento.
        </div>
        """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 5: TEST & DEBUG
    # =========================================================================
    with tab5:
        st.markdown("### 🛠 Test & Debug")
        
        col_test1, col_test2 = st.columns(2)
        
        with col_test1:
            st.markdown("#### 📱 Test Notifica Telegram")
            st.info("Verifica che il bot Telegram sia configurato correttamente.")
            
            if st.button("📤 Invia Segnale Test", type="primary"):
                msg = format_message(
                    date=last_row.name.strftime('%Y-%m-%d'),
                    price=last_row['Close'],
                    hmm_probs=[p_low, p_medium, p_high],
                    garch_vol=garch_vol_ann,
                    regime_label=REGIME_LABELS[last_row['HMM_State']],
                    signal_type=signal_type,
                    trend_prob=trend_p_high
                )
                
                with st.spinner("Invio in corso..."):
                    success = send_telegram_alert(msg)
                
                if success:
                    st.success("✅ Messaggio inviato con successo!")
                else:
                    st.error("❌ Errore nell'invio. Verifica le credenziali.")
        
        with col_test2:
            st.markdown("#### 📊 Stato Sistema")
            
            st.markdown(f"""
            | Parametro | Valore |
            |-----------|--------|
            | Dati caricati | {len(df):,} righe |
            | Ultima data | {df.index[-1].strftime('%Y-%m-%d')} |
            | Prima data | {df.index[0].strftime('%Y-%m-%d')} |
            | HMM Stati | {HMM_PARAMS['n_states']} |
            | GARCH | (1,1) |
            """)
        
        # Dati raw
        st.markdown("#### 📋 Ultimi Dati")
        
        display_cols = ['Close', 'Returns', 'GK_Vol', 'HMM_State', 'P_Low', 'P_Medium', 'P_High']
        st.dataframe(
            df[display_cols].tail(50).sort_index(ascending=False).round(4),
            use_container_width=True
        )
    
    # --- FOOTER ---
    st.markdown("""
    <div class="footer">
        <strong>Kriterion Quant - Volatility Regime Monitor</strong><br>
        Sviluppato per analisi e ricerca quantitativa | 
        <a href="https://kriterionquant.com">kriterionquant.com</a><br>
        <small>Dati: EODHD API | Modelli: HMM + GARCH | Framework: Streamlit</small>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
