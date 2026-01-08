# notifications.py - Sistema di Notifiche Telegram v2.0
# Kriterion Volatility Monitor

import requests
from utils import get_secret
from config import SIGNAL_CONFIG, REGIME_LABELS

def format_message(date, price, hmm_probs, garch_vol, regime_label, signal_type, trend_prob):
    """
    Formatta il messaggio per Telegram con layout professionale.
    
    Parameters:
    -----------
    date : str
        Data del segnale
    price : float
        Prezzo di chiusura SPY
    hmm_probs : list
        Probabilità [Low, Medium, High]
    garch_vol : float
        Previsione volatilità GARCH
    regime_label : str
        Etichetta regime corrente
    signal_type : str
        Tipo di segnale generato
    trend_prob : float
        Trend P(High Vol) ultimi 5 giorni
        
    Returns:
    --------
    str
        Messaggio formattato HTML
    """
    
    signal_info = SIGNAL_CONFIG.get(signal_type, SIGNAL_CONFIG['NEUTRAL'])
    icon = signal_info['icon']
    action = signal_info['action']
    vix_strategy = signal_info.get('vix_strategy', 'N/A')
    
    # Barre visive probabilità (5 blocchi max)
    def make_bar(prob, emoji):
        blocks = int(prob * 5)
        return emoji * blocks + "⬜" * (5 - blocks)
    
    p_low_bar = make_bar(hmm_probs[0], "🟩")
    p_med_bar = make_bar(hmm_probs[1], "🟨")
    p_high_bar = make_bar(hmm_probs[2], "🟥")
    
    # Indicatore trend
    if trend_prob > 0.10:
        trend_icon = "🔺"
    elif trend_prob < -0.10:
        trend_icon = "🔻"
    else:
        trend_icon = "➡️"
    
    # Calcola confidenza
    confidence = max(hmm_probs) * 100
    
    # Formatta volatilità
    garch_pct = garch_vol * 100
    
    msg = f"""
<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>
<b>📊 KRITERION VOLATILITY ALERT</b>
<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>

📅 <b>Data:</b> {date}
💰 <b>SPY Close:</b> ${price:.2f}

<b>━━━━ SEGNALE ━━━━</b>
{icon} <b>{signal_type}</b>

📌 <i>{action}</i>

<b>━━━━ HMM ANALYSIS ━━━━</b>
🤖 <b>Regime:</b> {regime_label}
🎯 <b>Confidenza:</b> {confidence:.1f}%

<b>Probabilità Regimi:</b>
🟢 Low:    {hmm_probs[0]*100:5.1f}% {p_low_bar}
🟡 Med:    {hmm_probs[1]*100:5.1f}% {p_med_bar}
🔴 High:   {hmm_probs[2]*100:5.1f}% {p_high_bar}

{trend_icon} <b>Trend P(High):</b> {trend_prob*100:+.1f}% (5gg)

<b>━━━━ GARCH FORECAST ━━━━</b>
📉 <b>Vol 1-step:</b> {garch_pct:.2f}%

<b>━━━━ VIX STRATEGY ━━━━</b>
📈 {vix_strategy}

<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>
<i>#SPY #Volatility #KriterionQuant</i>
<i>kriterionquant.com</i>
"""
    
    return msg.strip()


def format_daily_report(date, price, hmm_probs, garch_vol, regime_label, 
                        signal_type, trend_prob, daily_return=None):
    """
    Formatta un report giornaliero più completo.
    
    Parameters:
    -----------
    Stessi di format_message, più:
    daily_return : float, optional
        Rendimento giornaliero
        
    Returns:
    --------
    str
        Report formattato HTML
    """
    
    signal_info = SIGNAL_CONFIG.get(signal_type, SIGNAL_CONFIG['NEUTRAL'])
    icon = signal_info['icon']
    action = signal_info['action']
    description = signal_info.get('description', '')
    vix_strategy = signal_info.get('vix_strategy', 'N/A')
    
    # Calcola confidenza
    confidence = max(hmm_probs) * 100
    
    # Return formatting
    if daily_return is not None:
        ret_str = f"{daily_return*100:+.2f}%"
        ret_icon = "📈" if daily_return >= 0 else "📉"
    else:
        ret_str = "N/A"
        ret_icon = "📊"
    
    # Probabilità dominante
    probs_dict = {'Low': hmm_probs[0], 'Medium': hmm_probs[1], 'High': hmm_probs[2]}
    dominant = max(probs_dict, key=probs_dict.get)
    
    msg = f"""
<b>🌅 KRITERION DAILY REPORT</b>
<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>

📅 {date} | 💰 SPY: ${price:.2f} {ret_icon} {ret_str}

<b>⚡ SEGNALE: {icon} {signal_type}</b>
<i>{description}</i>

<b>📊 HMM Status</b>
├ Regime: {regime_label}
├ Confidenza: {confidence:.1f}%
├ P(Low): {hmm_probs[0]*100:.1f}%
├ P(Med): {hmm_probs[1]*100:.1f}%
├ P(High): {hmm_probs[2]*100:.1f}%
└ Trend: {trend_prob*100:+.1f}%

<b>📉 GARCH Forecast</b>
└ Vol (1-step): {garch_vol*100:.2f}%

<b>💡 Azione</b>
{action}

<b>📈 VIX</b>
{vix_strategy}

<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>
<i>#KriterionQuant #SPY #Volatility</i>
"""
    
    return msg.strip()


def send_telegram_alert(message, parse_mode='HTML'):
    """
    Invia il messaggio al canale Telegram configurato.
    
    Parameters:
    -----------
    message : str
        Messaggio da inviare
    parse_mode : str
        Modalità parsing ('HTML' o 'Markdown')
        
    Returns:
    --------
    bool
        True se invio riuscito, False altrimenti
    """
    
    bot_token = get_secret('TELEGRAM_BOT_TOKEN')
    chat_id = get_secret('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        print("⚠️ Credenziali Telegram mancanti. Messaggio non inviato.")
        print("   Configura TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID nei secrets.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': parse_mode,
        'disable_web_page_preview': True
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print("✅ Notifica Telegram inviata con successo.")
                return True
            else:
                print(f"❌ Errore Telegram API: {result.get('description', 'Unknown error')}")
                return False
        else:
            print(f"❌ Errore HTTP Telegram: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Timeout durante invio Telegram")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Errore connessione Telegram")
        return False
    except Exception as e:
        print(f"❌ Eccezione durante invio Telegram: {e}")
        return False


def send_error_alert(error_message, context=""):
    """
    Invia un alert di errore al canale Telegram.
    
    Parameters:
    -----------
    error_message : str
        Messaggio di errore
    context : str
        Contesto dell'errore
        
    Returns:
    --------
    bool
        True se invio riuscito
    """
    
    msg = f"""
<b>🚨 KRITERION ERROR ALERT</b>
<b>━━━━━━━━━━━━━━━━━━━━━━━━━</b>

⚠️ Si è verificato un errore nel sistema.

<b>Contesto:</b> {context if context else 'N/A'}

<b>Errore:</b>
<code>{error_message[:500]}</code>

<i>Verifica i logs per maggiori dettagli.</i>
"""
    
    return send_telegram_alert(msg)
