# notifications.py
import requests
from utils import get_secret
from config import SIGNAL_CONFIG

def format_message(date, price, hmm_probs, garch_vol, regime_label, signal_type, trend_prob):
    """
    Formatta il messaggio per Telegram usando HTML/Markdown supportato.
    """
    signal_info = SIGNAL_CONFIG.get(signal_type, SIGNAL_CONFIG['NEUTRAL'])
    icon = signal_info['icon']
    action = signal_info['action']
    
    # Formattazione barre probabilità (visuale testuale)
    p_low_bar = "🟩" * int(hmm_probs[0] * 5)
    p_med_bar = "🟨" * int(hmm_probs[1] * 5)
    p_high_bar = "🟥" * int(hmm_probs[2] * 5)

    msg = (
        f"<b>KRITERION VOLATILITY ALERT</b> {icon}\n"
        f"📅 <b>Data:</b> {date}\n"
        f"💲 <b>SPY Close:</b> ${price:.2f}\n"
        f"-------------------------------\n"
        f"<b>SEGNALE: {signal_type}</b>\n"
        f"<i>{action}</i>\n"
        f"-------------------------------\n"
        f"<b>🤖 HMM REGIME:</b> {regime_label}\n"
        f"Low Vol:  {hmm_probs[0]*100:.1f}% {p_low_bar}\n"
        f"Med Vol:  {hmm_probs[1]*100:.1f}% {p_med_bar}\n"
        f"High Vol: {hmm_probs[2]*100:.1f}% {p_high_bar}\n"
        f"Trend (5gg): {trend_prob*100:+.1f}%\n"
        f"-------------------------------\n"
        f"<b>📉 GARCH (1-step):</b> {garch_vol*100:.2f}%\n"
        f"-------------------------------\n"
        f"#SPY #Volatility #KriterionQuant"
    )
    return msg

def send_telegram_alert(message):
    """Invia il messaggio al canale Telegram configurato."""
    bot_token = get_secret('TELEGRAM_BOT_TOKEN')
    chat_id = get_secret('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        print("⚠️ Credenziali Telegram mancanti. Messaggio non inviato.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Notifica Telegram inviata con successo.")
            return True
        else:
            print(f"❌ Errore Telegram: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Eccezione durante invio Telegram: {e}")
        return False
