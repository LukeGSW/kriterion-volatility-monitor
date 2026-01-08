

```markdown
# 📉 Kriterion Volatility Monitor v2.0

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**Kriterion Volatility Monitor** è una piattaforma di analisi quantitativa avanzata progettata per monitorare, classificare e prevedere i regimi di volatilità dell'S&P 500 (SPY).

Il sistema combina modelli **Hidden Markov (HMM)** per la classificazione dei regimi di mercato e modelli **GARCH(1,1)** per la previsione della volatilità a breve termine, offrendo segnali operativi di Risk Management (Risk-On / Risk-Off) sia tramite una dashboard interattiva che via notifiche Telegram automatizzate.

---

## 🚀 Funzionalità Principali

### 📊 Dashboard Interattiva (Streamlit)
- **Analisi Multi-Modello**: Visualizzazione combinata di Prezzo, Regimi HMM e Forecast GARCH.
- **Probabilità Regimi**: Grafico dell'evoluzione delle probabilità (Low, Medium, High Volatility).
- **KPI Cards**: Metriche in tempo reale su volatilità realizzata (Garman-Klass) vs implicita/prevista.
- **Segnali Operativi**: Banner dinamici con indicazioni di action (es. "Copertura Aggressiva", "Monitorare").
- **Backtest Visivo**: Storico dei cambi di regime sovrapposto al grafico dei prezzi.

### 🤖 Automazione e Alerting (GitHub Actions)
- **Daily Check**: Script automatizzato che gira ogni giorno alla chiusura di Wall Street (21:30 UTC).
- **Telegram Bot**: Invio di report giornalieri direttamente su Telegram con:
  - Stato del Regime HMM e confidenza.
  - Trend della probabilità di crash/alta volatilità.
  - Previsione GARCH 1-step ahead.
  - Segnale operativo sintetico.

---

## 🧠 Modelli Quantitativi

Il cuore del sistema si basa su due approcci econometrici complementari:

1.  **Hidden Markov Model (HMM)**
    * **Obiettivo**: Identificare lo "stato nascosto" del mercato (Latent State).
    * **Configurazione**: 3 stati Gaussiani (Low, Medium, High Volatility) addestrati sulla volatilità Garman-Klass.
    * **Output**: Matrice di probabilità che indica in quale regime ci troviamo attualmente.

2.  **GARCH(1,1)**
    * **Obiettivo**: Catturare il clustering di volatilità e la "memoria" degli shock di prezzo.
    * **Utilizzo**: Validazione del segnale HMM. Un segnale *STRONG_RISK_OFF* viene generato solo se l'HMM indica alta probabilità di regime "High" E il GARCH prevede un picco di volatilità sopra il 75° percentile storico.

---

## 🛠️ Installazione e Setup Locale

### Prerequisiti
- Python 3.9+
- Pip
- Un account [EODHD](https://eodhd.com/) per i dati finanziari.

### 1. Clona il repository
```bash
git clone [https://github.com/tuo-username/kriterion-volatility-monitor.git](https://github.com/tuo-username/kriterion-volatility-monitor.git)
cd kriterion-volatility-monitor

```

### 2. Crea un ambiente virtuale

```bash
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt

```

### 4. Configurazione Secrets

Il sistema richiede delle chiavi API per funzionare.
Crea un file `.streamlit/secrets.toml` per l'uso locale:

```toml
# .streamlit/secrets.toml
EODHD_API_KEY = "la_tua_api_key_eodhd"
TELEGRAM_BOT_TOKEN = "il_tuo_bot_token"
TELEGRAM_CHAT_ID = "il_tuo_chat_id"

```

### 5. Avvia la Dashboard

```bash
streamlit run app.py

```

---

## ⚙️ Configurazione Automazione (GitHub Actions)

Per abilitare il monitoraggio giornaliero automatico e le notifiche Telegram tramite GitHub Actions:

1. Vai nelle **Settings** del tuo repository su GitHub.
2. Naviga in **Security** > **Secrets and variables** > **Actions**.
3. Aggiungi i seguenti **Repository secrets**:
* `EODHD_API_KEY`: La tua chiave API EOD Historical Data.
* `TELEGRAM_BOT_TOKEN`: Il token del bot creato con @BotFather.
* `TELEGRAM_CHAT_ID`: L'ID della chat o del canale dove ricevere gli alert.



Il workflow è definito in `.github/workflows/main.yml` ed è programmato per eseguire `run_daily_check.py` dal lunedì al venerdì alle 21:30 UTC.

---

## 📂 Struttura del Progetto

```text
kriterion-volatility-monitor/
├── .github/workflows/     # Configurazione CI/CD (GitHub Actions)
├── app.py                 # Entry point Dashboard Streamlit
├── config.py              # Parametri globali (Ticker, Soglie, Modelli)
├── data_loader.py         # Funzioni download dati e calcolo features (Garman-Klass)
├── models.py              # Logica Training HMM e GARCH
├── notifications.py       # Motore di formattazione e invio messaggi Telegram
├── run_daily_check.py     # Script per l'esecuzione batch giornaliera
├── utils.py               # Gestione sicura dei secrets (Env var vs Streamlit secrets)
└── requirements.txt       # Dipendenze Python

```

---

## ⚠️ Disclaimer

Questo software è fornito esclusivamente a scopo educativo e di ricerca.
**Non costituisce consulenza finanziaria.** Il trading di opzioni e strumenti finanziari comporta un elevato livello di rischio. L'autore non si assume alcuna responsabilità per eventuali perdite derivanti dall'uso dei segnali generati da questo codice.

---

## 🔗 Credits

Sviluppato da **Kriterion Quant**.

* Website: [kriterionquant.com](https://kriterionquant.com/)
* Data Provider: EOD Historical Data

Copyright © 2025 Kriterion Quant.

```

### Note sulle modifiche apportate per renderlo "Professionale":

1.  **Badges**: Ho aggiunto i badge all'inizio (Python, Streamlit, License) che danno subito un aspetto curato su GitHub.
2.  **Branding**: Ho utilizzato il nome "Kriterion Quant" e il link al sito web come da tue informazioni salvate, per collegare il repo alla tua identità professionale.
3.  **Sezione Secrets**: Ho spiegato chiaramente come gestire i segreti sia in locale (`secrets.toml`) che in cloud (GitHub Secrets), punto cruciale per far funzionare lo script `utils.py`.
4.  **Descrizione Modelli**: Ho sintetizzato il funzionamento di HMM e GARCH senza scendere troppo nel matematico, ma abbastanza per far capire che c'è sostanza quantitativa.
5.  **Struttura File**: L'albero delle directory aiuta chi guarda il repo a orientarsi velocemente tra logica (`models.py`) e presentazione (`app.py`).

```
