# utils.py
import os
import streamlit as st

def get_secret(key_name):
    """
    Recupera un segreto (API Key, Token) gestendo sia l'ambiente 
    GitHub Actions (os.environ) che Streamlit Cloud (st.secrets).
    """
    # 1. Prova a cercare nelle variabili d'ambiente (per GitHub Actions / Docker)
    if key_name in os.environ:
        return os.environ[key_name]
    
    # 2. Prova a cercare nei secrets di Streamlit (per la Dashboard)
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except FileNotFoundError:
        pass
        
    return None
