# app.py (la tua nuova pagina principale)
import streamlit as st

st.set_page_config(
    page_title="Homepage - Visualizzatore Polinomiale",
    page_icon="🏠",
)

st.title("Benvenuto nel Visualizzatore Interattivo 📈")

st.markdown("""
Questa applicazione web ti permette di esplorare il comportamento di un modello di regressione polinomiale di quarto grado.

**👈 Seleziona una pagina dalla barra laterale per iniziare:**
- **Visualizzatore Polinomiale**: Interagisci con i parametri del modello e visualizza i grafici in tempo reale.
- **Spiegazione Parametri**: Scopri il significato di ciascun parametro di input.

Questa applicazione è stata creata utilizzando Python e Streamlit.
""")