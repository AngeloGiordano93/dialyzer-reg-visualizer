import streamlit as st

# Imposta la configurazione della pagina (titolo e icona opzionale)
st.set_page_config(page_title="Spiegazione Parametri", page_icon="ℹ️")

st.title("Guida ai Parametri del Modello")

st.markdown("""
Questa pagina fornisce una breve spiegazione per ciascuno dei parametri di input utilizzati dal modello di regressione polinomiale per predire la 'Clearance'.
""")

st.header("Parametri di Input")

st.subheader("Qb: Flusso Sanguigno (Blood Flow Rate)")
st.write(
    "Rappresenta la velocità con cui il sangue fluisce attraverso il dializzatore, tipicamente misurata in millilitri al minuto (ml/min). Un flusso maggiore generalmente aumenta l'efficienza della dialisi."
)
st.markdown("---") # Crea una linea di separazione

st.subheader("Qd: Flusso del Dialisato (Dialysate Flow Rate)")
st.write(
    "Indica la velocità del fluido di dialisi che scorre in controcorrente al sangue. Anche questo parametro, se aumentato, tende a migliorare la rimozione delle tossine."
)
st.markdown("---")

st.subheader("theta: Frazione di Flusso (Flow Fraction)")
st.write(
    "È un parametro adimensionale che descrive..."
)
st.markdown("---")

# Aggiungi qui le spiegazioni per tutti gli altri parametri...
st.subheader("Lp: Lunghezza della Fibra (Fiber Length)")
# ...

st.subheader("eps_d: Frazione di Vuoto (Void Fraction)")
# ...

st.subheader("Pb: Pressione del Sangue (Blood Pressure)")
# ...

st.subheader("km: Coefficiente di Trasporto di Massa (Mass Transfer Coefficient)")
# ...

st.subheader("Am: Area della Membrana (Membrane Area)")
# ...

st.info("I valori di default nell'applicazione rappresentano un caso d'uso tipico o medio.")