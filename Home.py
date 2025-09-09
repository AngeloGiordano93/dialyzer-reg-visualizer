import streamlit as st

st.set_page_config(
    page_title="Homepage - Dialyzer Performance Visualizer",
    page_icon="🏠",
)

st.title("Welcome to the interactive dialyzer performance visualizer!")

st.markdown("""
This web application enables the visual exploration of how a dialyzer's performance—quantified by urea clearance—varies as a function of its operating and design parameters.
\nThe dialyzer clearance is computed by a fourth-degree polynomial model trained on synthetic data generated with a first-principles one-dimensional model.

**👈 Select one page from the side bar to begin:**
- **Performance Visualizer**: Interact with the model parameters and visualize the clearance variation graphically in real time.
- **Documentation**: Find more details about the model and its input parameters.

This web app was created using Python and Streamlit by **Angelo Giordano**, PhD student at the University of Palermo (2025).
""")