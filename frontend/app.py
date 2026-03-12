"""Main entry point for the Streamlit frontend application."""
import streamlit as st

st.set_page_config(
    page_title="FinSight - Earnings Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("FinSight")
st.markdown("Welcome to FinSight. Use the sidebar to navigate.")
st.sidebar.success("Select a page above.")
