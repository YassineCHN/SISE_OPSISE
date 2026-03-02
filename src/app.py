import streamlit as st
from modules.preprocessing import load_data

st.set_page_config(layout="wide")


@st.cache_data
def get_data():
    return load_data()


df = get_data()

st.write("Nombre total de lignes :", len(df))
