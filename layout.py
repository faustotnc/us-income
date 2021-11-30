import streamlit as st


def get_wide_container():
    _, wide_container, _ = st.columns((1, 4, 1))
    return wide_container


def get_sub_columns():
    _, subcol1, subcol2, _ = st.columns((1, 2, 2, 1))
    return (subcol1, subcol2)
