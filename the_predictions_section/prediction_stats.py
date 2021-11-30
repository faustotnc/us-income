import streamlit as st
import altair as alt
import pandas as pd


def show_prediction_stats(predicted_val, predicted_probability):
    st.metric(
        label="Predicted Income",
        value="Less than 50k" if predicted_val == 0 else "Greater/equal to 50k",
        delta="-Low Income" if predicted_val == 0 else "+High Income",
    )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("Probability distribution")

    prob_chart = alt.Chart(pd.DataFrame({
        "label": ["<= 50k", "> 50k"],
        "probability": [predicted_probability[0], predicted_probability[1]]
    })).mark_bar().encode(
        x="label",
        y="probability",
        tooltip=["probability"]
    ).configure_axisX(
        labelAngle=0
    )

    st.altair_chart(prob_chart, use_container_width=True)
