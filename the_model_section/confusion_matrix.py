import streamlit as st
import altair as alt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from layout import get_wide_container


def show_confusion_matrix(X_test, y_test, model, size):
    container = get_wide_container()

    container.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, model.predict(X_test))
    matrix = ConfusionMatrixDisplay(cm, display_labels=["0", "1"]).plot(cmap='BuGn').confusion_matrix

    d = pd.DataFrame({
        "x": ["<= 50k", "> 50k", "<= 50k", "> 50k"],
        "y": ["<= 50k", "<= 50k", "> 50k", "> 50k"],
        "Value": matrix.flat
    })

    plot = alt.Chart(d).mark_rect().encode(
        x=alt.X('x:O', title="Predicted Label"),
        y=alt.Y('y:O', title="True Label"),
        color='Value:Q',
        tooltip=["Value"]
    ).properties(
        width=size[0],
        height=size[1]
    ).configure_axisX(
        labelAngle=0
    )

    container.altair_chart(plot)
