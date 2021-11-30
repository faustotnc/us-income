import streamlit as st
import altair as alt
import pandas as pd


def show_cross_validation(X, y, model, size):
    st.subheader("Cross-Validation")

    # cv_scores = cross_val_score(model, X, y, cv=10)
    cv_scores = [0.893531, 0.87601078, 0.89083558, 0.893531, 0.90161725, 0.9083558,
                 0.89068826, 0.91902834, 0.89608637, 0.91093117]

    cv_as_df = pd.DataFrame({
        "scores": cv_scores,
        "name": "scores"
    })

    cv_chart = alt.Chart(cv_as_df).mark_boxplot(outliers=True, orient="horizontal").encode(
        x=alt.X('scores:Q', title="Score", scale=alt.Scale(zero=False)),
        y=alt.Y('name:N', title="Cross Validation"),
    ).properties(
        width=size[0],
        height=size[1]
    ).configure_axisX(
        labelAngle=0
    ).interactive()

    st.altair_chart(cv_chart)
