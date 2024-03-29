import streamlit as st
import altair as alt
import pandas as pd

from layout import get_sub_columns, get_wide_container


def show_smote_subsection(dataset, y):
    wide_container = get_wide_container()
    col1, col2 = get_sub_columns()

    with wide_container:
        st.subheader("Synthetic Minority Oversampling Technique (SMOTE)")
        st.markdown('''In the original dataset, individuals labeled as "> 50k" appear significantly less frequently than individuals labeled
        as "<= 50k". This demonstrates an imbalance in our dataset, which can lead the model to be biased towards incorrectly labeling new
        individuals as making more than $50k a year. To overcome this problem, we used a Synthetic Minority Oversampling Technique (SMOTE),
        which will increase the minority class by creating synthetic points that are similar to other points in that class.\n''')

    # Show the dataset without SMOTE
    with col1:
        st.text("\nBefore oversampling:")
        non_smote_data = dataset["income"].value_counts()

        non_smote_chart = alt.Chart(pd.DataFrame({
            "Class Labels": ["<= 50k", "> 50k"],
            "Number of Samples": [non_smote_data[0], non_smote_data[1]],
        })).mark_bar().encode(
            x=alt.X('Class Labels:N'),
            y='Number of Samples:Q',
            tooltip=["Number of Samples"]
        ).configure_axisX(
            labelAngle=0
        )

        st.altair_chart(non_smote_chart, use_container_width=True)

    # Show the dataset with SMOTE
    with col2:
        st.text("After oversampling:")
        smote_data = pd.DataFrame(y, columns=["income"])["income"].value_counts()

        smote_chart = alt.Chart(pd.DataFrame({
            "Class Labels": ["<= 50k", "> 50k"],
            "Number of Samples": [smote_data[0], smote_data[1]],
        })).mark_bar().encode(
            x=alt.X('Class Labels:N'),
            y='Number of Samples:Q',
            tooltip=["Number of Samples"]
        ).configure_axisX(
            labelAngle=0
        )

        st.altair_chart(smote_chart, use_container_width=True)
