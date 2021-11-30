import streamlit as st
import altair as alt
import pandas as pd


def show_smote_subsection(dataset, y):
    st.subheader("Synthetic Minority Oversampling Technique (SMOTE)")
    st.markdown('''In the original dataset, individuals labeled as "> 50k" appear significantly less frequently than individuals labeled
    as "<= 50k". This demonstrates an imbalance in our dataset, which can lead the model to be biased towards incorrectly labeling new
    individuals as making more than $50k a year. To overcome this problem, we used a Synthetic Minority Oversampling Technique (SMOTE),
    which will increase the minority class by creating synthetic points that are similar to other points in that class.\n''')

    # Show the dataset without SMOTE
    st.text("\nBefore oversampling:")
    non_smote_data = dataset["income"].value_counts().rename(index={0: "<= 50K", 1: "> 50K"}).to_dict()

    non_smote_chart = alt.Chart(pd.DataFrame({
        "Class Labels": non_smote_data.keys(),
        "Number of Samples": non_smote_data.values(),
    })).mark_bar().encode(
        x=alt.X('Class Labels:N'),
        y='Number of Samples:Q',
        tooltip=["Number of Samples"]
    ).configure_axisX(
        labelAngle=0
    )

    st.altair_chart(non_smote_chart, use_container_width=True)

    # Show the dataset with SMOTE
    st.text("After oversampling:")
    smote_data = pd.DataFrame(y, columns=["income"])["income"].value_counts().rename(index={0: "<= 50K", 1: "> 50K"}).to_dict()

    smote_chart = alt.Chart(pd.DataFrame({
        "Class Labels": smote_data.keys(),
        "Number of Samples": smote_data.values(),
    })).mark_bar().encode(
        x=alt.X('Class Labels:N'),
        y='Number of Samples:Q',
        tooltip=["Number of Samples"]
    ).configure_axisX(
        labelAngle=0
    )

    st.altair_chart(smote_chart, use_container_width=True)
