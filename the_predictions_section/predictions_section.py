import streamlit as st
import numpy as np

from layout import get_wide_container
from the_predictions_section.prediction_rules import show_prediction_rules
from the_predictions_section.prediction_stats import show_prediction_stats


def display_predictions_section(X_test, y_test, raw_dataset, dataset, model):
    _, c1, c2, c3, c4, _ = st.columns((1, 1, 1, 1, 1, 1))

    # Column 1
    age = c1.slider('How old are you?', 0, 100, 25)
    sex = c1.selectbox('What is your sex?', raw_dataset.sex.unique())
    race = c1.selectbox('What is your race?', raw_dataset.race.unique())

    # Column 2
    gain = c2.number_input('What is your capital gain?', step=1)
    loss = c2.number_input('What is your capital loss?', step=1)
    education = c2.selectbox('What is your highest level of eduction?', raw_dataset.education.unique())

    # Column 3
    work_class = c3.selectbox('What is your work class?', raw_dataset['workclass'].unique())
    occupation = c3.selectbox('What is your occupation?', raw_dataset.occupation.unique())
    hours_per_week = c3.number_input('How many hours per week do you work?', step=1)

    # Column 4
    marital = c4.selectbox('What is your marital status?', raw_dataset['marital-status'].unique())
    relationship = c4.selectbox('What is your relationship?', raw_dataset.relationship.unique())
    country = c4.selectbox('What is your native country?', raw_dataset['native-country'].unique())

    get_wide_container().subheader("Generated Prediction")

    # Compose the input vector
    col_names = list(dataset.drop('income', axis=1).columns)

    inp_vec = [np.zeros(len(col_names))]
    inp_vec[0][col_names.index("age")] = age
    inp_vec[0][col_names.index(f"sex_{sex}")] = 1
    inp_vec[0][col_names.index(f"race_{race}")] = 1
    inp_vec[0][col_names.index("capital-gain")] = gain
    inp_vec[0][col_names.index("capital-loss")] = loss
    inp_vec[0][col_names.index(f"education_{education}")] = 1
    inp_vec[0][col_names.index(f"workclass_{work_class}")] = 1
    inp_vec[0][col_names.index(f"occupation_{occupation}")] = 1
    inp_vec[0][col_names.index("hours-per-week")] = hours_per_week
    inp_vec[0][col_names.index(f"marital-status_{marital}")] = 1
    inp_vec[0][col_names.index(f"relationship_{relationship}")] = 1
    inp_vec[0][col_names.index(f"native-country_{country}")] = 1
    inp_vec = np.array(inp_vec)

    _, col1, _, col2, _ = st.columns((1, 1.3, 0.3, 2.1, 1))

    predicted_val = model.predict(inp_vec)[0]
    predicted_probability = model.predict_proba(inp_vec)[0]

    with col1:
        show_prediction_stats(predicted_val, predicted_probability)

    with col2:
        show_prediction_rules(dataset.drop("income", axis=1).columns, X_test, y_test, inp_vec, model)
