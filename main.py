import altair as alt
import streamlit as st
import pandas as pd
import joblib
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from layout import get_wide_container

# Page subsections
from the_data_section.smote import show_smote_subsection
from the_model_section.feature_importance import compute_feature_importance, show_average_feature_importance
from the_model_section.confusion_matrix import show_confusion_matrix
from the_model_section.cross_validation import show_cross_validation

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")


@st.cache
def prepare_data():
    # Prepare the data & necessary requirements
    df = pd.read_csv('./adult_with_col_names.csv')

    # Remove the "education-num" and "fnlwgt" columns since they do not add
    # any useful information to the algorithm.
    df.drop(['education-num', 'fnlwgt'], axis=1, inplace=True)

    def clean_value(x):
        '''Cleans up categorical, string-valued attributes.'''
        val = x.strip().lower()
        val = "unknown" if val == "?" else val
        return val

    for column in ['workclass', 'education', 'relationship', 'marital-status', 'occupation', 'race', 'sex', 'native-country', 'income']:
        df[column] = df[column].map(clean_value)

    # Convert the output label from categorical to numerical
    df['income'] = df.income.map(lambda x: 1 if x == ">50k" else 0)

    return df


@st.cache
def apply_smote(X, y):
    oversample = SMOTE(sampling_strategy='not majority', k_neighbors=256)
    return oversample.fit_resample(X, y)


raw_dataset = prepare_data()

# Converts categorical attributes to numerical attributes using a one-hot encoding.
dataset = pd.get_dummies(raw_dataset, columns=[
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
])

# The original X and Y (in case we need the original at any time)
raw_X = dataset.drop('income', axis=1).values
raw_y = dataset['income'].values

# The X and Y that will be used for most of the program
X = dataset.drop('income', axis=1).values
y = dataset['income'].values

# Convert inputs and outputs with SMOTE resampling
X, y = apply_smote(X, y)

# Divide the dataset into training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=24)

# Load the model
model = joblib.load('./trained_baseline.joblib')

feature_importance = compute_feature_importance(dataset, X_test, y_test, model)


# ========================================== Streamlit View
get_wide_container().title("U.S. Income Prediction")

# ======= THE DATA =======
get_wide_container().title("The Data 📊")
show_smote_subsection(dataset, y)

# ======= THE MODEL =======
get_wide_container().title("The Model 📦")

show_cross_validation(X, y, model, 300)

show_confusion_matrix(X_test, y_test, model, (350, 300))

show_average_feature_importance(feature_importance)

# ======= THE PREDICTIONS =======
get_wide_container().title("Predictions 🔮")

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


with col2:
    st.markdown("Rules used to make prediction:")

    numerical_attr = ["age", "capital-gain", "capital-loss", "hours-per-week"]

    estimator = model.base_estimator.fit(X_test, y_test)
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_indicator = estimator.decision_path(inp_vec)
    leaf_id = estimator.apply(inp_vec)

    sample_id = 0

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
    ]

    questions_md_table = """| Question | Answer |\n|-|-|"""

    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if inp_vec[sample_id, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        feature_name = dataset.drop("income", axis=1).columns[feature[node_id]]
        feature_val = inp_vec[sample_id, feature[node_id]]
        threshold_val = threshold[node_id]

        if (feature_name in numerical_attr):
            operator = "less than or equal to" if threshold_sign == "<=" else "greater than"
            val = feature_val <= threshold_val if threshold_sign == "<=" else feature_val > threshold_val

            q = f"Is **{feature_name}** {operator} {threshold_val}?"
            a = "Yes" if val else "No"

            questions_md_table += f"\n|{q}|{a}|"
        else:
            f_name, f_categorical_cal = feature_name.split("_")

            q = f"Is **{f_name}** equal to *{f_categorical_cal}*?"
            a = "No" if feature_val == 0 else "Yes"

            questions_md_table += f"\n|{q}|{a}|"

    st.markdown(questions_md_table)
