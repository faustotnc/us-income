from matplotlib.pyplot import step
import streamlit as st
import pandas as pd
import joblib
# from pprint import pprint

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
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
work = c3.number_input('How many hours per week do you work?', step=1)

# Column 4
marital = c4.selectbox('What is your marital status?', raw_dataset['marital-status'].unique())
relationship = c4.selectbox('What is your relationship?', raw_dataset.relationship.unique())
country = c4.selectbox('What is your native country?', raw_dataset['native-country'].unique())
