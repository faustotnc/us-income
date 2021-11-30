import streamlit as st
import altair as alt
import pandas as pd

from sklearn.inspection import permutation_importance

attributes = [
    'age',
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]


def compute_feature_importance(dataset, X_test, y_test, model):
    result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=24, n_jobs=-1)
    forest_importances = pd.Series(result.importances_mean, index=dataset.drop("income", axis=1).columns)
    forest_importances.sort_values(ascending=False, inplace=True)

    return forest_importances


def show_average_feature_importance(forest_importances):
    st.subheader("Feature Importance")
    st.markdown("Below are the relative average importance of each attribute in the dataset.")

    collected_importances = {
        # "feature_name": [importance_val1, importance_val2, ...]
    }

    for feature, importance in forest_importances.items():
        found = False

        # Find which attribute the feature belongs to
        for base in attributes:
            if feature.startswith(base):
                found = True

                # If the attribute is already in the dict,
                # append the value to the dict entry.
                if base in collected_importances:
                    collected_importances[base].append(importance)
                else:
                    collected_importances[base] = [importance]
                break

        if not found:
            print(f"Feature '{feature}' does not have a base.")

    avg_importance = {attr: sum(vals) / len(vals) for attr, vals in collected_importances.items()}

    chart = alt.Chart(pd.DataFrame({
        "attribute": avg_importance.keys(),
        "importance": avg_importance.values(),
    })).mark_bar().encode(
        x=alt.X("attribute:N", title="Attribute Names", scale=alt.Scale(zero=False)),
        y=alt.Y("importance:Q", title="Average Importance"),
        tooltip=["importance"]
    ).properties(
        height=400
    ).configure_axisX(
        labelAngle=-45
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
