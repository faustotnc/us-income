import streamlit as st
import altair as alt
import pandas as pd

from sklearn.inspection import permutation_importance

from layout import get_wide_container

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
    # result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=24, n_jobs=-1)
    # forest_importances = pd.Series(result.importances_mean, index=dataset.drop("income", axis=1).columns)
    # forest_importances.sort_values(ascending=False, inplace=True)

    # return forest_importances
    return {
        "capital-gain": 0.03612459546925566,
        "occupation_other-service": 0.017806094929881326,
        "age": 0.014468716289104627,
        "education_hs-grad": 0.01414509169363537,
        "marital-status_never-married": 0.01318096008629988,
        "sex_female": 0.013153991370010793,
        "occupation_craft-repair": 0.011758360302049626,
        "relationship_husband": 0.009000809061488684,
        "occupation_machine-op-inspct": 0.00840749730312837,
        "relationship_own-child": 0.00832659115426106,
        "education_some-college": 0.007706310679611639,
        "hours-per-week": 0.007632146709816617,
        "workclass_private": 0.007423139158576042,
        "sex_male": 0.0070051240560949305,
        "occupation_transport-moving": 0.0067758899676375215,
        "marital-status_married-civ-spouse": 0.00664778856526429,
        "occupation_handlers-cleaners": 0.006148867313915851,
        "occupation_farming-fishing": 0.0060005393743257866,
        "capital-loss": 0.005501618122977336,
        "workclass_self-emp-not-inc": 0.0050094390507011925,
        "education_7th-8th": 0.004894822006472499,
        "education_11th": 0.004348705501618122,
        "education_10th": 0.0038834951456310552,
        "relationship_unmarried": 0.0030946601941747587,
        "relationship_not-in-family": 0.0028519417475728116,
        "education_assoc-voc": 0.002265372168284796,
        "occupation_adm-clerical": 0.002042880258899682,
        "occupation_sales": 0.0018001618122977458,
        "workclass_state-gov": 0.0016046386192017203,
        "occupation_unknown": 0.0014967637540452994,
        "workclass_unknown": 0.0014563106796116553,
        "race_white": 0.0014563106796116442,
        "occupation_exec-managerial": 0.0012001078748651417,
        "education_9th": 0.0011663969795037699,
        "education_bachelors": 0.001004584681769144,
        "race_black": 0.0007820927723840243,
        "relationship_other-relative": 0.0007753505933117633,
        "workclass_local-gov": 0.0007281553398058249,
        "education_masters": 0.0007214131607335527,
        "native-country_united-states": 0.0007079288025889918,
        "education_assoc-acdm": 0.0006877022653721699,
        "native-country_mexico": 0.0006809600862998811,
        "occupation_prof-specialty": 0.0005663430420711935,
        "education_12th": 0.00048543689320388326,
        "workclass_self-emp-inc": 0.0004584681769147669,
        "education_doctorate": 0.00041801510248110627,
        "marital-status_married-spouse-absent": 0.00030339805825240764,
        "education_5th-6th": 0.00028317152103559675,
        "marital-status_divorced": 0.0002696871628910358,
        "race_amer-indian-eskimo": 0.0002696871628910302,
        "race_asian-pac-islander": 0.00018878101402373116,
        "workclass_federal-gov": 0.0001550701186623593,
        "native-country_vietnam": 0.00012810140237327628,
        "education_prof-school": 0.00012135922330098747,
        "native-country_india": 0.00010787486515644318,
        "occupation_tech-support": 8.090614886731573e-05,
        "marital-status_widowed": 2.0226537216821994e-05,
        "native-country_canada": 1.3484358144555398e-05,
        "native-country_philippines": 1.3484358144555398e-05,
        "occupation_priv-house-serv": 6.742179072277699e-06,
        "native-country_england": 6.742179072277699e-06,
        "native-country_cuba": 6.742179072277699e-06,
        "native-country_south": 0.0,
        "native-country_nicaragua": 0.0,
        "native-country_iran": 0.0,
        "native-country_ireland": 0.0,
        "native-country_jamaica": 0.0,
        "native-country_japan": 0.0,
        "native-country_laos": 0.0,
        "native-country_peru": 0.0,
        "native-country_outlying-us(guam-usvi-etc)": 0.0,
        "native-country_scotland": 0.0,
        "native-country_poland": 0.0,
        "native-country_trinadad&tobago": 0.0,
        "native-country_thailand": 0.0,
        "native-country_portugal": 0.0,
        "native-country_puerto-rico": 0.0,
        "native-country_hungary": 0.0,
        "native-country_taiwan": 0.0,
        "occupation_armed-forces": 0.0,
        "native-country_yugoslavia": 0.0,
        "native-country_hong": 0.0,
        "native-country_ecuador": 0.0,
        "marital-status_married-af-spouse": 0.0,
        "education_preschool": 0.0,
        "workclass_without-pay": 0.0,
        "workclass_never-worked": 0.0,
        "native-country_cambodia": 0.0,
        "native-country_honduras": 0.0,
        "native-country_columbia": 0.0,
        "native-country_china": 0.0,
        "native-country_france": 0.0,
        "native-country_greece": 0.0,
        "native-country_haiti": 0.0,
        "native-country_holand-netherlands": 0.0,
        "education_1st-4th": -5.551115123125783e-18,
        "native-country_dominican-republic": -6.742179072277699e-06,
        "native-country_guatemala": -6.742179072277699e-06,
        "native-country_italy": -6.742179072277699e-06,
        "native-country_el-salvador": -6.742179072277699e-06,
        "race_other": -4.045307443366064e-05,
        "marital-status_separated": -5.3937432578210484e-05,
        "occupation_protective-serv": -6.0679611650493735e-05,
        "native-country_germany": -0.00012135922330099857,
        "relationship_wife": -0.0001887810140237256,
        "native-country_unknown": -0.00022923408845737514,
    }


def show_average_feature_importance(forest_importances):
    with get_wide_container():
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
            labelAngle=-22.5
        )

        st.altair_chart(chart, use_container_width=True)
