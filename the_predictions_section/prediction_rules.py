import streamlit as st


def show_prediction_rules(all_feature_names, X_test, y_test, inp_vec, model):
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

        feature_name = all_feature_names[feature[node_id]]
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
