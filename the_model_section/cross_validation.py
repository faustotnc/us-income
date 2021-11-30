import streamlit as st
import altair as alt
import pandas as pd

from layout import get_sub_columns, get_wide_container


def show_cross_validation(X, y, model, height):
    wide_container = get_wide_container()
    _, semiwide_container, _, _ = st.columns((1, 2.75, 1.25, 1))
    _, col1, col2, _ = st.columns((1, 3, 2, 1))

    wide_container.subheader("Cross-Validation")

    # cv_scores = cross_val_score(model, X, y, cv=10)
    cv_scores = [0.77487864, 0.78964401, 0.79389159, 0.92495955, 0.95145631, 0.95024272,
                 0.95651294, 0.95570388, 0.95489482, 0.94822006]

    cv_as_df = pd.DataFrame({
        "scores": cv_scores,
        "name": "scores"
    })

    semiwide_container.markdown(f'''For the crossvalidation, we found the model to have an average accuracy
        of **{round(cv_as_df.scores.mean(), 3)}** with standardad deviation of **{round(cv_as_df.scores.std(), 3)}**. This
        was computed using a 10-fold crossvalidation on the entire dataset.''')

    with col1:
        stats = pd.DataFrame({
            "cv": ["cv1", "cv2", "cv3", "cv4", "cv5", "cv6", "cv7", "cv8", "cv9", "cv10"],
            "score": cv_scores
        })

        cs_stats_chart = alt.Chart(stats).mark_bar().encode(
            x=alt.X("cv", title="Cross-validation Run", sort=None),
            y=alt.Y("score", title="Accuracy Score"),
            tooltip=["score"]
        ).configure_axisX(
            labelAngle=0
        )

        st.altair_chart(cs_stats_chart, use_container_width=True)

    with col2:
        cv_chart = alt.Chart(cv_as_df).mark_boxplot(outliers=True, size=50).encode(
            y=alt.Y("scores:Q", title="Score", scale=alt.Scale(zero=False)),
            x=alt.X("name:N", title="Cross Validation"),
        ).properties(
            height=height
        ).configure_axisX(
            labelAngle=0
        ).interactive()

        st.altair_chart(cv_chart, use_container_width=True)
