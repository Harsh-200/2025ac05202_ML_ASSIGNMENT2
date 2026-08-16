import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .metric-label {
        font-size: 14px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

MODEL_DIR = "model"

TARGET = "Class"

MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.pkl",

    "Decision Tree":
        "decision_tree.pkl",

    "KNN":
        "knn.pkl",

    "Naive Bayes":
        "naive_bayes.pkl",

    "Random Forest":
        "random_forest.pkl"
}


# ============================================================
# EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = [

    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRatio",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌱 Dry Bean Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Model Comparison and Prediction Dashboard'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Information")

    st.write(
        """
        This application compares five machine learning
        classification models trained on the UCI Dry Bean Dataset.
        """
    )

    st.write("**Dataset:** UCI Dry Bean Dataset")

    st.write("**Features:** 16")

    st.write("**Classes:** 7")

    st.write("**Task:** Multiclass Classification")

    st.divider()

    st.header("Models")

    for model_name in MODEL_FILES:
        st.write(f"• {model_name}")


# ============================================================
# CSV UPLOAD
# ============================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload the test_data.csv file generated during model evaluation.",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to begin model evaluation."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    test_data = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV file: {e}"
    )

    st.stop()


# ============================================================
# VALIDATE DATA
# ============================================================

missing_features = [
    feature
    for feature in EXPECTED_FEATURES
    if feature not in test_data.columns
]

if missing_features:

    st.error(
        "The uploaded dataset is missing the following "
        f"required features: {missing_features}"
    )

    st.stop()


if TARGET not in test_data.columns:

    st.error(
        "The uploaded dataset must contain the "
        f"'{TARGET}' column for evaluation."
    )

    st.stop()


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

st.success(
    "Dataset uploaded successfully."
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Test Samples",
        len(test_data)
    )

with col2:

    st.metric(
        "Features",
        len(EXPECTED_FEATURES)
    )

with col3:

    st.metric(
        "Classes",
        test_data[TARGET].nunique()
    )


with st.expander("Preview Test Dataset"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


# ============================================================
# PREPARE TEST DATA
# ============================================================

X_test = test_data[
    EXPECTED_FEATURES
]

y_test = test_data[
    TARGET
]


# ============================================================
# MODEL SELECTION
# ============================================================

st.divider()

st.header("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a model to evaluate:",
    list(MODEL_FILES.keys())
)


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected_model]
)


if not os.path.exists(model_path):

    st.error(
        f"Model file not found: {model_path}"
    )

    st.stop()


try:

    model = joblib.load(
        model_path
    )

except Exception as e:

    st.error(
        f"Unable to load the selected model: {e}"
    )

    st.stop()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

try:

    y_pred = model.predict(
        X_test
    )

    y_proba = model.predict_proba(
        X_test
    )

except Exception as e:

    st.error(
        f"Prediction failed: {e}"
    )

    st.stop()


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="weighted"
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# DISPLAY MODEL
# ============================================================

st.success(
    f"Selected Model: **{selected_model}**"
)


# ============================================================
# DISPLAY METRICS
# ============================================================

st.header("3. Model Evaluation Metrics")


row1 = st.columns(3)

with row1[0]:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with row1[1]:

    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

with row1[2]:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


row2 = st.columns(3)

with row2[0]:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with row2[1]:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with row2[2]:

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.divider()

st.header("4. Confusion Matrix")


class_names = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)


fig, ax = plt.subplots(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax
)

ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

plt.tight_layout()

st.pyplot(
    fig
)

plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("5. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# PREDICTION RESULTS
# ============================================================

st.divider()

st.header("6. Prediction Results")

prediction_results = X_test.copy()

prediction_results[
    "Actual Class"
] = y_test.values

prediction_results[
    "Predicted Class"
] = y_pred

prediction_results[
    "Correct Prediction"
] = (
    y_test.values == y_pred
)

st.dataframe(
    prediction_results,
    use_container_width=True
)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.header("7. Model Comparison")

comparison_path = os.path.join(
    "outputs",
    "model_comparison.csv"
)


if os.path.exists(comparison_path):

    comparison_df = pd.read_csv(
        comparison_path
    )

    display_comparison = comparison_df.copy()

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    display_comparison[
        metric_columns
    ] = display_comparison[
        metric_columns
    ].round(4)

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Highlight best model
    # --------------------------------------------------------

    best_model_row = comparison_df.loc[
        comparison_df["F1"].idxmax()
    ]

    st.info(
        f"**Best overall model based on F1 Score:** "
        f"{best_model_row['ML Model Name']} "
        f"({best_model_row['F1']:.4f})"
    )

else:

    st.warning(
        "Model comparison file was not found."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Dry Bean Classification | BITS Pilani WILP "
    "| Machine Learning Classification Assignment"
)
