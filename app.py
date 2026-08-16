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
    page_title="Dry Bean Classification & Model Comparison",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is located in the repository root
MODEL_DIR = "model"

# Your comparison file is inside model/outputs/
COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "outputs",
    "model_comparison.csv"
)

TARGET = "Class"


# ============================================================
# MODEL FILES
# ============================================================

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
    "AspectRation",
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
# TITLE
# ============================================================

st.title("🌱 Dry Bean Classification & Model Comparison")

st.write(
    """
    An Interactive Machine Learning Classification Dashboard — 
    compare models and evaluate their performance on dry bean variety classification.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        This project compares five machine learning classification
        algorithms using the Dry Bean Dataset.
        """
    )

    st.write("**Dataset:** Dry Bean Dataset")
    st.write("**Problem:** Multiclass Classification")
    st.write("**Features:** 16")
    st.write("**Classes:** 7")

    st.divider()

    st.subheader("Models")

    for model_name in MODEL_FILES:
        st.write(f"• {model_name}")


# ============================================================
# STEP 1 — UPLOAD TEST DATA
# ============================================================

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload the test_data.csv generated during model evaluation.",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Upload test_data.csv to evaluate the classification models."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    test_data = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Error reading CSV file: {e}"
    )

    st.stop()


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    feature
    for feature in EXPECTED_FEATURES
    if feature not in test_data.columns
]


if missing_features:

    st.error(
        "The uploaded CSV is missing these required features:"
    )

    st.write(missing_features)

    st.stop()


if TARGET not in test_data.columns:

    st.error(
        "The uploaded CSV must contain the 'Class' column."
    )

    st.stop()


# ============================================================
# DATASET INFORMATION
# ============================================================

st.success(
    "Test dataset uploaded successfully."
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


with st.expander("Preview Uploaded Data"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


# ============================================================
# PREPARE DATA
# ============================================================

X_test = test_data[
    EXPECTED_FEATURES
]

y_test = test_data[
    TARGET
]


# ============================================================
# STEP 2 — MODEL SELECTION
# ============================================================

st.divider()

st.header("2. Select Model")

selected_model = st.selectbox(
    "Choose a classification model:",
    list(MODEL_FILES.keys())
)


# ============================================================
# MODEL PATH
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected_model]
)


# ============================================================
# CHECK MODEL EXISTS
# ============================================================

if not os.path.exists(model_path):

    st.error(
        f"""
        The selected model file was not found:

        {model_path}

        Please make sure all five .pkl files are uploaded
        inside the model folder.
        """
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(
        model_path
    )

except Exception as e:

    st.error(
        f"Unable to load model: {e}"
    )

    st.stop()


# ============================================================
# PREDICTION
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
# METRICS
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
# STEP 3 — DISPLAY METRICS
# ============================================================

st.divider()

st.header(
    f"3. Evaluation Metrics — {selected_model}"
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with col2:

    st.metric(
        "AUC",
        f"{auc:.4f}"
    )

with col3:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with col5:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with col6:

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# STEP 4 — CONFUSION MATRIX
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
    f"Confusion Matrix — {selected_model}"
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# STEP 5 — CLASSIFICATION REPORT
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
# STEP 6 — PREDICTION RESULTS
# ============================================================

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
# STEP 7 — ALL MODEL COMPARISON
# ============================================================

st.divider()

st.header("7. Model Comparison")


if os.path.exists(COMPARISON_PATH):

    comparison_df = pd.read_csv(
        COMPARISON_PATH
    )

    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    comparison_display = comparison_df.copy()

    comparison_display[
        metric_columns
    ] = comparison_display[
        metric_columns
    ].round(4)

    st.dataframe(
        comparison_display,
        use_container_width=True,
        hide_index=True
    )


    # Find best model based on F1
    best_model = comparison_df.loc[
        comparison_df["F1"].idxmax()
    ]


    st.success(
        f"Overall Winner: **{best_model['ML Model Name']}** "
        f"based on the highest F1 Score "
        f"({best_model['F1']:.4f})"
    )

else:

    st.warning(
        "model_comparison.csv was not found."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Dry Bean Classification | by Harshini J"
)
