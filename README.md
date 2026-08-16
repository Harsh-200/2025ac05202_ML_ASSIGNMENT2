# 2025ac05202_ML_ASSIGNMENT2
# 🌱 Dry Bean Variety Prediction & Model Evaluation

An Interactive Machine Learning Classification Dashboard

This project implements and compares different machine learning classification
models to predict the variety of dry beans based on their physical and
geometrical characteristics. An interactive Streamlit application is also
provided to evaluate the trained models using test data.

---

## a. Problem Statement

The objective of this project is to classify dry beans into their respective
varieties using machine learning classification algorithms.

The dataset contains different physical and geometrical measurements of dry
beans. These measurements are used as input features, while the bean variety
is used as the target variable.

Five machine learning classification models are implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models are evaluated using the following metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The main goal is to identify which model performs best for the selected
Dry Bean dataset.

---

## b. Dataset Description

### Dataset Name

**Dry Bean Dataset**

### Dataset Source

[The Dry Bean Dataset is a publicly available classification dataset.](https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip)

### Dataset Characteristics

- Total instances: **13,611**
- Number of input features: **16**
- Number of classes: **7**
- Problem type: **Multiclass Classification**
- Target column: **Class**

The seven bean varieties in the dataset are:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Features

The dataset contains numerical measurements describing the size, shape,
and geometrical properties of the beans. The 16 features used for prediction
are:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

### Target Variable

The target variable is:

`Class`

It represents the variety of the dry bean.

### Preprocessing

The following preprocessing steps were performed:

1. The dataset was loaded and inspected.
2. The target column `Class` was separated from the input features.
3. The data was divided into training and testing sets.
4. Feature scaling was applied where required by the model.
5. The same training and test data were used for all models so that their
   performance could be compared fairly.
6. A fixed random state was used to make the results reproducible.

---

## c. GitHub Repository Link

The complete project, including the source code, trained model files,
notebook, test data, requirements file, and README, is available in the
GitHub repository.

**GitHub Repository:**

[PASTE YOUR GITHUB REPOSITORY URL HERE]

### Repository Contents

```text
2025ac05202_ml_assignment2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── ML_Assignment2.ipynb
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    │
    └── outputs/
        └── model_comparison.csv

## d. Models Used

Five machine learning classification models were implemented on the Dry Bean
dataset. All models were trained and evaluated using the same dataset split
so that their performance could be compared fairly.

The models used are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Evaluation Metrics

The following six evaluation metrics were used to compare the models:

- **Accuracy** – Measures the overall percentage of correctly classified
  instances.
- **AUC** – Measures how well the model distinguishes between the different
  classes.
- **Precision** – Measures how many of the instances predicted as a class
  actually belong to that class.
- **Recall** – Measures how many of the actual instances of a class were
  correctly identified.
- **F1 Score** – Provides a balance between Precision and Recall.
- **MCC (Matthews Correlation Coefficient)** – Measures the quality of the
  classification, taking all parts of the confusion matrix into account.

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.919158 | 0.993449 | 0.919726 | 0.919158 | 0.919290 | 0.902314 |
| Decision Tree | 0.896641 | 0.936251 | 0.896507 | 0.896641 | 0.896412 | 0.875028 |
| KNN | 0.915467 | 0.981051 | 0.916315 | 0.915467 | 0.915652 | 0.897791 |
| Naive Bayes | 0.897010 | 0.989880 | 0.899653 | 0.897010 | 0.897199 | 0.876171 |
| Random Forest (Ensemble) | **0.919897** | 0.990709 | **0.919906** | **0.919897** | **0.919792** | **0.903097** |

---

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression performed very well, achieving 91.92% accuracy. It also achieved the highest AUC of 0.993449. This shows that even a relatively simple model was able to separate the different bean varieties effectively. |
| **Decision Tree** | Decision Tree achieved 89.66% accuracy, which was lower than the other models. Its AUC, F1 Score, and MCC were also comparatively lower, indicating that a single decision tree was less effective for this dataset. |
| **KNN** | KNN achieved 91.55% accuracy and performed quite well overall. Its results were close to Logistic Regression and Random Forest, although its overall metrics were slightly lower. |
| **Naive Bayes** | Naive Bayes achieved 89.70% accuracy. Although its accuracy was lower, its AUC of 0.989880 was quite high, showing that it was still able to distinguish between the bean classes reasonably well. |
| **Random Forest (Ensemble)** | Random Forest achieved the highest accuracy of 91.99%. It also had the highest Precision, Recall, F1 Score, and MCC. Its performance was slightly better than Logistic Regression across most metrics, making it the strongest overall model for this dataset. |

---

### Overall Winner for the Dataset

**Random Forest (Ensemble)** is the overall winner for this dataset.

It achieved the best results in **5 out of the 6 evaluation metrics**:

- Accuracy: **0.919897**
- Precision: **0.919906**
- Recall: **0.919897**
- F1 Score: **0.919792**
- MCC: **0.903097**

Logistic Regression achieved the highest AUC (**0.993449**), so its
performance was very close to Random Forest.

Overall, Random Forest was selected as the winner because it achieved the
highest scores across the majority of the evaluation metrics.

Streamlit Application

An interactive Streamlit application was developed to provide a simple
interface for testing and evaluating the trained machine learning models.

Live Streamlit App

https://8zn6uuo4rwpccdw4sgvqq3.streamlit.app/

Application Features

The Streamlit application provides the following features:

Upload test data in CSV format
Select a machine learning model using a dropdown
Display Accuracy
Display AUC
Display Precision
Display Recall
Display F1 Score
Display MCC
Display a confusion matrix
Display a classification report
View prediction results
Compare the performance of all implemented models

The application uses the saved trained models from the model directory,
so the models do not need to be retrained when the application is opened.
