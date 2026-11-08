# Breast Cancer Classification – 5 ML Models + Streamlit App

## a. Problem statement
Predict whether a breast tumour is **malignant or benign** from 30 numeric measurements of cell nuclei
computed from a digitised fine-needle-aspirate image. This is a binary classification problem. Five
classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same
data, compared on six metrics, and exposed through a Streamlit web app.

## b. Dataset description
- **Source:** Breast Cancer Wisconsin (Diagnostic) – UCI ML Repository (also bundled in `sklearn.datasets.load_breast_cancer`).
- **Instances:** 569 (≥ 500) &nbsp;|&nbsp; **Features:** 30 (≥ 12), all numeric, no missing values.
- **Features:** mean, standard error and "worst" value of 10 nucleus properties (radius, texture, perimeter,
  area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Target:** `target` – 0 = malignant (212), 1 = benign (357).
- **Split:** 80/20 stratified, `random_state=42` → 455 train / 114 test (`test_data.csv`).
- **Preprocessing:** `StandardScaler` inside a scikit-learn `Pipeline` for Logistic Regression, KNN and SVM
  (so the scaler is saved with the model). Decision Tree and Naive Bayes use raw features.

## c. GitHub repository link
<ADD YOUR GITHUB REPO URL HERE>

Live app: <ADD YOUR STREAMLIT COMMUNITY CLOUD URL HERE>

```
├── app.py                # Streamlit app
├── metrics_utils.py      # shared metric computation
├── train_models.py       # trains + saves the 5 models, writes test_data.csv / metrics.csv
├── requirements.txt
├── README.md
├── test_data.csv         # 114-row held-out test set (features + target)
├── full_dataset.csv
├── metrics.csv
└── model/                # logistic_regression, decision_tree, knn, naive_bayes, support_vector_machine (.joblib)
```
Run locally: `pip install -r requirements.txt && python train_models.py && streamlit run app.py`

## d. Models used
Metrics on the 114-row held-out test set (positive class = benign):

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|:-----------------------|---------:|-------:|----------:|-------:|-------:|-------:|
| Logistic Regression    | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree          | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN                    | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes            | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations
| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Best overall (tied with SVM on accuracy/F1/MCC, highest AUC 0.9954). The classes are close to linearly separable once features are standardised; simple, fast and interpretable. |
| Decision Tree | Weakest model (accuracy 0.921, AUC 0.916). Depth-limited tree (max_depth=5) is high-variance on a small dataset and its hard leaf probabilities give a lower AUC. |
| KNN | Solid (accuracy 0.956) after scaling; sensitive to the choice of k and to feature scaling, and slower at prediction time than the linear models. |
| Naive Bayes | Good AUC (0.988) but lower accuracy/MCC (0.939/0.868): the 30 features are strongly correlated, violating the independence assumption, which hurts calibrated decisions. |
| Support Vector Machine | Ties Logistic Regression (accuracy 0.9825, MCC 0.962); RBF kernel with scaled inputs handles the data very well, with a negligibly lower AUC. |
| **Overall winner** | **Logistic Regression** – equal-best on accuracy, precision, recall, F1 and MCC, highest AUC, and the simplest model. SVM is a very close second. |

*Caveat:* with only 114 test rows, one misclassification changes accuracy by ~0.9 points, so the top
two models are not statistically distinguishable.

## Streamlit app features
CSV upload (test data only; falls back to bundled `test_data.csv`), model-selection dropdown, evaluation
metrics, confusion matrix, classification report, and a comparison table of all five models. Saved models
are loaded from `model/`; nothing is retrained.

## BITS Virtual Lab
<INSERT ONE SCREENSHOT OF THE ASSIGNMENT RUNNING ON BITS VIRTUAL LAB HERE>
