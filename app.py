import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from metrics_utils import compute_metrics

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Breast Cancer Classification - 5 ML Models")
st.caption("Wisconsin Diagnostic Breast Cancer dataset (0 = malignant, 1 = benign). "
           "Pre-trained models are loaded; nothing is retrained on uploaded data.")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(BASE, "model", MODEL_FILES[name]))


uploaded = st.file_uploader("Upload test CSV (features + 'target' column)", type="csv")
if uploaded is None:
    st.info("No file uploaded - using the bundled test_data.csv.")
    df = pd.read_csv(os.path.join(BASE, "test_data.csv"))
else:
    df = pd.read_csv(uploaded)

if "target" not in df.columns:
    st.error("The CSV must contain a 'target' column.")
    st.stop()

X, y = df.drop(columns="target"), df["target"]
expected = list(load_model("Logistic Regression").feature_names_in_)
missing = [c for c in expected if c not in X.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()
X = X[expected]
st.write(f"Loaded **{len(df)}** rows, **{X.shape[1]}** features.")
with st.expander("Preview data"):
    st.dataframe(df.head())

choice = st.selectbox("Select model", list(MODEL_FILES))
model = load_model(choice)

m = compute_metrics(model, X, y)
st.subheader(f"Evaluation metrics - {choice}")
cols = st.columns(6)
for c, (k, v) in zip(cols, m.items()):
    c.metric(k, f"{v:.4f}")

left, right = st.columns(2)
pred = model.predict(X)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y, pred, target_names=["Malignant", "Benign"], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(3))

st.subheader("All models on this data")
allm = pd.DataFrame([{"ML Model Name": n, **compute_metrics(load_model(n), X, y)} for n in MODEL_FILES])
st.dataframe(allm.set_index("ML Model Name").round(4))
