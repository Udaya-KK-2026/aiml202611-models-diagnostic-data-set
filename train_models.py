"""Train 5 classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.

Outputs: model/*.joblib, test_data.csv, metrics.csv, full_dataset.csv
"""
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from metrics_utils import compute_metrics

data = load_breast_cancer(as_frame=True)
df = data.frame  # 30 features + 'target' (0 = malignant, 1 = benign)
df.to_csv("full_dataset.csv", index=False)

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])
test.to_csv("test_data.csv", index=False)
X_train, y_train = train.drop(columns="target"), train["target"]
X_test, y_test = test.drop(columns="target"), test["target"]

# Scaling is bundled in a Pipeline so the saved artifact is self-contained.
models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=42))]),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    fname = "model/" + name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, fname)
    rows.append({"ML Model Name": name, **compute_metrics(model, X_test, y_test)})

res = pd.DataFrame(rows).round(4)
res.to_csv("metrics.csv", index=False)
print(res.to_markdown(index=False) if hasattr(res, "to_markdown") else res)
