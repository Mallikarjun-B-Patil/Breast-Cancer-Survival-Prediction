import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, roc_auc_score
from joblib import dump

df = pd.read_csv("Breast Cancer METABRIC.csv")

df["Age at Diagnosis"] = df["Age at Diagnosis"].fillna(df["Age at Diagnosis"].mean())
#df["Age at Diagnosis"].fillna(df["Age at Diagnosis"].mean(), inplace=True)
df["Cohort"] = df["Cohort"].fillna(df["Cohort"].mode()[0])
#df["Cohort"].fillna(df["Cohort"].mode()[0], inplace=True)
df = df.drop(columns= ["Patient ID", "Sex", "Patient's Vital Status", "Overall Survival (Months)"], axis=1)
df = df.dropna(subset=["Overall Survival Status"]).reset_index(drop=True)

categorical = df.columns[df.dtypes == 'object']
le = LabelEncoder()
for cat in categorical:
    df[cat] = le.fit_transform(df[cat])

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

imputer = KNNImputer(n_neighbors=5, weights='distance')
df_imputed = imputer.fit_transform(df_scaled)
df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df.columns)
df = df_imputed

x = df.drop(columns= "Overall Survival Status", axis=1)
y = df["Overall Survival Status"]
y = y.map({0: "Deceased", 1: "Living"})
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

base_model = LogisticRegression()
grid_parameter = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
    'max_iter': [100, 200, 500],  # Iterations for convergence
    'tol': [1e-4, 1e-3, 1e-2]  # Tolerance for stopping criteria
}
grid_search = GridSearchCV(base_model, grid_parameter, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(x_train, y_train)

best_parameter = grid_search.best_params_
model = LogisticRegression(**best_parameter)
model.fit(x_train, y_train)

# Save the trained model to a file
#model.save('breast_cancer_model.h5')
dump(model, 'model.joblib')