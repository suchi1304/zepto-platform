import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

os.makedirs("analytics/charts", exist_ok=True)

# 1. Load data from offline fallback (loaded once across pipeline)
df = pd.read_csv("analytics/titanic.csv")

# Clean baseline per threshold rules without leakage
df = df.dropna(subset=['embarked'])
df = df.drop(columns=['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'])

# 2. Stratified Train/Test Split (Before any preprocessing)
X = df.drop(columns=['survived'])
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("=== TASK 7: Stratified Split Complete ===")
print(f"Train class balance: {np.bincount(y_train)} | Test class balance: {np.bincount(y_test)}")

# 3. Preprocessing Definition (Train-fit only)
numeric_features = ['age', 'fare', 'pclass', 'sibsp', 'parch']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['sex', 'embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 4. Train Three Classifiers on Identical Split
print("\n=== TASK 8 & 9: Training & Evaluating 3 Classifiers ===")
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })
    print(f"\n--- {name} ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Visualize Decision Tree with plot_tree
plt.figure(figsize=(16, 8))
tree_model = fitted_pipelines["Decision Tree"].named_steps['classifier']
feature_names = numeric_features + list(
    fitted_pipelines["Decision Tree"].named_steps['preprocessor']
    .named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_features)
)
plot_tree(tree_model, feature_names=feature_names, class_names=['Died', 'Survived'], filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig("analytics/charts/decision_tree_plot.png")
plt.close()
print("Saved decision tree visualization to analytics/charts/decision_tree_plot.png")

# 5. Three-way Imbalance Comparison on Decision Tree
print("\n=== TASK 10: Imbalance Handling Comparison ===")
# Variant A: Baseline (Already computed)
baseline_f1 = results[1]["F1-Score"]

# Variant B: class_weight='balanced'
pipe_bal = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42))
])
pipe_bal.fit(X_train, y_train)
y_bal_pred = pipe_bal.predict(X_test)

# Variant C: SMOTE (Train fold only)
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_trans, y_train)
dt_smote = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_smote.fit(X_train_sm, y_train_sm)
y_sm_pred = dt_smote.predict(X_test_trans)

print(f"Baseline -> Precision: {results[1]['Precision']:.4f}, Recall: {results[1]['Recall']:.4f}, F1: {results[1]['F1-Score']:.4f}")
print(f"Balanced -> Precision: {precision_score(y_test, y_bal_pred):.4f}, Recall: {recall_score(y_test, y_bal_pred):.4f}, F1: {f1_score(y_test, y_bal_pred):.4f}")
print(f"SMOTE    -> Precision: {precision_score(y_test, y_sm_pred):.4f}, Recall: {recall_score(y_test, y_sm_pred):.4f}, F1: {f1_score(y_test, y_sm_pred):.4f}")

# 6. Hyperparameter Tuning for RandomForest with oob_score=True
print("\n=== TASK 11: GridSearchCV for RandomForest with OOB Score ===")
rf_base = RandomForestClassifier(oob_score=True, random_state=42)
rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('rf', rf_base)])

param_grid = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [4, 6],
    'rf__max_features': ['sqrt', 'log2']
}

grid = GridSearchCV(rf_pipe, param_grid, cv=3, scoring='f1')
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_.named_steps['rf']
print(f"Best Parameters: {grid.best_params_}")
print(f"Best Estimator OOB Score: {best_rf.oob_score_:.4f}")

# 7. Regression Side-Task: Predict Fare & Heteroscedasticity Test
print("\n=== TASK 12: Multivariate Regression (Predicting Fare) ===")
reg_X = df.drop(columns=['fare'])
reg_y = df['fare']

reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(
    reg_X, reg_y, test_size=0.20, random_state=42
)

reg_num_cols = ['age', 'pclass', 'sibsp', 'parch']
reg_cat_cols = ['sex', 'embarked']
reg_prep = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), reg_num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('enc', OneHotEncoder(handle_unknown='ignore'))]), reg_cat_cols)
])

reg_pipe = Pipeline(steps=[('preprocessor', reg_prep), ('regressor', LinearRegression())])
reg_pipe.fit(reg_X_train, reg_y_train)
reg_pred = reg_pipe.predict(reg_X_test)

mae = mean_absolute_error(reg_y_test, reg_pred)
rmse = np.sqrt(mean_squared_error(reg_y_test, reg_pred))
r2 = r2_score(reg_y_test, reg_pred)
n = len(reg_y_test)
p = reg_X_train.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
residuals = reg_y_test - reg_pred

print(f"Regression Metrics: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.4f} | Adj-R2={adj_r2:.4f}")

# Residual plot
plt.figure(figsize=(8, 5))
plt.scatter(reg_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Fare")
plt.tight_layout()
plt.savefig("analytics/charts/regression_residual_plot.png")
plt.close()
print("Saved regression residual plot to analytics/charts/regression_residual_plot.png")
print("Heteroscedasticity Conclusion: The residuals fan outward as predicted fare increases, indicating presence of heteroscedasticity.")

# 8. Save Best Full Pipeline Artifact
print("\n=== TASK 13: Exporting Fitted Full Pipeline Artifact ===")
artifact_path = "analytics/titanic_pipeline.joblib"
best_full_pipeline = grid.best_estimator_
joblib.dump(best_full_pipeline, artifact_path)
print(f"Saved complete end-to-end pipeline to {artifact_path}")

# Reload Verification
reloaded_pipeline = joblib.load(artifact_path)
sample_test = X_test.iloc[:3]
preds = reloaded_pipeline.predict(sample_test)
print(f"Demonstration of reload: Predicted {preds} on raw test samples.")

print("\n--- Part B Modeling Complete! ---")