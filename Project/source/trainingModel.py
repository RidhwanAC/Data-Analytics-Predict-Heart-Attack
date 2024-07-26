# Step 1: Import necessary libraries
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib
import os

import warnings
warnings.simplefilter(action='ignore')

# Step 2: Load the dataset
df = pd.read_csv("datasets/heart.csv")
# Display the first few rows of the dataframe
print(df.head())
df.info()
print()

# Check for missing values
print(df.isnull().sum())
print()
print(df.describe())
print(f"Dataset shape: {df.shape}")

# Explore the distribution of the target variable
sns.countplot(x='output', data=df)
plt.title('Distribution of Heart Attack')
plt.show()

# Explore correlations between features
plt.figure(figsize=(7,7))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', annot_kws={"size": 8})
plt.title('Correlation Matrix', fontsize=10)
plt.xticks(fontsize=8) 
plt.yticks(fontsize=8) 
plt.tight_layout() 
plt.show()
print()

# Step 3: Data Preprocessing
# Split the data into features (X) and target variable (y)
X = df.drop('output', axis=1)
y = df['output']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Handle missing values (if any)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Hyperparameter Tuning and Cross-Validation for Multiple Models
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    },
    'Naive Bayes': {},
    'MLP Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'max_iter': [200, 300]
    }
}

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),       
    'MLP Neural Network': MLPClassifier() 
}
best_models = {}

for name, param_grid in param_grids.items():
    print(f"Performing Grid Search for {name}...")
    model = models[name]
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy for {name}: {grid_search.best_score_:.2f}")
    print()

# Evaluate the best model on the test set
for name, model in best_models.items():
    print(f"Evaluating {name} on the test set...")
    y_pred = model.predict(X_test)
    print(f"Test set accuracy for {name}: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

# Model Comparison
# Dictionary to store accuracies
accuracies = {}

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy

# Create a bar plot for accuracies
plt.figure(figsize=(10, 6))
plt.barh(list(accuracies.keys()), list(accuracies.values()), color='skyblue')
plt.xlabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xlim(0, 1)
plt.gca().invert_yaxis() 
plt.show()

# Generate ROC curves for each model
plt.figure(figsize=(10, 8))

for name, model in best_models.items():
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    else:  # For models that don't have predict_proba, use decision_function
        y_pred_prob = model.decision_function(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Step 5: Save the Best Model
best_model_name = max(accuracies, key=accuracies.get)
best_model = best_models[best_model_name]

model_filename = 'best_model.pkl'
scaler_filename = 'scaler.pkl'

if not os.path.exists(model_filename):
    joblib.dump(best_model, model_filename)
    print(f"Best model ({best_model_name}) saved as '{model_filename}'")
else:
    print(f"Model file '{model_filename}' already exists. Skipping save.")

if not os.path.exists(scaler_filename):
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved as '{scaler_filename}'")
else:
    print(f"Scaler file '{scaler_filename}' already exists. Skipping save.")
