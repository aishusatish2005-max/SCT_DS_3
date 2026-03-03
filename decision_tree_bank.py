# ================================
# TASK 03 - Decision Tree Classifier
# Bank Marketing Dataset
# ================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("bank-full.csv", sep=';')
print("Dataset loaded successfully!\n")

# 2. Preview Data
print("First 5 rows:")
print(df.head(), "\n")

# 3. Separate Features and Target
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})  # convert target to numeric

# 4. Identify Categorical and Numerical Columns
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(exclude='object').columns

# 5. Preprocessing (One-Hot Encoding)
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# 6. Split Data into Train & Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Build Decision Tree Model
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])

print("Training model...")
model.fit(X_train, y_train)
print("Model training completed!\n")

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy, "\n")

print("Classification Report:")
print(classification_report(y_test, y_pred), "\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

# 10. Visualize Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(model.named_steps['classifier'], filled=True, feature_names=None)
plt.title("Decision Tree - Bank Marketing Dataset")
plt.show()
