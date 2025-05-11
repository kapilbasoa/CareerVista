import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
import joblib

# Load dataset
file_path = "Dataset.csv"
df = pd.read_csv(file_path)

print("Dataset Overview:")
print(df.shape)

print(df.columns)

df.head(5)

print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)

# Encode categorical target column 'Role'
label_encoder = LabelEncoder()
if 'Role' in df.columns:
    df["Role"] = label_encoder.fit_transform(df["Role"])
else:
    raise ValueError("Role column not found!")

# Normalize only numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_features = [feat for feat in numerical_features if feat != "Role"]
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split dataset
X = df.drop("Role", axis=1)
y = df["Role"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData Preprocessing Completed. Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Heatmap of correlations
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Role distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=df["Role"], palette="viridis")
plt.xticks(rotation=45)
plt.title("Distribution of Career Roles")
plt.xlabel("Career Roles")
plt.ylabel("Count")
plt.show()

# Dimensionality reduction before clustering
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df[numerical_features])
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(pca_features)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=df["Cluster"], palette="viridis")
plt.title("K-Means Clusters (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", class_weight='balanced'),
    "SVM": SVC(kernel="linear", class_weight='balanced', probability=True)
}

# Train and evaluate models
for name, model in models.items():
    accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean() * 100
    print(f"{name} Accuracy: {accuracy:.4f}")
    model.fit(X_train, y_train)  # Fit once
    models[name] = model

best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Model Evaluation:\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


# # Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))  # Adjust figure size to give more space
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Rotate x-axis labels for clarity
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()  # Prevent clipping of tick-labels
plt.show()


# Classification report heatmap
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="viridis")
plt.title("Classification Report Heatmap")
plt.show()

# Hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)

# Feature importance plot
importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices], palette="coolwarm")
plt.title("Feature Importance in Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Save model, encoder, and scaler
joblib.dump(best_model, "6career_guidance_model.pkl")
joblib.dump(label_encoder, "6label_encoder.pkl")
joblib.dump(scaler, "6scaler.pkl")

import joblib
import numpy as np
import pandas as pd

# Load components
model = joblib.load("6career_guidance_model.pkl")
label_encoder = joblib.load("6label_encoder.pkl")
scaler = joblib.load("6scaler.pkl")

# Get feature names based on scaler (this ensures it matches exactly)
feature_names = scaler.feature_names_in_.tolist()

# Get top N important features
importances = model.feature_importances_
important_indices = np.argsort(importances)[::-1][:7]  # Top 7
important_features = [feature_names[i] for i in important_indices]

def get_valid_input(prompt):
    while True:
        try:
            value = float(input(f"{prompt} (0–8): "))
            if 0 <= value <= 8:
                return value
            else:
                print(" Please enter a number between 0 and 8.")
        except ValueError:
            print(" Invalid input. Please enter a number.")

def get_user_input():
    print("\nPlease enter the following values (from 0 to 8):")
    user_input_dict = {}
    for feature in important_features:
        value = get_valid_input(feature)
        user_input_dict[feature] = value
    return user_input_dict

def predict_career_with_top_features(user_input_dict):
    # Create a full feature dictionary with 0.0
    full_input_dict = {feature: 0.0 for feature in feature_names}

    # Update the dict with user-provided values
    for key, value in user_input_dict.items():
        full_input_dict[key] = value

    # Convert to DataFrame
    input_df = pd.DataFrame([full_input_dict])

    # Scale using the same scaler
    scaled_input = scaler.transform(input_df)

    # Predict probabilities
    probabilities = model.predict_proba(scaled_input)[0]
    top_3 = np.argsort(probabilities)[-3:][::-1]

    # Display predictions
    print("\nTop Career Recommendations:")
    for index in top_3:
        role = label_encoder.inverse_transform([index])[0]
        confidence = probabilities[index] * 100
        print(f"{role}: {confidence:.2f}% confidence")

    return label_encoder.inverse_transform([top_3[0]])[0]

# Run the process
user_input_dict = get_user_input()
predicted_role = predict_career_with_top_features(user_input_dict)
print(f"\nFinal Recommended Career Role: {predicted_role}")


