import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
# --- Step 2: Load and Explore the Dataset ---
url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nAttrition Distribution:")
print(df["Attrition"].value_counts())
attrition_rate = df["Attrition"].value_counts(normalize=True)
print(f"\nAttrition Rate: {attrition_rate['Yes']:.2%}")
# Quick visualization
plt.figure(figsize=(8, 5))
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.title("Attrition by Overtime Status")
plt.tight_layout()
plt.savefig("attrition_overview.png")
plt.close()
# --- Step 3: Preprocess the Data ---
# Drop uninformative columns
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])

# Encode target variable
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
print(f"\nCategorical columns to encode: {list(categorical_cols)}")

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}") 

print("\nAll columns are now numeric:")
print(df.dtypes.value_counts())
print("\nSaved: attrition_overview.png")
# --- Step 4: Train and Evaluate the Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Features for Attrition Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
print("\nSaved: feature_importance.png")
# --- Step 5: Explain Predictions with SHAP and Save the Model ---

# --- Step 5: Explain Predictions with SHAP and Save the Model ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Handle different SHAP versions dynamically (older list format vs. newer 3D array format)
import numpy as np
if isinstance(shap_values, list):
    # Older SHAP returns a list of 2D arrays (index 1 is Class 1)
    shap_values_class1 = shap_values[1]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    # Newer SHAP returns a 3D array: (samples, features, classes)
    shap_values_class1 = shap_values[:, :, 1]
else:
    # Fallback default
    shap_values_class1 = shap_values

# SHAP summary plot for class 1 (Left) - Using the version-safe variable
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()
print("\nSaved: shap_summary.png")

# Explain a single prediction
sample_index = 0
sample = X_test.iloc[[sample_index]]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print(f"\nSample Employee Prediction: {'Will Leave' if prediction == 1 else 'Will Stay'}")
print(f"Probability of Leaving: {probability[1]:.2%}")
print(f"Probability of Staying: {probability[0]:.2%}")

# Handle expected_value format dynamically (older list/array vs newer scalar)
expected_value_class1 = (
    explainer.expected_value[1] 
    if hasattr(explainer.expected_value, "__len__") 
    else explainer.expected_value
)

# SHAP force plot for single prediction - Using version-safe variables
shap.force_plot(
    expected_value_class1,
    shap_values_class1[sample_index],
    sample.iloc[0],
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.savefig("shap_individual.png", bbox_inches="tight")
plt.close()
print("Saved: shap_individual.png")
# Save model and features
joblib.dump(model, "attrition_model.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
print("\nModel saved as 'attrition_model.pkl'")
print("Feature list saved as 'model_features.pkl'")

# Verify by loading
loaded_model = joblib.load("attrition_model.pkl")
test_pred = loaded_model.predict(sample)
print(f"Verification - Loaded model prediction: {'Will Leave' if test_pred[0] == 1 else 'Will Stay'}")
# --- Secret Mission: Risk Scoring Function ---

# --- Secret Mission: Risk Scoring Function ---
def predict_attrition_risk(employee_data, model_path="attrition_model.pkl", features_path="model_features.pkl"):
    """
    Predict attrition risk for a single employee.
    """
    # Load model and features
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # Create DataFrame with correct column order
    employee_df = pd.DataFrame([employee_data])
    
    # Ensure all required columns exist (fill missing with 0)
    for col in feature_names:
        if col not in employee_df.columns:
            employee_df[col] = 0
    
    employee_df = employee_df[feature_names]
    
    # Predict probability
    probability = model.predict_proba(employee_df)[0][1]
    
    # Classify risk level
    if probability <= 0.30:
        risk_level = "Low Risk"
    elif probability <= 0.70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    
    # Get SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(employee_df)
    
    # Handle different SHAP versions dynamically (older list format vs. newer 3D array format)
    import numpy as np
    if isinstance(shap_values, list):
        # Older SHAP: list of 2D arrays (index 1 is Class 1, index 0 is first sample)
        shap_values_class1 = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Newer SHAP: 3D array (samples, features, classes)
        shap_values_class1 = shap_values[0, :, 1]
    else:
        # Fallback default
        shap_values_class1 = shap_values[0]
    
    # Get top 3 factors pushing toward leaving (class 1)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values_class1
    })
    top_factors = shap_df.reindex(
        shap_df["SHAP_Value"].abs().sort_values(ascending=False).index
    ).head(3)
    
    factors_list = []
    for _, row in top_factors.iterrows():
        direction = "increases" if row["SHAP_Value"] > 0 else "decreases"
        factors_list.append(f"{row['Feature']} ({direction} risk)")
    
    # CRITICAL: This return block must be aligned to the main function level
    return {
        "risk_level": risk_level,
        "probability": f"{probability:.2%}",
        "top_factors": factors_list
    }

    # Load model and features
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    
    # Create DataFrame with correct column order
    employee_df = pd.DataFrame([employee_data])
    
    # Ensure all required columns exist (fill missing with 0)
    for col in feature_names:
        if col not in employee_df.columns:
            employee_df[col] = 0
    
    employee_df = employee_df[feature_names]
    
    # Predict probability
    probability = model.predict_proba(employee_df)[0][1]
    
    # Classify risk level
    if probability <= 0.30:
        risk_level = "Low Risk"
    elif probability <= 0.70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
    

    # Get SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(employee_df)
    
    # Handle different SHAP versions dynamically (older list format vs. newer 3D array format)
    import numpy as np
    if isinstance(shap_values, list):
        # Older SHAP returns a list of 2D arrays (index 1 is Class 1, index 0 is first sample)
        shap_values_class1 = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # Newer SHAP returns a 3D array: (samples, features, classes)
        # We select the first sample (0), all features (:), and Class 1 (1)
        shap_values_class1 = shap_values[0, :, 1]
    else:
        # Fallback default
        shap_values_class1 = shap_values[0]
    
    # Get top 3 factors pushing toward leaving (class 1)
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_values_class1
    })
    top_factors = shap_df.reindex(
        shap_df["SHAP_Value"].abs().sort_values(ascending=False).index
    ).head(3)


# Test the function
sample_employee = {
    "Age": 26,
    "MonthlyIncome": 3000,
    "OverTime": 1,
    "JobSatisfaction": 1,
    "YearsAtCompany": 2,
    "TotalWorkingYears": 3,
    "DistanceFromHome": 15,
    "NumCompaniesWorked": 3,
    "JobLevel": 1,
    "EnvironmentSatisfaction": 2,
    "WorkLifeBalance": 2,
    "YearsSinceLastPromotion": 2,
    "YearsInCurrentRole": 1,
    "YearsWithCurrManager": 1,
    "MaritalStatus": 2,
    "BusinessTravel": 2,
    "Department": 2,
    "Education": 3,
    "EducationField": 1,
    "Gender": 1,
    "JobInvolvement": 3,
    "JobRole": 3,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 2,
    "StockOptionLevel": 0,
    "TrainingTimesLastYear": 2,
    "DailyRate": 800,
    "HourlyRate": 50,
    "MonthlyRate": 15000,
    "PercentSalaryHike": 12
}

print("\n--- Risk Assessment ---")
result = predict_attrition_risk(sample_employee)
print(f"Risk Level: {result['risk_level']}")
print(f"Probability of Leaving: {result['probability']}")
print(f"Top Factors:")
for factor in result["top_factors"]:
    print(f"  - {factor}")
