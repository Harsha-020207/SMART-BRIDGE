# ===========================================
# Credit Card Approval Prediction System
# Backend - Part 1
# ===========================================

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ===========================================
# Flask App
# ===========================================

app = Flask(__name__)

MODEL_FILE = "model.pkl"
DATASET = "credit_card.csv"


# ===========================================
# Load Dataset
# ===========================================

print("\nLoading Dataset...\n")

df = pd.read_csv(DATASET)

print(df.head())

print("\nDataset Loaded Successfully")


# ===========================================
# Remove Missing Values
# ===========================================

df.dropna(inplace=True)


# ===========================================
# Remove Duplicate Rows
# ===========================================

df.drop_duplicates(inplace=True)


# ===========================================
# Encode Categorical Columns
# ===========================================

encoder = LabelEncoder()

categorical_columns = [

    "Gender",
    "EmploymentType",
    "Education",
    "HousingType",
    "FamilyStatus",
    "PaymentHistory"

]

for column in categorical_columns:

    df[column] = encoder.fit_transform(df[column])


# ===========================================
# Features and Target
# ===========================================

X = df[[

    "Age",
    "Gender",
    "Income",
    "EmploymentType",
    "Education",
    "HousingType",
    "FamilyStatus",
    "EmploymentYears",
    "CreditScore",
    "LoanAmount",
    "PaymentHistory",
    "Dependents"

]]

y = df["Approved"]


# ===========================================
# Train Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42

)


print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ===========================================
# Models Dictionary
# ===========================================

models = {

    "Logistic Regression":

        LogisticRegression(max_iter=1000),

    "Decision Tree":

        DecisionTreeClassifier(),

    "Random Forest":

        RandomForestClassifier(

            n_estimators=200,
            random_state=42

        ),

    "XGBoost":

        XGBClassifier(

            use_label_encoder=False,
            eval_metric="logloss"

        )

}


# ===========================================
# Variables for Best Model
# ===========================================

best_model = None

best_accuracy = 0

best_model_name = ""


print("\nData Preprocessing Completed")

print("Ready For Training...\n")

# ===========================================
# Backend - Part 2
# Train Machine Learning Models
# ===========================================

print("=" * 50)
print("Training Machine Learning Models...")
print("=" * 50)

model_results = {}

# ===========================================
# Train Each Model
# ===========================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    model_results[name] = accuracy

    print(f"Accuracy : {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name


# ===========================================
# Model Comparison
# ===========================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

for model_name, accuracy in model_results.items():

    print(f"{model_name:<25} : {accuracy * 100:.2f}%")

print("=" * 50)

print(f"Best Model : {best_model_name}")

print(f"Best Accuracy : {best_accuracy * 100:.2f}%")

print("=" * 50)


# ===========================================
# Save Best Model
# ===========================================

joblib.dump(best_model, MODEL_FILE)

print(f"\nBest model saved as '{MODEL_FILE}'")


# ===========================================
# Load Saved Model
# ===========================================

loaded_model = joblib.load(MODEL_FILE)

print("Saved model loaded successfully.")


# ===========================================
# Feature Order
# (Must Match Frontend)
# ===========================================

FEATURE_NAMES = [

    "Age",
    "Gender",
    "Income",
    "EmploymentType",
    "Education",
    "HousingType",
    "FamilyStatus",
    "EmploymentYears",
    "CreditScore",
    "LoanAmount",
    "PaymentHistory",
    "Dependents"

]


# ===========================================
# Display Feature Information
# ===========================================

print("\nInput Features")

for index, feature in enumerate(FEATURE_NAMES, start=1):

    print(f"{index}. {feature}")


print("\nMachine Learning Backend Ready.")
# ===========================================
# Backend - Part 3
# Prediction Functions
# ===========================================

def calculate_risk(credit_score, loan_amount, payment_history):

    """
    Returns:
    Low
    Medium
    High
    """

    if payment_history == 0:
        return "High"

    if credit_score >= 750 and loan_amount < 300000:
        return "Low"

    if credit_score >= 650:
        return "Medium"

    return "High"


# ===========================================

def recommend_limit(income, credit_score):

    """
    Recommend Credit Card Limit
    """

    if credit_score >= 800:
        return int(income * 0.60)

    elif credit_score >= 750:
        return int(income * 0.45)

    elif credit_score >= 700:
        return int(income * 0.35)

    elif credit_score >= 650:
        return int(income * 0.25)

    else:
        return int(income * 0.10)


# ===========================================

def format_currency(amount):

    return "₹{:,.0f}".format(amount)


# ===========================================

def predict_credit_card(data):

    """
    Predict Credit Card Approval
    """

    sample = np.array([[
        data["age"],
        data["gender"],
        data["income"],
        data["employment"],
        data["education"],
        data["housing"],
        data["family"],
        data["experience"],
        data["credit"],
        data["loan"],
        data["history"],
        data["children"]
    ]])

    prediction = loaded_model.predict(sample)[0]

    probability = 90

    if hasattr(loaded_model, "predict_proba"):

        probability = round(

            loaded_model.predict_proba(sample)[0][1] * 100,

            2

        )

    status = "Approved" if prediction == 1 else "Rejected"

    risk = calculate_risk(

        data["credit"],
        data["loan"],
        data["history"]

    )

    limit = recommend_limit(

        data["income"],
        data["credit"]

    )

    response = {

        "status": status,

        "probability": probability,

        "risk": risk,

        "limit": format_currency(limit),

        "model": best_model_name,

        "accuracy": round(best_accuracy * 100, 2)

    }

    return response


# ===========================================
# Backend Ready
# ===========================================

print("=" * 50)
print("Prediction Engine Ready")
print("Risk Analysis Ready")
print("Credit Limit Recommendation Ready")
print("=" * 50)
# ===========================================
# Backend - Part 4
# Flask Routes
# ===========================================

@app.route("/")
def home():
    """
    Serve the frontend page.
    Make sure index.html is in the same folder as app.py.
    """
    return send_from_directory(".", "index.html")


# ===========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        required_fields = [

            "age",
            "gender",
            "income",
            "employment",
            "education",
            "housing",
            "family",
            "experience",
            "credit",
            "loan",
            "history",
            "children"

        ]

        # Check for missing fields
        for field in required_fields:

            if field not in data:

                return jsonify({

                    "error": f"Missing field: {field}"

                }), 400

        # Convert all values to numeric
        processed_data = {

            "age": int(data["age"]),
            "gender": int(data["gender"]),
            "income": float(data["income"]),
            "employment": int(data["employment"]),
            "education": int(data["education"]),
            "housing": int(data["housing"]),
            "family": int(data["family"]),
            "experience": float(data["experience"]),
            "credit": int(data["credit"]),
            "loan": float(data["loan"]),
            "history": int(data["history"]),
            "children": int(data["children"])

        }

        result = predict_credit_card(processed_data)

        return jsonify(result)

    except Exception as e:

        return jsonify({

            "status": "Error",
            "message": str(e)

        }), 500


# ===========================================
# Health Check
# ===========================================

@app.route("/health")
def health():

    return jsonify({

        "status": "Running",
        "application": "Credit Card Approval Prediction",
        "model": best_model_name,
        "accuracy": round(best_accuracy * 100, 2)

    })


# ===========================================
# Main Function
# ===========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(" Credit Card Approval Prediction System")
    print("=" * 60)
    print(f" Best Model    : {best_model_name}")
    print(f" Accuracy      : {best_accuracy * 100:.2f}%")
    print(" Server        : http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )