import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv(r"D:\MY PROJECTS\student perfomance predictor\data\student_performance.csv")

print(data.head())


data['internet'] = data['internet'].map({'Yes': 1, 'No': 0})
data['extra_classes'] = data['extra_classes'].map({'Yes': 1, 'No': 0})

print("Categorical values converted successfully!")
print(data.head())


X = data.drop('final_result', axis=1)
y = data['final_result']

print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data size:", X_train.shape)
print("Testing data size :", X_test.shape)



#train model
from sklearn.linear_model import LogisticRegression

# Create the model
lr_model = LogisticRegression()

# Train the model
lr_model.fit(X_train, y_train)

print("Logistic Regression model trained successfully!")





# STEP 6: Make predictions
y_pred = lr_model.predict(X_test)

# Import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 2))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)




# STEP 7: Visualize Confusion Matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save figure
plt.savefig("outputs/confusion_matrix.png")
plt.show()



# STEP 8: Train Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", round(rf_accuracy, 2))





# STEP 9: Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance.sort_values(by='Importance', ascending=False, inplace=True)
importance.to_csv("outputs/feature_importance.csv", index=False)
print("Feature importance saved:\n", importance)




# STEP 10: Save Random Forest model
import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(rf_model, "models/student_model.pkl")
print("Random Forest model saved as 'models/student_model.pkl'")



# STEP 11: Risk Level Function
def risk_level(prob):
    if prob >= 0.75:
        return "Low Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "High Risk"




import numpy as np

# Example new student: [study_time, attendance, previous_marks, internet(1/0), extra_classes(1/0)]
new_student = np.array([[3, 72, 55, 1, 0]])

# Probability of passing
probability = rf_model.predict_proba(new_student)[0][1]

# Prediction
prediction = rf_model.predict(new_student)[0]

print("Pass Probability:", round(probability, 2))
print("Risk Level:", risk_level(probability))

if prediction == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
