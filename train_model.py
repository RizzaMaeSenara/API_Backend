import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure folder exists for saving models
os.makedirs("trained_data", exist_ok=True)

# Sample dummy data loading - replace with your actual dataset CSV path
# Columns: Study_Hours_per_Week, Attendance_Rate, Past_Exam_Scores, Internet_Access_at_Home, Extracurricular_Activities, Final_Exam_Score, Result
# Result = 'Pass' or 'Fail'
df = pd.read_csv('csv/Student_Performance_Prediction.csv')

# Convert categorical inputs to numeric if needed, for example Internet_Access_at_Home ('Yes'/'No') and Extracurricular_Activities ('Yes'/'No')
df['Internet_Access_at_Home'] = df['Internet_Access_at_Home'].map({'Yes':1, 'No':0})
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes':1, 'No':0})

# Target encoding: Pass=1, Fail=0
df['Pass_Fail'] = df['Pass_Fail'].map({'Pass': 1, 'Fail': 0})

# Features and target
X = df[['Study_Hours_per_Week', 'Attendance_Rate', 'Past_Exam_Scores',
        'Internet_Access_at_Home', 'Extracurricular_Activities', 'Final_Exam_Score']]
y = df['Pass_Fail']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: scaling + MLP classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(hidden_layer_sizes=(64,32), activation='relu',
                                 max_iter=2000, early_stopping=True, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save model and features list
joblib.dump(pipeline, 'trained_data/model_cls.pkl')
joblib.dump(X.columns.tolist(), 'trained_data/model_features.pkl')

print("✅ Training complete. Model saved in 'trained_data/model_cls.pkl'")

