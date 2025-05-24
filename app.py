from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
# Load model and features list
model = joblib.load('trained_data/model_cls.pkl')
features = joblib.load('trained_data/model_features.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert input to DataFrame with proper column order
    input_df = pd.DataFrame([data])

    # Map categorical inputs to numeric if needed
    if 'Internet_Access_at_Home' in input_df.columns:
        input_df['Internet_Access_at_Home'] = input_df['Internet_Access_at_Home'].map({'Yes':1, 'No':0})
    if 'Extracurricular_Activities' in input_df.columns:
        input_df['Extracurricular_Activities'] = input_df['Extracurricular_Activities'].map({'Yes':1, 'No':0})

    # Reindex to ensure all expected features exist, fill missing with 0
    input_df = input_df.reindex(columns=features, fill_value=0)

    # Predict
    pred = model.predict(input_df)[0]

    # Convert 1/0 back to Pass/Fail
    result = 'Pass' if pred == 1 else 'Fail'

    return jsonify({"Prediction": result})

if __name__ == '__main__':
    app.run(debug=True)

