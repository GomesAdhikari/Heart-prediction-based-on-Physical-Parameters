import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Mapping for categorical features
SEX_MAPPING = {'Male': 1, 'Female': 0}
SMOKING_MAPPING = {'1': 1, '0': 0}
OBESITY_MAPPING = {'1': 1, '0': 0}

# Suggestions and precautions based on risk level
SUGGESTIONS = {
    'high': "It is highly recommended to consult a healthcare professional for a thorough examination and personalized advice.",
    'medium': "Consider adopting a healthier lifestyle, including a balanced diet and regular exercise. Consult a healthcare professional for personalized guidance.",
    'low': "Maintain a healthy lifestyle and continue regular check-ups. Stay informed about heart health and preventive measures."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/documents')
def documents():
    return render_template('documents.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect features from the form
        form_data = {
            'Age': float(request.form['Age']),
            'Sex': request.form['Sex'],
            'Cholesterol': float(request.form['Cholesterol']),
            'Smoking': request.form['Smoking'],
            'Obesity': request.form['Obesity'],
            'Exercise_Hours_Per_Week': float(request.form['Exercise_Hours_Per_Week']),
            'Stress_Level': float(request.form['Stress_Level']),
            'Sedentary_Hours_Per_Day': float(request.form['Sedentary_Hours_Per_Day']),
            'Income': float(request.form['Income']),
            'Triglycerides': float(request.form['Triglycerides']),
            'Physical_Activity_Days_Per_Week': float(request.form['Physical_Activity_Days_Per_Week']),
            'Systolic': float(request.form['Systolic']),
            'Diastolic': float(request.form['Diastolic'])
        }

        # Convert categorical features
        sex = SEX_MAPPING[request.form['Sex']]
        smoking = SMOKING_MAPPING[request.form['Smoking']]
        obesity = OBESITY_MAPPING[request.form['Obesity']]

        # Prepare the feature vector
        features = np.array([
            form_data['Age'],
            sex,
            form_data['Cholesterol'],
            smoking,
            obesity,
            form_data['Exercise_Hours_Per_Week'],
            form_data['Stress_Level'],
            form_data['Sedentary_Hours_Per_Day'],
            form_data['Income'],
            form_data['Triglycerides'],
            form_data['Physical_Activity_Days_Per_Week'],
            form_data['Systolic'],
            form_data['Diastolic']
        ]).reshape(1, -1)

        # Transform features using the loaded scaler
        scaled_features = scaler.transform(features)

        # Predict probabilities
        prediction_proba = model.predict_proba(scaled_features)
        risk_probability = prediction_proba[0][1] * 100

        # Determine risk level and provide suggestion
        if risk_probability > 70:
            risk_level = 'high'
        elif risk_probability > 30:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        suggestion = SUGGESTIONS[risk_level]
        formatted_probability = f'{risk_probability:.2f}'

    except Exception as e:
        formatted_probability = f'Error: {str(e)}'
        suggestion = "Please check your input and try again."

    # Render the result page with probability, suggestion, and user details
    return render_template('result.html',
                           probability=formatted_probability,
                           suggestion=suggestion,
                           form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
