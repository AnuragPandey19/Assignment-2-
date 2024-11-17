from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models
with open('C:/Users/upesc/OneDrive/Documents/elements of ai ml project/myenv/calorie_burn_predictor_model.pkl', 'rb') as f:
    calorie_burn_model = pickle.load(f)

with open('C:/Users/upesc/OneDrive/Documents/elements of ai ml project/myenv/calorie_predictor_model.pkl', 'rb') as f:
    calorie_predictor_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_calories_burned', methods=['POST'])
def predict_calories_burned():
    activity = request.form['activity']
    calories_per_kg = float(request.form['calories_per_kg'])

    # Prepare input for the model
    input_data = {'Activity': [activity], 'CaloriesPerKg': [calories_per_kg]}
    
    # Prediction
    prediction = calorie_burn_model.predict(pd.DataFrame(input_data))[0]
    
    return render_template('index.html', burn_result=round(prediction, 2))

@app.route('/predict_calories_needed', methods=['POST'])
def predict_calories_needed():
    bmi = float(request.form['bmi'])
    gender = request.form['gender']
    activity_level = request.form['activity_level']

    # Prepare input for the model
    input_data = {'BMI': [bmi], 'Gender': [gender], 'Activity Level': [activity_level]}
    
    # Prediction
    prediction = calorie_predictor_model.predict(pd.DataFrame(input_data))[0]
    
    return render_template('index.html', needed_result=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
