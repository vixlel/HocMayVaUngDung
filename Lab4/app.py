from flask import Flask, render_template, request
import pickle as pk
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load models and encoders
try:
    with open('decision_tree_model.pkl', 'rb') as dt_file:
        dt_model = pk.load(dt_file)
    with open('random_forest_model.pkl', 'rb') as rf_file:
        rf_model = pk.load(rf_file)
    with open('label_encoders.pkl', 'rb') as enc_file:
        label_encoders = pk.load(enc_file)
except FileNotFoundError:
    print("Error: Model file not found.")
    exit()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        bp = request.form['bp']
        cholesterol = request.form['cholesterol']
        na_to_k = float(request.form['na_to_k'])

        # Apply label encoders to input
        sex_encoded = label_encoders['Sex'].transform([sex])[0]
        bp_encoded = label_encoders['BP'].transform([bp])[0]
        cholesterol_encoded = label_encoders['Cholesterol'].transform([cholesterol])[0]

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex_encoded],
            'BP': [bp_encoded],
            'Cholesterol': [cholesterol_encoded],
            'Na_to_K': [na_to_k]
        })

        # Make predictions
        decision_tree_prediction = dt_model.predict(input_data)[0]
        random_forest_prediction = rf_model.predict(input_data)[0]

        # Convert prediction back to original labels
        decision_tree_pred_label = label_encoders['Drug'].inverse_transform([decision_tree_prediction])[0]
        random_forest_pred_label = label_encoders['Drug'].inverse_transform([random_forest_prediction])[0]

        return render_template('index.html',
                               decision_tree_result=decision_tree_pred_label,
                               random_forest_result=random_forest_pred_label)

if __name__ == '__main__':
    app.run(debug=True)
