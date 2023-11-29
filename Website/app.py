import warnings

# To ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
os.environ['FLASK_ENV'] = 'production'

from flask import Flask, request, jsonify, render_template
from model import predict
import webbrowser
import threading
import os
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        # Convert form data to a dictionary
        input_features = request.form.to_dict()

        # Convert numeric fields from string to float
        numeric_fields = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        for field in numeric_fields:
            input_features[field] = float(input_features.get(field, 0))

        # Call the predict function from model.py
        prediction = predict(input_features)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})
    
# Load the CSV file
def load_csv():
    # Replace 'heart.csv' with the path to your CSV file
    return pd.read_csv('heart.csv')

@app.route('/get-random-data', methods=['GET'])
def get_random_data():
    df = load_csv()
    random_row = df.sample().to_dict(orient='records')[0]
    return jsonify(random_row)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.25, open_browser).start()
    app.run(debug=True)
