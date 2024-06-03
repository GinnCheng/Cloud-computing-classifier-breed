"""
app.py: Flask web application for Sparkify churn prediction.

This module initializes a Flask web application that serves a machine learning model for predicting user churn for Sparkify. The application provides endpoints for submitting user data and getting churn predictions.

Usage:
    Run this script to start the web server:
    $ python app.py

Endpoints:
    - /: Renders the home page.
    - /predict: Accepts user data and returns churn prediction.
"""



from flask import Flask, render_template, request, redirect, url_for
from data_analysis import predicting_users
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        results = predicting_users(file_path)
        return render_template('results.html', results=results)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
