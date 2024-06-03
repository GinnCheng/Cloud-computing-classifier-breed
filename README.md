# Sparkify User Churn Prediction
Flask, AWS, Azure, GCP, ML, Classification, Apache cluster, HPC

## Project Overview

The goal is to predict user churn for a fictional music streaming service called Sparkify. Churn prediction is critical for subscription-based businesses to retain customers and improve their service.

## Motivation

Predicting user churn helps businesses identify users who are likely to cancel their subscription. This information can be used to take proactive measures to retain customers, such as targeted promotions, personalized offers, or improved customer service.

## Data

The dataset used for this project contains user activity logs for Sparkify. The data includes information such as user demographics, session details, page views, and the length of time users listened to songs. The target variable is `churn`, which indicates whether a user has canceled their subscription.

## Architecture

The project uses the following architecture:

1. **Data Preprocessing**: Cleaning and transforming the data using PySpark.
2. **Feature Engineering**: Creating relevant features for the prediction model.
3. **Model Training**: Training a machine learning model using PySpark's MLlib.
4. **Model Evaluation**: Evaluating the model's performance using appropriate metrics.
5. **Web Application**: Building a Flask web application to serve the model for real-time predictions.

## Web Application

A Flask web application is built to interact with the trained machine learning model. Users can submit their data through the web interface and get predictions about whether they are likely to churn.

## Installation

To run this project locally, follow these steps:

- **Set up a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

- **Set up PySpark**:
    Ensure you have Apache Spark installed and configured. Follow the instructions on the [official Spark documentation](https://spark.apache.org/docs/latest/index.html).

## Usage

### Training the Model

To train the model, run:

```bash
python sparkify_etl_model.py
