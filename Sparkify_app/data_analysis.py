from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder.appName('FlaskApp').getOrCreate()

# Load the pre-trained Spark model
model = PipelineModel.load('model/your_spark_model_directory')

def analyze_data(file_path):
    # Read the CSV file into a Pandas DataFrame
    data = pd.read_csv(file_path)

    # Convert the Pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(data)

    # Make predictions
    predictions = model.transform(spark_df)

    # Convert predictions to Pandas DataFrame for easy rendering
    predictions_df = predictions.toPandas()

    # Return the results as HTML
    return predictions_df.to_html()
