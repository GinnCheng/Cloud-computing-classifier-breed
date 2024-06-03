from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, count, sum as sql_sum, avg, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

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

def analysing_data(file_path='./mini_sparkify_event_data.json', output_path='./bestModelFinal'):

    # read the data via spark
    sc = spark.read.json(file_path)

    # Convert data types
    columns_to_cast = {
        "userId": "Integer",
        "length": "Double"
    }
    for column, dtype in columns_to_cast.items():
        sc = sc.withColumn(column, sc[column].cast(dtype))
    # compute the mean length for each userID
    mean_length = sc.groupBy('userId').agg(avg('length').alias('mean_length'))
    # join the mean lengths back to the original DataFrame
    sc_length = sc.join(mean_length, on='userId', how='left')
    # fill null values
    sc_filled = sc_length.withColumn(
        'length_filled', when(sc_length['length'].isNull(), sc_length['mean_length']).otherwise(sc_length['length'])
    )
    # drop columns and rename the length_filled to length
    sc_filled = sc_filled.drop('length', 'mean_length').withColumnRenamed('length_filled', 'length')
    sc = sc_filled
    # rename the column names
    from pyspark.sql.types import StringType
    to_lower_coln = udf(lambda x: x.lower(), StringType())
    for col in sc.columns:
        sc.withColumn(col, to_lower_coln(col))
    # select the relevant columns
    coln_selected = ['userId', 'gender', 'itemInSession', 'level', 'length']
    sc = sc.select(coln_selected)
    # drop na and duplicated
    sc_cl = sc.dropna()
    sc_cl = sc_cl.dropDuplicates()
    # indexing the strings
    for col in ['gender', 'itemInSession', 'level', 'length']:
        if sc_cl.select(col).dtypes[0][1] == 'string':
            print(col, ':', sc_cl.select(col).dtypes[0][1])
            indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
            sc_cl = indexer.fit(sc_cl).transform(sc_cl)
            sc_cl = sc_cl.drop(col).withColumnRenamed(col + "_index", col)

    # Make predictions
    predictions = model.transform(sc_cl)

    # Convert predictions to Pandas DataFrame for easy rendering
    predictions_df = predictions.toPandas()

    # list the potential downgrade and cancel confirmation
    predictions_df = predictions_df[predictions_df.churn in [1,2]]

    # Return the results as HTML
    return predictions_df.to_html()