"""
data_analysis.py: Load the machine learning model for churn prediction.

This module predicts user churn based on user activity data.

"""


# import libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, count, sum as sql_sum, avg, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import datetime
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.functions import split, explode
from pyspark.ml import PipelineModel
import shutil

def prediction_accuracy(row):
    if (row['page'] == 'Cancellation Confirmation') and (row['churn_pred'] == 2):
        result = True
    elif (row['page'] == 'Downgrade') and (row['churn_pred'] == 1):
        result = True
    else:
        result = False
    return result

def predicting_users(file_path='./mini_sparkify_event_data.json', model_path="./final_model.zip"):
    # Initialize Spark session
    spark = SparkSession.builder.appName('FlaskApp').getOrCreate()

    # Load the pre-trained Spark model
    shutil.unpack_archive(model_path, "./model")
    model = PipelineModel.load('./model')

    # read the data via spark
    print('reading the data ...')
    sc = spark.read.json(file_path)

    # Convert data types
    print('cleaning the data ...')
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
    coln_selected = ['userId', 'gender', 'itemInSession', 'level', 'length', 'page']
    sc = sc.select(coln_selected)
    # drop na and duplicated
    sc_cl = sc.dropna()
    sc_cl = sc_cl.dropDuplicates()
    # indexing the strings
    for col in ['gender', 'itemInSession', 'level', 'length']:
        if sc_cl.select(col).dtypes[0][1] == 'string':
            # print(col, ':', sc_cl.select(col).dtypes[0][1])
            indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
            sc_cl = indexer.fit(sc_cl).transform(sc_cl)
            sc_cl = sc_cl.drop(col).withColumnRenamed(col + "_index", col)

    # Make predictions
    print('predicting the data ...')
    predictions = model.transform(sc_cl)

    # Convert predictions to Pandas DataFrame for easy rendering
    predictions_df = predictions.toPandas()

    # list the potential downgrade and cancel confirmation
    predictions_df = predictions_df[predictions_df.churn_pred.isin([1, 2])]
    predictions_df.drop(columns=['features', 'rawPrediction', 'probability'], inplace=True)

    # check if the prediction accuracy
    predictions_df['correct_prediction'] =\
        predictions_df[['page', 'churn_pred']].apply(prediction_accuracy, axis=1)

    accuracy_percent = predictions_df.correct_prediction.sum()/predictions_df.shape[0]*100
    print(f'The accuracy is {accuracy_percent} %')

    # stop spark
    spark.stop()

    # Return the results as HTML
    # return predictions_df
    return predictions_df.to_html()

if __name__ == '__main__':
    file_path = '../../../Datasets/UdacityCapstoneSpark/mini_sparkify_event_data.json'
    pred = predicting_users(file_path=file_path)
    print(pred.page.unique())
    print(pred.correct_prediction.unique())
