'''
This script demonstrates a complete workflow for preprocessing data, training a machine learning model,
and saving the model using PySpark. The example is focused on analyzing user behavior data from the
Sparkify music streaming service to predict user churn.

The script is divided into several parts:
1. Library Imports
2. Data Loading
3. Data Wrangling Transformer Definition
4. Model Training Class Definition
5. Execution of the Pipeline

To analyse and train the data:
file_path: The location of the source data
output_path: The location that the model is saved

'''

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
import shutil

def loading_data(file_path='./mini_sparkify_event_data.json', output_path='./bestModelFinal'):
    # read the data via spark
    print('loading data ...')
    sc = spark.read.json(file_path)
    return sc, output_path


def analysing_data(sc):
    print('wrangling data ...')
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
    coln_selected = ['userId', 'gender', 'itemInSession', 'level', 'length', 'page']
    sc = sc.select(coln_selected)
    # drop na and duplicated
    sc_cl = sc.dropna()
    sc_cl = sc_cl.dropDuplicates()
    # add a column of churn
    to_label_churn = udf(lambda x: 2 if x == 'Cancellation Confirmation' else 1 if x == 'Downgrade' else 0)
    sc_cl = sc_cl.withColumn('churn', to_label_churn(sc_cl.page)).drop('page')
    # convert the data types
    columns_to_cast = {
        'churn': 'Integer',
    }
    for column, dtype in columns_to_cast.items():
        sc_cl = sc_cl.withColumn(column, sc_cl[column].cast(dtype))
    # indexing the strings
    for col in ['gender', 'itemInSession', 'level', 'length']:
        if sc_cl.select(col).dtypes[0][1] == 'string':
            # print(col, ':', sc_cl.select(col).dtypes[0][1])
            indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
            sc_cl = indexer.fit(sc_cl).transform(sc_cl)
            sc_cl = sc_cl.drop(col).withColumnRenamed(col + "_index", col)

    # Calculate the number of instances for each class
    class_counts = sc_cl.groupBy('churn').count().collect()
    churn_counts = [class_counts[0][1], class_counts[1][1], class_counts[2][1]]
    max_count = max(churn_counts)
    min_count = min(churn_counts)
    mid_count = [x for x in churn_counts if x not in [max_count, min_count]][0]
    # Calculate weights
    mid_ratio = 2 * 4.3 * min_count / mid_count
    max_ratio = 2 * 4.3 * min_count / max_count
    # select the data based on the churns
    try:
        churn_0 = sc_cl.where(sc_cl.churn == 0).sample(withReplacement=False, fraction=max_ratio, seed=42)
    except:
        churn_0 = sc_cl.where(sc_cl.churn == 0)
    try:
        churn_1 = sc_cl.where(sc_cl.churn == 1).sample(withReplacement=False, fraction=mid_ratio, seed=42)
    except:
        churn_1 = sc_cl.where(sc_cl.churn == 1)
    churn_2 = sc_cl.where(sc_cl.churn == 2)
    # merge the data
    sc_cl = churn_0.union(churn_1.union(churn_2))
    # Split the data into training and testing sets
    train_data, rest_data = sc_cl.randomSplit([0.7, 0.3], seed=42)
    valid_data, test_data = rest_data.randomSplit([0.5, 0.5], seed=42)

    return train_data, valid_data, test_data


class sparkify_model:
    def __init__(self, features=['gender', 'itemInSession', 'level', 'length']):
        self.features = features
        self.label = 'churn'

    def train_model(self, train_data, valid_data):
        print('training model ...')
        # Assemble features into a single vector
        assembler = VectorAssembler(inputCols=self.features, outputCol='features')
        # Define the classifier
        rf = RandomForestClassifier(featuresCol='features', labelCol=self.label, predictionCol='churn_pred')
        # Create the pipeline
        pipeline = Pipeline(stages=[assembler, rf])
        # Set Up Cross-Validation
        # Define the parameter grid
        paramGrid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [10, 20, 30])
            .addGrid(rf.maxDepth, [5, 10, 15])
            .addGrid(rf.maxBins, [32, 64, 128])
            .build())
        # Define the evaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol='churn',
            predictionCol='churn_pred',
            metricName='f1'
        )
        # Set up cross-validator
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=3)
        # Train the Model
        cvModel = crossval.fit(train_data)
        # Make predictions
        predictions = cvModel.transform(valid_data)
        # Evaluate the model
        accuracy = evaluator.evaluate(predictions)
        print(f"Model Accuracy: {accuracy}")
        # save the model
        bestModel = cvModel.bestModel
        return bestModel


if __name__ == '__main__':
    # create a Spark session
    spark = SparkSession.builder \
        .appName("Sparkify Data Analysis") \
        .getOrCreate()

    # load data
    file_path = './mini_sparkify_event_data.json'
    output_path = './bestModelFinal'
    sc, output_path = loading_data(file_path=file_path, output_path=output_path)
    # wrangle data
    train_data, valid_data, test_data = analysing_data(sc)
    # train data
    clf = sparkify_model()
    model = clf.train_model(train_data, valid_data)
    # save the model
    model.write().overwrite().save(output_path)
    # Archive the directory
    shutil.make_archive('final_model', 'zip', output_path)
    # Stop the Spark session
    spark.stop()