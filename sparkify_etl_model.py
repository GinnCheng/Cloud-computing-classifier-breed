# import libraries
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, count, sum as sql_sum, avg, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

# create a Spark session
spark = SparkSession.builder \
    .appName("Sparkify Data Analysis") \
    .getOrCreate()

file_path = "mini_sparkify_event_data.json"
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
    'length_filled',when(sc_length['length'].isNull(), sc_length['mean_length']).otherwise(sc_length['length'])
)
# drop columns and rename the length_filled to length
sc_filled = sc_filled.drop('length','mean_length').withColumnRenamed('length_filled', 'length')

sc = sc_filled

# rename the column names
from pyspark.sql.types import StringType
to_lower_coln = udf(lambda x: x.lower(), StringType())

for col in sc.columns:
    sc.withColumn(col, to_lower_coln(col))

# select the relevant columns
coln_selected = ['gender', 'itemInSession', 'level', 'location', 'length', 'method', 'page', 'sessionId', 'status', 'ts', 'userId']
sc_sl = sc.select(coln_selected)

# drop na and duplicated
sc_cl = sc_sl.dropna()
sc_cl = sc_cl.dropDuplicates()

# add a column of churn
to_label_churn = udf(lambda x: 2 if x == 'Cancellation Confirmation' else 1 if x == 'Downgrade' else 0)
sc_cl = sc_cl.withColumn('churn', to_label_churn(sc_cl.page))

# convert time
import datetime
get_year = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0). year)
get_month = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0). month)
get_day = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0). day)
get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000.0). hour)

sc_cl = sc_cl.withColumn('year', get_year(sc_cl.ts))
sc_cl = sc_cl.withColumn('month', get_month(sc_cl.ts))
sc_cl = sc_cl.withColumn('day', get_day(sc_cl.ts))
sc_cl = sc_cl.withColumn('hour', get_hour(sc_cl.ts))

sc_cl = sc_cl.drop('ts')

# convert the data types
columns_to_cast = {
    'churn': 'Integer',
    'year': 'Integer',
    'month': 'Integer',
    'day': 'Integer',
    'hour': 'Integer'
}

for column, dtype in columns_to_cast.items():
    sc_cl = sc_cl.withColumn(column, sc_cl[column].cast(dtype))

# select on the state as the location
to_extract_state = udf(lambda s: s.split(',')[1].strip() if ',' in s else s)

sc_cl = sc_cl.withColumn('state', to_extract_state(sc_cl.location))

from pyspark.sql.functions import split, explode
sc_cl2 = sc_cl.withColumn('states', split(sc_cl.state, "-"))
sc_cl2 = sc_cl2.withColumn('state', explode(sc_cl2.states))
sc_cl = sc_cl2.drop('states','location')

# indexing the strings
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# select the relevant columns
coln_selected = ['gender', 'itemInSession', 'level', 'length', 'churn']
sc_cl = sc_cl.select(coln_selected)

for col in sc_cl.columns:
    if sc_cl.select(col).dtypes[0][1] == 'string':
        print(col,':',sc_cl.select(col).dtypes[0][1])
        indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
        sc_cl = indexer.fit(sc_cl).transform(sc_cl)
        sc_cl = sc_cl.drop(col).withColumnRenamed(col + "_index", col)

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Calculate the number of instances for each class
class_counts = sc_cl.groupBy('churn').count().collect()
churn_counts = [class_counts[0][1],class_counts[1][1],class_counts[2][1]]
max_count = max(churn_counts)
min_count = min(churn_counts)
mid_count = [x for x in churn_counts if x not in [max_count, min_count]][0]

# Calculate weights
mid_ratio = 2*min_count/mid_count
max_ratio = 2*min_count/max_count

# select the data based on the churns
churn_0 = sc_cl.where(sc_cl.churn == 0).sample(withReplacement=False, fraction=max_ratio, seed=42)
churn_1 = sc_cl.where(sc_cl.churn == 1).sample(withReplacement=False, fraction=mid_ratio, seed=42)
churn_2 = sc_cl.where(sc_cl.churn == 2)
# merge the data
sc_cl = churn_0.union(churn_1.union(churn_2))

# Split the data into training and testing sets
train_data, rest_data = sc_cl.randomSplit([0.7, 0.3], seed=42)
valid_data, test_data = rest_data.randomSplit([0.5, 0.5], seed=42)


class sparkify_model:
    def __init__(self, features=['gender', 'itemInSession', 'level', 'length'],
                 impurity=['gini', 'entropy'],
                 maxDepth=[1, 3, 5]):
        self.features = features
        self.label = 'churn'
        self.impurity = impurity
        self.maxDepth = maxDepth

    def train_model(self, train_data=train_data, valid_data=valid_data):
        # Assemble features into a single vector
        assembler = VectorAssembler(inputCols=self.features, outputCol='features')
        # Define the classifier
        rf = RandomForestClassifier(featuresCol='features', labelCol=self.label, predictionCol='churn_pred')
        # Create the pipeline
        pipeline = Pipeline(stages=[assembler, rf])
        # Set Up Cross-Validation
        # Define the parameter grid
        paramGrid = ParamGridBuilder() \
            .addGrid(rf.impurity, self.impurity) \
            .addGrid(rf.maxDepth, self.maxDepth) \
            .build()
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
        bestModel.save('./bestModel')


sparkify_model().train_model()




