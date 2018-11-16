# Clustering Using Pyspark

# For this project we will attempt to use KMeans Clustering to cluster Mall customer dataset. We will make segmentation of those customers.

# Pyspark Initializasing
# to make pyspark importable as a regular library
import findspark
findspark.init()

import pyspark

from pyspark import SparkContext
sc = SparkContext.getOrCreate()

#initializasing SparkSession for creating Spark DataFrame
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# Load Libraries
# Data Frame spark profiling 
from pyspark.sql.types import IntegerType, StringType, DoubleType, ShortType, DecimalType
import pyspark.sql.functions as func
from pyspark.sql.functions import isnull
from pyspark.sql.functions import isnan, when, count, col, round
from pyspark.sql.functions import mean
from pyspark.sql.types import Row
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf


# Pandas DF operation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import array

# Modeling + Evaluation
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.sql.functions import when
from pyspark.sql import functions as F
from pyspark.sql.functions import avg
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Load Data to Spark DataFrame
#Initializing File Type
file_type = 'text'
path=r'Mall_Customers.csv'
delimeter=','

# Function load data
def load_data(file_type):
    """input type of file "text" or "parquet" and Return pyspark dataframe"""
    if file_type =="text": # use text as file type input
        df = spark.read.option("header", "true") \
        .option("delimeter",delimeter)\
        .option("inferSchema", "true") \
        .csv(path) 
    else:  
        df= spark.read.parquet("example.parquet") #path file that you want import
    return df
    
# Call function load_data
df = load_data(file_type)

# Check data
# check type of data
type(df)

#show 5 observation in DataFrame
df.show(15)

#Print Schema
df.printSchema()

#rename column name
df=df.withColumnRenamed('CustomerID','Id')
df=df.withColumnRenamed('Annual Income (k$)','AnnIncome')
df=df.withColumnRenamed('Spending Score (1-100)','SpendScore')

#check number of columns and name of columns
len(df.columns), df.columns

#Define categorical and nummerical variable
#Categorical and numerical variable
cat_cols = [item[0] for item in df.dtypes if item[1].startswith('string')] #just will select string data type
print("cat_cols:", cat_cols)
num_cols = [item[0] for item in df.dtypes if item[1].startswith('int') | item[1].startswith('double')] #just will select integer or double data type
print("num_cols:", num_cols)

#Select column 'Id' from num_cols
num_id=num_cols.pop(0)
print("num_id:", num_id)

#save column 'Id' in num_id variable
num_id=[num_id]

#print num_id
print(num_id)

#print num_cols
print(num_cols)

#count number of observation
df.count()

# Check summary statistic of numerical columns
df.select(num_cols).describe().show()

#Check Missing Value
#Check Missing Value in Pyspark Dataframe
def count_nulls(df_final):
    """Input pyspark dataframe and return list of columns with missing value and it's total value"""
    null_counts = []          #make an empty list to hold our results
    for col in df.dtypes:     #iterate through the column data types we saw above, e.g. ('C0', 'bigint')
        cname = col[0]        #splits out the column name, e.g. 'C0'    
        ctype = col[1]        #splits out the column type, e.g. 'bigint'
        nulls = df.where( df[cname].isNull() ).count() #check count of null in column name
        result = tuple([cname, nulls])  #new tuple, (column name, null count)
        null_counts.append(result)      #put the new tuple in our result list
    null_counts=[(x,y) for (x,y) in null_counts if y!=0]  #view just columns that have missing values
    return null_counts
    
#call function count_nulls
null_counts = count_nulls(df)
null_counts

#From null_counts, we just take information of columns name and save in list "list_cols_miss", like in the script below:
list_cols_miss=[x[0] for x in null_counts]
list_cols_miss

#Create dataframe which just has list_cols_miss
df_miss= df.select(*list_cols_miss)

#check type data in df_miss
df_miss.dtypes

#Define categorical columns and numerical columns which have missing value.
### for categorical columns
catcolums_miss=[item[0] for item in df_miss.dtypes if item[1].startswith('string')]  #will select name of column with string data type
print("catcolums_miss:", catcolums_miss)

### for numerical columns
numcolumns_miss = [item[0] for item in df_miss.dtypes if item[1].startswith('int') | item[1].startswith('double')] #will select name of column with integer or double data type
print("numcolumns_miss:", numcolumns_miss)

#Handle Missing Values
#Drop missing value
df_Nomiss=df.na.drop()

#fill missing value in categorical variable with most frequent
for x in catcolums_miss:
    mode=df_Nomiss.groupBy(x).count().sort(col("count").desc()).collect()[0][0] #group by based on categories and count each categories and sort descending then take the first value in column
    print(x, mode) #print name of columns and it's most categories 
    df = df.na.fill({x:mode}) #fill missing value in each columns with most frequent

#fill missing value in numerical variable with average
for i in numcolumns_miss:
    meanvalue = df.select(round(mean(i))).collect()[0][0] #calculate average in each numerical column
    print(i, meanvalue) #print name of columns and it's average value
    df=df.na.fill({i:meanvalue}) #fill missing value in each columns with it's average value

#Check Missing value after filling
null_counts = count_nulls(df)
null_counts

# EDA
# convert spark dataframe to pandas for visualization
df_pd2=df.toPandas()

#Barchart for categorical variable
plt.figure(figsize=(20,10))
plt.subplot(221)
sns.countplot(x='Genre', data=df_pd2, order=df_pd['Genre'].value_counts().index)
plt.show()

#density plot Age
#plt.figure(figsize=(24,5))
sns.distplot(df_pd2['Age'])
plt.show()

#density plot Annual Income
sns.distplot(df_pd2['AnnIncome'])
plt.show()

#density plot Spending Score (1-100)
sns.distplot(df_pd2['SpendScore'])
plt.show()

#Check outlier
#Check outlier in numerical variable: 'Age'
sns.boxplot(y="Age",data=df_pd2)
plt.show()

#Check outlier in numerical variable: 'AnnIncome'
sns.boxplot(y="AnnIncome",data=df_pd2)
plt.show()

#Check outlier in numerical variable: 'SpendScore'
sns.boxplot(y="SpendScore",data=df_pd2)
plt.show()

#Handle of outlier
#create quantile dataframe
def quantile(e):
    """Input is dataframe and return new dataframe with value of quantile from numerical columns"""
    percentiles = [0.25, 0.5, 0.75]
    quant=spark.createDataFrame(zip(percentiles, *e.approxQuantile(num_cols, percentiles, 0.0)),
                               ['percentile']+num_cols) #calculate quantile from pyspark dataframe, 0.0 is relativeError,
                                                        #The relative target precision to achieve (>= 0). If set to zero, 
                                                        #the exact quantiles are computed, which could be very expensive
                                                        #and aggregate the result with percentiles variable, 
                                                        #then create pyspark dataframe
    return quant

#call function quantile
quantile=quantile(df)

#function calculate upper side
def upper_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun upper value from the column"""
    q1 = b.select(c).collect()[0][0] #select value of q1 from the column
    q2 = b.select(c).collect()[1][0] #select value of q2 from the column
    q3 = b.select(c).collect()[2][0] #select value of q3 from the column
    IQR=q3-q1  #calculate the value of IQR
    upper= q3 + (IQR*1.5)   #calculate the value of upper side
    return upper

#function calculate lower side
def lower_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun lower value from the column"""
    q1 = b.select(c).collect()[0][0] #select value of q1 from the column
    q2 = b.select(c).collect()[1][0] #select value of q2 from the column
    q3 = b.select(c).collect()[2][0] #select value of q3 from the column
    IQR=q3-q1                   #calculate the value of IQR
    lower= q1 - (IQR*1.5)       #calculate the value of lower side
    return lower

#Replacing outlier
#function replacing values above upper side with upper side values
def replce_outlier_up2(d,col, value):
    """Input is name of numerical column and it's upper side value"""
    d=d.withColumn(col, F.when(d[col] > value , value).otherwise(d[col]))
    return d

#function replacing values under lower side with lower side values
def replce_outlier_low2(d,col, value):
    """Input is name of numerical column and it's lower side value"""
    d=d.withColumn(col, F.when(d[col] < value , value).otherwise(d[col]))
    return d

#call function to calculate lower side and replace value under lower side with value lower side
for i in num_cols:
    lower=lower_value(quantile,i)
    df=replce_outlier_low2(df, i, lower)

#call function to calculate upper side and replace value above upper side with value upper side
for x in num_cols:
    upper=upper_value(quantile,x)
    df=replce_outlier_up2(df, x, upper)

# Modelling K-Mean
# drop Genre from dataframe, we just used numerical variable for clustering
df_final2=df.drop('Genre')
df_final2.show(4)

#define columns for vector assembler processing
cols_assember=num_cols

#create vector assembler from cols_assember
assembler=VectorAssembler(inputCols=cols_assember, outputCol='features')

#transform vector assembler to dataset
final_data2 = assembler.transform(df_final2)

#Compute cost function
cost= np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df_final)
    cost[k] = model.computeCost(df_final)

#Plot the cost
fig, ax = plt.subplots(1,1, figsize=(10,7))
ax.plot(range(2,20), cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()


#Create model KMeans with K=5
k = 5
kmeans2 = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
model_k5 = kmeans2.fit(final_data2)
centers2 = model_k5.clusterCenters()

print("Cluster Centers: ")
for center in centers2:
    print(center)

#Assign cluster to the event in data
prediction2= model_k5.transform(final_data2).select('Id','prediction', 'Age', 'AnnIncome', 'SpendScore').collect()

#create dataframe 
prediction2=spark.createDataFrame(prediction2)

#show dataframe
prediction2.show()

#grouping by cluster prediction
prediction2.groupBy('prediction').count().show()

#Join prediction with original data
prediction3=prediction2.join(df, 'Id')
prediction3.show(6)

#grouping by cluster prediction and check average of age
prediction2.groupBy('prediction').agg({'Age':'mean'}).show()

#grouping by cluster prediction and check average of all numerical variable
prediction2.groupBy('prediction').avg().show()

#Visualize the result
#convert prediction to Pandas
pred_pd=prediction2.toPandas().set_index('Id')

#show pandas dataframe
pred_pd.head(5)

#Create 3d visualization
threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
threedee.scatter(pred_pd.Age, pred_pd.AnnIncome, pred_pd.SpendScore, c=pred_pd.prediction, cmap="seismic")
threedee.set_xlabel('Age')
threedee.set_ylabel('AnnIncome')
threedee.set_zlabel('SpendScore')
plt.show()


