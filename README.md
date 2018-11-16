# Clustering-Pyspark
This is a repository of clustering using pyspark

I tried to make a template of clustering machine learning using pyspark. Generally, the steps of clustering are same with the steps of classification and regression from load data, data cleansing and making a prediction. But the differences are the libraries, models and data that are used. I used K-Mean to create cluster. 

I use the same function as the function used in regression and classfication for data cleansing and data importing. 

To test my template, I use Mall customers dataset, this data represent customers income, age, sex and spending score in a Mall. From that dataset I will make segmentation of those customers.

In general, the steps of clustering machine learning are:

* Load Libraries

  The first step in applying clustering model is we have to load all libraries are needed. Below the capture of all libraries are needed in clustering: 
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/load_libraries.png)
  
  
* Load Dataset into Spark Dataframe

  Because we will work on spark environment so the dataset must be in spark dataframe. In this step, I created function to load data into spark dataframe. To run this function, first we have to define type of file of dataset (text or parquet) and path where dataset is stored and delimeter like ',' for example or other. 
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/load_dataset.png)
  

* Check the data
  
  After load data, lets do some check of the dataset such as numbers of columns, numbers of observations, names of columns, type of columns, etc. In this part, we also do some changes like rename columns name if the column name too long, change the data type if data type not in accordance or drop unnecessary column.
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/check_data.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/check_data2.png)
  
  
* Define categorical and numerical variables

  In this step, I tried to split the variables based on it's data types. If data types of variables is string will be saved in list called **cat_cols** and if data types of variables is integer or double will be saved in list called **num_cols**. This step applied to make easier in the following step so I don't need to define categorical and numerical variables manually.
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/define_categorical_nummerical_variables.png)
  

* Check Missing Values
  
  Sometimes the data are received is not clean. So, we need to check whether there are missing values or not. Output from this step is the name of columns which have missing values and the number of missing values. To check missing values, actually I created two method:

    - Using pandas dataframe,
    - Using pyspark dataframe. But the prefer method is method using pyspark dataframe so if dataset is too large we can still calculate / check missing values.
    ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/check_missing_values.png)
    ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/check_missing_values2.png)
    
    
* Handle Missing Values

  The approach that used to handle missing values between numerical and categorical variables is different. For numerical variables I fill the missing values with average in it's columns. While for categorical values I fill missing values use most frequent category in that column, therefore count categories which has max values in each columns is needed. 
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/handle_missing_values.png)
  
  
* EDA 

  Create distribution visualization in each variables to get some insight of dataset. 
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/EDA1.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/EDA2.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/EDA3.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/EDA4.png)
  
  
* Handle Outlier

  Outlier is observations that fall below lower side or above upper side.

  To handle outlier the approach is by replacing the value greater than upper side with upper side value and replacing the value lower than lower side with lower side value. So, we need calculate upper and lower side from quantile value, quantile is probability distribution of variable. In General, there are three quantile:

    - Q1 = the value that cut off 25% of the first data when it is sorted in ascending order.
    - Q2 = cut off data, or median, it's 50 % of the data
    - Q3 = the value that cut off 75% of the first data when it is sorted in ascending order.
    - IQR or interquartile range is range between Q1 and Q3. IQR = Q3 - Q1.
  Upper side = Q3 + 1.5 * IQR Lower side = Q1 - 1.5 * IQR

  To calculate quantile in pyspark dataframe I created a function and then created function to calculate uper side, lower side, replacing upper side and replacing lower side. function of replacing upper side and lower side will looping as much as numbers of numerical variables in dataset.
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/handle_outlier.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/handle_outlier2.png)
  
  
* Modelling using K-Mean
  
  Before modelling process, I just select 3 variables numerical from all data so the cluster can be visualized in 3D. Because we work in spark environment so vector assemble still needed to be applied in this data.
  Because K-Mean need define the value of K and to check the best value of K, I optimize of k, group fraction of the data for different k and look for an "elbow" in the cost function. Then, we plot the "elbow" and choose K with little gain.
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/modelling.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/modelling2.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/modelling3.png)
  
  
  From result of K-Mean above, we have 5 clustering of mall customer and save our cluster prediction in dataframe called **prediction2**. To check amount each cluster group dataframe (**prediction2**) by column **prediction** which contains type of cluster or to check average each variables in each cluster just do group by function, like picture below:
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/modelling4.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/modelling5.png)
  
  
  Now, lets try to see the visualization of those five cluster in 3D (age, AnnIncome and SpendScore).
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/visualize_clustering.png)
  ![alt text](https://github.com/elsyifa/Clustering-Pyspark/blob/master/Images/visualize_clustering2.png)
  
  **HOREEEE!!!, we got our clustering**
  
  For more details, please see my code.
