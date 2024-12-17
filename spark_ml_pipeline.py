from config import  AWS_ACCESS_KEY, AWS_SECRET_KEY
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when
from pyspark.ml.feature import Imputer, VectorAssembler
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


os.environ['PYSPARK_SUBMIT_ARGS'] = """
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 
  --conf spark.hadoop.fs.s3a.endpoint=s3.us-east-2.amazonaws.com 
  --conf spark.executor.extraJavaOptions=-Dcom.amazonaws.services.s3.enableV4=true 
  --conf spark.driver.extraJavaOptions=-Dcom.amazonaws.services.s3.enableV4=true 
  pyspark-shell
"""


spark = SparkSession.builder \
    .appName("PriceOptimizationWithMultipleModels") \
    .getOrCreate()


hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()


hadoopConf.set("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")  
hadoopConf.set("fs.s3a.access.key", AWS_ACCESS_KEY)  
hadoopConf.set("fs.s3a.secret.key", AWS_SECRET_KEY)  
hadoopConf.set("fs.s3a.path.style.access", "true")  
hadoopConf.set("fs.s3a.connection.ssl.enabled", "true")  
hadoopConf.set("fs.s3a.signature.version", "v4")  


s3_bucket = "data-engineering-project-2024-s3"
file_key = "price_optimization/filtered_product_category_data.csv"


df = spark.read \
    .option("header", "true") \
    .csv(f"s3a://{s3_bucket}/{file_key}")


df.show()


df = df.withColumn("product_title", regexp_replace("product_title", r'[\&amp;\#x27;]', ''))


df = df.replace(0, None)


df = df.withColumn("product_price", col("product_price").cast("float"))
df = df.withColumn("product_star_rating", col("product_star_rating").cast("float"))
df = df.withColumn("sales_volume", col("sales_volume").cast("float"))
df = df.withColumn("product_num_offers", col("product_num_offers").cast("int"))
df = df.withColumn("product_original_price", col("product_original_price").cast("float"))
df = df.withColumn("product_minimum_offer_price", col("product_minimum_offer_price").cast("float"))


df = df.withColumn("discount_percentage", 
                   (col("product_original_price") - col("product_price")) / col("product_original_price") * 100)


df = df.withColumn("discount_percentage", 
                   when(col("discount_percentage").isNull() | (col("discount_percentage") == float('inf')) | 
                        (col("discount_percentage") == float('-inf')), 0).otherwise(col("discount_percentage")))


df = df.withColumn("offer_intensity", col("product_num_offers") / (col("product_minimum_offer_price") + 1))

df = df.withColumn("has_reviews", when(col("product_star_rating") > 0, 1).otherwise(0))


df = df.withColumn("weighted_rating", col("product_star_rating") * col("sales_volume"))

df = df.withColumn("price_competitiveness", col("product_price") / (col("product_minimum_offer_price") + 1))


imputer = Imputer(inputCols=["product_price", "product_star_rating", "sales_volume", 
                             "product_num_offers", "product_original_price", "product_minimum_offer_price"],
                  outputCols=["product_price_imputed", "product_star_rating_imputed", "sales_volume_imputed",
                              "product_num_offers_imputed", "product_original_price_imputed", "product_minimum_offer_price_imputed"])

df = imputer.fit(df).transform(df)


scaled_cols = ["product_price", "sales_volume", "product_num_offers", "product_original_price", 
               "product_minimum_offer_price", "discount_percentage", "offer_intensity", "weighted_rating", 
               "price_competitiveness"]
assembler = VectorAssembler(inputCols=scaled_cols, outputCol="features")
df = assembler.transform(df)

pandas_df = df.select("product_star_rating", "sales_volume", "product_num_offers", "discount_percentage", 
                      "offer_intensity", "weighted_rating", "price_competitiveness", "product_price").toPandas()


X = pandas_df.drop(columns=["product_price"])
y = pandas_df["product_price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)  


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', max_depth=6, eta=0.1, subsample=0.8, colsample_bytree=0.8)
}


mae_values = []
rmse_values = []
feature_importances = {}

for model_name, model in models.items():
  
    model.fit(X_train_scaled, y_train)
    
 
    y_pred = model.predict(X_test_scaled)
    
   
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae_values.append(mae)
    rmse_values.append(rmse)

    print(f"Model: {model_name}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print("-" * 50)
    
  
    if model_name == "XGBoost":
        feature_importances[model_name] = model.feature_importances_


plt.figure(figsize=(10,6))
for model_name, model in models.items():
    if model_name == "XGBoost":
        xgb.plot_importance(model)
        plt.title(f'Feature Importance ({model_name})')
        plt.show()


for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, color='blue', label='Predictions')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Predicted vs Actual Product Price ({model_name})')
    plt.legend()
    plt.show()


for model_name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    plt.figure(figsize=(10,6))
    sns.histplot(y_pred, kde=True, color='green', label='Predicted Prices')
    sns.histplot(y_test, kde=True, color='red', label='Actual Prices')
    plt.title(f'Prediction vs Actual Price Distribution ({model_name})')
    plt.legend()
    plt.show()

plt.figure(figsize=(10,6))
plt.bar(models.keys(), mae_values, color='blue', alpha=0.7, label='MAE')
plt.bar(models.keys(), rmse_values, color='red', alpha=0.7, label='RMSE')
plt.xlabel('Model')
plt.ylabel('Error')
plt.title('Model Comparison: MAE vs RMSE')
plt.legend()
plt.show()


spark.stop()



