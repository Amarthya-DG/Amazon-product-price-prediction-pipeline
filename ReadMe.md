## Amazon Product Price Optimization Pipeline
Project Overview
This project fetches real-time product data from the Amazon API, processes and stores it in an Amazon S3 bucket, and builds a Machine Learning (ML) model to optimize product pricing. The ML models used include Linear Regression, Random Forest, and XGBoost. Key tasks include:

Fetching product data from the Amazon API.
Cleaning and preprocessing the data.
Uploading the processed data to an Amazon S3 bucket.
Building an ML pipeline for price prediction and optimization.
Comparing model performances and visualizing results.
Project Workflow
1. Data Fetching
File: api_fetch.py
Fetches product data using the RapidAPI Amazon API.
Filters and processes data fields such as:
Product price
Sales volume
Ratings
Discount percentage
Saves the processed data locally as filtered_product_category_data.csv.
2. Uploading Data to Amazon S3
File: upload_to_s3.py
Uploads the processed CSV file to an Amazon S3 bucket using the boto3 library.
Configurable S3 bucket and credentials are set in the config.py file.
3. Machine Learning Pipeline
File: spark_ml_pipeline.py
Reads the data directly from the S3 bucket into a PySpark DataFrame.
Cleans and transforms the data by:
Handling missing values
Generating derived features (e.g., discount_percentage, offer_intensity, price_competitiveness)
Imputing and scaling features.
Models Implemented:
Linear Regression
Random Forest Regressor
XGBoost Regressor
Performance Metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
Visualizations:
Feature importance for XGBoost.
Predicted vs Actual Prices scatter plot.
Price distribution comparison.
Model comparison (MAE vs RMSE bar chart).
Directory Structure
plaintext
Copy code
project-root/
│-- api_fetch.py
│-- upload_to_s3.py
│-- spark_ml_pipeline.py
│-- config.py
│-- filtered_product_category_data.csv  # Local processed file (optional)
│-- README.md

Setup Instructions
1. Prerequisites
Ensure you have the following installed using pip install -r requirements.txt:

Python (3.8+)
PySpark
boto3
scikit-learn
xgboost
pandas
seaborn
matplotlib
RapidAPI account with access to Amazon Product API.
AWS credentials for S3 access.

2. Installation
Install required libraries:

pip install boto3 requests pandas pyspark scikit-learn xgboost matplotlib seaborn

3. Configuration
Create a config.py file in the root directory with the following credentials:


# config.py
API_KEY = "your_rapidapi_key"
API_HOST = "real-time-amazon-data.p.rapidapi.com"

AWS_ACCESS_KEY = "your_aws_access_key"
AWS_SECRET_KEY = "your_aws_secret_key"
S3_BUCKET_NAME = "your_s3_bucket_name"

4. Execution
Fetch Data:

python api_fetch.py
Upload to S3:

python upload_to_s3.py
Run the ML Pipeline:

spark-submit spark_ml_pipeline.py
Results
The project generates MAE and RMSE scores for all models and visualizations such as:
Predicted vs Actual Prices scatter plots.
Prediction vs Actual Price Distributions.
Model performance comparison (bar chart).
Feature importance for XGBoost.
