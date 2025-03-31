import os

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from pyspark.ml.feature import Imputer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import AWS_ACCESS_KEY, AWS_SECRET_KEY

print("Amazon Product Price Optimization Tool")
print("=====================================")

# Configure Spark session with AWS credentials
os.environ["PYSPARK_SUBMIT_ARGS"] = """
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262 
  --conf spark.hadoop.fs.s3a.endpoint=s3.us-east-2.amazonaws.com 
  --conf spark.executor.extraJavaOptions=-Dcom.amazonaws.services.s3.enableV4=true 
  --conf spark.driver.extraJavaOptions=-Dcom.amazonaws.services.s3.enableV4=true 
  pyspark-shell
"""

spark = SparkSession.builder.appName("PriceOptimizationTool").getOrCreate()

hadoopConf = spark.sparkContext._jsc.hadoopConfiguration()
hadoopConf.set("fs.s3a.endpoint", "s3.us-east-2.amazonaws.com")
hadoopConf.set("fs.s3a.access.key", AWS_ACCESS_KEY)
hadoopConf.set("fs.s3a.secret.key", AWS_SECRET_KEY)
hadoopConf.set("fs.s3a.path.style.access", "true")
hadoopConf.set("fs.s3a.connection.ssl.enabled", "true")
hadoopConf.set("fs.s3a.signature.version", "v4")

# Load data from S3
s3_bucket = "data-engineering-project-2024-s3"
file_key = "price_optimization/filtered_product_category_data.csv"

print(f"Loading data from S3: s3a://{s3_bucket}/{file_key}")
df = spark.read.option("header", "true").csv(f"s3a://{s3_bucket}/{file_key}")

# Data cleaning and preprocessing (similar to spark_ml_pipeline.py)
df = df.withColumn(
    "product_title", regexp_replace("product_title", r"[\&amp;\#x27;]", "")
)
df = df.replace(0, None)

# Cast columns to appropriate types
df = df.withColumn("product_price", col("product_price").cast("float"))
df = df.withColumn("product_star_rating", col("product_star_rating").cast("float"))
df = df.withColumn("sales_volume", col("sales_volume").cast("float"))
df = df.withColumn("product_num_offers", col("product_num_offers").cast("int"))
df = df.withColumn(
    "product_original_price", col("product_original_price").cast("float")
)
df = df.withColumn(
    "product_minimum_offer_price", col("product_minimum_offer_price").cast("float")
)

# Feature engineering
df = df.withColumn(
    "discount_percentage",
    (col("product_original_price") - col("product_price"))
    / col("product_original_price")
    * 100,
)
df = df.withColumn(
    "discount_percentage",
    when(
        col("discount_percentage").isNull()
        | (col("discount_percentage") == float("inf"))
        | (col("discount_percentage") == float("-inf")),
        0,
    ).otherwise(col("discount_percentage")),
)
df = df.withColumn(
    "offer_intensity",
    col("product_num_offers") / (col("product_minimum_offer_price") + 1),
)
df = df.withColumn("has_reviews", when(col("product_star_rating") > 0, 1).otherwise(0))
df = df.withColumn("weighted_rating", col("product_star_rating") * col("sales_volume"))
df = df.withColumn(
    "price_competitiveness",
    col("product_price") / (col("product_minimum_offer_price") + 1),
)

# Handle missing values
imputer = Imputer(
    inputCols=[
        "product_price",
        "product_star_rating",
        "sales_volume",
        "product_num_offers",
        "product_original_price",
        "product_minimum_offer_price",
    ],
    outputCols=[
        "product_price_imputed",
        "product_star_rating_imputed",
        "sales_volume_imputed",
        "product_num_offers_imputed",
        "product_original_price_imputed",
        "product_minimum_offer_price_imputed",
    ],
)
df = imputer.fit(df).transform(df)

# Select and assemble features
assembler = VectorAssembler(
    inputCols=[
        "product_star_rating",
        "product_num_offers",
        "product_original_price",
        "product_minimum_offer_price",
        "discount_percentage",
        "offer_intensity",
        "weighted_rating",
        "price_competitiveness",
    ],
    outputCol="features",
)
df = assembler.transform(df)

# Convert to pandas for easier manipulation
print("Converting data to pandas DataFrame...")
pandas_df = df.select(
    "product_title",
    "product_price",
    "sales_volume",
    "product_original_price",
    "product_minimum_offer_price",
    "product_star_rating",
    "product_num_offers",
    "discount_percentage",
    "offer_intensity",
    "weighted_rating",
    "price_competitiveness",
).toPandas()

print(f"Total products loaded: {len(pandas_df)}")

# ----- SALES VOLUME PREDICTION (KEY DIFFERENCE FROM ORIGINAL PIPELINE) -----
# This is critical for price optimization as we need to predict how sales change with price

print("\nTraining sales volume prediction model...")
# Prepare data for sales volume prediction
sales_X = pandas_df.drop(columns=["sales_volume", "product_title"])
sales_y = pandas_df["sales_volume"]

# Split data for training and testing
sales_X_train, sales_X_test, sales_y_train, sales_y_test = train_test_split(
    sales_X, sales_y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
sales_X_train_scaled = scaler.fit_transform(sales_X_train)
sales_X_test_scaled = scaler.transform(sales_X_test)

# Train sales volume prediction model
sales_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

sales_model.fit(sales_X_train_scaled, sales_y_train)

# Evaluate model
sales_y_pred = sales_model.predict(sales_X_test_scaled)
sales_mae = mean_absolute_error(sales_y_test, sales_y_pred)
sales_rmse = mean_squared_error(sales_y_test, sales_y_pred, squared=False)

print(f"Sales Volume Prediction Model - MAE: {sales_mae:.2f}, RMSE: {sales_rmse:.2f}")

# ----- PRICE OPTIMIZATION FUNCTIONS -----


def estimate_profit(price, product_features, original_price, min_competitor_price):
    """
    Estimate profit for a given price point

    Args:
        price: The price to test
        product_features: Features of the product
        original_price: Original price of the product
        min_competitor_price: Minimum competitor price

    Returns:
        tuple: (estimated_profit, predicted_sales)
    """
    # Assume cost is 60% of original price (40% margin)
    # In a real implementation, this would come from actual cost data
    estimated_cost = original_price * 0.6

    # Create a copy of features with the new price
    features_copy = product_features.copy()
    features_copy["product_price"] = price

    # Update price-dependent features
    features_copy["discount_percentage"] = (
        (original_price - price) / original_price
    ) * 100
    features_copy["price_competitiveness"] = price / (min_competitor_price + 1)

    # Drop title and sales_volume as they're not used for prediction
    if "product_title" in features_copy:
        features_copy = features_copy.drop("product_title")
    if "sales_volume" in features_copy:
        features_copy = features_copy.drop("sales_volume")

    # Scale features
    features_array = np.array([features_copy.values])
    features_scaled = scaler.transform(features_array)

    # Predict expected sales at this price
    predicted_sales = max(0, sales_model.predict(features_scaled)[0])

    # Calculate profit
    profit = (price - estimated_cost) * predicted_sales
    return profit, predicted_sales


def find_optimal_price(product_data):
    """
    Find the profit-maximizing price for a product

    Args:
        product_data: DataFrame row containing product data

    Returns:
        dict: Optimization results
    """
    # Extract necessary data
    original_price = product_data["product_original_price"]
    min_competitor_price = product_data["product_minimum_offer_price"]
    current_price = product_data["product_price"]

    # Test price range: 60% to 130% of current price
    # This range can be adjusted based on business constraints
    min_price = max(
        current_price * 0.6, original_price * 0.4
    )  # Ensure price covers estimated costs
    max_price = min(
        current_price * 1.3, original_price * 1.2
    )  # Cap at 120% of original price

    # Initialize variables
    best_profit = -float("inf")
    optimal_price = current_price
    best_sales = 0
    current_profit = 0

    # Store price-profit data for visualization
    price_points = []
    profits = []
    sales_volumes = []

    # Search for optimal price
    for test_price in np.linspace(min_price, max_price, 30):
        profit, sales = estimate_profit(
            test_price, product_data, original_price, min_competitor_price
        )

        # Store data for visualization
        price_points.append(test_price)
        profits.append(profit)
        sales_volumes.append(sales)

        # Save current price profit for comparison
        if abs(test_price - current_price) < 0.01:
            current_profit = profit

        # Update if better profit found
        if profit > best_profit:
            best_profit = profit
            optimal_price = test_price
            best_sales = sales

    # Calculate profit improvement
    profit_improvement = 0
    if current_profit > 0:
        profit_improvement = ((best_profit - current_profit) / current_profit) * 100

    return {
        "product_title": product_data.get("product_title", "Unknown Product"),
        "current_price": current_price,
        "optimal_price": optimal_price,
        "price_change_percentage": ((optimal_price - current_price) / current_price)
        * 100,
        "current_profit_estimate": current_profit,
        "optimal_profit_estimate": best_profit,
        "profit_improvement_percentage": profit_improvement,
        "projected_sales": best_sales,
        "price_points": price_points,
        "profits": profits,
        "sales_volumes": sales_volumes,
    }


# ----- RUN OPTIMIZATION -----

print("\nRunning price optimization...")
# Sample 5 products for demonstration
# In production, you would process all products or a specific segment
sample_products = pandas_df.sample(min(5, len(pandas_df)))
optimization_results = []

for idx, product in sample_products.iterrows():
    print(
        f"Optimizing price for product: {product.get('product_title', 'Unknown')[:50]}..."
    )
    result = find_optimal_price(product)
    optimization_results.append(result)

# ----- DISPLAY RESULTS -----

print("\n--- Price Optimization Results ---")
for i, result in enumerate(optimization_results):
    print(f"\nProduct {i + 1}: {result['product_title'][:50]}...")
    print(f"  Current Price: ${result['current_price']:.2f}")
    print(
        f"  Recommended Price: ${result['optimal_price']:.2f} ({result['price_change_percentage']:.1f}% change)"
    )
    print(
        f"  Estimated Profit Increase: {result['profit_improvement_percentage']:.1f}%"
    )
    print(f"  Projected Sales at New Price: {result['projected_sales']:.0f} units")

# ----- VISUALIZE RESULTS -----

# Price-Profit Curve for first product
if optimization_results:
    result = optimization_results[0]

    plt.figure(figsize=(12, 8))

    # Plot 1: Price-Profit Curve
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(result["price_points"], result["profits"], "b-", label="Estimated Profit")
    ax1.set_xlabel("Price ($)")
    ax1.set_ylabel("Profit ($)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.axvline(
        x=result["current_price"], color="r", linestyle="--", label="Current Price"
    )
    ax1.axvline(
        x=result["optimal_price"], color="g", linestyle="--", label="Optimal Price"
    )
    ax1.set_title(f"Price Optimization Analysis for {result['product_title'][:40]}...")
    ax1.legend()

    # Plot 2: Price-Sales Volume Curve
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(
        result["price_points"],
        result["sales_volumes"],
        "orange",
        label="Projected Sales",
    )
    ax2.set_xlabel("Price ($)")
    ax2.set_ylabel("Sales Volume (units)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.axvline(
        x=result["current_price"], color="r", linestyle="--", label="Current Price"
    )
    ax2.axvline(
        x=result["optimal_price"], color="g", linestyle="--", label="Optimal Price"
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig("price_optimization_curves.png")
    plt.show()

    # Summary of all optimizations
    plt.figure(figsize=(10, 6))

    product_names = [
        result["product_title"][:20] + "..." for result in optimization_results
    ]
    current_prices = [result["current_price"] for result in optimization_results]
    optimal_prices = [result["optimal_price"] for result in optimization_results]

    x = range(len(product_names))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        current_prices,
        width,
        label="Current Price",
        color="blue",
    )
    plt.bar(
        [i + width / 2 for i in x],
        optimal_prices,
        width,
        label="Optimal Price",
        color="green",
    )

    plt.xlabel("Products")
    plt.ylabel("Price ($)")
    plt.title("Current vs Optimal Prices")
    plt.xticks(x, product_names, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig("price_comparison.png")
    plt.show()

print(
    "\nOptimization complete. Results saved to price_optimization_curves.png and price_comparison.png"
)
spark.stop()
