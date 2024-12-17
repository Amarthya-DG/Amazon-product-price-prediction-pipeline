import requests
import pandas as pd
import re
from config import API_KEY, API_HOST

def fetch_and_filter_product_category_data():
    url = "https://real-time-amazon-data.p.rapidapi.com/products-by-category"
    querystring = {
        "category_id": "2478868012", 
        "page": "1",
        "country": "US",
        "sort_by": "RELEVANCE",
        "product_condition": "ALL",
        "is_prime": "false",
        "deals_and_discounts": "NONE"
    }

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_HOST
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        products = data.get("data", {}).get("products", [])

        filtered_data = []
        for product in products:
            product_price = product.get("product_price")
            product_original_price = product.get("product_original_price")
            sales_volume = product.get("sales_volume", "")

           
            numeric_sales_volume = 0
            if sales_volume:
                match = re.search(r'(\d+)([KkMm]?)', sales_volume)
                if match:
                    number = int(match.group(1))
                    multiplier = match.group(2).lower()
                    if multiplier == 'k':
                        numeric_sales_volume = number * 1000
                    elif multiplier == 'm':
                        numeric_sales_volume = number * 1000000
                    else:
                        numeric_sales_volume = number

            filtered_data.append({
                "asin": product.get("asin"),
                "product_title": product.get("product_title"),
                "product_price": float(product_price.replace("$", "")) if product_price else 0.0,
                "product_star_rating": float(product.get("product_star_rating", "0.0")) if product.get("product_star_rating") else 0.0,
                "sales_volume": numeric_sales_volume,
                "product_num_offers": int(product.get("product_num_offers", "0")) if product.get("product_num_offers") else 0,
                "product_original_price": float(product_original_price.replace("$", "")) if product_original_price else 0.0,
                "product_minimum_offer_price": float(product.get("product_minimum_offer_price", "0.0").replace("$", "")) if product.get("product_minimum_offer_price") else 0.0
            })

      
        df = pd.DataFrame(filtered_data)


        df.to_csv("filtered_product_category_data.csv", index=False)
        print("Filtered data saved to filtered_product_category_data.csv.")
    else:
        print(f"Failed to fetch Product Category data: {response.status_code}")

if __name__ == "__main__":
    fetch_and_filter_product_category_data()
