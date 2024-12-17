import boto3
from botocore.exceptions import NoCredentialsError, BotoCoreError
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET_NAME

def upload_to_s3(file_name, bucket_name, s3_key):
    """
    Upload a file to an S3 bucket.

    :param file_name: The path to the file to upload
    :param bucket_name: The name of the S3 bucket
    :param s3_key: The S3 object key (i.e., the path in the bucket)
    :return: True if file was uploaded, else False
    """

    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    try:
        s3.upload_file(file_name, bucket_name, s3_key)
        print(f"File '{file_name}' successfully uploaded to 's3://{bucket_name}/{s3_key}'.")
        return True
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
    except NoCredentialsError:
        print("Error: AWS credentials not available.")
    except BotoCoreError as e:
        print(f"Error: Failed to upload to S3 due to {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False

if __name__ == "__main__":
 
    file_name = "filtered_product_category_data.csv"
    s3_key = "price_optimization/filtered_product_category_data.csv" 
    
    
    success = upload_to_s3(file_name, S3_BUCKET_NAME, s3_key)
    
    if success:
        print("Upload completed successfully.")
    else:
        print("Upload failed.")
