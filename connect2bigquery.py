from google.cloud import bigquery
import pandas as pd
import os

# Set Google Cloud credentials (if not set as environment variable)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/Meera/Downloads/mydbconnet-4a43d2bddd92.json"

# Initialize BigQuery client
client = bigquery.Client()

# Define project, dataset, and table
project_id = "mydbconnet"
dataset_id = "bigquery-public-data" 
table_id = "austin_311.311_service_requests"

# Construct table reference
table_ref = f"{project_id}.{dataset_id}.{table_id}"

# Run a sample query
query = f"SELECT * FROM `{table_ref}` LIMIT 10"
query_job = client.query(query)  # Execute query

# Convert to Pandas DataFrame
df = query_job.to_dataframe()

# Print results
print(df)


