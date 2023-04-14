from google.cloud import bigquery

client = bigquery.Client()

# construct schemas 
job_config = bigquery.SourceFormat(
    schema=[
        bigquery.SchemaField("post_title", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("post_content", "STRING"),
        bigquery.SchemaField("")
    ]
)