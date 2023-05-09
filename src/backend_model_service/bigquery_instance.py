import os
from typing import Any
import pandas as pd 
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


# ----------------------------------------------------------------------
#           big query instance
# ----------------------------------------------------------------------
class AITABigQueryInstance():
    def __init__(self) -> None:
        self.client = bigquery.Client()
        self.proj_name = "bonion"
        self.dataset = "AITA_dataset"

        self.post_table_id = "post_table"
        self.comment_table_id = "comment_table"
        self.reply_table = "reply_table"

    # ----------------------------------------------------------------------
    #           helper functions
    # ----------------------------------------------------------------------
    def gcp_table_exists(self,  table_id: str):
        try:
            self.client.get_table(table_id)  # Make an API request.
            return True
        except NotFound:
            return False
        

    # ----------------------------------------------------------------------
    #           get queries
    # ----------------------------------------------------------------------
    def get_post_data(self):

        query_for_post_data = """
            SELECT reddit_post_id AS post_id, post_title, post_self_text AS post_content, posts.upvotes AS post_upvotes, 
                    comment_id, content AS comment_content, comments.upvotes AS comment_upvotes
            FROM {} posts JOIN {} comments ON (posts.reddit_post_id = SUBSTRING(comments.parent_id, 4, 100));
        """.format(self.post_table_id, self.comment_table_id)

        post_join_comment_df = pd.read_gbq(query_for_post_data, project_id=self.proj_name)
        return post_join_comment_df

    
    # ----------------------------------------------------------------------
    #           post queries
    # ----------------------------------------------------------------------