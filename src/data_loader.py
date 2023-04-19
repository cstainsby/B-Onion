# used for loading in data from praw and converting it to JSONL styling
# which can be used with open ai's fine tuning

import praw 
import pandas as pd

import praw_instance





if __name__ == "__main__":
    praw_inst = praw_instance.PrawInstance()
    subreddit_name = "amitheasshole"


    top_posts = praw_instance.get_top_by_subreddit(praw_inst, subreddit_name, limit=100)
    


