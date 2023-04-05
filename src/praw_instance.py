import praw 
import csv 
import os


src_path = os.path.dirname(os.path.abspath(__file__))

class PrawInstance():
    def __init__(self) -> None:
        cred_filepath =  src_path + "/creds/praw_creds.csv"

        with open(cred_filepath, "r") as f_in:
            reader = csv.DictReader(f_in)
            praw_creds = next(reader)

        client_id = praw_creds["client_id"]
        client_secret = praw_creds["client_secret"]
        username = praw_creds["username"]
        password = praw_creds["password"]
        user_agent = praw_creds["user_agent"]

        self.reddit_inst = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent
        )

    def __call__(self) -> praw.Reddit:
        return self.reddit_inst




if __name__ == "__main__":
    # run specifically the PrawInstance for functionality check
    praw_inst = PrawInstance()

    hot_in_rpython = praw_inst().subreddit("python").hot(limit=5)

    for submission in hot_in_rpython:
        print("Title: {}, upvote: {}, downvote: {}, have_we_visited: {}"
              .format(submission.title, submission.ups, submission.downs, submission.visited))

