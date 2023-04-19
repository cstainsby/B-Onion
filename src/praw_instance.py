import praw 
import csv 
import os

src_path = os.path.dirname(os.path.abspath(__file__))

class PrawInstance():
    def __init__(self) -> None:

        client_id = os.environ.get("praw_client_id")
        client_secret = os.environ.get("praw_client_secret")
        username = os.environ.get("praw_username")
        password = os.environ.get("praw_password")
        user_agent = "bonion"

        self.reddit_inst = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent
        )

    def __call__(self) -> praw.Reddit:
        return self.reddit_inst


def get_hot_by_subreddit(praw_inst: PrawInstance, subreddit_name: str = "", limit: int = 7):
    """Gets the top posts on a subreddit
        RETURNS: dict of dict"""
    hot_by_sub = {}
    if praw_inst and subreddit_name != "":
        post_obj_lst = [hot_post for hot_post in praw_inst().subreddit(subreddit_name).hot(limit=limit) if not hot_post.stickied]
        for post in post_obj_lst:
            hot_by_sub[post] = {
                "title": post.title,
                "upvotes": post.ups,
                "downvotes": post.downs,
                "selftext": post.selftext
            }
    return hot_by_sub

def get_top_by_subreddit(praw_inst: PrawInstance, subreddit_name: str = "", limit: int = 7):
    """Gets the top posts on a subreddit
        RETURNS: dict of dict"""
    top_by_sub = {}
    if praw_inst and subreddit_name != "":
        post_obj_lst = [top_post for top_post in praw_inst().subreddit(subreddit_name).top(limit=limit) if not top_post.stickied]
        for post in post_obj_lst:
            top_by_sub[post] = {
                "title": post.title,
                "upvotes": post.ups,
                "downvotes": post.downs,
                "selftext": post.selftext
            }
    return top_by_sub

def get_post_by_id(praw_inst: PrawInstance, post_id: str):
    return praw_inst().submission(id=post_id)

def post_edition_to_reddit(praw_inst: PrawInstance, subreddit_name: str, edition: dict):
    if "title" in edition and "content" in edition:
        praw_inst().subreddit(subreddit_name).submit(
            title=edition["title"],
            selftext=edition["content"]
        )

if __name__ == "__main__":
    # run specifically the PrawInstance for functionality check
    praw_inst = PrawInstance()

    hot_in_rpython = get_hot_by_subreddit(praw_inst, "python")

    for submission in hot_in_rpython:
        print("Title: {}, upvote: {}, downvote: {}, have_we_visited: {}"
              .format(submission.title, submission.ups, submission.downs, submission.visited))

