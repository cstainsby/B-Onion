import praw 
import os
import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
#           analysis helper functions
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
#           getter functions for praw data
# ----------------------------------------------------------------------
def get_hot_by_subreddit(praw_inst: PrawInstance, subreddit_name: str = "", limit: int = 7) -> dict:
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

def get_top_by_subreddit(praw_inst: PrawInstance, subreddit_name: str = "", limit: int = 7) -> dict:
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
                "selftext": post.selftext,
                "numcomments": post.num_comments
            }
    return top_by_sub

def get_top_comments_by_post_id(praw_inst: PrawInstance, post_id: str, comment_limit: int = 5):
    """Getter for a set number of comments 
        RETURNS dict {
            comment_id: {
                content,
                upvotes
            }
        }"""
    comments_and_replies_dict = {}

    post = get_post_by_id(praw_inst, post_id)
    comments = post.comments 

    for i, comment in enumerate(comments):
        if i >= comment_limit:
            break
        com_id = comment.id
        com_content = comment.body
        com_upvotes = comment.score

        comments_and_replies_dict[com_id] = {
            "content": com_content,
            "upvotes": com_upvotes
        }
    
    return comments_and_replies_dict


def get_top_comments_and_top_replies_by_post_id(praw_inst: PrawInstance, post_id: str, comment_limit: int = 5, reply_limit: int = 5):
    """Getter for a set number of comments 
        RETURNS dict {
            comment_id: {
                content,
                upvotes,
                replies: {
                    reply_id: {
                        content,
                        upvotes
                    }
                }
            }
        }"""
    comments_and_replies_dict = {}

    post = get_post_by_id(praw_inst, post_id)
    comments = post.comments.list()

    for i, comment in enumerate(comments):
        if i >= comment_limit:
            break
        com_id = comment.id
        com_content = comment.body
        com_upvotes = comment.score

        comments_and_replies_dict[com_id] = {
            "content": com_content,
            "upvotes": com_upvotes,
            "replies": {}
        }

        replies_at_i = comment.replies.list()
        if len(comment.replies) > 0:
            for j, reply in enumerate(replies_at_i):
                if j >= comment_limit:
                    break

                rep_id = reply.id
                rep_content = reply.body
                rep_upvotes = reply.score
                    
                comments_and_replies_dict[com_id][rep_id] = {
                    "content": rep_content,
                    "upvotes": rep_upvotes
                }
    
    return comments_and_replies_dict



# ----------------------------------------------------------------------
#           analysis helper functions
# ----------------------------------------------------------------------
def post_dict_to_df(post_dict: dict):
    """A helper function for working with the post dictionary returned by get_top_by_subreddit and get_hot_by_subreddit"""

    data = []

    for post_id_key, post_values in post_dict.items():
        post_title = post_values["title"]
        post_self_text = post_values["selftext"]
        post_upvotes = post_values["upvotes"]
        post_num_comments = post_values["numcomments"]

        new_row = [str(post_id_key), str(post_title), str(post_self_text), int(post_upvotes), int(post_num_comments)]

        data.append(new_row)
    
    data = np.array(data)
    
    print("full data shape", data.shape)

    post_df = pd.DataFrame(
        data=data,
        columns=["reddit_post_id", "post_title", "post_self_text", "upvotes", "num_responses"]
    )
    return post_df


def comment_dict_to_df(comment_id: dict):
    return

def get_post_by_id(praw_inst: PrawInstance, post_id: str):
    """Getter for a reddit post given its post_id"""
    return praw_inst().submission(id=post_id)



# ----------------------------------------------------------------------
#           in-app reddit interaction functions
# ----------------------------------------------------------------------
def post_edition_to_reddit(praw_inst: PrawInstance, subreddit_name: str, edition: dict):
    """Post an edition to reddit as a post within a subreddit"""
    if "title" in edition and "content" in edition:
        praw_inst().subreddit(subreddit_name).submit(
            title=edition["title"],
            selftext=edition["content"]
        )

def post_comment_to_post(praw_inst: PrawInstance, post_id: str, comment_content: str):
    """Post a comment to a given post id"""
    return

if __name__ == "__main__":
    # run specifically the PrawInstance for functionality check
    praw_inst = PrawInstance()

    hot_in_rpython = get_hot_by_subreddit(praw_inst, "python")

    for submission in hot_in_rpython:
        print("Title: {}, upvote: {}, downvote: {}, have_we_visited: {}"
              .format(submission.title, submission.ups, submission.downs, submission.visited))

