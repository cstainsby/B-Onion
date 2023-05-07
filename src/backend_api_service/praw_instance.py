import praw 
from praw.models import MoreComments
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
        if isinstance(comment, MoreComments) or comment.stickied:
            continue
        if i >= comment_limit:
            break
        com_id = comment.id
        com_content = comment.body
        com_upvotes = comment.score
        com_parent_id = comment.parent_id

        comments_and_replies_dict[com_id] = {
            "content": com_content,
            "upvotes": com_upvotes,
            "parent_id": com_parent_id,
            "replies": {}
        }

        replies_at_i = comment.replies.list()
        # print("replies at i", replies_at_i)
        for j, reply in enumerate(replies_at_i):
            if isinstance(reply, MoreComments):
                continue
            if j >= reply_limit:
                break

            rep_id = reply.id
            rep_content = reply.body
            rep_upvotes = reply.score
            rep_parent_id = reply.parent_id
                
            comments_and_replies_dict[com_id]["replies"][rep_id] = {
                "content": rep_content,
                "upvotes": rep_upvotes,
                "parent_id": rep_parent_id
            }
            # print("added reply", comments_and_replies_dict[com_id][rep_id])
    
    return comments_and_replies_dict

def get_post_by_id(praw_inst: PrawInstance, post_id: str):
    """Getter for a reddit post given its post_id"""
    return praw_inst().submission(id=post_id)

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

    post_df = pd.DataFrame(
        data=data,
        columns=["reddit_post_id", "post_title", "post_self_text", "upvotes", "num_responses"]
    )
    return post_df


def comment_and_reply_dict_to_df(comments_and_replies_dict: dict):
    """Helper function for converting dict data structure returned by get_top_comments_and_top_replies_by_post_id
        into two dataframes, one for comments, one for replies"""
    
    comment_data = []
    reply_data = []

    for comment_id, comment_values in comments_and_replies_dict.items():
        com_content = comment_values["content"]
        com_upvotes = comment_values["upvotes"]
        com_parent_id = comment_values["parent_id"]
        com_replies = comment_values["replies"]

        new_com_row = [str(comment_id), str(com_parent_id), str(com_content), int(com_upvotes)]
        comment_data.append(new_com_row)

        for reply_id, reply_values in dict(com_replies).items():
            rep_content = reply_values["content"]
            rep_upvotes = reply_values["upvotes"]
            rep_parent_id = reply_values["parent_id"]

            new_rep_row = [str(reply_id), str(rep_parent_id), str(rep_content), int(rep_upvotes)]
            reply_data.append(new_rep_row)

    comments_df = pd.DataFrame(
        data=comment_data,
        columns=["comment_id", "parent_id", "content", "upvotes"]
    )
    replies_df = pd.DataFrame(
        data=reply_data,
        columns=["reply_id", "parent_id", "content", "upvotes"]
    )

    return comments_df, replies_df



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

