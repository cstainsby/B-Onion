import os
import json 
from pathlib import Path
# from dotenv import load_dotenv
from flask import Flask, render_template, request, Response

import praw_instance as praw_instance



app_root_path = Path(__file__).absolute().parent 
local_env_path = app_root_path / Path("./settings.env")

# if there is a settings.env file to reach out to for importing environment variables
#   do it 
# this will only happen in local builds
# if local_env_path.exists():
#     print("loading environment variables from local file")
#     load_dotenv(dotenv_path=local_env_path)



app = Flask(__name__)
praw_inst = praw_instance.PrawInstance()




@app.route("/", methods=["GET"])
def home():
    endpoints = [
        { 
            "name": "Post To Reddit",
            "url": "/reddit/submission/post",
            "methods": "POST",
            "description": """This endpoint will take a request with items:
                subreddit_name: str,
                title: str,
                content: str
                It will take these items and make a post to the chosen subreddit
                """
        },
        { 
            "name": "Comment on Post In Reddit",
            "url": "/reddit/comment/post/",
            "methods": "POST",
            "description": """This endpoint will take a request with items:
                submission_id: str,
                content: str
                It will take these items and make a comment on the post/submission specified
                """
        },
    ]
    return render_template("index.html", title='Reddit Analysis Data API', endpoints=endpoints)

# ---------------------------------------------------------------------------
#   GET endpoints
# ---------------------------------------------------------------------------

@app.route("/subreddit/amitheasshole", methods=["GET"])
def browse_amitheasshole():
    pinned_subreddits = [
        "amitheasshole"
    ]

    # this dictionary will be used to store the various 
    trending_dict = {}

    # add pinned subreddits to webpage
    trending_dict["reddit"] = {}
    for pinned_reddit_sub in pinned_subreddits:
        trending_dict["reddit"][pinned_reddit_sub] = praw_instance.get_hot_by_subreddit(praw_inst, pinned_reddit_sub)
    
    return render_template("home.html", trending_dict=trending_dict)

@app.route("/reddit/post/<post_id>", methods=["GET"])
def get_post_by_id(post_id):
    post = praw_instance.get_post_by_id(praw_inst, post_id)
    print("POST", post)
    post_json = json.dumps(post)
    res = Response(response=post_json, mimetype="application/json")
    return res

@app.route("/reddit/hot/<subreddit_name>", methods=["GET"])
def get_hot_posts_from_AITA(subreddit_name):
    hot_posts = praw_instance.get_hot_by_subreddit(praw_inst, subreddit_name)
    reformated_posts = {}
    for key, val in hot_posts.items():
        reformated_posts[key.id] = val
    post_json = json.dumps(reformated_posts)
    res = Response(response=post_json, mimetype="application/json")
    return res

# ---------------------------------------------------------------------------
#   POST endpoints
# ---------------------------------------------------------------------------
@app.route("/reddit/submission/post", methods=["POST"])
def make_submission_to_reddit():
    req = request.get_json()

    subreddit_name = req["subreddit_name"]
    edition_dict = {
        "title": req["title"],
        "content": req["content"]
    }

    praw_instance.post_edition_to_reddit(praw_inst, subreddit_name, edition_dict)

    res = Response(mimetype="application/json")
    return res

@app.route("/reddit/comment/post/", methods=["POST"])
def make_comment_on_submission_to_reddit():
    req = request.get_json()

    submission_id = req["submission_id"]
    content = req["content"]

    praw_instance.post_comment_to_post(praw_inst, submission_id, content)

    res = Response(mimetype="application/json")
    return res


if __name__ == '__main__':
    DEBUG_MODE = True if os.environ.get("FLASK_DEBUG") and os.environ.get("FLASK_DEBUG") == "true" else False
    app.run(debug=True, port=8000)