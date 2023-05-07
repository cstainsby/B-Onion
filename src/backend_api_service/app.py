import os
import json 
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, Response

# from google_secret_access import access_secret_version
import backend_api_service.app_utils as app_utils
import backend_api_service.praw_instance as praw_instance
# import transformer_model.model as model
# import transformer_model.model_utils as model_utils


# all of the webpage functionality 
# add bot button 
#   - settings (account?)
#   - modal where you can describe bot activity, 
#        how often 
#        where 
#        description of tone (maybe tags) 
#        initiated by cloud functions 
#       


app_root_path = Path(__file__).absolute().parent 
local_env_path = app_root_path / Path("./settings.env")

# if there is a settings.env file to reach out to for importing environment variables
#   do it 
# this will only happen in local builds
print(local_env_path.exists())
if local_env_path.exists():
    print("loading environment variables from local file")
    load_dotenv(dotenv_path=local_env_path)



app = Flask(__name__)
praw_inst = praw_instance.PrawInstance()

openai_model_def = "text-davinci-003"
openai.api_key = os.environ.get("openai_access_token")




# ---------------------------------------------------------------------------
#   GET endpoints
# ---------------------------------------------------------------------------

@app.route("/subreddit/amitheasshole", methods=["GET"])
def browse_amitheasshole():
    pinned_subreddits =[
        "AskReddit",
        "amitheasshole",
        "MaliciousCompliance",
        "WritingPrompts"
    ]

    # this dictionary will be used to store the various 
    trending_dict = {}

    # add pinned subreddits to webpage
    trending_dict["reddit"] = {}
    for pinned_reddit_sub in pinned_subreddits:
        trending_dict["reddit"][pinned_reddit_sub] = praw_instance.get_hot_by_subreddit(praw_inst, pinned_reddit_sub)
    
    return render_template("home.html", trending_dict=trending_dict)


@app.route("/", methods=["GET"])
def home():
    return render_template("project_overview.html")




# ---------------------------------------------------------------------------
#   POST endpoints
# ---------------------------------------------------------------------------

@app.route("/openai/prompt", methods=["POST"])
def send_open_ai_prompt():
    """
    input in body includes:
        current_prompt: a string of the current user input prompt,
        editions: all current editions in a list
    """
    req = request.get_json()
    # this prompt will contain everything that will inform how to generate the text
    prompt = req["prompt_contents"]
    currentEditions = req["editions"]

    # scan for any tags in the prompt
    edition_ids_to_include = app_utils.scan_for_edition_tag_ids(prompt)
    reddit_ids_to_include = app_utils.scan_for_reddit_tags(prompt)

    # process any reddit tags if they exist
    reddit_edition_strs = []
    if len(reddit_ids_to_include) > 0:
        reddit_posts_to_id = [
            praw_instance.get_post_by_id(praw_inst, post_id) for post_id in reddit_ids_to_include
            ]
        reddit_edition_strs = [
            app_utils.reddit_post_to_edition_str(reddit_post) for reddit_post in reddit_posts_to_id
        ]

    # process tagged editions if there are any
    editions_strs = []
    if len(edition_ids_to_include) > 0:
        editions_to_include_dict = app_utils.get_included_editions(currentEditions, edition_ids_to_include)
        editions_strs = app_utils.editions_to_strs(editions_to_include_dict)


    # format all linked editions into something that can be processed by our defined input style
    formated_edition_str = app_utils.formated_req_edition_str(
        editions_strs + reddit_edition_strs
        )

    openai_prompt = """
    Your job will be to help the user create text based on the prompts they provide.  

    An Edition will be defined as
        Edition:
            id: an integer
            title: a string title for the edition
            content: a string of content for the edition

    Your text responses should be formatted to include
            a single Edition with nothing after:
                should be the the same format listed above, without an id 

        current_prompt:
            {current_prompt}
        referenced_editions:
            {editions_str}
    """.format(current_prompt=prompt, editions_str=formated_edition_str)

    openai_res = openai.Completion.create(
        model=openai_model_def,
        prompt=openai_prompt,
        temperature=0.5,
        max_tokens=300)

    openai_res_text = app_utils.get_text_from_openai_res(openai_res)

#     openai_res_text = """
# Edition:
#     title: "The Mythical Beast"
#     content: "Long ago, there lived a mythical creature that was said to be the most powerful being in the land. It had the ability to control the elements and could manipulate the very fabric of reality. One day, a brave adventurer set out to find the creature and prove its existence. After a long and perilous journey, the adventurer found the creature and was able to capture it. The creature granted the adventurer three wishes, and with them, the adventurer's dreams came true."
#     """
#     openai_res_text = app_utils.clean_res_str(openai_res_text)

    
    formated_openai_res = app_utils.get_edition_items_from_openai_res_text(openai_res_text)

    res = Response(json.dumps(formated_openai_res), mimetype="application/json")
    return res

@app.route("/reddit/post", methods=["POST"])
def post_to_reddit():
    """
    input in body includes:
        the edition which should be posted
    """
    req = request.get_json()

    subreddit_name = req["subreddit_name"]
    edition_dict = {
        "title": req["title"],
        "content": req["content"]
    }

    praw_instance.post_edition_to_reddit(praw_inst, subreddit_name, edition_dict)

    res = Response(mimetype="application/json")
    return res

if __name__ == '__main__':
    DEBUG_MODE = True if os.environ.get("FLASK_DEBUG") and os.environ.get("FLASK_DEBUG") == "true" else False
    app.run(debug=True, port=3000)