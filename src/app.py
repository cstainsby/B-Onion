import os
import json 
import openai
from flask import Flask, render_template, request, Response

import app_utils
import praw_instance as praw_instance
import transformer_model.model as model
import transformer_model.model_utils as model_utils


# all of the webpage functionality 
# add bot button 
#   - settings (account?)
#   - modal where you can describe bot activity, 
#        how often 
#        where 
#        description of tone (maybe tags) 
#        initiated by cloud functions 
#       



# app filepath
backend_path = os.path.dirname(os.path.realpath(__file__))

# static_frontend_filepath = backend_path + "/static"
# if not os.path.exists(static_frontend_filepath):
#     static_frontend_filepath = backend_path + "/templates"

app = Flask(__name__)
praw_inst = praw_instance.PrawInstance()

openai_model_def = "text-davinci-003"
openai.api_key = os.environ["openai_token"]

@app.route("/", methods=["GET"])
def home():
    # this dictionary will be used to store the various 
    trending_dict = {}

    trending_dict["reddit"] = {}
    trending_dict["reddit"]["python"] = praw_instance.get_hot_by_subreddit(praw_inst, "python")

    return render_template("home.html", trending_dict=trending_dict)


@app.route("/openai/init", methods=["POST"])
def initialize_open_ai_prompting():
    # this message will prime the ongoing openai session for 
    prompt_structure_init_message = """
        Your job will be to help the user create articles or social media 
        posts based on the prompts they provide.  

        An Edition will be defined as
            Edition:
                id: an integer
                title: a string title for the edition
                content: a string of content for the edition

        For the following prompts, they will all be in the form 
            current_prompt: 
                a string describing the instructions,

            referenced_editions:
                a list of editions, each with the above described contents
                Note that this can be empty, if there is no referenced_editions,
                    you can assume that it is empty
                This will be used by you to gain context related to the users 
                    current prompt.

                
        Your responses should be formatted to include
            an Edition:
                should be the the same format listed above, without an id    

        Note that your responses should only ever contain formatted Edition's, nothing else.        
    """

    openai_res = openai.Completion.create(
        model=openai_model_def,
        prompt=prompt_structure_init_message,
        temperature=0,
        max_tokens=5)

    res = Response(json.dumps(openai_res), mimetype="application/json")
    return res


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

    # scan for any edition tags in the prompt, this will 
    edition_ids_to_include = app_utils.scan_for_edition_tag_ids(prompt)
    editions_to_include_dict = app_utils.get_included_editions(currentEditions, edition_ids_to_include)
    editions_strs = app_utils.editions_to_strs(editions_to_include_dict)
    formated_edition_str = app_utils.formated_req_edition_str(editions_strs)

    openai_prompt = """
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

    res = Response(json.dumps(openai_res), mimetype="application/json")
    return res

if __name__ == '__main__':
    app.run(debug=True, port=3000)