import os
import json 
import queue
import torch
from flask import Flask, render_template, request, Response

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

# import vocab
SAVE_PATH = backend_path + "/transformer_model/vocab_save/" + "embeddings_vocab.pt"
vocab_obj = torch.load(SAVE_PATH)

# import model
SAVE_PATH = backend_path + "/transformer_model/model_save/" + "glove_emb_model.pt"
transformer = torch.load(SAVE_PATH)



@app.route("/", methods=["GET"])
def home():
    # this dictionary will be used to store the various 
    trending_dict = {}

    trending_dict["reddit"] = {}
    trending_dict["reddit"]["python"] = praw_instance.get_hot_by_subreddit(praw_inst, "python")

    return render_template("home.html", trending_dict=trending_dict)




# @app.route('/trans-end-stream', methods=["POST"])
# def trans_stream_end():
#     print("data:",request.get_json())
#     prompt_text = str(request.get_json()["prompt"]).lower().split(" ")

#     def stream_trans_text(prompt_text):

#         if prompt_text:
#             start_seq = prompt_text

#             for word in model_utils.generate_text(
#                 model=transformer,
#                 vocab=vocab_obj,
#                 start_str=start_seq,
#                 max_length=50):
#                 print("generated word", word)
#                 yield 'data: {}\n\n'.format(word) 
#         else:
#             yield "Error: unprocessable input given"

#     return Response(stream_trans_text(prompt_text), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, port=3000)