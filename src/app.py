import os
import json 
import queue
import torch
from flask import Flask, render_template, request, Response
from transformer_model import TransformerModel, generate_text

app = Flask(__name__)

# app filepath
app_path = os.path.dirname(os.path.realpath(__file__))

# import vocab
SAVE_PATH = app_path + "/transformer_model/vocab_save/" + "embeddings_vocab.pt"
vocab_obj = torch.load(SAVE_PATH)

# import model
SAVE_PATH = app_path + "/transformer_model/model_save/" + "glove_emb_model.pt"
model = torch.load(SAVE_PATH)


# generator queue definition 
# https://maxhalford.github.io/blog/flask-sse-no-deps/
class generatorPubSub():

    def __init__(self):
        self.listeners = []

    def listen(self):
        q = queue.Queue(maxsize=5)
        self.listeners.append(q)
        return q

    def announce(self, msg):
        for i in reversed(range(len(self.listeners))):
            try:
                self.listeners[i].put_nowait(msg)
            except queue.Full:
                del self.listeners[i]


def format_sse(data: str, event=None) -> str:
    msg = f'data: {data}\n\n'
    if event is not None:
        msg = f'event: {event}\n{msg}'
    return msg




@app.route("/", methods=["GET"])
def hello_world():
    return render_template("base.html")

@app.route('/trans-end-stream', methods=["POST"])
def trans_stream_end():
    print("data:",request.get_json())
    prompt_text = str(request.get_json()["prompt"]).lower().split(" ")

    def stream_trans_text(prompt_text):

        if prompt_text:
            start_seq = prompt_text

            for word in generate_text(
                model=model,
                vocab=vocab_obj,
                start_seq=start_seq,
                max_length=10):
                print("generated word", word)
                yield 'data: {}\n\n'.format(word) 
        else:
            yield "Error: unprocessable input given"

    return Response(stream_trans_text(prompt_text), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, port=8000)