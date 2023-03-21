import json 
import zipfile 
import torch
from flask import Flask, render_template, request
# import transformer_model

app = Flask(__name__)


# Add model
# model = transformer_model.TransformerModel(
#   embedding_dim = 300,
#   hidden_dim = 300,
#   num_layers = 2,
#   num_heads = 4,
#   vocab_len = 400004,
#   max_len = 100,
#   dropout_p = 0.2
# )


# with zipfile.ZipFile('vocab_mapping_dict.zip', 'r') as f:
#   f.extractall('extracted_files')

# with open("extracted_files/data.json", "r") as f:
#   vocab_mappings = json.load(f)

# reload state into model
# model.load_state_dict(torch.load("model_save/GloVeDemo.pt"))


@app.route("/", methods=["GET"])
def hello_world():
    return render_template("base.html")

@app.route('/trans-end', methods=["POST"])
def my_endpoint():
    print("data:",request.get_json())
    prompt_text = str(request.get_json()["prompt"]).lower().split(" ")
    
    # Do something here
    if prompt_text:
      start_seq = prompt_text
    else:
      start_seq = ["the", "quick", "brown"]

    res = start_seq
    res = " ".join(res)
    return res

if __name__ == '__main__':
    app.run(debug=True, port=8000)