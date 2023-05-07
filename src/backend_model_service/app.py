from flask import Flask, request, jsonify, render_template
from text_gen_model import TextGenModelManager

app = Flask(__name__)
# model_manager = TextGenModelManager()

@app.route("/", methods=["GET"])
def home_view():
    # model_ids = model_manager.get_existing_fine_tuned_models()
    model_ids = ["test"]

    return render_template("index.html", model_ids=model_ids)

# @app.route("/", methods=["POST"])
# def generate_text():
#     try:
#         # get prompts from request body
#         data = request.json
#         prompts = data.get("prompts")

#         # generate text for each prompt
#         generated_texts = model_manager.generate_text_for_each_prompt(prompts)

#         # return generated texts
#         return jsonify({"generated_texts": generated_texts})
#     except Exception as e:
#         # handle errors
#         return jsonify({"error": str(e)}), 500
    

@app.route("/models/<model_id>", methods=["GET"])
def model_details(model_id):
    return render_template("model.html", model_id=model_id)

@app.route("/models/<model_id>", methods=["POST"])
def generate_text(model_id):
    pass

if __name__ == '__main__':
    app.run(debug=True, port=8080) 