from flask import Flask, request, jsonify, render_template
from text_gen_model import TextGenModelManager
from saved_models import model_storer

app = Flask(__name__)
# model_manager = TextGenModelManager()

@app.route("/", methods=["GET"])
def home_view():
    # model_ids = model_manager.get_existing_fine_tuned_models()
    model_ids = model_storer.get_existing_fine_tuned_models()

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
def get_model_details(model_id):
    model_details = model_storer.get_model_info(model_id)
    return render_template("model.html", model_details=model_details)

@app.route("/models/<model_id>", methods=["POST"])
def generate_text(model_id):
    pass

@app.route("/create_model", methods=["POST"])
def create_model():
    data = request.form
    model_name = data["model_name"]
    model_type = data["model_type"]
    model_desc = data["model_desc"]

    # Create a new model using the TextGenModelManager class
    model_manager = TextGenModelManager()
    model = model_manager.model

    # Save the model to disk
    model_output_dir = os.path.join(model_manager.model_output_dir, model_name)
    model.save_pretrained(model_output_dir)

    # Return a response
    response = {
        "status": "success",
        "message": "Model created successfully",
        "model_name": model_name,
        "model_type": model_type,
        "model_desc": model_desc,
        "model_output_dir": model_output_dir
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=8080) 