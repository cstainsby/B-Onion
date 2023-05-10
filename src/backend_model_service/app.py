from flask import Flask, request, jsonify, render_template
from saved_models import model_storer
import AITAclassmodel_utils
import app_utils
import requests

app = Flask(__name__)


# class model info 


@app.route("/", methods=["GET"])
def home_view():
    model_ids = model_storer.get_existing_fine_tuned_models()

    return render_template("index.html", model_ids=model_ids)
    

@app.route("/models/<model_name>", methods=["GET"])
def get_model_details(model_name):
    model_details = model_storer.get_model_info(model_name)

    text = request.args.get("text")

    trending_posts = requests.get(f"https://backend-api-service-zkidffnq6a-uc.a.run.app/reddit/hot/amitheasshole").json()

    if text is not None:
        reddit_ids_to_include = app_utils.scan_for_reddit_tags(text)

        # process any reddit tags if they exist
        reddit_str = ""
        if len(reddit_ids_to_include) > 0:
            reddit_posts_to_id = [
                requests.get(f"https://backend-api-service-zkidffnq6a-uc.a.run.app/reddit/post/{post_id}") for post_id in reddit_ids_to_include
                ]

            # just going to include the first liked reddit post
            reddit_str = "title: {}\ncontent: {}".format(reddit_posts_to_id[0].json()["title"], reddit_posts_to_id[0].json()["selftext"])
            # reddit_edition_strs = [
            #     app_utils.reddit_post_to_edition_str(reddit_post) for reddit_post in reddit_posts_to_id if reddit_post.ok
            # ]
            text = reddit_str

        aita_class_model, aita_tokenizer = AITAclassmodel_utils.get_model_and_tokenizer()
        
        if model_name == "AITAclassmodel":
            class_label = AITAclassmodel_utils.predict(aita_class_model, aita_tokenizer, text)
    else:
        class_label = None

    model_details = model_storer.get_model_info(model_name)
    return render_template("model.html", model_details=model_details, class_label=class_label, trending_posts=trending_posts)


@app.route("/models/<model_name>", methods=["POST"])
def use_model(model_name):
    print("test")
    req = request.get_json()
    print(req)
    text = req["text"]

    print("WAITING")
    aita_class_model, aita_tokenizer = AITAclassmodel_utils.get_model_and_tokenizer()
    
    if model_name == "AITAclassmodel":
        class_label = AITAclassmodel_utils.predict(aita_class_model, aita_tokenizer, text)
        print("PREDICTED CLASS LABEL", class_label)
    model_details = model_storer.get_model_info(model_name)
    return render_template("model.html", model_details=model_details, class_label=class_label)




if __name__ == '__main__':
    app.run(debug=True, port=8080) 