import torch 
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


labels = ['nta', 'yta', 'info', 'nah', 'esh', 'ywbta']
id2label = {0: 'nta', 1: 'yta', 2: 'info', 3: 'nah', 4: 'esh', 5: 'ywbta'}
label2id = {'nta': 0, 'yta': 1, 'info': 2, 'nah': 3, 'esh': 4, 'ywbta': 5}

def get_model_and_tokenizer():
    aita_class_model = AutoModelForSequenceClassification.from_pretrained("./saved_models/store/AITAclassmodel/", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained("./saved_models/store/AITAclassmodel/")

    return aita_class_model, tokenizer


def predict(aita_class_model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, text: str):
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    encoding = {k: v.to(aita_class_model.device) for k,v in encoding.items()}

    outputs = aita_class_model(**encoding)

    logits = outputs.logits

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    return predicted_labels[0] 