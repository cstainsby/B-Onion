from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
import os

import evaluate

class TextGenModelManager():
    def __init__(self) -> None:
        checkpoint = "gpt2-large"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

        self.penalty_alpha = 0.6
        self.top_k = 4
        self.max_new_tokens = 100

        self.model_output_dir = "saved_models/text_gen"

    def get_existing_fine_tuned_models(self) -> list:
        """
        Returns a list of all text gen fine-tuned models.
        
        RETURNS:
            selected_model: list
                a list of all saved model names
        """
        fine_tune_ids = []

        # Get all filenames in the directory
        filenames = os.listdir(self.model_output_dir)

        # Iterate over the filenames and split them into their base names and extensions
        for filename in filenames:
            basename, extension = os.path.splitext(filename)
            fine_tune_ids.append(basename)
            
        return fine_tune_ids

    
    def get_model_from_path(self, model_fname: str = ""):
        """
        Returns a fine-tuned model from the specified path, or the base model if no path is provided.

        PARAMS:
            model_fname: str
                name of the fine-tuned model file in the saved models directory, without the '.pt' extension

        RETURNS:
            selected_model: transformers.PreTrainedModel
                the selected fine-tuned model or the base model if no path is provided or the file doesn't exist
        """
        selected_model = None 

        if model_fname is not None:
            model_path = f"{self.model_output_dir}/{model_fname}.pt"
            if os.path.exists(model_path):
                selected_model = AutoModelForCausalLM.from_pretrained(model_path)
            else: 
                selected_model = self.model
        
        return selected_model

    def generate_text_for_each_prompt(self, inputs: list, model_fname= "") -> list:
        """
        Pass in a list of strings for a given model to use for text generation

        PARAMS:
            inputs: list of str 
                what the model is generating off of
            model_fname: str
                which model to use, if no file named model_fname in saved_models/, 
                generate using the base model
        RETURNS:
            decoded_outputs: list
                list of generated strings for each input
        """
        # get model
        selected_model = self.get_model_from_path(model_fname)

        # generate output tensor
        outputs = selected_model.generate(
            **inputs, 
            penalty_alpha=self.penalty_alpha, 
            top_k=self.top_k, 
            max_new_tokens=self.max_new_tokens
        )
        
        # decode tensor
        decoded_outputs_list = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoded_outputs_list
    
    def fine_tune_and_save(self, model_fname: str, train_dataset, output_dir, num_train_epochs=1) -> str:
        """
        Fine-tune the GPT-2 model on a given dataset and save the fine-tuned model as a .pt file.
        
        PARAMS:
            model_fname: str
                Name of the fine-tuned model file to save.
            train_dataset: Dataset
                Training dataset to use for fine-tuning the model.
            output_dir: str
                Directory to save the fine-tuned model and training artifacts.
            num_train_epochs: int, optional
                Number of training epochs to run. Defaults to 1.
                
        RETURNS:
            output_model_path: str
                Path to the saved fine-tuned model file.
        """
        
        # define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=1,
            save_total_limit=2,
            overwrite_output_dir=True
        )

        # create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        # fine-tune the model
        trainer.train() 

        # save the fine-tuned model
        output_model_path = f"{self.model_output_dir}/{model_fname}.pt"
        self.model.save_pretrained(output_model_path)
        
        # return the path to the saved model
        return output_model_path

    def evaluate_model(self, model_fname: str, test_set: list) -> dict:
        """
        Evaluates the performance of a fine-tuned model using the evaluate package.

        PARAMS:
            model_path: str
                path to the fine-tuned model
            test_set: list of tuples
                test set to evaluate the model on, where each tuple contains a prompt and a target response

        RETURNS:
            metrics_dict: dict
                dictionary containing the evaluation metrics
        """

        # get the model
        selected_model = self.get_model_from_path(model_fname) 

        # define the evaluation metrics
        metrics = [evaluate.Perplexity(), evaluate.BLEU(ngram=2), evaluate.ROUGE(), evaluate.SelfBLEU(ngram=2)]

        # evaluate the model
        evaluator = evaluate.Evaluator(model=selected_model, tokenizer=self.tokenizer, metrics=metrics)
        results = evaluator.evaluate(test_set)

        # parse the evaluation results
        metrics_dict = {}
        for metric, result in zip(metrics, results):
            metric_name = type(metric).__name__
            metrics_dict[metric_name] = result

        return metrics_dict