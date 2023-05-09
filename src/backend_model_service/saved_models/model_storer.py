""" This file will contain load and save functions for models
    This system is temporary, the are more robust ways of building this out
    however that is out of scope for this project

    NOTE: the models will be uniquly identifiable a unique integer key
"""

import csv
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

store_root_path = Path(__file__).absolute().parent 
info_csv_path = store_root_path / Path("./model_info_guide.csv")

defined_model_types = [
    "Text Generation",
    "Text Classification",
    "misc"
]

def find_last_id():
    """
    Finds the id of the model in the last row of a CSV file.
    """
    with open(info_csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        last_row = None
        for row in csv_reader:
            last_row = row
        if last_row is not None:
            return int(last_row['id'])
        else:
            return None
        

def get_file_name(model_id: int, model_name: str) -> str:
    """
    Helper which creates a filename based on model id and name
    """
    return f'{model_name}_{model_id}.pkl'


def read_csv():
    """
    Reads a CSV file and returns a list of dictionaries where each dictionary represents a row in the CSV file.
    """
    with open(info_csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = []
        for row in csv_reader:
            rows.append(row)
        return rows

def write_csv(rows):
    """
    Writes a list of dictionaries to a CSV file where each dictionary represents a row in the CSV file.
    NOTE: only for initial creation of file
     """
    with open(info_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['id', 'model_name', 'model_type', 'model_path', 'model_desc']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_model_info_to_csv(model_name, model_type, model_path, model_desc):
    """
    Appends a new row to a CSV file with the specified columns.
    """
    last_id = find_last_id()
    if last_id is None:
        new_id = 0
    else:
        new_id = last_id + 1
    with open(info_csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['id', 'model_name', 'model_type', 'model_path', 'model_desc']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow({'id': new_id, 'model_name': model_name, 'model_type': model_type, 'model_path': model_path, 'model_desc': model_desc })

def save_model_pickle(model: AutoModelForCausalLM, file_path):
    """
    Pickles and saves a model to the ./store/ directory.
    """
    model.save


def add_model_to_store_and_csv(model, model_name, model_type):
    """
    Saves a model to the ./store/ directory, updates the CSV file with the model details,
    and returns the new model ID.
    """
    last_id = find_last_id()

    # create filename and path
    file_name = get_file_name(last_id + 1, model_name)
    file_path = store_root_path / Path(f'/store/{file_name}')

    save_model_pickle(model, file_path)

    # switch model_type to misc if provided model type not supported
    valid_model_names = defined_model_types[:-1]
    if model_type not in valid_model_names:
        model_type = "misc"

    add_model_info_to_csv(info_csv_path, model_name, model_type, str(file_path))

def load_model(model_id):
    """
    Unpickles and returns a model from the ./store/ directory based on its ID in the CSV file.
    """
    model_path = None

    with open(info_csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if int(row['id']) == model_id:
                model_path = row["model_path"]
        else:
            raise ValueError(f"No model found with ID {model_id}.")

    # Unpickle the model from the retrieved path
    with open(model_path, mode='rb') as file:
        model = pickle.load(file)

    return model

def get_model_info(model_id: int):
    """
    Search for the row in the CSV file with the matching ID.
    """
    # Search for the row in the CSV file with the matching ID
    with open(info_csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if int(row['id']) == int(model_id):
                print("found")
                return row
        else:
            raise ValueError(f"No model found with ID {model_id}.")
    
    return None

def get_existing_fine_tuned_models() -> list:
        """
        Returns a list of all text gen fine-tuned models.
        """
        fine_tune_ids = []
        
        # Get all filenames in the directory
        model_save_loc = str(store_root_path) + "/store/"
        print("MODEL SAVE LOC", model_save_loc)
        filenames = os.listdir(model_save_loc)

        # Iterate over the filenames and split them into their base names and extensions
        for filename in filenames:
            basename, extension = os.path.splitext(filename)
            fine_tune_ids.append(basename)
            
        return fine_tune_ids


if __name__ == "__main__":
    # create the file with nothing in it
    write_csv([])
    add_model_info_to_csv("test","Text Classification","None","This is my test text classification model")
    add_model_info_to_csv("test","Text Generation","None","This is my test text generation model")
    model_info = get_model_info(0)

    print("MODEL INFO", model_info)