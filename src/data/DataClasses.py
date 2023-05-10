
import pandas as pd 
from pathlib import Path

class AITAData():
    """This object will be used for working with a joined post comment dataset which will be stored in csv. 
    This will be useful when attempting to train the sentiment classification model from the stored csv,

    Data will cols will be:
        1. post_title
        2. post_content
        3. class
    NOTE: classification labels will be described by the self.labels_to_full dict

    Link to how it will be used: https://platform.openai.com/docs/guides/fine-tuning
    
    Useful command: openai tools fine_tunes.prepare_data -f <LOCAL_FILE>
    """
    def __init__(self) -> None:
        self.data_path = Path(__file__).absolute().parent 


        self.question_labels_to_full = {
            "AITA": "am I the asshole",
            "WIBTA": "would I be the asshole"
        }

        self.class_labels_to_full = {
            "YTA": "you're the asshole",
            "YWBTA": "you would be the asshole",
            "NTA": "not the asshole",
            "YWNBTA": "you would not be the asshole",
            "ESH": "everyone sucks here",
            "NAH": "no assholes here",
            "INFO": "not enough info"        
        }

        self.question_labels = self.question_labels_to_full.keys()
        self.class_labels = self.class_labels_to_full.keys()

    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)
    

    def format_data_for_classification(self, post_title: str, post_content: str, classification: str = None):
        """Format a posts data into a prompt completion pair which can be used by openai"""
        prompt = "title: {}\ncontent: {}".format(post_title, post_content)
        class_id = "{}".format(classification) if classification else "None"
        return prompt, class_id


    def store_post_to_class_data(self, filename: str, classification_df: pd.DataFrame) -> None:
        """Stores a pandas df into a csv, this pandas df should be the same shape/same cols as the stored data
            
        PARAMS:
            write_type(str): tells how to store the data"""
        out_file_path = str(self.data_path / Path("store/{}.csv".format(filename)))

        out_data = pd.DataFrame({
            "prompt": [],
            "class": []
        })

        for _, row in classification_df.iterrows():
            title = row["post_title"]
            content = row["post_content"]
            classification = row["class"]

            prompt, class_id = self.format_data_for_classification(title, content, classification)

            out_data.loc[len(out_data.index)] = [prompt, class_id]

        out_data.to_csv(out_file_path, index=False)

    def store_post_and_class_to_text_data(self, filename: str, classification_df: pd.DataFrame) -> None:
        """Stores a pandas df into a csv, this pandas df should be the same shape/same cols as the stored data
            
        PARAMS:
            write_type(str): tells how to store the data"""
        out_file_path = str(self.data_path / Path("store/{}.csv".format(filename)))

        out_data = pd.DataFrame({
            "prompt": [],
            "completion": []
        })

        for _, row in classification_df.iterrows():
            title = row["post_title"]
            content = row["post_content"]
            classification = row["class"]
            comment_content = row["comment_content"]

            prompt = "title: {}\ncontent: {}\nclassification: {}".format(title, content, classification)
            completion = "{}".format(comment_content)

            out_data.loc[len(out_data.index)] = [prompt, completion]

        out_data.to_csv(out_file_path, index=False)

        
