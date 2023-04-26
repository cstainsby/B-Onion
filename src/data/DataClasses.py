
import pandas as pd 
from pathlib import Path

class PostCommentData():
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
        self.data_path = str(Path(__file__).absolute().parent / Path("store/post_comment_class.csv"))

        self.labels_to_full = {
            "YTA": "you're the asshole",
            "YWBTA": "you would be the asshole",
            "NTA": "not the asshole",
            "YWNBTA": "you would not be the asshole",
            "ESH": "everyone sucks here",
            "NAH": "no assholes here",
            "INFO": "not enough info"        
        }
        self.labels = self.labels_to_full.keys()

    def read_data(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path)

    def store_data_for_training(self, classification_df: pd.DataFrame) -> None:
        """Stores a pandas df into a csv, this pandas df should be the same shape/same cols as the stored data
            
        PARAMS:
            write_type(str): tells how to store the data"""
        out_data = pd.DataFrame({
            "prompt": [],
            "completion": []
        })

        for _, row in classification_df.iterrows():
            title = row["post_title"]
            content = row["post_content"]
            classification = row["class"]

            prompt = "title: {}\ncontent: {}".format(title, content)
            completion = "{}".format(classification)

            out_data.loc[len(out_data.index)] = [prompt, completion]

        out_data.to_csv(self.data_path, index=False)

        
