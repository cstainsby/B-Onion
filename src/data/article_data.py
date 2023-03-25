import os 
import pandas as pd

from .data import extract_zipfile, data_dir_path


articles_filepath = data_dir_path + "/articles.csv"

class ArticlesIter():
    def __init__(self, batch_size: int = 1000) -> None:
        self.iter = pd.read_csv(articles_filepath, chunksize=batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except:
            raise StopIteration

if __name__ == "__main__":
    zip_path = data_dir_path + '/articles.zip'
    dest_dir = data_dir_path + '/.'

    if not os.path.exists(articles_filepath):
        print("articles csv doesn't exist, extracting now")
        extract_zipfile(zip_path, dest_dir)
