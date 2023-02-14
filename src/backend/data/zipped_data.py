"""A Script file which unpacks downloaded zip files into their own sub-folders within the data_repo folder
"""


import os
import zipfile
from zipfile import ZipFile
import pandas as pd


def extract_zip_by_name(zipfile_name: str) -> None:
    """Function to extract a zipfile given the name of it
        NOTE: the zipfile must be within the data_repo folder
    """
    path_to_data_folder = os.path.dirname(os.path.realpath(__file__))
    path_to_data_repo_folder = path_to_data_folder + "/data_repo/"
    path_to_zipfile = path_to_data_repo_folder + zipfile_name + ".zip"

    with ZipFile(path_to_zipfile, mode="r") as archive:
        new_unzip_folder = path_to_data_repo_folder + "/" + zipfile_name
        os.mkdir(new_unzip_folder)

        archive.extractall(new_unzip_folder)



if __name__ == "__main__":
    extract_zip_by_name("glove.6B")