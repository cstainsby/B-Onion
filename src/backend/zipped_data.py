import os
import zipfile
from zipfile import ZipFile
import pandas as pd

""" Class for loading in data from zip files


"""
class ZippedData():
    def __init__(self, zipfile_name: str) -> None:
        self.zip_file_contents = None   # the handle to the zipfile 
        self.data = None                # the data of a specific file within the zip folder

        # SETUP
        path_to_data_folder = os.path.dirname(os.path.realpath(__file__))
        path_to_zipfile = path_to_data_folder + "/data_repo/" + zipfile_name + ".zip"

        try:
            with ZipFile(path_to_zipfile, mode="r") as archive:
                self.zip_file_contents = archive.extractall()
                print(self.zip_file_contents)
        except zipfile.BadZipFile as err:
            print("Zip file couldn't be accessed")
            print(err)
    

class CsvZippedData(ZippedData):
    def __init__(self, zipfile_name: str, csv_file_name: str = "") -> None:
        super().__init__(zipfile_name)

        self.data = None
        print("check")
        # read in csv file
        if self.zip_file_contents:
            print(self.zip_file_contents)


            try:
                with self.zip_file_handle.open(csv_file_name) as csv_file:
                                               
                    print(csv_file)
                    # self.data = csv_file
            except FileNotFoundError as err:
                print("csv couldn't be found")
                print(err)

        



if __name__ == "__main__":
    c = CsvZippedData("onion_or_not_classifications", "OnionOrNot.csv")