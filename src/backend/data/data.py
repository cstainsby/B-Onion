
import os

class GloVeDataset():
  def __init__(self, dataset_label: str) -> None:
    self.data_dir_label = "glove.6B"
    self.data = None
    
    # LOAD IN DATA
    path_to_data_folder = os.path.dirname(os.path.realpath(__file__))
    path_to_specified_data = path_to_data_folder + "/data_repo/" + self.data_dir_label + "/"  + dataset_label + ".txt"
      

    with open(path_to_specified_data, 'r') as glove_data:
      print(glove_data.readline())
    

if __name__ == "__main__":
  ds = GloVeDataset("glove.6B.50d")