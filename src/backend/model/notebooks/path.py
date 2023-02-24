import sys 
import os

def get_model_folder_path():
  model_path = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir))
  return model_path