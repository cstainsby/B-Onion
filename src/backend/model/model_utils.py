
import torch
from torch import nn
from torch import optim

from model import TransformerModel

def save_model_state(model: nn.Module, save_name: str = ""):
  """Using pytorch's provided pickle functionality, save a model's state dictionary to the model_save/ directory
  """
  SAVE_PATH = "model_save/" + save_name + ".pt"
  state_dict = model.state_dict()

  torch.save(state_dict, SAVE_PATH)

def load_model_state(path_to_model: str):
  """Using pytorch's provided loading
  """
  SAVE_PATH = "model_save"

  model = TransformerModel()
  model.load_state_dict(torch.load())

def print_model_state_dict(model: nn.Module):
  print("Model State Dict:")
  for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def print_optimizer_state_dict(optimizer: optim.Optimizer):
  print("Optimizer's state_dict:")
  for var_name in optimizer.state_dict():
      print(var_name, "\t", optimizer.state_dict()[var_name])