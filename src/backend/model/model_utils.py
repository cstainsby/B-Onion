
import torch
from torch import nn
from torch import optim

from model import TransformerModel


# ------------------------------------------------------
#     savers and loaders
# ------------------------------------------------------
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



# ------------------------------------------------------
#     savers and loaders
# ------------------------------------------------------


# tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
# sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

def create_training_sequences(max_sequence_length, tokenized_training_data):
    # Create sequences of length max_sequence_length + 1
    # The last token of each sequence is the target token
    sequences = []
    for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
        sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])
    return sequences

def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    # Tokenize the training data
    tokenized_training_data = tokenizer.tokenize(training_data)
    for _ in range(max_sequence_length):
        # Prepend padding tokens
        tokenized_training_data.insert(0, tokenizer.character_to_token('<pad>'))
    return tokenized_training_data