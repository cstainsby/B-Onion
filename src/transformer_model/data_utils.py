import torch
from torchtext.vocab import vocab, Vocab

from collections import Counter
import string


class DataProcessor():
  """DataProcessor class
  
  the declaration of state is necessary in order to remeber our vocab, tokenizer, etc
  This will speed up our training loop by limiting the amount of repeated lookups"""
  def __init__(self, vocab: Vocab, tokenizer) -> None:
    self.tokenizer = tokenizer
    self.vocab_stoi = vocab.get_stoi()
    # self.vocab_itos = vocab.get_itos()

  
  def encode_tokens(self, decoded_token_sequence):
    return [self.encode_token(token) for token in decoded_token_sequence]
  
  def decode_tokens(self, encoded_token_sequence):
    return [self.decode_token(token) for token in encoded_token_sequence]

  def tokenize_raw_data(self, data: str):
    tokenized_data = self.tokenizer(data)

    # set all tokens to lowercase 
    tokenized_data = [token.lower() for token in tokenized_data]

    # validate tokens with vocab, if a token doesn't exist within the vocab then set it to <unk>
    tokenized_data = [token if token in self.vocab_stoi else "<unk>" for token in tokenized_data]

    # append <sos> to the beginning of sequence and <eos> to the end
    tokenized_data = ["<sos>"] + tokenized_data + ["<eos>"]

    return tokenized_data
  

  # --------------------------------------------------------------------------------------------------
  #   Helper Functions
  # --------------------------------------------------------------------------------------------------

  def encode_token(self, decoded_token: str):
    return self.vocab_stoi[decoded_token]

  # def decode_token(self, encoded_token: int):
  #   return self.vocab_itos[encoded_token]


# def build_vocab(tokenizer, filepath=None, word_list:list=None):
#   counter = Counter()

#   if filepath:
#     with open(filepath, encoding="utf8") as in_file:
#       for string in in_file:
#         counter.update(tokenizer(string))

#   elif word_list:
#     # word list is a list with all unique words within vocab
#     for word in word_list:
#       counter.update(tokenizer(word))
    
#   return vocab(counter, specials=["<pad>", "<unk>", "<sos>", "<eos>"])

