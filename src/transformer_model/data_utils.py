import torch
from torchtext.vocab import vocab, Vocab

from collections import Counter
import string




def build_vocab(tokenizer, filepath=None, word_list:list=None):
  counter = Counter()

  if filepath:
    with open(filepath, encoding="utf8") as in_file:
      for string in in_file:
        counter.update(tokenizer(string))

  elif word_list:
    # word list is a list with all unique words within vocab
    for word in word_list:
      counter.update(tokenizer(word))
    
  return vocab(counter, specials=["<pad>", "<unk>", "<sos>", "<eos>"])


# --------------------------------------------------------------------------------------------------
#   Helper Functions
# --------------------------------------------------------------------------------------------------

def encode_token(vocab: Vocab, decoded_token: str):
  return vocab.get_stoi()[decoded_token]

def decode_token(vocab: Vocab, encoded_token: int):
  return vocab.get_itos()[encoded_token]