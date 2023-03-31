from spacy.tokenizer import Tokenizer 
import torch
from torchtext.data import get_tokenizer
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


def tokenize_raw_data(data: str, vocab: Vocab):
  tokenizer = get_tokenizer("spacy")
  tokenized_data = tokenizer(data)

  # set all tokens to lowercase 
  tokenized_data = [token.lower() for token in tokenized_data]

  # validate tokens with vocab, if a token doesn't exist within the vocab then set it to <unk>
  vocab_protected_tokens = [token if token in vocab.get_itos() else "<unk>" for token in tokenized_data]

  # append <sos> to the beginning of sequence and <eos> to the end
  full_sequence = ["<sos>"] + vocab_protected_tokens + ["<eos>"]

  return full_sequence


# --------------------------------------------------------------------------------------------------
#   Helper Functions
# --------------------------------------------------------------------------------------------------

def encode_token(vocab: Vocab, decoded_token: str):
  return vocab.get_stoi()[decoded_token]

def decode_token(vocab: Vocab, encoded_token: int):
  return vocab.get_itos()[encoded_token]

# def add_token_to_vocab():
#   pass