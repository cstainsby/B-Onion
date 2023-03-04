
import numpy as np
import random
from typing import Tuple

import torch
from torch.utils.data import IterableDataset
from torch import Tensor
from torch.autograd import Variable

from torchtext.vocab import Vocab


class Batch():
    """Object for holding a batch of data with mask during training.
    
    A Batch includes 
      - src
      - tgt
      - src_mask
      - tgt_mask
      - n_tokens"""
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for _ in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)

        
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# ------------------------------------------------------
#     data functions
# ------------------------------------------------------

# def generate_random_data(n):
#     """
#     X: src 
#     y: tgt
#     """
    
#     # NOTE: 0 and 1 are reserved for start of sentence and end of sentence respectivley
#     SOS_token = np.array([0])
#     EOS_token = np.array([1])
#     length = 8 

#     number_of_tokens = 3 + 2

#     data = []

#     for i in range(n):
#         X = np.concatenate((SOS_token, np.array([random.randint(2, number_of_tokens) for _ in range(length)]), EOS_token))
#         y = np.concatenate((SOS_token, np.array([random.randint(2, number_of_tokens) for _ in range(length)]), EOS_token))
#         data.append([X, y])

#     # for _ in range(n):
#     #     X = np.concatenate((SOS_token, np.array([2 for _ in range(number_of_tokens)]), EOS_token))
#     #     y = np.concatenate((SOS_token, np.array([2 for _ in range(number_of_tokens)]), EOS_token))
#     #     data.append([X, y])

#     # for a third of the data, only 1's
#     # 1,1,1,1,1,1 -> 1,1,1,1,1
#     # for i in range(n // 3):
#     #     X = np.concatenate((SOS_token, np.ones(length), EOS_token))
#     #     y = np.concatenate((SOS_token, np.ones(length), EOS_token))
#     #     data.append([X, y])

#     # # 0,0,0,0 -> 0,0,0,0
#     # for i in range(n // 3):
#     #     X = np.concatenate((SOS_token, np.zeros(length), EOS_token))
#     #     y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
#     #     data.append([X, y])

#     # # 1,0,1,0 -> 1,0,1,0,1
#     # for i in range(n // 3):
#     #     X = np.zeros(length)
#     #     start = random.randint(0, 1)

#     #     X[start::2] = 1

#     #     y = np.zeros(length)
#     #     if X[-1] == 0:
#     #         y[::2] = 1
#     #     else:
#     #         y[1::2] = 1

#     #     X = np.concatenate((SOS_token, X, EOS_token))
#     #     y = np.concatenate((SOS_token, y, EOS_token))

#     #     data.append([X, y])

#     np.random.shuffle(data)

#     return data


# def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
#     batches = []
#     for idx in range(0, len(data), batch_size):
#         # We make sure we dont get the last bit if its not batch_size size
#         if idx + batch_size < len(data):
#             # Here you would need to get the max length of the batch,
#             # and normalize the length with the PAD token.
#             if padding:
#                 max_batch_length = 0

#                 # Get longest sentence in batch
#                 for seq in data[idx : idx + batch_size]:
#                     if len(seq) > max_batch_length:
#                         max_batch_length = len(seq)

#                 # Append X padding tokens until it reaches the max length
#                 for seq_idx in range(batch_size):
#                     remaining_length = max_batch_length - len(data[idx + seq_idx])
#                     data[idx + seq_idx] += [padding_token] * remaining_length

#             batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

#     print(f"{len(batches)} batches of size {batch_size}")

#     return batches


def data_process(raw_text_iter: IterableDataset, vocab: Vocab, tokenizer) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source: Tensor, i: int, bptt: int = 35) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
        bptt: used to define subdivide chunk length (e.g. bptt = 35 -> chunks of length 35)

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target