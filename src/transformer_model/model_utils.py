import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.vocab import Vocab

from .model import TransformerModel
from .data_utils import encode_token, decode_token, tokenize_raw_data


def train_batch(model: TransformerModel, vocab: Vocab, optimizer, criterion, batch, device):
    model.train()
    total_loss = 0
    num_tokens = 0

    start_time = time.time()

    for i, raw_batch_strs in enumerate(batch):
        curr_time = time.time()

        if i > 9 and i % 10 == 0:
            print("i:", i, "time to train 10 instances:", curr_time - start_time)

        full_tokenized_sequence = tokenize_raw_data(data=raw_batch_strs, vocab=vocab)

        # convert tokens to encodings
        encoded_tokens = [encode_token(vocab, token) for token in full_tokenized_sequence]

        # Convert sentence to tensor and move to device
        sentence_tensor = torch.tensor(encoded_tokens, dtype=torch.long).unsqueeze(0).to(device)
        # Clear gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(sentence_tensor)
        # Flatten output and target for loss calculation
        output = output.view(-1, output.size(-1))
        target = sentence_tensor.view(-1)
        # Calculate loss
        loss = criterion(output, target)
        # Backward pass
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # Update parameters
        optimizer.step()
        # Update loss and token count
        total_loss += loss.item()
        num_tokens += len(encoded_tokens)
        # Print progress
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {total_loss/num_tokens:.4f}")


    print("Total time:", time.time() - start_time)

    # Return average loss over all tokens
    return total_loss / num_tokens


def generate_text(model: TransformerModel, vocab: Vocab, start_str: str, max_length: int=100, temperature: float=1.0):

    start_seq = tokenize_raw_data(data=start_str, vocab=vocab)

    model.eval()
    gen_seq = start_seq

    with torch.no_grad():
        # Check input sequence length
        if len(gen_seq) > model.max_len:
            raise ValueError(f"Input sequence must be at least {model.max_len} words long")

        for _ in range(max_length):
            in_text_token_ids = [encode_token(vocab, word) for word in gen_seq]
            sequence_length = len(in_text_token_ids)

            curr_seq = torch.tensor(in_text_token_ids).unsqueeze(0)

            logits = model(curr_seq)
            logits = logits[0, -1, :] / temperature
            prob = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
            curr_seq = torch.cat([curr_seq, next_token.unsqueeze(0)], dim=-1)
            if next_token == encode_token(vocab, "<eos>") or sequence_length >= max_length:
                return
            else:
                generated_text = decode_token(vocab, next_token)
                yield generated_text

# --------------------------------------------------------------------------------------------------
#   Model State Interaction Functions
# --------------------------------------------------------------------------------------------------

def save_model_state(model: nn.Module, save_name: str = ""):
    """Using pytorch's provided pickle functionality, save a model's state dictionary to the model_save/ directory
    """
    SAVE_PATH = "model_save/" + save_name + ".pt"
    state_dict = model.state_dict()

    torch.save(state_dict, SAVE_PATH)

def load_model_state(model: nn.Module, save_name: str):
    """Using pytorch's provided loading
    """
    SAVE_PATH = "model_save/" + save_name + ".pt"
    return model.load_state_dict(torch.load(SAVE_PATH))

def print_model_state_dict(model: nn.Module):
    print("Model State Dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def print_optimizer_state_dict(optimizer: optim.Optimizer):
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
