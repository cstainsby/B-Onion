import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.vocab import Vocab

from .model import TransformerModel
from .data_utils import encode_token, decode_token, build_vocab


def train_model(model: TransformerModel, vocab: Vocab, optimizer, criterion, train_data, device):
    model.train()
    total_loss = 0
    num_tokens = 0
    for i, tokenized_strings in enumerate(train_data):

        # convert tokens to encodings
        encoded_tokens = [encode_token(vocab, token) for token in tokenized_strings]

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
    # Return average loss over all tokens
    return total_loss / num_tokens


# def train(model, train_loader, val_loader, epochs, lr):
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(epochs):
#         # Training
#         model.train()
#         train_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()
#             src, target = batch
#             output = model(src)
#             loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         train_loss /= len(train_loader)

#         # Evaluation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 src, target = batch
#                 output = model(src)
#                 loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
#                 val_loss += loss.item()

#         val_loss /= len(val_loader)

#         print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')

#     print('Training finished!')




def generate_text(model: TransformerModel, vocab: Vocab, start_seq: list, max_length: int=100, temperature: float=1.0):
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
                break
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
