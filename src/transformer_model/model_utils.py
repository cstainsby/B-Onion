import torch
import torch.nn as nn
from torch import optim

import torch.nn.functional as F
from torchtext.vocab import Vocab

from model import TransformerModel
from data_utils import encode_token, decode_token


def train_model(model: TransformerModel, optimizer, criterion, train_loader, device):
    model.train()  # Set the model to training mode
    
    total_loss = 0
    for i, (src, tgt) in enumerate(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt[:, :-1])  # Predict up to the second last token
        
        # Flatten the output and target tensors to be 2D
        output = output.view(-1, output.shape[-1])
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"Batch {i + 1}, loss={loss.item()}")
    
    return total_loss / len(train_loader)


def generate_text(model: TransformerModel, vocab: Vocab, in_text, max_length=100, temperature=1.0):
    model.eval()
    
    in_text_tokens = [encode_token(vocab, word) for word in in_text]
    input_seq = torch.tensor(in_text_tokens).unsqueeze(0)

    output_seq = input_seq.clone().detach()

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_seq, output_seq)
            logits = logits[0, -1, :] / temperature
            prob = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=-1)

            if next_token == encode_token(vocab, "<eos>"):
                break
            else:
                output_seq = torch.cat([output_seq, next_token.unsqueeze(0)], dim=-1)
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
