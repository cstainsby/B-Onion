import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchtext.vocab import Vocab
from torchtext.data import get_tokenizer


import transformer_model.model as model
import transformer_model.data_utils as data_utils


def train_batch(model: model.TransformerModel, data_processor: data_utils.DataProcessor, optimizer, criterion, batch, device):
    model.train()

    # initialize needed content
    total_loss = 0
    num_tokens = 0
    start_time = time.time()

    torch.cuda.empty_cache()

    for i, raw_batch_strs in enumerate(batch):

        tokenized_data = data_processor.tokenize_raw_data(data=raw_batch_strs)

        encoded_tokenized_data = data_processor.encode_tokens(decoded_token_sequence=tokenized_data)

        # Convert sentence to tensor and move to device
        sentence_tensor = torch.tensor(encoded_tokenized_data, dtype=torch.long).unsqueeze(0).to(device)

        # Clear gradients
        model.zero_grad()
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
        num_tokens += len(encoded_tokenized_data)

        if device.type == "cuda":
            torch.cuda.empty_cache()

        
    print(f" - Loss = {total_loss/num_tokens:.4f}")
    print(" - Memory Allocated:" + f"Current CUDA memory usage: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.2f} GB")


    print(" - Total time:", time.time() - start_time)

    # Return average loss over all tokens
    return total_loss / num_tokens


def generate_text(model: model.TransformerModel, vocab: Vocab, start_str: str, max_length: int=100, temperature: float=1.0):
    """
    Generates text from a trained transformer model and a starting string.

    Args:
        model (nn.Module): trained transformer model
        vocab (Vocab): vocabulary object
        start_str (str): starting string for text generation
        max_length (int): maximum number of words to generate
        temperature (float): softmax temperature value for sampling. Default=1.0.

    Yields:
        str: generated words
    """

    # initialize needed content
    tokenizer = get_tokenizer("spacy") # as a standard for all training I will be using spacy
    data_processor = data_utils.DataProcessor(vocab=vocab, tokenizer=tokenizer)

    with torch.no_grad():

        # tokenize the incoming string data 
        # remove last string which is an <eos> created by the tokenize function
        start_seq = data_processor.tokenize_raw_data(data=start_str)[:-1]
        start_seq_tokens = data_processor.encode_tokens(start_seq)
        model_input_tensor = torch.tensor(start_seq_tokens, dtype=torch.long, device=model.device).unsqueeze(0)

        for _ in range(max_length):
            out = model(model_input_tensor, memory=None)

            # Get last predicted token and apply temperature
            logits = out[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1).squeeze()
            generated_token = torch.multinomial(probs, num_samples=1).item()

            # Add new token to input sequence
            model_input_tensor = torch.cat([model_input_tensor, torch.tensor([[generated_token]], dtype=torch.long, device=model.device)], dim=-1)

            # check if <eos> is hit
            #       if so exit function
            #       else yield the next token
            if generated_token == data_processor.encode_token("<eos>"):
                return
            else:
                generated_text = data_processor.decode_token(generated_token)
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
