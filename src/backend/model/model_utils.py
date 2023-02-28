
import time
import random
import numpy as np

import torch
from torch import nn
from torch import optim

# from backend.model.inDepthModel import TransformerModel


def predict(model, input_sequence, max_length=15, SOS_token=2, EOS_token=3):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    
    y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

    num_tokens = len(input_sequence[0])

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)
        
        pred = model(input_sequence, y_input, tgt_mask)
        
        next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()

def fit_model(model: nn.Module, optimizer, loss_fn, num_epochs: int):
    for epoch in range(num_epochs):
        print("EPOCH: {}".format(epoch))
        model.train() # pytorch provided function

        run_epoch(
            data_iter=7,
            model=model,
            loss_function=loss_fn,
            optimizer=optimizer
        )

        model.eval()
        print()



def run_epoch(data_iter: int, model, loss_function):
    """Function for running a single epoch"""
    start = time.time()
    total_tokens = total_loss = tokens = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(
            src=batch.src,
            tgt=batch.tgt,
            src_mask=batch.src_mask,
            tgt_mask=batch.tgt_mask
        )
        loss =  loss_function(out, batch.tgt_y, batch.num_tokens)
        total_loss += loss
        total_tokens += batch.num_tokens

        if i % 50 == 1: 
            elapsed = time.time() - start
            print("Epoch Step: {} Loss: {} Tokens per sec: {}".format(i, loss / batch.num_tokens, tokens / elapsed))
        
        start = time.time()
        tokens = 0 

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






def generate_random_data(n):
    SOS_token = np.array([2])
    EOS_token = np.array([3])
    length = 8

    data = []

    # 1,1,1,1,1,1 -> 1,1,1,1,1
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.ones(length), EOS_token))
        y = np.concatenate((SOS_token, np.ones(length), EOS_token))
        data.append([X, y])

    # 0,0,0,0 -> 0,0,0,0
    for i in range(n // 3):
        X = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        y = np.concatenate((SOS_token, np.zeros(length), EOS_token))
        data.append([X, y])

    # 1,0,1,0 -> 1,0,1,0,1
    for i in range(n // 3):
        X = np.zeros(length)
        start = random.randint(0, 1)

        X[start::2] = 1

        y = np.zeros(length)
        if X[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1

        X = np.concatenate((SOS_token, X, EOS_token))
        y = np.concatenate((SOS_token, y, EOS_token))

        data.append([X, y])

    np.random.shuffle(data)

    return data


def batchify_data(data, batch_size=16, padding=False, padding_token=-1):
    batches = []
    for idx in range(0, len(data), batch_size):
        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):
            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx : idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_bath_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx : idx + batch_size]).astype(np.int64))

    print(f"{len(batches)} batches of size {batch_size}")

    return batches


train_data = generate_random_data(9000)
val_data = generate_random_data(3000)

train_dataloader = batchify_data(train_data)
val_dataloader = batchify_data(val_data)