#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
import pandas as pd


class char_tokenizer:
    """
    a very simple char-based tokenizer. the tokenizer turns a string into a list of integers.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        # TODO: calculate the vocab size and create a dictionary that maps each character to a unique integer
        self.n_vocab = len(corpus)
        self.ix_to_char = corpus
        self.char_to_ix = {char: i for i, char in enumerate(corpus)}
        # End of your code

    def encode(self, string: str):
        # TODO: convert a string into a list of integers and return, using the dictionary you created above
        return [self.char_to_ix[char] for char in string]
        # End of your code
 
    def decode(self, codes: List[int]):
        # TODO: convert a list of integers into a string and return, using the dictionary you created above
        return ''.join([self.ix_to_char[i] for i in codes])
        # End of your code

class Head(nn.Module):
    """single head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        # TODO: create three linear layers, Key, Query, and Value, each of which maps from n_embd to head_size
        #       and assign them to self.Key, self.Query, and self.Value, respectively
        self.Key = nn.Linear(n_embd, head_size)
        self.Query = nn.Linear(n_embd, head_size)
        self.Value = nn.Linear(n_embd, head_size)
        self.head_size = head_size
        # End of your code
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))


    def forward(self, inputs):
        # TODO: implement the forward function of the head
        #       the input is a tensor of shape (batch, time, n_embd)
        #       the output should be a tensor of shape (batch, time, head_size)
        #       you may use the tril buffer defined above to mask out the upper triangular part of the affinity matrix

        key = self.Key(inputs)
        query = self.Query(inputs)
        value = self.Value(inputs)

        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_size ** 0.5)

        # Get the time dimension size from the inputs tensor
        time_size = inputs.shape[1]

        # Create a tril tensor dynamically based on the time dimension size
        tril = torch.tril(torch.ones(time_size, time_size, device=inputs.device))

        # Apply the dynamic mask to the scores tensor
        scores = scores * tril - 1e12 * (1 - tril)

        # Compute the softmax of the scores
        attn_weights = torch.softmax(scores, dim=-1)

        # Compute the weighted sum of the value matrix using the attention weights
        out = torch.matmul(attn_weights, value)

        # Output is a tensor of shape (batch, time, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        #TODO: implement heads and projection
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, n_embd)
        # End of your code
    def forward(self, inputs):
        #TODO: implement the forward function of the multi-head attention
        
        out = torch.cat([head(inputs) for head in self.heads], dim=-1)

        return self.projection(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        #TODO: implement the feed-forward network

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

        # End of your code

    def forward(self, inputs):
        return self.net(inputs)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # TODO: implement the block of transformer using the MultiHeadAttention and 
        # FeedForward modules, along with the layer normalization layers
        self.attention = MultiHeadAttention(n_heads, n_embd//n_heads)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

        # End of your code
    def forward(self, x):
        #TODO: implement the forward function of the block, you may refer to the docs of this experiment
        x = self.attention(x) + x
        x = self.norm1(x)
        x = self.ff(x) + x
        x = self.norm2(x)
        # End of your code
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: create the embedding table, the stack of blocks, the layer normalization layer, 
        # and the linear layers.
        self.wte = nn.Embedding(n_vocab, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, n_vocab, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        # End of your code
    def get_positional_embeddings(self, inputs):
        seq_len = inputs.size(1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pos_enc = torch.zeros(1, seq_len, n_embd)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        return pos_enc.to(inputs.device)
    
    def forward(self, inputs, labels=None):
        # TODO: implement the forward function of the transformer

        # inputs:(batch, context)

        # embedding:(batch, context, embedding)
        embedding = self.wte(inputs)+self.get_positional_embeddings(inputs)

        # attens:(batch, context, embedding)
        attens = self.blocks(embedding)

        # attens:(batch, context, embedding)
        attens = self.norm(attens)
        # logits:(batch, context, attens)
        logits = self.lm_head(attens)
        # End of your code

        # compute the loss
        
        if labels is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)
            labels = labels.view(batch * time)
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        generated_tokens = []
        for _ in range(max_new_tokens):
            # Truncate the input sequence to keep the same length
            if inputs.size(1) > block_size:
                inputs = inputs[:, -block_size:]
            logits, _ = self.forward(inputs)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            new_token = torch.multinomial(probabilities, num_samples=1)
            generated_tokens.append(new_token)
            inputs = torch.cat((inputs, new_token), dim=1)
        return torch.cat(generated_tokens, dim=1)
    

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def generate(model):
    context = torch.zeros((1, 1), device=device, dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))

def myGenerate(model):
    text = "I would I were a Roman; "
    context = torch.tensor(encode(text),device=device, dtype=torch.long)
    context = torch.stack([context])
    print(text,decode(model.generate(context, max_new_tokens=200)[0].tolist()))


def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_eval = float('inf')
    for iter in range(max_iters):
        
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            df.loc[len(df)] = [iter, losses['train'].item(), losses['val'].item()]
            if losses["val"] < best_eval:
                best_eval = losses["val"]
                torch.save(model.state_dict(), 'model.pt')

        inputs, labels = get_batch("train")

        logits, loss = model(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


# define the hyperparameters
batch_size = 16
block_size = 256
max_iters = 5000 # set the number of training iterations as you like
eval_interval = 50
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 128
n_heads = 8
n_layers = 8
df = pd.DataFrame(columns=['iter', 'train_loss', 'val_loss'])

# read the dataset
with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))

# initialize the vocabulary
tokenizer = char_tokenizer(chars)
encode = tokenizer.encode
decode = tokenizer.decode
n_vocab = tokenizer.n_vocab

# separate the dataset into train and validation
train_data = torch.tensor(encode(text[: -len(text) // 10]), dtype=torch.long)
val_data = torch.tensor(encode(text[-len(text) // 10 :]), dtype=torch.long)

# define the model
model = Transformer().to(device)
train(model)
generate(model)

# TODO: train the model and generate samples
# my Imput: I would I were a Roman;
myGenerate(model)
df.to_csv('loss.csv', index=False)