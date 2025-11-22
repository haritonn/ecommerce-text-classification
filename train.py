import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.rnn import pad_sequence

import preprocessing
from dataset import ecomDataset
from model import RNNModel


def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, labels


column_names = ["class", "text"]
text_data = pd.read_csv("data/ecommerceDataset.csv", name=column_names, heading=None)
dataset = ecomDataset(text_data)

train_size = int(len(text_data) * 0.8)
test_size = len(text_data) - train_size

train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_data, batch_size=32, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)

_, input_dim = preprocessing.get_embeddings(preprocessing.preprocess_text(text_data))

model = RNNModel(input_dim, hidden_dim=120, layer_dim=5)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# todo: train script
