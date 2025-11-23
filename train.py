import os

import pandas as pd
import torch
import torch.nn as nn
import tqdm
import yaml
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import preprocessing
from dataset import ecomDataset
from model import RNNModel

# Reading config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def collate_fn(batch):
    """
    Function, designed to pad every element of sequence to equal size.
    Padding vaule: 0
    """
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels


# Dataset preparing
column_names = ["class", "text"]
text_data = pd.read_csv("data/ecommerceDataset.csv", names=column_names, header=None)
dataset = ecomDataset(text_data)

train_size = int(len(text_data) * config["data"]["train_split"])
test_size = len(text_data) - train_size

train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(
    train_data,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
)

_, input_dim = preprocessing.get_embeddings(preprocessing.preprocess_text(text_data))


device = "cuda" if torch.cuda.is_available() else "cpu"
model = RNNModel(
    input_dim,
    hidden_dim=config["model"]["hidden_dim"],
    layer_dim=config["model"]["layer_dim"],
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

total_loss = []

# Train process
for epoch in range(config["training"]["num_epochs"]):
    pbar_prefix = f"Epoch {epoch + 1} | 10"
    current_epoch_loss = float(0)
    loop = tqdm.tqdm(train_loader, desc=pbar_prefix, leave=False)
    for embs, labels in loop:
        embs, labels = embs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(embs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        current_epoch_loss += loss.item()
        loop.set_postfix(loss=current_epoch_loss / (loop.n + 1))

    epoch_loss = current_epoch_loss / len(train_loader)
    total_loss.append(epoch_loss)


# Visualization
os.makedirs(config["paths"]["result_dir"], exist_ok=True)

epochs = range(1, len(total_loss) + 1)
fig = plt.figure(figsize=(6, 8))
plt.plot(epochs, total_loss, color="red")
plt.title("Train process")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.savefig(os.path.join(config["paths"]["result_dir"], "train_results.png"))
plt.close(fig)
