import torch
from torch.utils.data import Dataset

import preprocessing


class ecomDataset(Dataset):
    """
    Simple class for pt Dataset, using functions from preprocessing.
    """
    def __init__(self, data_table):
        processed_labels = preprocessing.label_enc(data_table)
        self.texts = data_table["text"]
        self.labels = processed_labels["class"]
        embeddings, _ = preprocessing.get_embeddings(
            preprocessing.preprocess_text(processed_labels)
        )
        self.embeddings = embeddings["embeddings"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        idx_emb = self.embeddings[idx]
        idx_class = self.labels[idx]

        emb_tensor = torch.tensor(idx_emb, dtype=torch.float32)
        class_tensor = torch.tensor(idx_class, dtype=torch.long)

        return emb_tensor, class_tensor
