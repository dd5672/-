# dataset.py
import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id, max_len):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.samples = self._load_data(file_path)

    def _load_data(self, path):
        samples = []
        tokens, labels = [], []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        samples.append((tokens, labels))
                        tokens, labels = [], []
                else:
                    char, label = line.split()
                    tokens.append(char)
                    labels.append(self.label2id[label])

        if tokens:
            samples.append((tokens, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, labels = self.samples[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }
