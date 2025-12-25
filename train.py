# train.py
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

from dataset import NERDataset
from model import ALP2025_NER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ethanyt/guwenbert-base"
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-5
MAX_LEN = 256
NER_PATH = "./ner"

def load_labels(path):
    labels = {"O"}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labels.add(line.split()[1])
    labels = sorted(labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label

label2id, id2label = load_labels(f"{NER_PATH}/train.txt")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = ALP2025_NER(MODEL_NAME, len(label2id)).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

train_dataset = NERDataset(f"{NER_PATH}/train.txt", tokenizer, label2id, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()

        loss = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["labels"].to(DEVICE)
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), f"alp2025_epoch{epoch+1}.pt")
