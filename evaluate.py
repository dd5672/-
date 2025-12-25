import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import classification_report, f1_score

from dataset import NERDataset
from model import ALP2025_NER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ethanyt/guwenbert-base"
MODEL_PATH = "alp2025_epoch20.pt"
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

model = ALP2025_NER(MODEL_NAME, len(label2id))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
test_dataset = NERDataset(f"{NER_PATH}/test.txt", tokenizer, label2id, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=16)
y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        preds = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE)
        )
        for i in range(len(preds)):
            true_seq, pred_seq = [], []
            for j in range(len(preds[i])):
                if batch["labels"][i][j] != -100 and batch["attention_mask"][i][j] == 1:
                    true_seq.append(id2label[batch["labels"][i][j].item()])
                    pred_seq.append(id2label[preds[i][j].item()])
            y_true.append(true_seq)
            y_pred.append(pred_seq)

print(classification_report(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))