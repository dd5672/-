import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import NERDataset
from model import ALP2025_NER

# ======================
# 基本配置
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ethanyt/guwenbert-base"
NER_PATH = "./ner"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 20
MODEL_PREFIX = "alp2025_epoch"

# ======================
# 标签
# ======================
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

# ======================
# 数据
# ======================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = NERDataset(
    f"{NER_PATH}/train.txt",
    tokenizer,
    label2id,
    MAX_LEN
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ======================
# 评估函数（严格匹配你的模型）
# ======================
def evaluate_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # ---- Loss ----
            loss = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            # ---- Accuracy ----
            preds = model(input_ids, attention_mask)

            mask = (labels != -100) & (attention_mask == 1)
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# ======================
# 主流程
# ======================
loss_list = []
acc_list = []

for epoch in range(1, EPOCHS + 1):
    print(f"Evaluating epoch {epoch}...")

    model = ALP2025_NER(MODEL_NAME, len(label2id))
    model.load_state_dict(
        torch.load(f"{MODEL_PREFIX}{epoch}.pt", map_location=DEVICE)
    )
    model.to(DEVICE)

    loss, acc = evaluate_epoch(model, dataloader)
    loss_list.append(loss)
    acc_list.append(acc)

    print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")

# ======================
# 绘图
# ======================
epochs = list(range(1, EPOCHS + 1))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss_list, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, acc_list, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Curve")
plt.grid(True)

plt.tight_layout()
plt.show()
