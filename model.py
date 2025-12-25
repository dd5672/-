# model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class ALP2025_NER(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.bilstm = nn.LSTM(
            hidden,
            hidden // 2,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = outputs.last_hidden_state        # [B, T, H]
        x, _ = self.bilstm(x)                # [B, T, H]
        logits = self.classifier(x)          # [B, T, C]

        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss
        else:
            return torch.argmax(logits, dim=-1)
