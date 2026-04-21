# backend/model.py

import torch
import torch.nn as nn
from transformers import AutoModel

structured_cols_len = 10  # number of structured features

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.text_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

        self.struct_net = nn.Sequential(
            nn.Linear(structured_cols_len, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, structured):
        text_output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_output.last_hidden_state[:, 0, :]

        struct_out = self.struct_net(structured)

        combined = torch.cat((text_emb, struct_out), dim=1)
        return self.classifier(combined)