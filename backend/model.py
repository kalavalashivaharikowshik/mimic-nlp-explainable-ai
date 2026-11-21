"""Hybrid BioBERT + MLP mortality model.

Parameter names are unchanged from the original model so the published
``hybrid_model.pth`` weights load without conversion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from .config import N_FEATURES, TEXT_MODEL_NAME

STRUCT_HIDDEN = 64
STRUCT_OUT = 32
CLASSIFIER_HIDDEN = 128


class HybridModel(nn.Module):
    def __init__(
        self,
        text_config: PretrainedConfig | None = None,
        n_structured: int = N_FEATURES,
        pretrained: bool = False,
    ):
        """
        Args:
            text_config: transformer config. Defaults to the BioBERT config.
            n_structured: number of structured (vitals/labs) features.
            pretrained: download pretrained BioBERT weights. Not needed when a
                full state dict is loaded afterwards (that would download
                ~430 MB only to overwrite it), so it is off by default.
        """
        super().__init__()

        if pretrained:
            self.text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, attn_implementation="eager")
        else:
            if text_config is None:
                text_config = AutoConfig.from_pretrained(TEXT_MODEL_NAME)
            # "eager" attention is required to return attention weights.
            self.text_model = AutoModel.from_config(text_config, attn_implementation="eager")

        hidden = self.text_model.config.hidden_size  # 768 for BioBERT-base

        self.struct_net = nn.Sequential(
            nn.Linear(n_structured, STRUCT_HIDDEN),
            nn.ReLU(),
            nn.Linear(STRUCT_HIDDEN, STRUCT_OUT),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden + STRUCT_OUT, CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(CLASSIFIER_HIDDEN, 1),
        )

    # ---- building blocks (used separately for fast explanations) ----
    def encode_text(self, input_ids, attention_mask, output_attentions: bool = False):
        """Return the [CLS] embedding (and optionally all attention maps)."""
        out = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        cls = out.last_hidden_state[:, 0, :]
        return (cls, out.attentions) if output_attentions else (cls, None)

    def head(self, text_emb: torch.Tensor, structured: torch.Tensor) -> torch.Tensor:
        """Fuse a text embedding with (already scaled) structured features -> logits."""
        if text_emb.shape[0] == 1 and structured.shape[0] > 1:
            text_emb = text_emb.expand(structured.shape[0], -1)
        struct_out = self.struct_net(structured)
        return self.classifier(torch.cat((text_emb, struct_out), dim=1))

    def forward(self, input_ids, attention_mask, structured):
        text_emb, _ = self.encode_text(input_ids, attention_mask)
        return self.head(text_emb, structured)
