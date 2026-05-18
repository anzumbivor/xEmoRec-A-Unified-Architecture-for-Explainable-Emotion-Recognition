"""Model architecture used by the training and latest adaptive-XAI workflow."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


class ImprovedDeepEmotionModel(nn.Module):
    """RoBERTa -> CNN -> BiLSTM -> CLS-query multi-head attention classifier.

    The RoBERTa encoder is frozen, matching the notebooks. The trainable layers
    are the CNN, BiLSTM, CLS-query attention, layer norm, dropout, and final
    linear classifier. The `return_attn` and `return_embeddings` flags expose the
    intermediate tensors required by attention-based explanations and Integrated
    Gradients.
    """

    def __init__(
        self,
        num_classes: int = 6,
        dropout: float = 0.3,
        roberta_name: str = "roberta-base",
    ) -> None:
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.cnn = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.cls_query = nn.Parameter(torch.randn(1, 1, 512))
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(512)
        self.fc = nn.Linear(512, num_classes)

    def _mha_with_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-head attention weights when supported by the installed PyTorch.

        Newer PyTorch versions support `average_attn_weights=False`. The fallback
        keeps compatibility with older environments by adding the head dimension.
        """
        try:
            attn_out, attn_weights = self.attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        except TypeError:
            attn_out, attn_weights = self.attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
            )
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(1)
        return attn_out, attn_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attn: bool = False,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        roberta_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = roberta_out.last_hidden_state  # [batch, seq_len, 768]

        x = embeddings.transpose(1, 2)              # [batch, 768, seq_len]
        x = F.relu(self.cnn(x))                     # [batch, 128, seq_len]
        x = self.maxpool(x)                         # [batch, 128, pooled_len]
        x = x.transpose(1, 2)                       # [batch, pooled_len, 128]

        pooled_mask = attention_mask[:, ::2]
        pooled_mask = pooled_mask[:, : x.size(1)]
        key_padding_mask = ~pooled_mask.bool()

        x, _ = self.bilstm(x)                       # [batch, pooled_len, 512]
        query = self.cls_query.expand(x.size(0), -1, -1)
        attn_out, attn_weights = self._mha_with_weights(
            query, x, x, key_padding_mask=key_padding_mask
        )

        pooled = attn_out.squeeze(1)                # [batch, 512]
        pooled = self.layernorm(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)                    # [batch, num_classes]

        out = {"logits": logits}
        if return_attn:
            out["attn_weights"] = attn_weights
            out["pooled_mask"] = pooled_mask
        if return_embeddings:
            out["embeddings"] = embeddings
            out["emb_mask"] = attention_mask.bool()
        return out


def forward_from_embeddings(
    model: ImprovedDeepEmotionModel,
    embeddings: torch.Tensor,
    emb_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward pass from embeddings onward for Integrated Gradients."""
    x = embeddings.transpose(1, 2)
    x = F.relu(model.cnn(x))
    x = model.maxpool(x)
    x = x.transpose(1, 2)

    pooled_mask = emb_mask[:, ::2]
    pooled_mask = pooled_mask[:, : x.size(1)]
    key_padding_mask = ~pooled_mask.bool()

    x, _ = model.bilstm(x)
    query = model.cls_query.expand(x.size(0), -1, -1)
    attn_out, _ = model._mha_with_weights(query, x, x, key_padding_mask=key_padding_mask)

    pooled = attn_out.squeeze(1)
    pooled = model.layernorm(pooled)
    pooled = model.dropout(pooled)
    return model.fc(pooled)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
