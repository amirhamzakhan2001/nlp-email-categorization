# models/embedding_models/qwen.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class QwenEmbedder:
    """
    Inference-only wrapper for Qwen embedding models.
    """

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        cache_dir=None,
        device: str | None = None,
        max_length: int = 8192,
    ):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            use_fast=True,
            padding_side="left",
        )

        self.model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if self.device.type == "cuda" else None,
        ).to(self.device)

        self.model.eval()
        self.max_length = max_length

    @staticmethod
    def _mean_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            enc = {k: v.to(self.device) for k, v in enc.items()}

            outputs = self.model(**enc, return_dict=True)
            pooled = self._mean_pool(
                outputs.last_hidden_state,
                enc["attention_mask"],
            )

            vectors.append(pooled.cpu().numpy().astype(np.float32))

            del enc, outputs, pooled

        return np.vstack(vectors)