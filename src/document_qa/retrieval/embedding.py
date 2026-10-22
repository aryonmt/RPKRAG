import hashlib
from pathlib import Path
from typing import Any, List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from llama_index.core.embeddings import BaseEmbedding

from document_qa.core.config import settings
from document_qa.ingestion.normalizer import normalize_for_search


class EmbeddingInputTooLong(Exception):
    """Raised when the input text exceeds the maximum allowed tokens."""
    pass


class LocalOnnxEmbedding(BaseEmbedding):
    """
    Custom embedding class using ONNX Runtime and the local tokenizer.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        
        model_dir = Path(settings.EMBEDDING_MODEL_DIR)
        onnx_path = model_dir / "model_optimized.onnx"
        tokenizer_path = model_dir / "tokenizer.json"

        if not onnx_path.exists() or not tokenizer_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_dir}")

        # Load tokenizer
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.no_truncation()

        # Load ONNX model
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.EMBEDDING_THREADS
        self._session = ort.InferenceSession(
            str(onnx_path), 
            sess_options=opts, 
            providers=["CPUExecutionProvider"]
        )
        
        # Discover input names dynamically (some models need token_type_ids, some don't)
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name

    def _mean_pooling(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Pools the token embeddings into a single sentence embedding."""
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
        
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        
        return sum_embeddings / sum_mask

    def _get_text_embedding(self, text: str) -> List[float]:
        """Processes a single text string."""
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Processes a batch of texts."""
        if not texts:
            return []

        # Normalize and check length
        normalized_texts = [normalize_for_search(t) for t in texts]
        
        encodings = self._tokenizer.encode_batch(normalized_texts)
        
        # Ensure no text exceeds the limit (including special tokens)
        max_len = 0
        for enc in encodings:
            if len(enc.ids) > settings.EMBEDDING_MAX_TOKENS:
                raise EmbeddingInputTooLong(f"Text exceeds {settings.EMBEDDING_MAX_TOKENS} tokens.")
            max_len = max(max_len, len(enc.ids))

        # Pad to max_len in the batch
        input_ids = []
        attention_masks = []
        
        for enc in encodings:
            pad_len = max_len - len(enc.ids)
            input_ids.append(enc.ids + [0] * pad_len) # Assuming 0 is [PAD]
            attention_masks.append(enc.attention_mask + [0] * pad_len)

        inputs = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_masks, dtype=np.int64)
        }
        
        # Add token_type_ids if the model expects it
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        # Run inference
        outputs = self._session.run([self._output_name], inputs)
        last_hidden_state = outputs[0]

        # Pooling
        if len(last_hidden_state.shape) == 3:
            # Shape: (batch, seq_len, hidden_dim)
            embeddings = self._mean_pooling(last_hidden_state, inputs["attention_mask"])
        else:
            # Shape: (batch, hidden_dim) - already pooled
            embeddings = last_hidden_state

        # Verify dimension
        if embeddings.shape[1] != settings.EMBEDDING_DIM:
            raise ValueError(f"Expected dim {settings.EMBEDDING_DIM}, got {embeddings.shape[1]}")

        return embeddings.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Alias for query embedding."""
        return self._get_text_embedding(query)

    def _aget_query_embedding(self, query: str) -> List[float]:
        """Async alias for query embedding."""
        return self._get_text_embedding(query)


class FakeEmbedding(BaseEmbedding):
    """
    A deterministic fake embedding for fast offline tests.
    It hashes words to generate a pseudo-random but repeatable vector.
    """
    def _get_text_embedding(self, text: str) -> List[float]:
        # Create a deterministic vector based on the hash of the text
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        # Repeat hash to get 384 floats
        floats = []
        for i in range(384):
            val = hash_bytes[i % len(hash_bytes)] / 255.0
            floats.append(val)
        # Normalize L2
        norm = np.linalg.norm(floats)
        return (np.array(floats) / norm).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)