#!/usr/bin/env python3
"""
CodeBERT model encoder module for Phase 5.
Tokenizes source code and generates 768-dimensional embeddings using sliding window chunking and mean pooling.
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List

class CodeBERTEncoder:
    """
    Encoder class to load CodeBERT and extract embeddings from source code text.
    Uses MPS (Apple Silicon GPU) if available, otherwise falls back to CPU.
    """
    def __init__(self, model_name: str = "microsoft/codebert-base", chunk_size: int = 512, overlap: int = 256) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_tokens = chunk_size - 2  # 510 tokens for content
        
        # Configure GPU acceleration for M1 Mac
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"[*] Loading CodeBERT model '{model_name}' on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[+] Model loaded successfully.")

    def encode_code(self, code_text: str) -> np.ndarray:
        """
        Tokenizes the source code, applies sliding window chunking for long files,
        runs model inference, and aggregates chunk embeddings via mean pooling.
        """
        if not code_text or not code_text.strip():
            return np.zeros(768, dtype=np.float32)
            
        # 1. Tokenize source code without special tokens first
        tokens = self.tokenizer.encode(code_text, add_special_tokens=False)
        
        if len(tokens) == 0:
            return np.zeros(768, dtype=np.float32)
            
        # 2. Divide tokens into chunks using a sliding window
        chunks: List[List[int]] = []
        i = 0
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            # Form complete chunk with [CLS] and [SEP]
            full_chunk = [cls_id] + chunk_tokens + [sep_id]
            chunks.append(full_chunk)
            
            # Stop if the current chunk captures the rest of the tokens
            if i + self.max_tokens >= len(tokens):
                break
                
            # Advance sliding window by (max_tokens - overlap)
            i += (self.max_tokens - self.overlap)
            
        # 3. Generate hidden states for each chunk and apply mean pooling
        chunk_embs = []
        
        # Process chunks in small batches to save GPU memory on M1 Air (8GB)
        batch_size = 8
        for batch_idx in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_idx : batch_idx + batch_size]
            
            # Since chunks might have slightly different lengths (especially the final one),
            # pad them to the maximum length in the batch
            max_len = max(len(c) for c in batch_chunks)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
            
            padded_chunks = [c + [pad_id] * (max_len - len(c)) for c in batch_chunks]
            
            input_ids = torch.tensor(padded_chunks, dtype=torch.long).to(self.device)
            attention_mask = (input_ids != pad_id).long().to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # hidden_states size: (batch_size, seq_len, 768)
                hidden_states = outputs.last_hidden_state
                
                # Apply mean pooling over active tokens (excluding pads) for each chunk in the batch
                for b in range(len(batch_chunks)):
                    mask = attention_mask[b].unsqueeze(-1)  # (seq_len, 1)
                    active_hidden = hidden_states[b] * mask  # (seq_len, 768)
                    active_sum = active_hidden.sum(dim=0)  # (768,)
                    active_count = mask.sum().clamp(min=1)  # scalar
                    
                    mean_pool = (active_sum / active_count).cpu().numpy()
                    chunk_embs.append(mean_pool)
                    
        if not chunk_embs:
            return np.zeros(768, dtype=np.float32)
            
        # 4. Average across all chunks to generate one file embedding
        file_embedding = np.mean(chunk_embs, axis=0)
        return file_embedding.astype(np.float32)

if __name__ == "__main__":
    # Test execution
    encoder = CodeBERTEncoder()
    test_code = "def add(a, b):\n    return a + b"
    emb = encoder.encode_code(test_code)
    print(f"Test embedding shape: {emb.shape}")
    print(f"Sample values (first 5): {emb[:5]}")
