#!/usr/bin/env python3
"""
Embedding generator module for Phase 5.
Sequentially embeds source code files using CodeBERTEncoder.
Implements memory-efficient disk-backed numpy pre-allocation and checkpoints.
"""

import os
import json
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Set

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_DIR
from embeddings.codebert_encoder import CodeBERTEncoder

INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

def generate_embeddings() -> None:
    """
    Loads clean source dataset, initializes CodeBERTEncoder,
    and runs batch-wise embedding generation with resume checkpoint support.
    """
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    
    clean_csv_path = os.path.join(INTERMEDIATE_DIR, "clean_source_dataset.csv")
    if not os.path.exists(clean_csv_path):
        raise FileNotFoundError(f"Clean source dataset not found: {clean_csv_path}. Run preprocessor first.")
        
    df = pd.read_csv(clean_csv_path)
    num_files = len(df)
    
    if num_files == 0:
        print("[!] Warning: Clean source dataset is empty. No embeddings to generate.")
        return
        
    print(f"[*] Total files to embed: {num_files}")
    
    # 1. Initialize outputs and pre-allocate npy array using memmap
    npy_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    if not os.path.exists(npy_path):
        print(f"[*] Pre-allocating zero-filled numpy array on disk at {npy_path}...")
        # Create a zero array and save it to set up shape and file size
        empty_arr = np.zeros((num_files, 768), dtype=np.float32)
        np.save(npy_path, empty_arr)
        
    # Load with mmap_mode r+ so updates go straight to disk and don't eat RAM
    embeddings_mmap = np.load(npy_path, mmap_mode="r+")
    
    # 2. Load Checkpoint
    checkpoint_path = os.path.join(EMBEDDINGS_DIR, "checkpoint.json")
    processed_keys: Set[str] = set()
    
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                ckpt = json.load(f)
                processed_keys = set(ckpt.get("processed_keys", []))
            print(f"[+] Found checkpoint. Resuming with {len(processed_keys)} files already processed.")
        except Exception as e:
            print(f"[!] Warning: Failed to load checkpoint ({e}). Starting fresh.")
            
    # 3. Load encoder
    encoder = CodeBERTEncoder()
    
    # Batch parameters
    batch_size = 16
    checkpoint_interval = 16
    
    print("\n[*] Starting batch embedding generation...")
    
    dirty = False
    
    for idx, row in df.iterrows():
        repo = row["repository_name"]
        path = row["file_path"]
        key = f"{repo}:{path}"
        
        # Skip if already processed
        if key in processed_keys:
            continue
            
        code = str(row["source_code"])
        
        try:
            # Generate 768-D vector
            vec = encoder.encode_code(code)
            # Write directly to mapped memory
            embeddings_mmap[idx] = vec
            processed_keys.add(key)
            dirty = True
        except Exception as e:
            print(f"[!] Error generating embedding for {key}: {e}")
            continue
            
        processed_count = len(processed_keys)
        
        # Periodically save checkpoint and flush memmap
        if processed_count % checkpoint_interval == 0 and dirty:
            # Flush changes to disk
            embeddings_mmap.flush()
            
            # Save checkpoint JSON
            with open(checkpoint_path, "w") as f:
                json.dump({"processed_keys": list(processed_keys)}, f)
                
            print(f"[+] Progress: {processed_count} / {num_files} files embedded. Checkpoint saved.")
            dirty = False
            
    # Final flush and save checkpoint
    if dirty:
        embeddings_mmap.flush()
        with open(checkpoint_path, "w") as f:
            json.dump({"processed_keys": list(processed_keys)}, f)
            
    print(f"\n[+] Embedding generation complete. Processed {len(processed_keys)} / {num_files} files.")

if __name__ == "__main__":
    generate_embeddings()
