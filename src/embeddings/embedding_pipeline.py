#!/usr/bin/env python3
"""
Master orchestrator script for the Phase 5 Embedding Generation Pipeline.
Runs extraction, preprocessing, CodeBERT embedding generation, formatting, label merging, and analysis.
"""

import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_DIR, ensure_dirs_exist
from embeddings.code_extractor import extract_source_files
from embeddings.code_preprocessor import run_preprocessing
from embeddings.embedding_generator import generate_embeddings
from embeddings.embedding_storage import compile_storage
from embeddings.embedding_dataset_builder import build_embedding_dataset
from embeddings.embedding_analysis import analyze_embeddings

def run_embedding_pipeline() -> None:
    """
    Sequences all stages of Phase 5 end-to-end.
    """
    print("="*60)
    print("Starting Phase 5 CodeBERT Embedding Generation Pipeline")
    print("="*60)
    
    # 0. Ensure directories exist
    ensure_dirs_exist()
    
    # 1. Source Code Extraction
    print("\n[STAGE 1/6] Extracting source code files...")
    extract_source_files()
    
    # 2. Source Code Cleaning
    print("\n[STAGE 2/6] Cleaning and preprocessing source code...")
    run_preprocessing()
    
    # 3. CodeBERT Embedding Generation
    print("\n[STAGE 3/6] Generating embeddings via CodeBERT...")
    generate_embeddings()
    
    # 4. Storage Compile
    print("\n[STAGE 4/6] Compiling storage formats (.parquet, metadata CSV)...")
    compile_storage()
    
    # 5. Build Final Labeled Parquet
    print("\n[STAGE 5/6] Building final embedding training dataset...")
    build_embedding_dataset()
    
    # 6. Run Statistical Analysis
    print("\n[STAGE 6/6] Executing statistical analysis and generating report...")
    analyze_embeddings()
    
    print("\n" + "="*60)
    print("Phase 5 Embedding Generation Pipeline Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    run_embedding_pipeline()
