"""
Embedding model configuration.

The model is instantiated ONCE by app/main.py at startup via the lifespan
context manager and shared across all service functions as a parameter.

This module exists to centralise the model name so it can be referenced
consistently by scripts (e.g. scripts/precompute_embeddings.py) without
triggering a model download at import time.
"""

MODEL_NAME = "all-MiniLM-L6-v2"
