#!/usr/bin/env python3
"""Test script to verify library loading in SLURM environment"""

import sys
import os

print("=== Environment Test ===")
print(f"Python: {sys.executable}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'NOT SET')}")
print()

try:
    import fairseq2
    print("✓ fairseq2 imported successfully")
except Exception as e:
    print(f"✗ Failed to import fairseq2: {e}")
    sys.exit(1)

try:
    from stopes.modules.preprocess.sonar_text_embedding import SonarTextEmbedderConfig
    print("✓ sonar_text_embedding imported successfully")
except Exception as e:
    print(f"✗ Failed to import sonar_text_embedding: {e}")
    sys.exit(1)

print()
print("=== All imports successful ===")

