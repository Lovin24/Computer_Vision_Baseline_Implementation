import os
import subprocess
import torch
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.append('src')

from feature_extractor_i3d import I3DFeatureExtractor
from config import PreprocessConfig
from caption_dataset import SimpleTokenizer

def main():
    test_dir = Path("data/Test")
    out_dir = Path("data/features/i3d/Test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    weights_path = Path("data/weights/rgb_imagenet.pt")
    
    print("1. Checking/Building Vocab...")
    vocab_file = "data/vocab.json"
    if not os.path.exists(vocab_file):
        tokenizer = SimpleTokenizer()
        tokenizer.build_vocab([
            "a person is walking", 
            "an anomaly is detected",
            "someone is stealing",
            "a fight breaks out",
            "normal behavior"
        ], min_freq=1)
        tokenizer.save_vocab(vocab_file)
        print(f"Created dummy vocab at {vocab_file}")
    
    print("\n2. Extracting I3D Features for Test Videos...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We load extractor only if we need to extract features
    extractor = None
    config = PreprocessConfig()
    
    mp4_files = list(test_dir.glob("*.mp4"))
    for mp4 in mp4_files:
        out_file = out_dir / f"{mp4.stem}.npy"
        if not out_file.exists():
            if extractor is None:
                extractor = I3DFeatureExtractor(weights_path=weights_path, device=device)
            print(f"Extracting: {mp4.name} -> {out_file}")
            feats = extractor.extract_video_features(mp4, config)
            np.save(str(out_file), feats)
        else:
            print(f"Already extracted: {out_file}")

    print("\n3. Running End-to-End Inference...")
    for mp4 in mp4_files:
        npy_path = out_dir / f"{mp4.stem}.npy"
        print(f"\n=============================================")
        print(f"Testing Video: {mp4.name}")
        
        # Run the pipeline_inference script
        cmd = [
            "python", "src/pipeline_inference.py",
            "--feature_path", str(npy_path),
            "--vocab_file", vocab_file,
            "--threshold", "0.5" # Lowered threshold to see if we can trigger generation for testing
        ]
        
        subprocess.run(cmd)

if __name__ == '__main__':
    main()
