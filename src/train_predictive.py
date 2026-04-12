import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm

from predictive_transformer import PredictiveTransformer

class NormalVideoDataset(Dataset):
    def __init__(self, features_dir):
        super().__init__()
        self.features_dir = Path(features_dir)
        self.normal_dir = self.features_dir / "NormalVideos"
        
        if not self.normal_dir.exists():
            raise FileNotFoundError(f"Not found: {self.normal_dir}")
            
        self.file_paths = list(self.normal_dir.rglob("*.npy"))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        features = np.load(file_path)
        return torch.FloatTensor(features)

def train_predictive(features_dir, epochs=50, batch_size=32, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    
    dataset = NormalVideoDataset(features_dir)
    if len(dataset) == 0:
        print("Empty dataset")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = PredictiveTransformer(feature_dim=1024).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for features in progress_bar:
            features = features.to(device)
            optimizer.zero_grad()
            
            predictions = model(features)
            
            preds = predictions[:, :-1, :]
            targets = features[:, 1:, :]
            
            loss = criterion(preds, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        
    weights_dir = Path("data/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "predictive_transformer.pth"
    
    torch.save(model.state_dict(), weights_path)
    print("Saved weights!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    train_predictive(
        features_dir=args.features_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )