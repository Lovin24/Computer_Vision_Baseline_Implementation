import os
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from predictive_transformer import PredictiveTransformer, calculate_anomaly_scores

def evaluate_predictive(features_dir, weights_path, output_dir, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    model = PredictiveTransformer(feature_dim=1024)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    features_dir_path = Path(features_dir)
    all_files = list(features_dir_path.rglob("*.npy"))
    if len(all_files) == 0:
        print("No files found")
        return

    y_true = []
    y_scores = []
    normal_scores = []
    anomaly_scores_list = []
    
    with torch.no_grad():
        for file_path in tqdm(all_files, desc="Eval"):
            is_normal = "Normal" in file_path.parts or "NormalVideos" in file_path.parts
            label = 0 if is_normal else 1
            
            features = np.load(file_path)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            
            predictions = model(features_tensor)
            mse_scores = calculate_anomaly_scores(predictions, features_tensor)
            
            video_score = torch.max(mse_scores).item()
            
            y_true.append(label)
            y_scores.append(video_score)
            
            if label == 0:
                normal_scores.append(video_score)
            else:
                anomaly_scores_list.append(video_score)

    if len(set(y_true)) > 1:
        auc_score = roc_auc_score(y_true, y_scores)
        print(f"AUC: {auc_score:.4f}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    if normal_scores:
        sns.kdeplot(normal_scores, fill=True, label="Normal", color="blue", alpha=0.5)
    if anomaly_scores_list:
        sns.kdeplot(anomaly_scores_list, fill=True, label="Anomaly", color="red", alpha=0.5)
        
    plt.title("MSE Scores")
    plt.legend()
    
    plot_path = output_dir_path / "predictive_evaluation.png"
    plt.savefig(plot_path)
    plt.close()
    
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, default="data/features/i3d")
    parser.add_argument("--weights", type=str, default="data/weights/predictive_transformer.pth")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    evaluate_predictive(
        features_dir=args.features_dir,
        weights_path=args.weights,
        output_dir=args.output_dir,
        device=args.device
    )