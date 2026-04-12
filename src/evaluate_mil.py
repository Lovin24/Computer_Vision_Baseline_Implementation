"""
Evaluate the MIL Anomaly Classifier (Pipeline 1) with I3D features.
Computes ROC-AUC, PR-AUC, F1-Score, and generates plots.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
from tqdm import tqdm

from src.mil_classifier import AnomalyClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def evaluate_mil(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    model = AnomalyClassifier(input_dim=args.input_dim)
    if args.weights.is_file():
        model.load_state_dict(torch.load(args.weights, map_location=device))
        log.info(f"Loaded weights from {args.weights}")
    else:
        log.warning(f"Weights file not found at {args.weights}. Using randomly initialized model.")
    
    model.to(device)
    model.eval()

    annotations = {}
    has_annotations = False
    if args.annotations and args.annotations.is_file():
        with open(args.annotations, "r") as f:
            annotations = json.load(f)
        log.info(f"Loaded annotations from {args.annotations}")
        has_annotations = True
    else:
        log.info("No temporal annotations provided. Segment-level metrics will be skipped. Evaluating at video-level only.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_scores = []
    all_gts = []
    
    video_scores = []
    video_gts = []
    
    sample_timeline_data = None
    
    features_dir = Path(args.features_dir)
    npy_files = list(features_dir.rglob("*.npy"))
    
    if not npy_files:
        log.error(f"No .npy feature files found in {features_dir}")
        return

    log.info(f"Evaluating {len(npy_files)} videos...")
    
    for npy_file in tqdm(npy_files, desc="Evaluating"):
        category = npy_file.parent.name
        video_name = npy_file.stem
        is_normal = (category == "NormalVideos")
        
        features = np.load(str(npy_file)).astype(np.float32)
        if features.shape != (32, args.input_dim):
            log.warning(f"Skipping {video_name} with shape {features.shape}")
            continue
            
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            scores = model(features_tensor).squeeze(0).cpu().numpy()
            
        v_gt = 0 if is_normal else 1
        v_score = np.max(scores)
        video_gts.append(v_gt)
        video_scores.append(v_score)
        
        if has_annotations:
            gt = np.zeros(32, dtype=np.int32)
            if not is_normal:
                if video_name in annotations:
                    anom_indices = annotations[video_name]
                    gt[anom_indices] = 1
                else:
                    log.warning(f"Missing annotations for anomalous video {video_name}. Assuming GT=0 for all segments (will penalize segment metrics).")
                    
                if sample_timeline_data is None:
                    sample_timeline_data = {
                        "name": video_name,
                        "scores": scores,
                        "gt": gt
                    }
                    
            all_scores.append(scores)
            all_gts.append(gt)

    if not video_scores:
        log.error("No valid features evaluated.")
        return

    video_scores = np.array(video_scores)
    video_gts = np.array(video_gts)
    video_roc_auc = roc_auc_score(video_gts, video_scores)
    
    if has_annotations and all_scores:
        all_scores = np.concatenate(all_scores)
        all_gts = np.concatenate(all_gts)
        seg_roc_auc = roc_auc_score(all_gts, all_scores)
        seg_pr_auc = average_precision_score(all_gts, all_scores)
        preds = (all_scores >= 0.5).astype(np.int32)
        seg_f1 = f1_score(all_gts, preds)
    else:
        seg_roc_auc, seg_pr_auc, seg_f1 = None, None, None

    print("\n" + "="*50)
    print(f" MIL Classifier Evaluation Summary")
    print("="*50)
    print(f" Total Videos Evaluated   : {len(video_scores)}")
    print(f" Video-Level ROC-AUC      : {video_roc_auc:.4f}")
    print("-" * 50)
    
    if has_annotations:
        print(f" Total Segments Evaluated : {len(all_scores)}")
        print(f" Segment-Level ROC-AUC    : {seg_roc_auc:.4f}")
        print(f" Segment-Level PR-AUC     : {seg_pr_auc:.4f}")
        print(f" Segment-Level F1 (thr=0.5): {seg_f1:.4f}")
    else:
        print(f" Segment-Level Metrics    : SKIPPED (No Annotations)")
    print("="*50 + "\n")

    sns.set_theme(style="whitegrid")
    
    fpr_vid, tpr_vid, _ = roc_curve(video_gts, video_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_vid, tpr_vid, label=f'Video ROC (AUC = {video_roc_auc:.4f})', color='purple', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Video-Level ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(args.output_dir / 'video_roc_curve.png', dpi=300)
    plt.close()
    
    if has_annotations and all_scores is not None:
        fpr, tpr, _ = roc_curve(all_gts, all_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Segment ROC (AUC = {seg_roc_auc:.4f})', color='b', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Segment-Level ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(args.output_dir / 'roc_curve.png', dpi=300)
        plt.close()
        
        precision, recall, _ = precision_recall_curve(all_gts, all_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Segment PR (AP = {seg_pr_auc:.4f})', color='g', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Segment-Level PR Curve')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(args.output_dir / 'pr_curve.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        normal_scores = all_scores[all_gts == 0]
        anom_scores = all_scores[all_gts == 1]
        
        sns.kdeplot(normal_scores, label="Normal Segments (GT=0)", fill=True, color='blue', alpha=0.5)
        if len(anom_scores) > 0:
            sns.kdeplot(anom_scores, label="Anomalous Segments (GT=1)", fill=True, color='red', alpha=0.5)
            
        plt.xlabel("Predicted Anomaly Score")
        plt.ylabel("Density")
        plt.title("Segment Score Distribution (Normal vs. Anomalous)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / 'score_distribution.png', dpi=300)
        plt.close()
        
        if sample_timeline_data is not None:
            plt.figure(figsize=(10, 4))
            timeline_scores = sample_timeline_data["scores"]
            timeline_gt = sample_timeline_data["gt"]
            segments = np.arange(32)
            
            plt.plot(segments, timeline_scores, marker='o', linestyle='-', color='b', label='Predicted Score')
            
            in_anomaly = False
            start_idx = 0
            for i in range(32):
                if timeline_gt[i] == 1 and not in_anomaly:
                    in_anomaly = True
                    start_idx = i
                elif timeline_gt[i] == 0 and in_anomaly:
                    in_anomaly = False
                    plt.axvspan(start_idx - 0.5, i - 0.5, color='red', alpha=0.3, label='GT Anomaly' if start_idx==0 else "")
            if in_anomaly:
                plt.axvspan(start_idx - 0.5, 31.5, color='red', alpha=0.3, label='GT Anomaly' if start_idx==0 else "")
                
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            plt.ylim(0, 1.05)
            plt.xlim(-0.5, 31.5)
            plt.xlabel("Segment Index")
            plt.ylabel("Anomaly Score")
            plt.title(f"Sample Timeline: {sample_timeline_data['name']}")
            plt.tight_layout()
            plt.savefig(args.output_dir / 'sample_timeline.png', dpi=300)
            plt.close()
            
    log.info(f"Evaluation complete. Visualizations saved to {args.output_dir}")

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MIL Classifier (Pipeline 1)")
    parser.add_argument(
        "--features-dir", 
        type=Path, 
        default=Path("data/features/i3d"),
        help="Path to the I3D .npy features directory"
    )
    parser.add_argument(
        "--weights", 
        type=Path, 
        default=Path("data/weights/mil_classifier.pth"),
        help="Path to the trained AnomalyClassifier weights"
    )
    parser.add_argument(
        "--input-dim", 
        type=int, 
        default=1024,
        help="Dimension of I3D features (default 1024)"
    )
    parser.add_argument(
        "--annotations", 
        type=Path, 
        default=None,
        help="Path to JSON dictionary mapping anomalous video names to anomalous segment indices"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("results/"),
        help="Path to save the plots (default results/)"
    )
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    evaluate_mil(args)

if __name__ == "__main__":
    main()
