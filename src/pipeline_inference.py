import argparse
import os
import sys
import torch
import numpy as np

# Ensure we can import from the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from mil_classifier import AnomalyClassifier
except Exception as e:
    print(f"Warning: Could not import AnomalyClassifier from mil_classifier.py. Ensure the file exists. Error: {e}")
    # Fallback dummy class if it doesn't exist just so script doesn't crash on syntax/import check
    import torch.nn as nn
    class AnomalyClassifier(nn.Module):
        def __init__(self, input_dim=1024):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            return self.sigmoid(self.fc(x)).squeeze(-1)

from caption_transformer import CaptionTransformer
from caption_dataset import SimpleTokenizer

def generate_caption(model, visual_features, tokenizer, max_length=50, device='cpu'):
    """
    Autoregressive generation method for Phase 2.
    """
    model.eval()
    
    # Start with <SOS> token
    sos_token = tokenizer.vocab.get(tokenizer.SOS)
    eos_token = tokenizer.vocab.get(tokenizer.EOS)
    
    # Initialize sequence: shape [1, 1]
    current_sequence = torch.LongTensor([[sos_token]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Generate causal mask for the current sequence length
            seq_len = current_sequence.size(1)
            
            # Forward pass
            logits = model(visual_features=visual_features, captions=current_sequence)
            
            # Get logits for the last predicted token
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            # Append predicted token
            next_token_tensor = torch.LongTensor([[next_token_id]]).to(device)
            current_sequence = torch.cat([current_sequence, next_token_tensor], dim=1)
            
            # Break early if <EOS> is predicted
            if next_token_id == eos_token:
                break
                
    # Decode the final sequence (skip <SOS> at index 0 and <EOS> if it's there)
    token_ids = current_sequence[0].cpu().numpy().tolist()
    generated_text = tokenizer.decode(token_ids)
    
    return generated_text

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 1. Load I3D Features
    # ---------------------------------------------------------
    if not os.path.exists(args.feature_path):
        print(f"Error: Feature file not found at {args.feature_path}")
        return
        
    features = np.load(args.feature_path) # Expected: [32, 1024]
    
    # Prepare tensor for Phase 1: [batch_size=1, num_segments=32, feature_dim=1024]
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
    
    # ---------------------------------------------------------
    # 2. Phase 1: Detection
    # ---------------------------------------------------------
    print("\n--- Running Phase 1 (Anomaly Detection) ---")
    mil_model = AnomalyClassifier(input_dim=1024).to(device)
    
    if os.path.exists(args.mil_weights):
        mil_model.load_state_dict(torch.load(args.mil_weights, map_location=device))
        mil_model.eval()
    else:
        print(f"Warning: MIL weights not found at {args.mil_weights}. Proceeding with initialized weights.")
        mil_model.eval()
        
    with torch.no_grad():
        # Get anomaly scores for all 32 segments
        scores = mil_model(features_tensor) 
        scores = scores.squeeze().cpu().numpy() # Shape should be [32]
        
    max_score = float(np.max(scores))
    
    if max_score < args.threshold:
        print(f"Status: Normal. No caption needed. (Max Score: {max_score:.2f} < {args.threshold})")
        return
        
    # ---------------------------------------------------------
    # 3. The Bridge
    # ---------------------------------------------------------
    peak_index = int(np.argmax(scores))
    print(f"Anomaly detected at segment index: {peak_index} with score: {max_score:.2f}")
    
    # Extract window of I3D features around peak
    start_idx = max(0, peak_index - args.window_size)
    end_idx = min(len(features), peak_index + args.window_size + 1)
    
    anomaly_features = features[start_idx:end_idx] 
    print(f"Extracted context window from segment {start_idx} to {end_idx - 1} (Total: {len(anomaly_features)} segments)")
    
    # Prepare tensor for Phase 2: [1, window_len, 1024]
    anomaly_tensor = torch.FloatTensor(anomaly_features).unsqueeze(0).to(device)
    
    # ---------------------------------------------------------
    # 4. Phase 2: Captioning
    # ---------------------------------------------------------
    print("\n--- Running Phase 2 (Caption Generation) ---")
    
    if not os.path.exists(args.vocab_file):
        print(f"Error: Vocabulary file not found at {args.vocab_file}. Run training first.")
        return
        
    tokenizer = SimpleTokenizer(vocab_file=args.vocab_file)
    vocab_size = len(tokenizer.vocab)
    
    caption_model = CaptionTransformer(
        vocab_size=vocab_size,
        feature_dim=1024,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    
    if os.path.exists(args.caption_weights):
        caption_model.load_state_dict(torch.load(args.caption_weights, map_location=device))
        caption_model.eval()
    else:
        print(f"Warning: Caption weights not found at {args.caption_weights}. Proceeding with untrained model.")
        caption_model.eval()
        
    # Autoregressive Generation
    generated_text = generate_caption(
        caption_model, 
        anomaly_tensor, 
        tokenizer, 
        max_length=args.max_length, 
        device=device
    )
    
    # ---------------------------------------------------------
    # 5. Final Output Print
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"[ALERT] Anomaly Detected with Confidence: {max_score:.2f}")
    print(f"[REPORT] System generated caption: {generated_text}")
    print("="*50 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-End Inference: Anomaly Detection to Captioning")
    
    # I/O Paths
    parser.add_argument('--feature_path', type=str, required=True, help='Path to input .npy I3D feature file')
    parser.add_argument('--mil_weights', type=str, default='data/weights/mil_classifier.pth', help='Path to MIL classifier weights')
    parser.add_argument('--caption_weights', type=str, default='data/weights/caption_transformer.pth', help='Path to Caption Transformer weights')
    parser.add_argument('--vocab_file', type=str, default='data/vocab.json', help='Path to tokenizer vocabulary file')
    
    # Inference parameters
    parser.add_argument('--threshold', type=float, default=0.75, help='Score threshold to trigger caption generation')
    parser.add_argument('--window_size', type=int, default=2, help='Segments to extract around peak (e.g., 2 -> peak-2 to peak+2)')
    
    # Caption Transformer hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=50, help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    main(args)
