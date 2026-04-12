import torch
import torch.nn as nn
import math
import argparse

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class PredictiveTransformer(nn.Module):
    def __init__(self, feature_dim=1024, num_heads=8, num_layers=4, dropout=0.1):
        super(PredictiveTransformer, self).__init__()
        self.feature_dim = feature_dim
        
        self.pos_encoder = PositionalEncoding(d_model=feature_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.projection = nn.Linear(feature_dim, feature_dim)
        
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        
        x = self.pos_encoder(x)
        mask = self._generate_causal_mask(seq_len, x.device)
        output = self.transformer_encoder(x, mask=mask, is_causal=True)
        
        output = self.projection(output)
        return output

def calculate_anomaly_scores(predictions, actual_features):
    pred_shifted = predictions[:, :-1, :]
    actual_shifted = actual_features[:, 1:, :]
    
    mse = torch.mean((pred_shifted - actual_shifted) ** 2, dim=-1)
    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        print("Testing transformer...")
        batch_size = 2
        seq_len = 32
        feature_dim = 1024
        
        dummy_input = torch.randn(batch_size, seq_len, feature_dim)
        print(f"Input: {dummy_input.shape}")
        
        model = PredictiveTransformer(feature_dim=feature_dim)
        predictions = model(dummy_input)
        print(f"Output: {predictions.shape}")
        
        scores = calculate_anomaly_scores(predictions, dummy_input)
        print(f"Scores: {scores.shape}")
