import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import our custom modules
from caption_dataset import CaptionDataset, collate_fn, SimpleTokenizer
from caption_transformer import CaptionTransformer

def train_captioning(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Tokenizer and Dataset
    tokenizer = SimpleTokenizer(vocab_file=args.vocab_file if os.path.exists(args.vocab_file) else None)
    
    # Temporary dataset load to build vocab if it doesn't exist
    temp_dataset = CaptionDataset(args.captions_file, args.features_dir, tokenizer, max_length=args.max_length)
    if not os.path.exists(args.vocab_file):
        print("Building vocabulary...")
        # Assuming captions are stored under 'caption' or 'text' key
        captions = [item.get('caption', item.get('text', '')) for item in temp_dataset.data]
        tokenizer.build_vocab(captions, min_freq=2)
        
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(args.vocab_file) or '.', exist_ok=True)
        tokenizer.save_vocab(args.vocab_file)
        print(f"Vocabulary saved to {args.vocab_file}. Size: {len(tokenizer.vocab)}")
        
    # Re-initialize dataset with the fully populated tokenizer
    dataset = CaptionDataset(args.captions_file, args.features_dir, tokenizer, max_length=args.max_length)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # 2. Initialize Model
    vocab_size = len(tokenizer.vocab)
    model = CaptionTransformer(
        vocab_size=vocab_size,
        feature_dim=1024,
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)
    
    # 3. Loss, Optimizer, Scheduler
    pad_idx = tokenizer.vocab[tokenizer.PAD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_loss = float('inf')
    
    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            features = batch['features'].to(device) # [B, 32, 1024]
            captions = batch['captions'].to(device) # [B, seq_len]
            
            # Teacher forcing: 
            # Input to decoder is all words except the last
            # Target for loss is all words except the first (<SOS>)
            tgt_input = captions[:, :-1]
            tgt_target = captions[:, 1:]
            
            # Create padding mask for the input sequence (True where padding exists)
            tgt_padding_mask = (tgt_input == pad_idx)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(
                visual_features=features, 
                captions=tgt_input, 
                tgt_padding_mask=tgt_padding_mask
            )
            
            # Reshape logits and targets for CrossEntropyLoss
            # logits shape: [B, seq_len - 1, vocab_size] -> [B * (seq_len - 1), vocab_size]
            # tgt_target shape: [B, seq_len - 1] -> [B * (seq_len - 1)]
            loss = criterion(logits.reshape(-1, vocab_size), tgt_target.reshape(-1))
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping (good practice for Transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")
        
        # Save best model (include vocab so inference can decode correctly)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"New best loss! Saving model to {args.save_path}")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab': tokenizer.vocab,
                'inverse_vocab': tokenizer.inverse_vocab,
                'vocab_size': vocab_size,
                'hidden_dim': args.hidden_dim,
                'nhead': args.nhead,
                'num_layers': args.num_layers,
                'epoch': epoch,
                'loss': avg_loss,
            }
            torch.save(checkpoint, args.save_path)
            
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Caption Transformer")
    parser.add_argument('--captions_file', type=str, default='data/captions.json', help='Path to captions file')
    parser.add_argument('--features_dir', type=str, default='data/features/i3d/', help='Directory containing I3D features')
    parser.add_argument('--vocab_file', type=str, default='data/vocab.json', help='Path to save/load tokenizer vocab')
    parser.add_argument('--save_path', type=str, default='data/weights/caption_transformer.pth', help='Path to save best model weights')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_length', type=int, default=50, help='Max length for captions')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    
    args = parser.parse_args()
    
    train_captioning(args)
