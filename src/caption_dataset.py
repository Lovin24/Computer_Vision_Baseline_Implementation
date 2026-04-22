import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import Counter
import re
import torch.nn.functional as F

class SimpleTokenizer:
    def __init__(self, vocab_file=None):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_file = vocab_file
        self.PAD = "<PAD>"
        self.SOS = "<SOS>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"
        self.special_tokens = [self.PAD, self.SOS, self.EOS, self.UNK]
        
        # Initialize with special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def build_vocab(self, captions, min_freq=1):
        """Builds vocabulary from a list of caption strings."""
        counter = Counter()
        for caption in captions:
            tokens = self._tokenize(caption)
            counter.update(tokens)
            
        idx = len(self.special_tokens)
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.vocab:
                self.vocab[word] = idx
                self.inverse_vocab[idx] = word
                idx += 1
                
    def _tokenize(self, text):
        # Basic word tokenization, making lowercase and removing punctuation
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def encode(self, text):
        """Encodes a string into a list of token IDs."""
        tokens = self._tokenize(text)
        token_ids = [self.vocab.get(self.SOS)]
        token_ids.extend([self.vocab.get(token, self.vocab.get(self.UNK)) for token in tokens])
        token_ids.append(self.vocab.get(self.EOS))
        return token_ids

    def decode(self, token_ids):
        """Decodes a list of token IDs back into a string."""
        words = []
        for token_id in token_ids:
            # Handle tensors
            if torch.is_tensor(token_id):
                token_id = token_id.item()
                
            word = self.inverse_vocab.get(token_id, self.UNK)
            if word in [self.PAD, self.SOS, self.EOS]:
                continue
            words.append(word)
        return " ".join(words)

    def save_vocab(self, filepath):
        """Saves vocabulary to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, filepath):
        """Loads vocabulary from a JSON file."""
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
            self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}

class CaptionDataset(Dataset):
    def __init__(self, captions_file, features_dir, tokenizer, max_length=None):
        """
        Args:
            captions_file (str): Path to JSON or CSV file containing captions and video info.
            features_dir (str): Directory containing .npy I3D feature files.
            tokenizer (SimpleTokenizer): Tokenizer to encode the text.
            max_length (int): Optional max length for the caption tokens.
        """
        self.features_dir = features_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_captions(captions_file)
        
        # Pre-index all .npy files in the features directory and subdirectories
        self.feature_paths = {}
        for root, dirs, files in os.walk(self.features_dir):
            for file in files:
                if file.endswith('.npy'):
                    vid = file.replace('.npy', '').replace('_i3d', '')
                    self.feature_paths[vid] = os.path.join(root, file)
                    
    def _load_captions(self, filepath):
        data = []
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                raw_data = json.load(f)
                # Handle both list of dicts or dict of dicts
                if isinstance(raw_data, dict):
                    for video_id, info in raw_data.items():
                        info['video_id'] = video_id
                        data.append(info)
                elif isinstance(raw_data, list):
                    data = raw_data
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            data = df.to_dict('records')
        else:
            raise ValueError("Unsupported file format. Please provide .json or .csv")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Determine video id / filename
        video_id = item.get('video_id', item.get('video', f'unknown_{idx}'))
        
        # Load features using pre-indexed paths
        feature_path = self.feature_paths.get(video_id)
        if not feature_path:
            # Fallback exact matching if not in index (e.g. Test videos)
            feature_path = os.path.join(self.features_dir, f"{video_id}.npy")
            
        if feature_path and os.path.exists(feature_path):
            features = np.load(feature_path) # Expected shape (32, 1024)
        else:
            # If not found, return zero tensor of expected shape
            features = np.zeros((32, 1024), dtype=np.float32)
            # print(f"Warning: Features for {video_id} not found.")
            
        # Expected features shape is (32, 1024) for pre-extracted I3D features.
        num_segments = features.shape[0]
        
        start_time = item.get('start') or item.get('start_time')
        end_time = item.get('end') or item.get('end_time')
        total_duration = item.get('duration') or item.get('total_duration')
        
        if start_time is not None and end_time is not None and total_duration is not None:
            # Map timestamp to segment indices
            start_idx = int((start_time / total_duration) * num_segments)
            end_idx = int((end_time / total_duration) * num_segments)
            
            start_idx = max(0, min(start_idx, num_segments - 1))
            end_idx = max(0, min(end_idx, num_segments))
            if start_idx == end_idx:
                end_idx = min(start_idx + 1, num_segments)
                
            features = features[start_idx:end_idx]
        elif 'anomaly_frames' in item:
            # Placeholder if annotations provide specific anomaly frame numbers
            pass 
        else:
            # If no timestamps exist, pool or take the features corresponding to the anomaly.
            # Here we simply keep all 32 segments. Downstream collate_fn or model can pool.
            pass
            
        # Process caption
        caption = item.get('caption', item.get('text', ''))
        tokens = self.tokenizer.encode(caption)
        
        if self.max_length:
            tokens = tokens[:self.max_length]
            
        return {
            'video_id': video_id,
            'features': torch.FloatTensor(features),
            'caption': torch.LongTensor(tokens)
        }

def collate_fn(batch):
    """
    Collate function to handle variable length captions and visual features.
    Pads text sequences to the max length in the batch.
    Pads or interpolates visual features to a fixed temporal length.
    """
    # Sort batch by caption length in descending order (useful for RNNs if needed)
    batch.sort(key=lambda x: len(x['caption']), reverse=True)
    
    video_ids = [item['video_id'] for item in batch]
    features_list = [item['features'] for item in batch]
    captions_list = [item['caption'] for item in batch]
    
    # --- Pad Captions ---
    lengths = [len(cap) for cap in captions_list]
    max_cap_len = max(lengths)
    
    # We pad with 0 assuming <PAD> token id is 0
    padded_captions = torch.zeros(len(batch), max_cap_len).long()
    for i, cap in enumerate(captions_list):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]
        
    # --- Handle Visual Features ---
    # We interpolate features to a fixed size of 32 segments if they differ
    FIXED_NUM_SEGMENTS = 32
    processed_features = []
    
    for feat in features_list:
        if feat.shape[0] != FIXED_NUM_SEGMENTS and feat.shape[0] > 0:
            # Shape is (T, 1024). We need to interpolate it to (32, 1024)
            # PyTorch interpolate expects (batch, channels, seq_len)
            feat = feat.unsqueeze(0).transpose(1, 2) # (1, 1024, T)
            feat = F.interpolate(feat, size=FIXED_NUM_SEGMENTS, mode='linear', align_corners=False)
            feat = feat.transpose(1, 2).squeeze(0) # (32, 1024)
        elif feat.shape[0] == 0:
            feat = torch.zeros(FIXED_NUM_SEGMENTS, feat.shape[-1])
        processed_features.append(feat)
        
    padded_features = torch.stack(processed_features, dim=0) # (B, 32, 1024)
        
    return {
        'video_ids': video_ids,
        'features': padded_features, # Shape: (Batch, 32, 1024)
        'captions': padded_captions, # Shape: (Batch, max_cap_len)
        'caption_lengths': torch.LongTensor(lengths)
    }
