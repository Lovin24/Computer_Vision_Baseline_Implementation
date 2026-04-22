import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Shape: (1, max_len, d_model) for batch_first
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CaptionTransformer(nn.Module):
    def __init__(self, vocab_size, feature_dim=1024, hidden_dim=512, nhead=8, num_layers=4, dropout=0.1, max_seq_length=500):
        """
        Args:
            vocab_size: Size of the vocabulary.
            feature_dim: Dimensionality of the input visual features (default: 1024 for I3D).
            hidden_dim: Dimensionality of the transformer and embeddings (default: 512).
            nhead: Number of heads in the multiheadattention models.
            num_layers: Number of sub-encoder-layers in the encoder.
            dropout: Dropout value.
            max_seq_length: Maximum allowed sequence length for positional encoding.
        """
        super(CaptionTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # 1. Vision Projection Layer
        self.visual_projection = nn.Linear(feature_dim, hidden_dim)
        
        # 2. Text Embedding & Positional Encoding
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len=max_seq_length)
        
        # 3. The Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True # Enables (batch, seq, feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final Output Layer mapping hidden_dim to vocab_size
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with a uniform distribution."""
        initrange = 0.1
        self.text_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.visual_projection.bias.data.zero_()
        self.visual_projection.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz, device='cpu'):
        """
        4. Causal Text Mask
        Generates a causal mask to prevent attention to future tokens.
        Output is upper-triangular matrix of -inf, with zeros on the diagonal.
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, visual_features, captions, tgt_padding_mask=None):
        """
        5. Forward Pass
        
        Args:
            visual_features: Tensor, shape [batch_size, num_segments, feature_dim]
            captions: Tensor, shape [batch_size, seq_len]
            tgt_padding_mask: Tensor, shape [batch_size, seq_len] indicating padded elements (True for pad).
        
        Returns:
            logits: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = captions.size()
        
        # Transform visual features: [batch_size, num_segments, hidden_dim]
        memory = self.visual_projection(visual_features)
        
        # Embed text and scale by sqrt(hidden_dim): [batch_size, seq_len, hidden_dim]
        tgt = self.text_embedding(captions) * math.sqrt(self.hidden_dim)
        
        # Apply positional encoding
        tgt = self.pos_encoder(tgt)
        
        # Generate causal mask for the target sequence
        tgt_mask = self.generate_square_subsequent_mask(seq_len, device=captions.device)
        
        # Pass through Transformer Decoder
        output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project back to vocabulary dimension
        logits = self.fc_out(output)
        
        return logits
