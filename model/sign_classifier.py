"""
Sign Language Classifier Models

Combines RAFT optical flow features with MediaPipe landmarks using LSTM or Transformer.
Optimized for M4 MacBook with MPS acceleration.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from einops import rearrange


class LSTMSignClassifier(nn.Module):
    """LSTM-based sign language classifier combining flow and landmark features."""
    
    def __init__(
        self,
        num_classes: int = 15,
        landmark_dim: int = 258,
        flow_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM classifier.
        
        Args:
            num_classes: Number of sign language classes
            landmark_dim: Dimension of landmark features (258 for MediaPipe)
            flow_dim: Dimension of optical flow features
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.landmark_dim = landmark_dim
        self.flow_dim = flow_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        
        # Landmark encoder (LSTM for temporal modeling)
        self.landmark_lstm = nn.LSTM(
            input_size=landmark_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Flow encoder (LSTM for temporal modeling)
        self.flow_lstm = nn.LSTM(
            input_size=flow_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate final feature dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        flow_output_dim = (hidden_dim // 2) * (2 if bidirectional else 1)
        combined_dim = lstm_output_dim + flow_output_dim
        
        # Fusion and classification layers
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, landmarks: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            landmarks: (batch, num_frames, 258) landmark features
            flow: (batch, num_frames-1, flow_dim) flow features
            
        Returns:
            logits: (batch, num_classes)
        """
        # Encode landmarks
        landmark_out, (landmark_h, _) = self.landmark_lstm(landmarks)
        
        # Take the last hidden state (or mean of all states)
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            landmark_features = torch.cat([landmark_h[-2], landmark_h[-1]], dim=1)
        else:
            landmark_features = landmark_h[-1]
        
        # Encode flow
        flow_out, (flow_h, _) = self.flow_lstm(flow)
        
        if self.bidirectional:
            flow_features = torch.cat([flow_h[-2], flow_h[-1]], dim=1)
        else:
            flow_features = flow_h[-1]
        
        # Combine features
        combined = torch.cat([landmark_features, flow_features], dim=1)
        
        # Fusion and classification
        fused = self.fusion(combined)
        logits = self.classifier(fused)
        
        return logits


class TransformerSignClassifier(nn.Module):
    """Transformer-based sign language classifier."""
    
    def __init__(
        self,
        num_classes: int = 15,
        landmark_dim: int = 258,
        flow_dim: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer classifier.
        
        Args:
            num_classes: Number of sign language classes
            landmark_dim: Dimension of landmark features
            flow_dim: Dimension of optical flow features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Input projections
        self.landmark_projection = nn.Linear(landmark_dim, d_model)
        self.flow_projection = nn.Linear(flow_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, landmarks: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            landmarks: (batch, num_frames, 258) landmark features
            flow: (batch, num_frames-1, flow_dim) flow features
            
        Returns:
            logits: (batch, num_classes)
        """
        # Project inputs to d_model
        landmark_embedded = self.landmark_projection(landmarks)
        flow_embedded = self.flow_projection(flow)
        
        # Pad flow to match landmark sequence length
        # Add a zero frame at the end to align with landmarks
        batch_size = flow_embedded.shape[0]
        zero_pad = torch.zeros(batch_size, 1, self.d_model, device=flow.device)
        flow_embedded = torch.cat([flow_embedded, zero_pad], dim=1)
        
        # Combine landmark and flow features
        combined = landmark_embedded + flow_embedded  # Element-wise addition
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Transformer encoding
        encoded = self.transformer_encoder(combined)
        
        # Global average pooling over time
        pooled = encoded.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridSignClassifier(nn.Module):
    """Hybrid model using both LSTM and attention mechanisms."""
    
    def __init__(
        self,
        num_classes: int = 15,
        landmark_dim: int = 258,
        flow_dim: int = 32,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            num_classes: Number of sign language classes
            landmark_dim: Dimension of landmark features
            flow_dim: Dimension of optical flow features
            hidden_dim: Hidden dimension
            num_lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        # Feature encoders
        self.landmark_encoder = nn.LSTM(
            landmark_dim, hidden_dim, num_lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        self.flow_encoder = nn.LSTM(
            flow_dim, hidden_dim // 2, num_lstm_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2 + hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, landmarks: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            landmarks: (batch, num_frames, 258)
            flow: (batch, num_frames-1, flow_dim)
            
        Returns:
            logits: (batch, num_classes)
        """
        # Encode features
        landmark_out, _ = self.landmark_encoder(landmarks)
        flow_out, _ = self.flow_encoder(flow)
        
        # Pad flow to match landmark length
        batch_size = flow_out.shape[0]
        zero_pad = torch.zeros(batch_size, 1, flow_out.shape[2], device=flow.device)
        flow_out = torch.cat([flow_out, zero_pad], dim=1)
        
        # Concatenate features
        combined = torch.cat([landmark_out, flow_out], dim=-1)
        
        # Apply self-attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Global average pooling
        pooled = attended.mean(dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


def create_model(
    model_type: str = "lstm",
    num_classes: int = 15,
    device: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: "lstm", "transformer", or "hybrid"
        num_classes: Number of classes
        device: Device to place model on
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
    """
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    device = torch.device(device)
    
    # Create model
    if model_type == "lstm":
        model = LSTMSignClassifier(num_classes=num_classes, **kwargs)
    elif model_type == "transformer":
        model = TransformerSignClassifier(num_classes=num_classes, **kwargs)
    elif model_type == "hybrid":
        model = HybridSignClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    print(f"Created {model_type} model on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Sign Language Classifier Models Demo")
    print("=" * 60)
    
    # Test models with dummy data
    batch_size = 4
    num_frames = 30
    landmark_dim = 258
    flow_dim = 32
    num_classes = 15
    
    landmarks = torch.randn(batch_size, num_frames, landmark_dim)
    flow = torch.randn(batch_size, num_frames - 1, flow_dim)
    
    print("\nTesting LSTM model...")
    lstm_model = create_model("lstm", num_classes=num_classes)
    lstm_out = lstm_model(landmarks, flow)
    print(f"Output shape: {lstm_out.shape}")
    
    print("\nTesting Transformer model...")
    transformer_model = create_model("transformer", num_classes=num_classes)
    transformer_out = transformer_model(landmarks, flow)
    print(f"Output shape: {transformer_out.shape}")
    
    print("\nTesting Hybrid model...")
    hybrid_model = create_model("hybrid", num_classes=num_classes)
    hybrid_out = hybrid_model(landmarks, flow)
    print(f"Output shape: {hybrid_out.shape}")
    
    print("\n" + "=" * 60)
    print("All models tested successfully!")
    print("=" * 60)
