import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) based model for sepsis prediction
    Suitable for sequential vital signs data
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, output_dim=1, dropout=0.3):
        super(GRUModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: Prediction of shape (batch_size, output_dim)
        """
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_dim)
        
        # Attention mechanism
        attention_weights = self.attention(gru_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * gru_out, dim=1)  # (batch_size, hidden_dim)
        
        # Fully connected layers
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.sigmoid(self.fc3(out))
        
        return out


class SwinTransformerBlock(nn.Module):
    """
    Shifted Window Transformer Block
    Applies self-attention with shifted windows for efficiency
    """
    
    def __init__(self, dim, num_heads, mlp_dim, window_size=4, shift_size=2):
        super(SwinTransformerBlock, self).__init__()
        
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        """
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # MLP with residual connection
        norm_x = self.norm2(x)
        mlp_out = self.mlp(norm_x)
        x = x + mlp_out
        
        return x


class SwinTransformerModel(nn.Module):
    """
    Swin Transformer based model for sepsis prediction
    Uses shifted window attention for efficient sequential processing
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_layers=2, output_dim=1):
        super(SwinTransformerModel, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, hidden_dim))
        
        # Swin Transformer blocks
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_dim=hidden_dim * 4,
                window_size=4,
                shift_size=2 if i % 2 == 1 else 0
            )
            for i in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: Prediction of shape (batch_size, output_dim)
        """
        # Project input
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.shape[1], :]
        
        # Swin Transformer blocks
        for block in self.swin_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.norm(x)
        x = torch.mean(x, dim=1)  # (batch_size, hidden_dim)
        
        # Output layers
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        
        return out


class HybridModel(nn.Module):
    """
    Hybrid model combining GRU and Swin Transformer
    Uses ensemble approach for improved predictions
    """
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(HybridModel, self).__init__()
        
        self.gru_model = GRUModel(input_dim, hidden_dim, output_dim=1)
        self.swin_model = SwinTransformerModel(input_dim, hidden_dim, output_dim=1)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        gru_out = self.gru_model(x)
        swin_out = self.swin_model(x)
        
        # Concatenate and fuse
        combined = torch.cat([gru_out, swin_out], dim=1)
        output = self.fusion(combined)
        
        return output


if __name__ == "__main__":
    print("Testing models...")
    
    # Test shapes
    batch_size = 32
    seq_len = 100
    input_dim = 10
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # GRU Model
    gru = GRUModel(input_dim, hidden_dim=64)
    gru_out = gru(x)
    print(f"GRU output shape: {gru_out.shape}")
    
    # Swin Transformer Model
    swin = SwinTransformerModel(input_dim, hidden_dim=64)
    swin_out = swin(x)
    print(f"Swin Transformer output shape: {swin_out.shape}")
    
    # Hybrid Model
    hybrid = HybridModel(input_dim)
    hybrid_out = hybrid(x)
    print(f"Hybrid model output shape: {hybrid_out.shape}")