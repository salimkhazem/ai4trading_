import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Adds positional information to input embeddings.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, dim_feedforward: int, num_layers: int, num_classes: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of the input features.
            d_model: The embedding dimension (must be divisible by nhead).
            nhead: Number of attention heads.
            dim_feedforward: Dimension of the feed-forward network model.
            num_layers: Number of Transformer encoder layers.
            num_classes: Number of output classes.
            dropout: Dropout rate.
        """
        super().__init__()
        self.d_model = d_model # Store d_model for potential use
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Create the encoder layer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Process (batch, seq, feature) inputs
        )
        
        # Create the full encoder stack also with batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # Final classifier layer - input is d_model after pooling
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Forward pass through the Transformer model.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # Input shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        # Shape: (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)
        # Shape: (batch_size, seq_len, d_model)

        # Pass through Transformer encoder (expects batch_first)
        x = self.transformer_encoder(x)
        # Output shape: (batch_size, seq_len, d_model)

        # --- Aggregation --- #
        # Mean pooling over the sequence dimension
        x = x.mean(dim=1)
        # Shape: (batch_size, d_model)

        # Pass through final classifier layer
        output = self.fc(x)
        # Shape: (batch_size, num_classes)

        return output



