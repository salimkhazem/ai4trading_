import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, rnn_type: str, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        """
        Args:
            input_dim: Number of features in the input embeddings.
            hidden_dim: Number of hidden units in each LSTM layer.
            num_layers: Number of LSTM layers.
            output_dim: Number of output dimensions (e.g., 1 for regression).
            dropout: Dropout rate for regularization.
        """
        super(RNN, self).__init__()
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'")    
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through the RNN model.
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).
        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        # Pass through RNN
        rnn_out, _ = self.rnn(x)  # Output shape: (batch_size, sequence_length, hidden_dim)
        rnn_out = rnn_out[:, -1, :]  # Take the last time step's output
        
        # Pass through fully connected layer
        output = self.fc(rnn_out)  # Output shape: (batch_size, output_dim)
        return output
