import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, kernel_size=3, dropout=0.0):
        super(TCN, self).__init__()
        # List of convolutional layers
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            out_channels = hidden_dim if i < num_layers - 1 else output_dim
            dilation = 2 ** i  # Exponentially increasing dilation factor
            
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=dilation, dilation=dilation)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        # Stack all layers
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        return x.mean(dim=2)
