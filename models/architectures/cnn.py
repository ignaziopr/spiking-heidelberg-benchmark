import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    """
    CNN Classifier based on paper specifications
    """

    def __init__(self, input_channels=64, output_size=10, dropout=0.2):
        super(CNNClassifier, self).__init__()

        self.input_channels = input_channels
        self.output_size = output_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(11, 11), padding=5)

        self.block1 = self._make_conv_block(32, 32)
        self.block2 = self._make_conv_block(32, 32)
        self.block3 = self._make_conv_block(32, 32)

        self.fc1 = nn.Linear(32 * 12 * 8, 128)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_size)

    def _make_conv_block(self, in_channels, out_channels):
        """
        Create a conv block: 2 conv layers + batch norm + ReLU + max pool + dropout
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Dropout2d(0.2)
        )

    def forward(self, x):
        """
        Forward pass
        x: (batch_size, time_steps, channels) - treat as 2D image (time x channels)
        """
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x
