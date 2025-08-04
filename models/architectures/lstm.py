import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    LSTM Classifier based on paper specifications
    """

    def __init__(self, input_size=700, hidden_size=128, output_size=10, dropout=0.2):
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass
        x: (batch_size, time_steps, input_size)
        """

        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        last_output = lstm_out[:, -1, :]  # Last time step

        batch_size, time_steps, hidden_size = lstm_out.shape
        lstm_out_reshaped = lstm_out.reshape(-1, hidden_size)
        all_outputs = self.fc(lstm_out_reshaped)
        # (batch_size, time_steps, output_size)
        all_outputs = all_outputs.reshape(
            batch_size, time_steps, self.output_size)

        last_output = self.fc(last_output)  # (batch_size, output_size)

        return {
            'last_time_step': last_output,
            'all_time_steps': all_outputs
        }
