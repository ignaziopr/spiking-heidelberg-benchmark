import torch
import torch.nn as nn


class LSTMCNNLoss(nn.Module):
    """
    Loss function for LSTM and CNN (standard cross entropy)
    Can handle both last-time-step and max-over-time for LSTM
    """

    def __init__(self, loss_type='last_time_step'):
        super(LSTMCNNLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.loss_type = loss_type  # either 'last_time_step' or 'max_over_time'

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):  # LSTM case
            if self.loss_type == 'last_time_step':
                return self.criterion(outputs['last_time_step'], targets)
            else:  # max_over_time
                max_outputs, _ = torch.max(outputs['all_time_steps'], dim=1)
                return self.criterion(max_outputs, targets)
        else:  # CNN case
            return self.criterion(outputs, targets)
