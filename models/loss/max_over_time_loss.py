import torch
import torch.nn as nn


class MaxOverTimeLoss(nn.Module):
    """
    Max-over-time loss function as described in the paper
    """

    def __init__(self):
        super(MaxOverTimeLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, readout_potentials, targets):
        """
        readout_potentials: (batch_size, time_steps, output_size)
        targets: (batch_size,)
        """
        max_potentials, _ = torch.max(readout_potentials, dim=1)

        return self.criterion(max_potentials, targets)
