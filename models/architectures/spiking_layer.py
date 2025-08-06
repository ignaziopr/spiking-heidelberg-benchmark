import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.architectures.lif_neuron import LIFNeuron


class SpikingLayer(nn.Module):
    """
    A layer of spiking neurons with synaptic connections
    """

    def __init__(self, input_size, output_size, dt=1e-3, recurrent=False):
        super(SpikingLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dt = dt
        self.recurrent = recurrent

        self.weight = nn.Parameter(torch.randn(output_size, input_size))

        if recurrent:
            self.recurrent_weight = nn.Parameter(
                torch.randn(output_size, output_size))

        self.lif = LIFNeuron(dt=dt)

        self.reset_parameters()

        self.spike_counts = []

    def reset_parameters(self):
        weight_scale = 0.2

        nn.init.normal_(self.weight, mean=0.0,
                        std=weight_scale/math.sqrt(self.input_size))

        if self.recurrent:
            nn.init.normal_(self.recurrent_weight, mean=0.0,
                            std=weight_scale/math.sqrt(self.output_size))

    def forward(self, spike_input, membrane_potential, synaptic_current, spike_state=None):

        input_current = F.linear(spike_input, self.weight)

        if self.recurrent and spike_state is not None:
            input_current += F.linear(spike_state, self.recurrent_weight)

        new_membrane_potential, new_synaptic_current, spikes = self.lif(
            input_current, membrane_potential, synaptic_current
        )

        self.spike_counts.append(spikes)

        return new_membrane_potential, new_synaptic_current, spikes

    def reset_spike_counts(self):
        """Call at the start of each forward pass"""
        self.spike_counts = []
