import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.architectures.lif_neuron import LIFNeuron
from models.architectures.surrogate_gradient import SurrogateGradient


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
        self.surrogate = SurrogateGradient.apply

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization (Section II-D3)
        k = 1.0 / self.input_size
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)

        if self.recurrent:
            k_rec = 1.0 / self.output_size
            bound_rec = math.sqrt(k_rec)
            nn.init.uniform_(self.recurrent_weight, -bound_rec, bound_rec)

    def forward(self, spike_input, membrane_potential, synaptic_current, spike_state=None):

        input_current = F.linear(spike_input, self.weight)

        if self.recurrent and spike_state is not None:
            input_current += F.linear(spike_state, self.recurrent_weight)

        new_membrane_potential, new_synaptic_current, raw_spikes = self.lif(
            input_current, membrane_potential, synaptic_current
        )

        spikes = self.surrogate(new_membrane_potential - self.lif.u_thresh)

        return new_membrane_potential, new_synaptic_current, spikes
