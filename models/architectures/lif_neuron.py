import torch.nn as nn
import matplotlib.pyplot as plt
import math
from models.architectures.surrogate_gradient import SurrogateGradient


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Neuron implementation
    Based on the paper's specifications in Section II-D2
    """

    def __init__(self, tau_mem=10e-3, tau_syn=5e-3, u_thresh=1.0, u_leak=0.0, dt=1e-3):
        super(LIFNeuron, self).__init__()
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.u_thresh = u_thresh
        self.u_leak = u_leak
        self.dt = dt

        self.lambda_mem = math.exp(-dt / tau_mem)
        self.kappa = math.exp(-dt / tau_syn)

        self.surrogate = SurrogateGradient.apply

    def forward(self, input_current, membrane_potential, synaptic_current):
        """
        Forward pass of LIF neuron
        Returns: new_membrane_potential, new_synaptic_current, spikes
        """
        new_synaptic_current = self.kappa * synaptic_current + input_current

        mthr = membrane_potential - self.u_thresh

        spikes = self.surrogate(mthr)

        rst = spikes.detach()

        new_membrane_potential = (
            self.lambda_mem * membrane_potential + new_synaptic_current) * (1.0 - rst)

        return new_membrane_potential, new_synaptic_current, spikes
