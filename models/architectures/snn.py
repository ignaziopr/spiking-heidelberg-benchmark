import torch
import torch.nn as nn
import math
from models.architectures.spiking_layer import SpikingLayer


class SpikingNeuralNetwork(nn.Module):
    """
    Multi-layer spiking neural network
    """

    def __init__(self, input_size=700, hidden_size=128, output_size=10,
                 num_layers=1, recurrent=False, dt=1e-3, T=100):
        super(SpikingNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.recurrent = recurrent
        self.dt = dt
        self.T = T

        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(SpikingLayer(
                input_size, hidden_size, dt, recurrent))
        else:
            self.layers.append(SpikingLayer(
                input_size, hidden_size, dt, False))
            for _ in range(num_layers - 2):
                self.layers.append(SpikingLayer(
                    hidden_size, hidden_size, dt, False))
            self.layers.append(SpikingLayer(
                hidden_size, hidden_size, dt, recurrent))

        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, spike_trains):
        """
        Forward pass through the spiking network
        spike_trains: (batch_size, time_steps, input_size)
        """
        batch_size, time_steps, _ = spike_trains.shape

        membrane_potentials = []
        synaptic_currents = []
        spike_states = []

        for i, layer in enumerate(self.layers):
            membrane_potentials.append(torch.zeros(
                batch_size, layer.output_size, device=spike_trains.device))
            synaptic_currents.append(torch.zeros(
                batch_size, layer.output_size, device=spike_trains.device))
            spike_states.append(torch.zeros(
                batch_size, layer.output_size, device=spike_trains.device))

        readout_potential = torch.zeros(
            batch_size, self.output_size, device=spike_trains.device)
        tau_readout = 20e-3
        lambda_readout = math.exp(-self.dt / tau_readout)

        # Store outputs for max-over-time loss
        readout_potentials = []

        for t in range(time_steps):
            current_input = spike_trains[:, t, :]

            for i, layer in enumerate(self.layers):
                if i == 0:
                    layer_input = current_input
                else:
                    layer_input = spike_states[i-1]

                if layer.recurrent:
                    recurrent_input = spike_states[i]
                else:
                    recurrent_input = None

                membrane_potentials[i], synaptic_currents[i], spike_states[i] = layer(
                    layer_input, membrane_potentials[i], synaptic_currents[i], recurrent_input
                )

            readout_input = self.readout(spike_states[-1])
            readout_potential = lambda_readout * readout_potential + \
                (1 - lambda_readout) * readout_input
            readout_potentials.append(readout_potential.clone())

        # Stack readout potentials and return for max-over-time loss
        # (batch_size, time_steps, output_size)
        readout_potentials = torch.stack(readout_potentials, dim=1)

        return readout_potentials
