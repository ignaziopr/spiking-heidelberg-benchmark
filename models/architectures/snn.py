import torch
import torch.nn as nn
import math
from models.architectures.spiking_layer import SpikingLayer


class SpikingNeuralNetwork(nn.Module):
    """
    Multi-layer spiking neural network
    """

    def __init__(self, input_size=700, hidden_size=128, output_size=10,
                 num_layers=1, recurrent=False, dt=1e-3, T=100,
                 reg_strength=1e-7):
        super(SpikingNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.recurrent = recurrent
        self.dt = dt
        self.T = T
        self.reg_strength = reg_strength

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

        tau_syn = 5e-3
        tau_mem = 10e-3
        effective_dt = 1e-3
        self.alpha = math.exp(-effective_dt / tau_syn)
        self.beta = math.exp(-effective_dt / tau_mem)

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

            layer.reset_spike_counts()

            membrane_potentials.append(torch.zeros(
                batch_size, layer.output_size, device=spike_trains.device))
            synaptic_currents.append(torch.zeros(
                batch_size, layer.output_size, device=spike_trains.device))
            spike_states.append(torch.zeros(
                batch_size, layer.output_size, device=spike_trains.device))

        all_spikes = []

        # Forward pass through time
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

            all_spikes.append(spike_states[-1].clone())

        spk_rec = torch.stack(all_spikes, dim=1)  # (batch, time, hidden)

        h2 = torch.einsum("abc,cd->abd", spk_rec, self.readout.weight.T)

        flt = torch.zeros(batch_size, self.output_size,
                          device=spike_trains.device)
        out = torch.zeros(batch_size, self.output_size,
                          device=spike_trains.device)
        out_rec = [out.clone()]

        for t in range(time_steps):
            new_flt = self.alpha * flt + h2[:, t]
            new_out = self.beta * out + flt

            flt = new_flt
            out = new_out
            out_rec.append(out.clone())

        readout_potentials = torch.stack(out_rec, dim=1) + self.readout.bias

        reg_loss = self.compute_spike_regularization()

        return readout_potentials, reg_loss

    def compute_spike_regularization(self):
        """
        Paper's exact regularization implementation
        """
        total_reg_loss = 0.0

        for layer in self.layers:
            if len(layer.spike_counts) == 0:
                continue

            spks = torch.stack(layer.spike_counts, dim=0)

            l1_loss = self.reg_strength * torch.mean(spks)
            l2_loss = self.reg_strength * \
                torch.mean(torch.sum(torch.sum(spks, dim=0), dim=0)**2)

            total_reg_loss += l1_loss + l2_loss

        return total_reg_loss
