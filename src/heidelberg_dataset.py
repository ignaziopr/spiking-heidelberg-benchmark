import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class HeidelbergDatasetCached(Dataset):
    """Fully cached version - loads all data into memory at init"""

    def __init__(self, hdf5_file, partition='train', dt=1e-3, T_max=1.0, model_type='snn'):
        self.partition = partition
        self.dt = dt
        self.T_max = T_max
        self.model_type = model_type

        print(f"Loading {partition} dataset into memory...")

        with h5py.File(hdf5_file, 'r') as f:
            self.spike_times_list = []
            self.spike_units_list = []

            spikes_group = f['spikes']
            times_vlarray = spikes_group['times']
            units_vlarray = spikes_group['units']

            for i in range(len(times_vlarray)):
                self.spike_times_list.append(times_vlarray[i])
                self.spike_units_list.append(units_vlarray[i])

            self.labels = f['labels'][:]

        self.n_channels = 700
        if model_type == 'snn':
            self.n_timesteps = int(self.T_max / self.dt)
        else:
            self.bin_size = 10e-3
            self.n_timesteps = int(self.T_max / self.bin_size)

        if model_type == 'cnn':
            self.n_channels_binned = 64

        print(f"✓ Loaded {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spike_times = self.spike_times_list[idx]
        spike_units = self.spike_units_list[idx]

        if self.model_type == 'snn':
            data = self._convert_to_dense_single(spike_times, spike_units)
        elif self.model_type == 'lstm':
            data = self._convert_to_lstm_single(spike_times, spike_units)
        elif self.model_type == 'cnn':
            data = self._convert_to_cnn_single(spike_times, spike_units)

        return torch.FloatTensor(data), torch.LongTensor([self.labels[idx]])[0]

    def _convert_to_dense_single(self, spike_times, spike_units):
        spike_train = np.zeros(
            (self.n_timesteps, self.n_channels), dtype=np.float32)
        if len(spike_times) > 0:
            time_bins = (spike_times / self.dt).astype(int)
            time_bins = np.clip(time_bins, 0, self.n_timesteps - 1)
            valid_units = np.clip(spike_units, 0, self.n_channels - 1)
            spike_train[time_bins, valid_units] = 1.0
        return spike_train

    def _convert_to_lstm_single(self, spike_times, spike_units):
        binned_spikes = np.zeros(
            (self.n_timesteps, self.n_channels), dtype=np.float32)
        if len(spike_times) > 0:
            time_bins = (spike_times / self.bin_size).astype(int)
            time_bins = np.clip(time_bins, 0, self.n_timesteps - 1)
            valid_units = np.clip(spike_units, 0, self.n_channels - 1)
            for t_bin, unit in zip(time_bins, valid_units):
                binned_spikes[t_bin, unit] += 1.0
        return binned_spikes

    def _convert_to_cnn_single(self, spike_times, spike_units):
        temp_binned = np.zeros(
            (self.n_timesteps, self.n_channels), dtype=np.float32)
        if len(spike_times) > 0:
            time_bins = (spike_times / self.bin_size).astype(int)
            time_bins = np.clip(time_bins, 0, self.n_timesteps - 1)
            valid_units = np.clip(spike_units, 0, self.n_channels - 1)
            for t_bin, unit in zip(time_bins, valid_units):
                temp_binned[t_bin, unit] += 1.0

        spatial_binned = np.zeros(
            (self.n_timesteps, self.n_channels_binned), dtype=np.float32)
        channels_per_bin = self.n_channels // self.n_channels_binned
        for i in range(self.n_channels_binned):
            start_ch = i * channels_per_bin
            end_ch = min((i + 1) * channels_per_bin, self.n_channels)
            spatial_binned[:, i] = np.sum(
                temp_binned[:, start_ch:end_ch], axis=1)

        return spatial_binned
