import torch
import numpy as np
import h5py
from torch.utils.data import Dataset


class HeidelbergDataset(Dataset):
    """
    Dataset class for Heidelberg Spiking Data
    """

    def __init__(self, hdf5_file, partition='train', dt=1e-3, T_max=1.0, model_type='snn'):
        self.file_path = hdf5_file
        self.partition = partition
        self.dt = dt
        self.T_max = T_max
        self.model_type = model_type  # 'snn', 'lstm', or 'cnn'

        # Load data based on the actual file structure
        with h5py.File(hdf5_file, 'r') as f:
            # The spikes group contains times and units as VLArrays
            spike_times_list = []
            spike_units_list = []

            # Read VLArrays - each element is an array of spike times/units for one sample
            spikes_group = f['spikes']
            times_vlarray = spikes_group['times']
            units_vlarray = spikes_group['units']

            # Convert VLArrays to lists
            for i in range(len(times_vlarray)):
                spike_times_list.append(times_vlarray[i])
                spike_units_list.append(units_vlarray[i])

            # Load labels
            self.labels = f['labels'][:]

        self.spike_times_list = spike_times_list
        self.spike_units_list = spike_units_list

        # Convert to appropriate representation based on model type
        if model_type == 'snn':
            self.data = self._convert_to_dense_representation()
        elif model_type == 'lstm':
            self.data = self._convert_to_lstm_representation()
        elif model_type == 'cnn':
            self.data = self._convert_to_cnn_representation()

    def _convert_to_dense_representation(self):
        """
        Convert sparse spike representation to dense time series for SNNs
        """
        n_samples = len(self.spike_times_list)
        n_channels = 700  # As specified in paper (Nch = 700 BCs)
        n_timesteps = int(self.T_max / self.dt)

        spike_trains = np.zeros((n_samples, n_timesteps, n_channels))

        for i in range(n_samples):
            sample_times = self.spike_times_list[i]
            sample_units = self.spike_units_list[i]

            if len(sample_times) > 0:  # Check if there are spikes in this sample
                # Convert spike times to time bins
                time_bins = (sample_times / self.dt).astype(int)
                time_bins = np.clip(time_bins, 0, n_timesteps - 1)

                # Make sure units are within valid range
                valid_units = np.clip(sample_units, 0, n_channels - 1)

                # Set spikes in the dense representation
                spike_trains[i, time_bins, valid_units] = 1.0

        return spike_trains

    def _convert_to_lstm_representation(self):
        """
        Convert to LSTM representation: binned in time bins of 10ms
        """
        n_samples = len(self.spike_times_list)
        n_channels = 700  # Nch = 700 BCs
        bin_size = 10e-3  # 10ms bins as specified
        n_timesteps = int(self.T_max / bin_size)

        binned_spikes = np.zeros((n_samples, n_timesteps, n_channels))

        for i in range(n_samples):
            sample_times = self.spike_times_list[i]
            sample_units = self.spike_units_list[i]

            if len(sample_times) > 0:
                # Convert spike times to time bins (10ms bins)
                time_bins = (sample_times / bin_size).astype(int)
                time_bins = np.clip(time_bins, 0, n_timesteps - 1)

                # Make sure units are within valid range
                valid_units = np.clip(sample_units, 0, n_channels - 1)

                # Count spikes in each bin
                for t_bin, unit in zip(time_bins, valid_units):
                    binned_spikes[i, t_bin, unit] += 1.0

        return binned_spikes

    def _convert_to_cnn_representation(self):
        """
        Convert to CNN representation: binned in time (10ms) and space (64 units)
        """
        n_samples = len(self.spike_times_list)
        n_channels_original = 700
        n_channels_binned = 64  # Spatial binning to 64 units as specified
        bin_size = 10e-3  # 10ms bins as specified
        n_timesteps = int(self.T_max / bin_size)

        # First create temporal bins
        temp_binned = np.zeros((n_samples, n_timesteps, n_channels_original))

        for i in range(n_samples):
            sample_times = self.spike_times_list[i]
            sample_units = self.spike_units_list[i]

            if len(sample_times) > 0:
                # Convert spike times to time bins (10ms bins)
                time_bins = (sample_times / bin_size).astype(int)
                time_bins = np.clip(time_bins, 0, n_timesteps - 1)

                # Make sure units are within valid range
                valid_units = np.clip(sample_units, 0, n_channels_original - 1)

                # Count spikes in each bin
                for t_bin, unit in zip(time_bins, valid_units):
                    temp_binned[i, t_bin, unit] += 1.0

        # Now spatially bin from 700 to 64 channels
        spatial_binned = np.zeros((n_samples, n_timesteps, n_channels_binned))
        # ~10.9, so we'll group channels
        channels_per_bin = n_channels_original // n_channels_binned

        for i in range(n_channels_binned):
            start_ch = i * channels_per_bin
            end_ch = min((i + 1) * channels_per_bin, n_channels_original)
            spatial_binned[:, :, i] = np.sum(
                temp_binned[:, :, start_ch:end_ch], axis=2)

        return spatial_binned

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])[0]
