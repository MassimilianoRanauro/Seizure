import torch
from torch.utils.data import Dataset, Sampler

from collections import defaultdict

import data.utils as utils
import numpy as np
import os

class SeizureDataset(Dataset):
    def __init__(self, input_dir:str=None, files_record:list[str]=None, time_step_size:int=1, max_seq_len:int=12, use_fft:bool=True, preprocess_data:str=None, method:str="plv", top_k:int=None):
        """
        Args:
            input_dir (str):            Directory to data files
            files_record (list[str]):   List of simple files with line records like `(file_name, index, bool_value)` if `preprocess_data` is None
                                        or `(file_name, bool_value)` if `preprocess_data` is not None where
                - `file_name` is the name of a *.npy file
                - `index` is a number which is between `0` and `time_duration_file/max_seq_len`
                - `bool_value` is `0` or `1` and corrispond to the absence or presence of a seizure
            time_step_size (int):       Duration of each time step in seconds for FFT analysis. Used only if `use_fft` is True
            max_seq_len (int):          Total duration of the output EEG clip in seconds
            use_fft (bool):             Use the Fast Fourier Transform when obtain the slice from the file
            
            preprocess_data (str):      Directory to the preprocess data. If it is not None `input_dir`, `time_step_size`, `max_seq_len`, `use_fft` will not be considered
            
            method (str):               How to compute the adjacency matrix
                - `cross`: for the scaled Laplacian matrix of the normalized cross-correlation
                - `plv`: for the Phase Locking Value
            top_k (int):                Maintain only the `top_k` higher value when compute the adjacency matrix
        """
        self.input_dir = input_dir
        self.preprocess_data = preprocess_data
        
        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.use_fft = use_fft

        possibilities= ["cross", "plv"]
        if method not in possibilities:
            raise ValueError("Current method '{}' do not exists. Choose between: '{}'".format(method, "', '".join(possibilities)))
        self.method= method
        self.top_k = top_k
        
        self.file_info= list()
        for file in files_record:
            datas= self._read_preprocess_data_data(file) if (preprocess_data is not None) else self._read_input_dir_data(file)
            self.file_info.extend(datas)
        
        if (preprocess_data is not None):
            self._target= [target for _,target in self.file_info]
        else:
            self._target= [target for _,_,target in self.file_info]

    def __len__(self):
        return len(self.file_info)
    
    def target(self) -> list[int]:
        """Target of the data"""
        return self._target
    
    def _read_input_dir_data(self, file:str):
        """Given a simple file returns the file informations accoring to the constructor when `preprocess_data` is None"""
        data= list()
        
        file_name:str=None
        index:str=None
        has_seizure:str=None
        
        with open(file, "r") as f:
            for line in f.readlines():
                file_name, index, has_seizure = line.split(",")
                data.append((file_name.strip(), int(index.strip()), bool(int(has_seizure.strip()))))
        
        return data
    
    def _read_preprocess_data_data(self, file:str):
        """Given a simple file returns the file informations accoring to the constructor when `preprocess_data` is not None"""
        data= list()
        
        file_name:str=None
        has_seizure:str=None
        
        with open(file, "r") as f:
            for line in f.readlines():
                file_name, has_seizure = line.split(",")
                data.append((file_name.strip(), bool(int(has_seizure.strip()))))
        
        return data

    def __getitem__(self, index:int):
        """
        Args:
            index (int):    Index in [0, 1, ..., size_of_dataset-1]
            
        Returns:
            tuple (Tensor, Tensor, Tensor):     The triplets is:
                - Feature/node matrix with shape (max_seq_len, num_channels, feature_dim)
                - Target of the current graph
                - Adjacency matrix with shape (num_channels, num_channels)
        
        Notes:
        ------
            The number of channels depend on the file. If `use_fft` is False the `feature_dim` is equal to `frequency`
            otherwise it is equal to `frequency/2`
        """
        # compute EEG clip
        if (self.preprocess_data is not None):
            npy_file_name, has_seizure = self.file_info[index]    
            eeg_clip = np.load(npy_file_name)
        else:
            npy_file_name, clip_idx, has_seizure = self.file_info[index]
            resample_sig_dir = os.path.join(self.input_dir, npy_file_name)
            eeg_clip = utils.compute_slice_matrix(
                file_name       =   resample_sig_dir,
                clip_idx        =   clip_idx,
                time_step_size  =   self.time_step_size,
                clip_len        =   self.max_seq_len,
                use_fft         =   self.use_fft
            )
        
        curr_feature = eeg_clip.copy()

        # generate adjacency matrix
        if self.method=="cross":
            adj_cross_corr = utils.cross_correlation(curr_feature, self.top_k)
            adj = utils.normalize_laplacian_spectrum(adj_cross_corr)
        elif self.method=="plv":
            # from (clip_len, num_channels, feature_dim) to (num_channels, clip_len*feature_dim)
            curr_feature= curr_feature.transpose((1,0,2)).reshape(curr_feature.shape[1], -1)
            adj= utils.compute_plv_matrix(curr_feature)
        else:
            raise NotImplementedError(f"Current method '{self.method}' not implemented yet")
        
        # transform in tensor all numpy arrays
        x = torch.FloatTensor(eeg_clip)
        y = torch.FloatTensor([has_seizure])
        adj= torch.FloatTensor(adj)

        return (x, y, adj)


class SeizureSampler(Sampler):
    def __init__(self, targets:list, batch_size:int, n_per_class:int, seed:int=None):
        """
        Custom Sampler ensuring each batch contains at least N samples of each class.
        
        Args:
            targets (list):     Target list of the dataset
            batch_size (int):   The batch size for each iteration.
            n_per_class (int):  Minimum number of samples per class in each batch.
            seed (int):         Random seed for reproducibility. If None the seed will be not custom initialize
        
        Raises:
            ValueError: If any class has fewer than n_per_class samples.
            ValueError: If batch_size < n_per_class * num_classes.
        """
        if seed:
            np.random.seed(seed)
        
        self.batch_size = batch_size
        self.n_per_class = n_per_class
        
        # Get targets/labels from dataset
        self.targets = np.array(targets)
        
        # Get unique classes and organize indices by class
        self.classes = np.unique(self.targets)
        self.num_classes = len(self.classes)
        self.class_to_indices = defaultdict(list)
        
        for idx,label in enumerate(self.targets):
            self.class_to_indices[label].append(idx)
        
        # Convert to numpy arrays for efficient sampling
        for label in self.class_to_indices:
            self.class_to_indices[label] = np.array(self.class_to_indices[label])
        
        # Validate constraints
        for label in self.classes:
            n_samples = len(self.class_to_indices[label])
            if n_samples < n_per_class:
                raise ValueError(f"Class {label} has only {n_samples} samples, but n_per_class={n_per_class} is required!")
        
        # Check if batch size is sufficient
        min_batch_size = n_per_class * self.num_classes
        if batch_size < min_batch_size:
            raise ValueError(f"Batch size ({batch_size}) must be at least ({min_batch_size}) to fit ({n_per_class}) samples of each of the ({self.num_classes}) classes")
        
        self.dataset_size = len(targets)
        self.num_batches = self.dataset_size // self.batch_size
        
    def __iter__(self):
        """Iterator to yield indices for each batch while ensuring each batch has at least n_per_class samples from each class."""
        # Create shuffled indices for each class
        class_indices_shuffled = {}
        class_positions = {}
        
        for label in self.classes:
            indices = self.class_to_indices[label].copy()
            np.random.shuffle(indices)
            class_indices_shuffled[label] = indices
            class_positions[label] = 0
        
        # Track which indices have been used globally
        all_indices = np.arange(self.dataset_size)
        np.random.shuffle(all_indices)
        global_position = 0
        
        all_batches = []
        
        for _ in range(self.num_batches):
            batch = []
            used_in_batch = set()
            
            # First, ensure n_per_class samples from each class
            for label in self.classes:
                class_idx = class_indices_shuffled[label]
                pos = class_positions[label]
                
                # Use modulo to wrap around and reuse samples if needed
                for _ in range(self.n_per_class):
                    idx = class_idx[pos % len(class_idx)]
                    batch.append(idx)
                    used_in_batch.add(idx)
                    pos += 1
                
                class_positions[label] = pos
            
            # Fill remaining spots in batch with any samples
            remaining_slots = self.batch_size - len(batch)
            
            if remaining_slots > 0:
                # Get candidates from unused global indices
                end_pos = min(global_position + remaining_slots + self.batch_size, self.dataset_size)
                candidates = all_indices[global_position:end_pos]
                
                # Filter out indices already in batch
                available = np.setdiff1d(candidates, list(used_in_batch), assume_unique=True)
                
                # Take what we need
                n_to_take = min(remaining_slots, len(available))
                if n_to_take > 0:
                    batch.extend(available[:n_to_take].tolist())
                    global_position += len(candidates)
                
                # If still not enough, sample randomly from entire dataset
                remaining_slots = remaining_slots - n_to_take
                if remaining_slots > 0:
                    all_available = np.setdiff1d(all_indices, list(batch))
                    if len(all_available) > 0:
                        extra = np.random.choice(all_available, size=min(remaining_slots, len(all_available)), replace=False)
                        batch.extend(extra.tolist())
            
            all_batches.append(batch)
        
        # Flatten and yield indices
        for batch in all_batches:
            for idx in batch:
                yield idx
    
    def __len__(self):
        """Return the total number of samples that will be yielded."""
        return self.num_batches * self.batch_size
