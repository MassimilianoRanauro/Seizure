import numpy as np

from scipy.fftpack import fft
from scipy.sparse import linalg
from scipy.sparse.csgraph import laplacian
from mne_features.bivariate import compute_phase_lock_val

FREQUENCY_CHB_MIT= 256

def normalize_laplacian_spectrum(adj_mx:np.ndarray, lambda_value:float=2) -> np.ndarray:
    """
    Compute scaled Laplacian matrix for graph convolutional networks.\\
    The scaled Laplacian is defined as:\\
    `(2 / lambda_max) * L - I`\\
    where:
    - `L` is the normalized Laplacian
    - `I` is the identity matrix
    
    Args:
        adj_mx (np.ndarray):        Adjacency matrix with shape (num_nodes, num_nodes)
        lambda_value (float):       Maximum eigenvalue for scaling. If None, computed automatically
    
    Returns:
        laplacian (np.ndarray):     Scaled Laplacian matrix with shape (num_nodes, num_nodes)
    """
    is_symmetric= np.allclose(adj_mx, adj_mx.T, rtol=1e-10, atol=1e-12)
    
    L= laplacian(adj_mx, normed=True, symmetrized=(not is_symmetric))
    
    lambda_value= lambda_value if lambda_value else linalg.eigsh(L, 1, which='LM', return_eigenvectors=False)[0]
    lambda_value= max(lambda_value, 1e-8) # for numerical stability

    I = np.eye(L.shape[0])
    L = (2 / lambda_value) * L - I
    
    return L

def keep_topk(adj_mat:np.ndarray, top_k:int, directed:bool=True) -> np.ndarray:
    """
    Helper function to sparsen the adjacency matrix by keeping top-k neighbors for each node.
    
    Args:
        adj_mat (np.ndarray):   Adjacency matrix with size (num_nodes, num_nodes)
        top_k (int):            Number of higher value neighbors for each node to maintain
        directed (bool):        If the graph is direct or undirect
    Returns:
        adj_mat (np.ndarray):   Sparse adjacency matrix with size (num_nodes, num_nodes)
    """
    num_nodes = adj_mat.shape[0]
    
    # Set values that are not of top-k neighbors to 0
    adj_mat_noSelfEdge = adj_mat.copy()
    np.fill_diagonal(adj_mat_noSelfEdge, 0.0)

    # Find top-k indices for each row
    top_k_idx = np.argsort(-adj_mat_noSelfEdge, axis=-1)[:, :top_k]
    
    # Create mask for the top_k indeces
    mask = np.zeros_like(adj_mat, dtype=bool)
    row_indices = np.arange(num_nodes).reshape((num_nodes, 1))
    mask[row_indices, top_k_idx] = True
    np.fill_diagonal(mask, True)
    
    # If the graph is undirect the mask must be symmetric
    mask= mask if directed else np.logical_or(mask, mask.T)
    
    return (mask * adj_mat)

def cross_correlation(eeg_clip:np.ndarray, top_k:int=None) -> np.ndarray:
    """
    Compute adjacency matrix using normalized cross-correlation between EEG channels
    
    Args:
        eeg_clip (np.ndarray):  EEG signal with shape (seq_len, num_nodes, input_dim)
        top_k (int):            Number of strongest connections to maintain per node. If None, all connections are maintained
        
    Returns:
        adj_mat (np.ndarray):   Absolute normalized cross-correlation adjacency matrix with shape (num_nodes, num_nodes)
    
    Notes:
    ------
        The resulting adjacency matrix is symmetric with self-connections set to 1. Cross-correlations are computed between flattened time-feature dimensions
    """
    num_nodes = eeg_clip.shape[1]
    adj_mat = np.eye(num_nodes, num_nodes, dtype=np.float32)  # diagonal is 1

    # reshape from (seq_len, num_nodes, input_dim) to (num_nodes, seq_len*input_dim)
    eeg_flat = np.transpose(eeg_clip, (1, 0, 2)).reshape((num_nodes, -1))

    # Pre-normalize all signals, handling zero-norm cases
    signals_norm = eeg_flat / np.linalg.norm(eeg_flat, axis=1, keepdims=True)
    signals_norm = np.where(np.isnan(signals_norm), 0, signals_norm)
    
    # Compute cross-correlation between pre-normalized signals ensuring diagonal self-correlation equal to 1
    adj_mat = np.abs( np.dot(signals_norm, signals_norm.T) )
    np.fill_diagonal(adj_mat, 1.0)

    if top_k is not None:
        adj_mat = keep_topk(adj_mat, top_k=top_k, directed=True)

    return adj_mat

def compute_FFT(signals:np.ndarray, n:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of the signal
    
    Args:
        signals (np.ndarray):   Signals with size (num_channels, num_data_points)
        n (int):                Length of positive frequency terms of fourier transform
    Returns:
        tuple (np.ndarray, np.ndarray): 
            - Log amplitude of FFT of signals with size (num_channels, num_data_points//2)
            - Phase spectrum of FFT of signals with size (num_channels, num_data_points//2)
    """
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = n // 2
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0
    
    FT = np.log(amp)
    P = np.angle(fourier_signal)

    return FT, P

def compute_slice_matrix(file_name:str, clip_idx:int, time_step_size:int=1, clip_len:int=60, use_fft:bool=False) -> np.ndarray:
    """
    Extract and process an EEG clip from an HDF5 file.
    The function extracts a clip of specified length from resampled EEG data and optionally applies FFT processing to generate time-frequency representations.
    
    Args:
        file_name (str):        Path to *.npy file containing the signal
        clip_idx (int):         Index of the clip to extract (0-based). Maximum value depends on signal length and clip duration
        time_step_size (int):   Duration of each time step in seconds for FFT analysis
        clip_len (int):         Total duration of the EEG clip in seconds
        use_fft (bool):         If True, apply FFT to generate time-frequency representation
    Returns:
        eeg_clip (np.ndarray):  EEG clip with shape:
            - Without FFT: (clip_len, num_channels, FREQUENCY_CHB_MIT)
            - With FFT: (clip_len, num_channels, FREQUENCY_CHB_MIT//2)
    """
    signal_array= np.load(file_name)

    # calculate physical dimensions
    physical_clip_len = FREQUENCY_CHB_MIT * clip_len
    physical_time_step_size = FREQUENCY_CHB_MIT * time_step_size
    
    start_window = clip_idx * physical_clip_len
    end_window = start_window + physical_clip_len
    
    # extract clipped signal (num_channels, physical_clip_len)
    clipped_signal = signal_array[:, start_window:end_window]
    
    # create empty clip of size (clip_len, num_channels, feature_dim)
    feature_dim= FREQUENCY_CHB_MIT//2 if use_fft else FREQUENCY_CHB_MIT
    eeg_clip= np.empty((clip_len, signal_array.shape[0], feature_dim))
    
    # if not use the FFT then the output has only different shape and different order of the axis
    # reshape from (num_signal, clip_len*feature_dim) to (clip_len, num_signal, feature_dim)
    if not use_fft:
        eeg_clip= clipped_signal.reshape(signal_array.shape[0], clip_len, feature_dim).transpose((1, 0, 2))

    # if use the FFT then is necessary to compute the FFT for each time step
    else:
        for t in range(clip_len):
            start_time_step = t*physical_time_step_size
            end_time_step = start_time_step + physical_time_step_size
            
            eeg_clip[t], _ = compute_FFT(signals=clipped_signal[:, start_time_step:end_time_step], n=physical_time_step_size)

    return eeg_clip

def compute_plv_matrix(graph: np.ndarray) -> np.ndarray:
    """Compute connectivity matrix via usage of PLV from MNE implementation.
    Args:
        graph: (np.ndarray) Single graph with shape [nodes,features] where features represent consecutive time samples and nodes represent electrodes in EEG.
        
    Returns:
        plv_matrix: (np.ndarray) PLV matrix of the input graph.
        
    Notes:
    -----
        See https://github.com/szmazurek/sano_eeg/blob/main/src/utils/utils.py#L695 for more detail
    """
    plv_conn_vector = compute_phase_lock_val(graph)

    n = int(np.sqrt(2 * len(plv_conn_vector))) + 1

    # Reshape the flattened array into a square matrix
    upper_triangular = np.zeros((n, n))
    upper_triangular[np.triu_indices(n, k=1)] = plv_conn_vector

    # Create an empty matrix for the complete symmetric matrix
    symmetric_matrix = np.zeros((n, n))

    # Fill the upper triangular part (including the diagonal)
    symmetric_matrix[np.triu_indices(n)] = upper_triangular[np.triu_indices(n)]

    # Fill the lower triangular part by mirroring the upper triangular
    plv_matrix = ( symmetric_matrix + symmetric_matrix.T - np.diag(np.diag(symmetric_matrix)) )

    # Add 1 to the diagonal elements
    np.fill_diagonal(plv_matrix, 1)
    return plv_matrix
