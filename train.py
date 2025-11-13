# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORT
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from utils.Checkpoint_manager import CheckPoint
from utils.constants_eeg import *
from utils.constants_main import *
from utils.metrics_classes import *
from utils.metric import Metrics

from data.dataloader import SeizureDataset, SeizureSampler

from model.ASGPFmodel import SGLCModel_classification
from model.loss_functions import *

# from torchvision.ops import sigmoid_focal_loss
from torch.nn.functional import one_hot
from train_args import parse_arguments

from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import logging
import random
import os
import re


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GLOBAL VARIABLE
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# it is the 10% of the batch size
MIN_SAMPLE_PER_CLASS= int( 0.1 * BATCH_SIZE )
MIN_SAMPLER_PER_BATCH= int( 0.1 * BATCH_SIZE )
PERCENTAGE_INCREASE_LOSS= 0.5

DEVICE= "cpu"
NUM_SEIZURE_DATA= 0
NUM_NOT_SEIZURE_DATA= 0
START_EPOCH= 0

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_subset_targets(subset:Subset, original_targets:list):
    """
    Extract targets for a Subset based on its indices
    
    Args:
        subset (Subset):            The subset for which to extract targets
        original_targets (list):    A list containing the targets (labels) for the *entire* original dataset *before* it was split. The order must match the original dataset's order

    Returns:
        (list):                     A new list containing only the targets that correspond to the samples in the provided `Subset`
    """
    return [original_targets[i] for i in subset.indices]

def dataset_split(dataset:Dataset, targets:list, split_sizes:list[float], min_per_class:int=1, seed:int=None) -> list[Subset]:
    """
    Split a dataset into multiple splits ensuring each class has at least `min_per_class` samples in each split.
    
    Args:
        dataset (Dataset):          The dataset to split
        split_sizes (list(float)):  List of proportions for each split (e.g., [0.7, 0.15, 0.15] for train/val/test). Automatically normalize split sizes to sum to 1
        targets (list):             A list or array-like object containing the class label for each sample in the `dataset`.
                                    The length of `targets` must be equal to `len(dataset)`, where `targets[i]` corresponds to the label of `dataset[i]`
        min_per_class (int):        Minimum samples per class in each split
        seed (int):                 Random seed for reproducibility. If None the seed will be not custom initialize
    
    Returns:
        subsets (list(Subset)):     List of Subset datasets
    """
    if (targets is None) or (not isinstance(targets, (list, tuple))):
        raise TypeError(f"Expected 'targets' to be a list or tuple, but got {type(targets)}")
    if len(dataset) != len(targets):
        raise ValueError(f"Length mismatch: 'dataset' has {len(dataset)} samples, but 'targets' has {len(targets)} samples.")
    if (split_sizes is None) or (not isinstance(split_sizes, (list, tuple))) or (not split_sizes):
        raise TypeError(f"Expected 'split_sizes' to be a not empty list or tuple")
    
    if seed:
        random.seed(seed)
        
    # Normalize split sizes to sum to 1
    total = sum(split_sizes)
    split_sizes = [s / total for s in split_sizes]
    n_splits = len(split_sizes)
    
    # Group sample indices by class
    class_to_indices = defaultdict(list)
    for idx,label in enumerate(targets):
        class_to_indices[label].append(idx)
    
    # Initialize split indices
    split_indices = [[] for _ in range(n_splits)]
    
    # Step 1: Ensure each class has at least min_per_class samples in all splits
    for cls,indices in class_to_indices.items():
        indices_copy = indices.copy()
        random.shuffle(indices_copy)
        n = len(indices_copy)
        
        if n < n_splits * min_per_class:
            raise ValueError(f"Class '{cls}' has too few samples ({n}) to guarantee {min_per_class} in each of {n_splits} splits")
        
        # Allocate guaranteed samples to each split
        idx_pos = 0
        for split_idx in range(n_splits):
            split_indices[split_idx].extend(indices_copy[idx_pos:idx_pos + min_per_class])
            idx_pos += min_per_class
        
        # Remaining samples for this class
        remaining = indices_copy[idx_pos:]
        
        # Distribute remaining samples proportionally
        for split_idx in range(n_splits):
            # Calculate how many more samples this split should get
            target_count = int(split_sizes[split_idx] * n)
            current_count = len([i for i in split_indices[split_idx] if targets[i] == cls])
            needed = max(0, target_count - current_count)
            
            # Take samples from remaining
            take = min(needed, len(remaining))
            split_indices[split_idx].extend(remaining[:take])
            remaining = remaining[take:]
        
        # If any samples still remain, add them to the first split
        if remaining:
            split_indices[0].extend(remaining)
    
    # Build Subsets
    datasets = [Subset(dataset, indices) for indices in split_indices]
    
    return datasets

def get_model(dir:str, specific_num:int=None) -> tuple[str, int]:
    """
    Search for model files with numeric identifiers in a directory tree.\\
    Recursively searches the specified directory and all subdirectories for files matching the pattern: `<prefix>_<number>.<extension>`\\
    The function extracts the numeric identifier from each matching filename and either:
    - Returns the file with the highest number (default behavior)
    - Returns the file with a specific number if requested
    
    Args:
        dir (str):          Root directory path to search. All subdirectories will be recursively traversed.
        specific_num (int): If provided, returns the first file found with this exact number instead of searching for the maximum
    
    Returns:
        filename (str):     Full filepath (including directory path) of the matching model file. If no matching files exist returnsa an empty string
        epoch (int):        Number found for the full filepath
    
    Examples:
        >>> get_model("models/")
        "models/checkpoint/model_15.pth"  # Highest number found
        
        >>> get_model("models/", specific_num=10)
        "models/checkpoint/model_10.pth"  # Specific number requested
        
        >>> get_model("empty_dir/")
        ""  # No matching files
    """
    pattern= re.compile(r'_(\d+)\.(\D+)$')
    
    curr_epoch= 0
    output_filename= ""
    
    for path, _, files in os.walk(dir):
        for filename in files:
            match = re.search(pattern, filename)
            if match:
                number = int(match.group(1))
                if number > curr_epoch:
                    curr_epoch= number
                    output_filename= os.path.join(path, filename)
                if (specific_num is not None) and (specific_num==number):
                    return os.path.join(path, filename), curr_epoch
    
    return output_filename, curr_epoch

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TRAINING & EVALUATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def train_or_eval(data_loader:DataLoader, model:SGLCModel_classification, optimizer:torch.optim.Optimizer, show_progress:bool=False, verbose:bool=False) -> list[tuple[str, float]]:
    """
    Unique function to train and evaluate the model. The operation is only one, so the training is for only one epoch.\\
    The use of training mode or evaluation mode depend on the `optimizer` parameter. During the evaluation mode there are no
    optimizer so, when the parameter is None the function create a dummy optimizer and set to `eval()` the model instead of `train()`. 

    Args:
        data_loader (DataLoader):           Data on which compute operations for the model
        model (SGLCModel_classification):   Model to train or to evaluate
        optimizer (torch.optim.Optimizer):  Optimizer used for training. If it is None then the evaluation is applyed
        show_progress (bool):               Show the progress bar. The progress bar is removed when terminated
        verbose (bool):                     Useful for printing information during the execution

    Returns:
        list(tuple(str, float)):            The list has as many tuples as used metrics. Each tuple is 
    """
    is_training = optimizer is not None
    model.train(is_training)
    
    # constants used inside the loop
    seizure_weight = 1.0 / NUM_SEIZURE_DATA
    non_seizure_weight = 1.0 / NUM_NOT_SEIZURE_DATA
    #weights= torch.Tensor([non_seizure_weight, seizure_weight]).to(device=DEVICE)
    pos_weight= torch.Tensor([NUM_NOT_SEIZURE_DATA / NUM_SEIZURE_DATA]).to(device=DEVICE)
    
    # init metrics
    average= Average_Meter()
    accuracy= Accuracy_Meter([non_seizure_weight, seizure_weight], num_classes=NUM_CLASSES)
    conf_matrix= ConfusionMatrix_Meter(NUM_CLASSES)
    
    # enable or not the gradients
    with (torch.enable_grad() if is_training else torch.no_grad()):
        for x,target,adj in (tqdm(data_loader, desc=f"{'Train' if model.training else 'Eval'} current epoch", leave=False) if show_progress else data_loader):
            x:Tensor= x.to(device=DEVICE)
            target:Tensor= target.to(device=DEVICE)
            adj:Tensor= adj.to(device=DEVICE)
            
            result, node_matrix, adj_matrix = model.forward(x, adj)
            
            # calculate loss and update model (if optimizer is not None)
            #damp_factor= torch.where(target==1, seizure_weight, non_seizure_weight)     # damping factor used to weight the focal loss
            
            # reshape from (batch_size, seq_length, num_nodes, input_dim) to (batch_size, num_nodes, seq_length*input_dim) for smoothness_loss_func
            node_matrix_for_smooth= node_matrix.transpose(dim0=1, dim1=2)#.contiguous().clone()
            node_matrix_for_smooth= node_matrix_for_smooth.reshape(node_matrix_for_smooth.size(0), node_matrix_for_smooth.size(1), -1)
            
            # reshape from (batch_size, 1) to (batch_size) and transform in one-hot encoder for the focal loss (the original dtype is keep)
            target_one_hot= one_hot(target.squeeze(-1).to(dtype=torch.int64), num_classes=NUM_CLASSES)
            target_one_hot= target_one_hot.to(dtype=target.dtype)

            # loss_pred= F.binary_cross_entropy(result, target_one_hot, weight=weights, reduction="none")
            loss_pred= F.binary_cross_entropy_with_logits(result, target_one_hot, pos_weight=pos_weight, reduction="none").sum(dim=1)
            
            loss_smooth= smoothness_loss_func(node_matrix_for_smooth, adj_matrix)
            loss_degree= degree_regularization_loss_func(adj_matrix)
            loss_sparsity= sparsity_loss_func(adj_matrix)
            
            total_loss= (1+PERCENTAGE_INCREASE_LOSS) * (loss_pred + DAMP_SMOOTH*loss_smooth + DAMP_DEGREE*loss_degree + DAMP_SPARSITY*loss_sparsity)
            
            if is_training:
                optimizer.zero_grad()
                total_loss.mean().backward()
                optimizer.step()
            
            # update metrics [from (batch_size,1) to (batch_size)]
            target= target.squeeze(-1)
            average.update(total_loss)
            accuracy.update(result, target)
            conf_matrix.update(result, target)
    
    model.eval()
    
    metrics= [
        average.get_metric(),
        #accuracy.get_metric(),
        *accuracy.get_class_accuracy(),
        *accuracy.get_avg_target_prob(),
        conf_matrix.get_precision(),
        conf_matrix.get_recall(),
        conf_matrix.get_f1_score()
    ]
    
    # print metrics during execution
    if verbose:
        mode= "Train" if is_training else "Eval"
        max_len= max(len(name) for name,_ in metrics)
        print(f"\n{mode} mode:")
        for name,value in metrics:
            print(f"{name:<{max_len}} --> {value:.6f}")
    
    return metrics

def eval(data_loader:DataLoader, model:SGLCModel_classification, verbose:bool=True, show_progress:bool=False):
    """For more info see the function :func:`train_or_eval`"""
    return train_or_eval(data_loader=data_loader, model=model, optimizer=None, verbose=verbose, show_progress=show_progress)

def train_epoch(data_loader:DataLoader, model:SGLCModel_classification, optimizer:torch.optim.Optimizer, verbose:bool=True, show_progress:bool=False):
    """For more info see the function :func:`train_or_eval`"""
    return train_or_eval(data_loader=data_loader, model=model, optimizer=optimizer, verbose=verbose, show_progress=show_progress)

def train(train_loader:DataLoader, val_loader:DataLoader, model:SGLCModel_classification, optimizer:torch.optim.Optimizer, num_epochs:int, verbose:bool=True, show_epoch_progress:bool=False):
    """
    Train the model and evaluate its performance. Use static parameters from :data:`utils.constants_main` and :data:`utils.constants_eeg`
    to implement some operations.\\
    Each time the model improves by :const:`PERCENTAGE_MARGIN`% both model and metrics will be saved. They will be saved also
    if for :const:`MAX_NUM_EPOCHS` the model has no saves and at the last iteration. The path where model will be saved 
    is defined by :const:`MODEL_PARTIAL_PATH`\\_ `epoch_number` with extention :const:`MODEL_EXTENTION` and the path where the metrics 
    will be saved will be `name_metric`\\_ `epoch_number` with extention :const:`METRICS_EXTENTION`.\\
    The metric used to is the first returned by :func:`train_or_eval`

    Args:
        train_loader (DataLoader):          Data on which train the model
        val_loader (DataLoader):            Data on which evaluate the model
        model (SGLCModel_classification):   Model to train or to evaluate
        optimizer (torch.optim.Optimizer):  Optimizer used for training
        num_epochs (int):                   Number of epochs for trainig
        verbose (bool):                     Useful for printing information during the execution of the evaluation method
        show_progress (bool):               Show the inner progress bar. The progress bar is removed when terminated
    """
    # using a dataloader of one batch with one item compute dynamically: number of metrics and name of metrics
    single_item_dataset= Subset(val_loader.dataset, indices=[0])
    single_item_dataloader = DataLoader(single_item_dataset, batch_size=1, shuffle=False)
    dummy_metrics= eval(single_item_dataloader, model, verbose=False, show_progress=False)
    metrics_name= [name for name,_ in dummy_metrics]
    num_metrics= len( metrics_name )
    
    # generate the metrics array
    array_train= np.empty((num_epochs, num_metrics), dtype=np.float64)
    array_val= np.empty((num_epochs, num_metrics), dtype=np.float64)
    
    # real training
    checkpoint_observer= CheckPoint(best_k=BEST_K_MODELS, each_spacing=MAX_NUM_EPOCHS, total_epochs=num_epochs, higher_is_better=False)
    checkpoint_observer.margin= PERCENTAGE_MARGIN
    
    for epoch_num in tqdm(range(num_epochs), desc="Progress", unit="epoch"):
        # print(f"Epoch {epoch_num+1}/{num_epochs}")
        metrics_train= train_epoch(train_loader, model, optimizer, verbose=False, show_progress=(epoch_num==0))
        metrics_val= eval(val_loader, model, verbose=verbose, show_progress=False)

        array_train[epoch_num]= np.array([value for _,value in metrics_train])
        array_val[epoch_num]= np.array([value for _,value in metrics_val])
        
        used_metric= metrics_val[0][1]
        
        # conditions to save the model
        saved_files= []
        if checkpoint_observer.check_saving(used_metric):
            file_path= f"{MODEL_PARTIAL_PATH}_{epoch_num+START_EPOCH+1}.{MODEL_EXTENTION}"
            saved_files.append(file_path)
            model.save(file_path)
            for index,name in enumerate(metrics_name):
                position_old= os.path.join(METRICS_SAVE_FOLDER, f"epoch_{epoch_num+START_EPOCH}", f"{name}_{epoch_num+START_EPOCH}.{METRICS_EXTENTION}")
                position= os.path.join(METRICS_SAVE_FOLDER, f"epoch_{epoch_num+START_EPOCH+1}", f"{name}_{epoch_num+START_EPOCH+1}.{METRICS_EXTENTION}")
                saved_files.append(position)
                if os.path.exists(position_old):
                    Metrics.save(position, *Metrics.fusion(position_old, train_metric=array_train[0:epoch_num+1, index], val_metric=array_val[0:epoch_num+1, index]))
                else:
                    Metrics.save(position, train_metric=array_train[0:epoch_num+1, index], val_metric=array_val[0:epoch_num+1, index])
            
            checkpoint_observer.update_saving(used_metric, saved_files)
        
        # delete obsolete model and check saved
        checkpoint_observer.delete_obsolete_checkpoints(auto_delete=True)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    # take input from command line
    input_dir, files_record, save_num, do_train, num_epochs, verbose, preprocess_dir = parse_arguments()
    
    # load dataset
    LOGGER.info("Loading dataset with at least ({}) samples for class in a batch of ({})...".format(MIN_SAMPLER_PER_BATCH, BATCH_SIZE))
    dataset= SeizureDataset(
        input_dir= input_dir,
        files_record= files_record,
        
        time_step_size= TIME_STEP_SIZE,
        max_seq_len= MAX_SEQ_LEN,
        
        preprocess_data=preprocess_dir,
        
        use_fft= USE_FFT,
        top_k= TOP_K
    )
    training_set, test_set = dataset_split(dataset, dataset.target(), split_sizes=[1-PERCENTAGE_TEST_SPLIT, PERCENTAGE_TEST_SPLIT], min_per_class=MIN_SAMPLE_PER_CLASS, seed=RANDOM_STATE)
    
    training_sampler= SeizureSampler(get_subset_targets(training_set, dataset.target()), batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)
    test_sampler= SeizureSampler(get_subset_targets(test_set, dataset.target()), batch_size=BATCH_SIZE, n_per_class=MIN_SAMPLER_PER_BATCH, seed=RANDOM_STATE)
    
    training_loader= DataLoader(training_set, batch_size=BATCH_SIZE, sampler=training_sampler, num_workers=NUM_WORKERS)
    test_loader= DataLoader(test_set, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=NUM_WORKERS)
    
    # load model if exists
    LOGGER.info("Loading model...")
    global DEVICE
    DEVICE= 'cuda' if (torch.cuda.is_available() and USE_CUDA) else 'cpu'
    LOGGER.info(f"Using {DEVICE} device...")
    
    filename, num_epoch = get_model(MODEL_SAVE_FOLDER, specific_num=save_num)
    if len(filename)==0 and (not do_train):
        raise ValueError(f"Evaluation stopped, model not present in the '{MODEL_SAVE_FOLDER}' folder")
    if len(filename)!=0:
        global START_EPOCH
        START_EPOCH= num_epoch
        model= SGLCModel_classification.load(filename, device=DEVICE)
        LOGGER.info(f"Loaded '{os.path.basename(filename)}'...")
    else:        
        feature_matrix, _, _ = dataset[0]
        num_nodes= feature_matrix.size(1)
        input_dim= feature_matrix.size(2)
                
        model= SGLCModel_classification(
            num_classes= NUM_CLASSES,
            
            num_cells= NUM_CELLS,
            input_dim= input_dim,
            num_nodes= num_nodes,
            
            hidden_dim_GL= HIDDEN_DIM_GL,
            hidden_dim_GGNN= HIDDEN_DIM_GNN,
            
            graph_skip_conn= GRAPH_SKIP_CONN,
            
            dropout= DROPOUT,
            epsilon= EPSILON,
            num_heads= NUM_HEADS,
            num_steps= NUM_STEPS,
            use_GATv2= USE_GATv2,
            
            device= DEVICE
        )
        
    
    # set the number of seizure and not seizure data
    global NUM_NOT_SEIZURE_DATA, NUM_SEIZURE_DATA
    if (preprocess_dir is not None):
        seizure_bool_array= np.array([has_seizure for _,has_seizure in dataset.file_info], dtype=np.int8)
    else:
        seizure_bool_array= np.array([has_seizure for _,_,has_seizure in dataset.file_info], dtype=np.int8)
    NUM_NOT_SEIZURE_DATA= np.count_nonzero(seizure_bool_array)
    NUM_SEIZURE_DATA = len(seizure_bool_array)-np.count_nonzero(seizure_bool_array)
    if do_train and (NUM_NOT_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data without seizure")
    if do_train and (NUM_SEIZURE_DATA==0):
        raise ValueError(f"Training aborted, no data with seizure")
    
    # start train or evaluation
    if do_train:
        optimizer= torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train(training_loader, test_loader, model, optimizer, num_epochs=num_epochs, verbose=verbose, show_epoch_progress=True)
    else:
        eval(test_loader, model, verbose=True, show_progress=True)

if __name__=='__main__':
    main()
