import torch
from torch import Tensor

def nan_mean(tensor:Tensor, dim:int, threshold:float=0.0) -> Tensor:
    """
    Calculate the mean of a tensor along a specific dimension, ignoring all values below the given threshold

    Args:
        tensor (Tensor):    The input tensor
        dim (int):          The dimension along which to compute the mean
        threshold (float):  Values below this threshold are treated as NaN and excluded

    Returns:
        Tensor:             Mean along the specified dimension with below-threshold values ignored
    """
    filtered_matrix= torch.where(tensor > threshold, tensor, torch.nan)
    result_nan= torch.nanmean(filtered_matrix, dim=dim)
    result= torch.nan_to_num(result_nan, nan=0.0)
    return result

def nan_mean_row_col(matrix:Tensor, threshold:float=0.0, batch_first:bool=True) -> Tensor:
    """
    Calculate the mean of a square matrix concatenated with its transpose. Computes the mean along rows of the combined matrix `[matrix || matrix^T]`

    Args:
        matrix (Tensor):    Input square tensor
        threshold (float):  Values below this threshold are excluded from mean calculation
        batch_first (bool): If True the input has batch dimension first (B, C, C) otherwise the input has no batch dimension (C, C)

    Returns:
        Tensor:             Row means of the concatenated matrix `[matrix || matrix^T]` with shape: (B, C) if `batch_first=True` or (C,) if `batch_first=False`
    """
    dim_row= 0 + int(batch_first)
    dim_col= 1 + int(batch_first)
    
    fusion_matrix= torch.cat(
        [
            matrix,
            matrix.transpose(dim_row, dim_col)
        ],
        dim=dim_row
    )
    
    mean= nan_mean(fusion_matrix, dim=dim_row, threshold=threshold)
    
    return mean
