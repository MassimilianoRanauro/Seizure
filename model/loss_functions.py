import torch
from torch import Tensor

def smoothness_loss_func(node_matrix:Tensor, adj_matrix:Tensor):
    """
    Loss function to encourage smoothness on the graph and it measure the scaled Dirichlet energy.
    The formula is\\
    `1/(2n^2) * tr( X^T * L_dir * X )`\\
    where:
    - `n` is the number of nodes
    - `X` is the node features matrix
    - `tr()` is the trace of a matrix
    - `L_dir` is a directed graph Laplacian, calculated as `(D_o + D_i - 2*A)`
    - `A` is the adjacency matrix
    - `D_o` is the diagonal out-degree matrix
    - `D_i` is the diagonal in-degree matrix
    
    Args:
        node_matrix (Tensor):   Node features matrix with size (num_nodes, input_dim) or (batch_size, num_nodes, input_dim)
        adj_matrix (Tensor):    Adjacency matrix with size (num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
    
    Returns:
        Tensor:                 Scalar loss value if no batch or loss value with size (batch_size) if batch
    """    
    out_degree_dim= -1
    in_degree_dim= -2
    
    out_degree_vector= adj_matrix.sum(dim=out_degree_dim)
    in_degree_vector= adj_matrix.sum(dim=in_degree_dim)
    
    out_degree= torch.diag_embed( out_degree_vector )
    in_degree= torch.diag_embed( in_degree_vector )
    
    num_nodes= adj_matrix.shape[-1]
    damping_factor= 1/(num_nodes**2)
    
    node_matrix_transpose= node_matrix.transpose(in_degree_dim, out_degree_dim)
    
    degree_matrix= (out_degree + in_degree - 2*adj_matrix)
    first_product= torch.matmul(node_matrix_transpose, degree_matrix)
    second_product= torch.matmul(first_product, node_matrix)
    
    dirichlet_energy= 0.5 * second_product
    diag = torch.diagonal(dirichlet_energy, dim1=in_degree_dim, dim2=out_degree_dim).clone()

    return damping_factor * diag.sum(dim=out_degree_dim)

def degree_regularization_loss_func(adj_matrix:Tensor):
    """
    Loss function to avoid focus the attention based on the number of neighborhood  node. The formula is\\
    `- 1/n * one^T * log(A * one)`\\
    where:
    - `n` is the number of nodes
    - `A` is the adjacency matrix
    - `one` is a vector of ones
    
    Args:
        adj_matrix (Tensor):    Adjacency matrix with size (num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
    
    Returns:
        Tensor:                 Scalar loss value if no batch or loss value with size (batch_size) if batch
    """
    device_number= adj_matrix.get_device()
    device= "cpu" if (device_number==-1) else f"cuda:{device_number}"

    num_nodes= adj_matrix.shape[-1]
    damping_factor= -1/num_nodes
    
    batch_size= adj_matrix.shape[0] if len(adj_matrix.shape)==3 else 1
    all_ones_col= torch.ones((batch_size, num_nodes, 1), device=device)
    all_ones_row= torch.ones((batch_size, 1, num_nodes), device=device)
    
    first_product= torch.log( torch.matmul(adj_matrix, all_ones_col) + 1e-10 )
    second_product= torch.matmul( all_ones_row, first_product )
    
    return damping_factor * second_product.squeeze(-1).squeeze(-1).squeeze(-1)

def sparsity_loss_func(adj_matrix:Tensor):
    """
    Loss function to reduce redundant informations to encourage sparsity in the learned adjacency matrix. The formula is\\
    `- 1/(n^2) * ||A||`\\
    where:
    - `n` is the number of nodes
    - `A` is the adjacency matrix
    - `||.||` is the Frobenius norm (it is the square root of the sum of the squares of its elements)
    
    Args:
        adj_matrix (Tensor):    Adjacency matrix with size (num_nodes, num_nodes) or (batch_size, num_nodes, num_nodes)
    
    Returns:
        Tensor:                 Scalar loss value if no batch or loss value with size (batch_size) if batch
    """
    num_nodes= adj_matrix.shape[-1]
    damping_factor= 1/(num_nodes**2)
    
    return damping_factor * torch.norm(adj_matrix, p='fro', dim=(-2, -1))