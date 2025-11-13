import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.constants_eeg import VERY_SMALL_NUMBER, INF

from torch_geometric.nn import GAT
from torch_geometric.utils import dense_to_sparse
'''
class GraphLearner(nn.Module):
    """
    Graph learning layer that computes a new learned adjacency matrix
    """
    def __init__(self, input_size:int, hidden_size:int, dropout:float=0.5, epsilon:float=None, num_heads:int=16, use_GATv2:bool=False, device:str=None):
        """
        Use the multi-head GAT layers to learn a new representation of the adjacency matrix.
        Args:
            input_size (int):       Dimension of input node features
            hidden_size (int):      Hidden dimension for attention computation
            dropout (float):        Dropout probability applied in the attention layer
            epsilon (float):        Threshold for deleting weak connections in the learned graph. If None, no deleting is applied
            num_heads (int):        Number of heads for multi-head attention
            use_GATv2 (bool):       Use GATV2 instead of GAT for the multi-head attention
            device (str):           Device to place the model on
        """
        super(GraphLearner, self).__init__()
        self.epsilon = epsilon
        self.num_heads = num_heads

        self.att = nn.ModuleList([EEGGraphAttentionLayer(input_size, hidden_size, dropout, use_v2=use_GATv2, device=device) for _ in range(num_heads)])        
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, context:Tensor, adj:Tensor) -> Tensor:
        """
        Compute the attention scores for each head and for each batch by:
        1. Compute attention scores for each head independently
        2. Average attention across all heads
        3. Apply epsilon thresholding if specified (delete weak edges)
        4. Apply softplus activation for non-negative edge weights
        
        Args:
            context (Tensor):   Node features matrix with size (batch_size, num_nodes*input_size)
            adj (Tensor):       Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            
        Returns:
            attention (Tensor): Adjacency attention matrix with size (batch_size, num_nodes, num_nodes)
        """
        # Reshape: (batch_size, num_nodes*input_size) --> (batch_size, num_nodes, input_size)
        context = context.reshape(adj.size(0), adj.size(1), -1)
        
        attention = []
        for head_index in range(self.num_heads):
            attention_head = []
            for i in range(context.size(0)):
                h = context[i]
                ad = adj[i]
                attention_ = self.att[head_index](h, ad)
                attention_head.append(attention_)
            attention_head = torch.stack(attention_head, 0)
            attention.append(attention_head)

        attention = torch.mean(torch.stack(attention, 0), 0)

        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, -INF)

        attention = F.softplus(attention)

        return attention

    def build_epsilon_neighbourhood(self, attention:Tensor, epsilon:float, markoff_value:float):
        """
        Modify the attention matrix by setting values below epsilon to a marker. The function is equivalent to
        - `mask= (attention > epsilon).detach().float()`
        - `return torch.where(mask==1, attention, markoff_value)`
        
        Args:
            attention (Tensor):     Attention matrix
            epsilon (float):        Threshold value
            markoff_value (float):  Value to assign to delete edges
            
        Returns:
            Tensor:                 Masked attention matrix with same shape as input
        """
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

class EEGGraphAttentionLayer(nn.Module):
    """
    Unified Graph Attention Layer supporting both GAT and GATv2 architectures, similar to https://arxiv.org/abs/1710.10903 and https://arxiv.org/abs/2105.14491
    """
    def __init__(self, in_features:int, out_features:int, dropout:float=0.5, alpha:float=0.2, use_v2:bool=False, device:str=None):
        """
        If GAT layer, similar to https://arxiv.org/abs/1710.10903
        - Generate two learnable matrix `W` and `a` which have dimension `(in_features, out_features)` and `(2*out_features, 1)` to learn where focus the attention.
        
        If GATv2 layer, similar to https://arxiv.org/abs/2105.14491
        - Generate two learnable matrix `W` and `a` which have dimension `(2*in_features, out_features)` and `(2*out_features, 1)` to learn where focus the attention.
        Args:
            in_features (int):      Dimension of input node features
            out_features (int):     Dimension chosen for hidden representation
            dropout (float):        Dropout probability applied to the final normalized attention scores
            alpha (float):          Slope of `torch.nn.LeakyReLU` function
            use_v2 (bool):          If True uses GATv2, otherwise uses GAT
            device (str):           Device to place the model on
        """
        super(EEGGraphAttentionLayer, self).__init__()

        if use_v2:
            self._compute_attention = self._compute_attention_v2
            self.w = nn.Parameter(torch.empty(size=(2*in_features, out_features), device=device))
            self.a = nn.Parameter(torch.empty(size=(out_features, 1), device=device))
        else:
            self._compute_attention = self._compute_attention_v1
            self.w = nn.Parameter(torch.empty(size=(in_features, out_features), device=device))
            self.a = nn.Parameter(torch.empty(size=(2*out_features, 1), device=device))
        
        nn.init.xavier_uniform_(self.w.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout= nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def _prepare_attentional_mechanism_input(self, tensor:Tensor):
        """
        Generates a tensor containing the Cartesian product of the first dimension of the input
            :param  tensor (Tensor):                Tensor of shape `(num_nodes, other_dimension)`
            :return Cartesian_product (Tensor):     Tensor of shape `(num_nodes, num_nodes, 2*other_dimension)`
        """
        N = tensor.size()[0]
        tensor_repeated_in_chunks = tensor.repeat_interleave(N, dim=0)
        tensor_repeated_alternating = tensor.repeat(N, 1)

        all_combinations_matrix = torch.cat([tensor_repeated_in_chunks, tensor_repeated_alternating], dim=1)
        
        return all_combinations_matrix.view(N, N, -1)
    
    def _compute_attention_v1(self, h: Tensor) -> Tensor:
        """
        Attention mechanism of GAT.\\
        Calculate the attention as a matrix of size `(num_nodes, num_nodes)` calculating for each element of the matrix `e_(i,j)=LeakyReLU(a^T*[W*h_i||W*h_j])`:
        - `Wh = h × W` --> `(num_nodes, in_features) × (in_features, out_features)`
        - `cartesian_product` as the cartesian product of `Wh` --> `(num_nodes, num_nodes, 2*out_features)`
        - `coefficient = LeakyReLU( cartesian_product × a )`
        
            :param  h (Tensor):         Node features matrix with size (num_nodes, in_features)
            :return attention (Tensor): Adjacency attention matrix with size (num_nodes, num_nodes)
        """
        Wh = torch.matmul(h, self.w)
        cartesian_product = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu( torch.matmul(cartesian_product, self.a ).squeeze(2))
        
        return e
    
    def _compute_attention_v2(self, h: Tensor) -> Tensor:
        """
        Attention mechanism of GATv2.\\
        Calculate the attention as a matrix of size `(num_nodes, num_nodes)` calculating for each element of the matrix `e_(i,j)=a^T*LeakyReLU(W*[h_i||h_j])`:
        - `cartesian_product` as the cartesian product of `h` --> `(num_nodes, num_nodes, 2*in_features)`
        - `Wh = LeakyReLU( cartesian_product × W )` --> `(num_nodes, num_nodes, 2*in_features) × (2*in_features, out_features)`
        - `coefficient =  Wh × a`
        
            :param  h (Tensor):         Node features matrix with size (num_nodes, in_features)
            :return attention (Tensor): Adjacency attention matrix with size (num_nodes, num_nodes)
        """
        cartesian_product = self._prepare_attentional_mechanism_input(h)
        Wh = self.leakyrelu( torch.matmul(cartesian_product, self.w) )
        e = torch.matmul(Wh, self.a).squeeze(2)
        
        return e
    
    def forward(self, h:Tensor, adj:Tensor) -> Tensor:
        """
        Calculate the attention as a matrix of size `(num_nodes, num_nodes)` calculating for each element of the matrix:
        - `e_(i,j)=LeakyReLU(a^T*[W*h_i||W*h_j])` if use GAT:
            - `Wh = h × W` --> `(num_nodes, in_features) × (in_features, out_features)`
            - `cartesian_product` as the cartesian product of `Wh` --> `(num_nodes, num_nodes, 2*out_features)`
            - `coefficient = LeakyReLU( cartesian_product × a )`
        - `e_(i,j)=a^T*LeakyReLU(W*[h_i||h_j])` if use GATv2
            - `cartesian_product` as the cartesian product of `h` --> `(num_nodes, num_nodes, 2*in_features)`
            - `Wh = LeakyReLU( cartesian_product × W )` --> `(num_nodes, num_nodes, 2*in_features) × (2*in_features, out_features)`
        
        - make zero all values corresponding to not positive value in the adjacency matrix (it is used a low value but not zero)
        
        Args:
            h (Tensor):         Node features matrix with size (num_nodes, in_features)
            adj (Tensor):       Adjacency matrix with size (num_nodes, num_nodes)
            
        Returns:
            attention (Tensor): Normalized djacency attention matrix with size (num_nodes, num_nodes)
        """
        # print()
        # print("ACTUAL {} -\tREAL {}".format("(num_nodes, in_features)", h.shape))
        # print("ACTUAL {} -\tREAL {}".format("(num_nodes, num_nodes)", adj.shape))
        e= self._compute_attention(h)

        zero_vec = -VERY_SMALL_NUMBER * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        return attention
'''

class GraphLearner(nn.Module):
    """
    Graph learning layer that computes a new learned adjacency matrix
    """
    def __init__(self, input_size:int, hidden_size:int, num_layers:int=3, num_nodes:int=21, dropout:float=0.5, epsilon:float=None, num_heads:int=16, use_GATv2:bool=False, device:str=None, a=1):
        """
        Use the multi-head GAT layers to learn a new representation of the adjacency matrix.
        Args:
            input_size (int):       Dimension of input node features
            hidden_size (int):      Hidden dimension for attention computation
            num_layers (int):       Number of message passing layers in the GAT module
            num_nodes (int):        Number of nodes in input graph
            dropout (float):        Dropout probability applied in the attention layer
            epsilon (float):        Threshold for deleting weak connections in the learned graph. If None, no deleting is applied
            num_heads (int):        Number of heads for multi-head attention
            use_GATv2 (bool):       Use GATV2 instead of GAT for the multi-head attention
            device (str):           Device to place the model on
        """
        super(GraphLearner, self).__init__()
        self.epsilon = epsilon
        self.num_heads = num_heads
        self.a= a
        
        self.att= GAT(
            in_channels=input_size,
            hidden_channels=hidden_size,
            num_layers=num_layers,
            out_channels=num_nodes,
            dropout=dropout,
            v2=use_GATv2,
            heads=num_heads,
            edge_dim=1
        ).to(device=device)
        
        for param in self.att.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self, context:Tensor, adj:Tensor) -> Tensor:
        """
        Compute the attention scores for each head and for each batch by:
        1. Compute attention scores for each head independently
        2. Average attention across all heads
        3. Apply epsilon thresholding if specified (delete weak edges)
        4. Apply sigmoid activation for non-negative edge weights
        
        Args:
            context (Tensor):   Node features matrix with size (batch_size, num_nodes*input_size)
            adj (Tensor):       Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            
        Returns:
            attention (Tensor): Adjacency attention matrix with size (batch_size, num_nodes, num_nodes)
        """
        # Reshape: (batch_size, num_nodes*input_size) --> (batch_size, num_nodes, input_size)
        context = context.reshape(adj.size(0), adj.size(1), -1)
        context = context + 0.01 * torch.randn_like(context)        # perturbatione
        
        attention = []
        for batch_idx in range(context.size(0)):
            x = context[batch_idx]
            edge_index, edge_attr = dense_to_sparse(adj[batch_idx])
            
            attention_ = self.att(x, edge_index, edge_attr=edge_attr)
            attention.append(attention_)
        
        attention= torch.stack(attention, dim=0)

        # Modify the attention matrix by setting values below epsilon to a marker
        if self.epsilon is not None:
            mask= (attention > self.epsilon).detach().float()
            attention= torch.where(mask==1, attention, -INF)

        #attention = F.softplus(attention)
        attention= F.sigmoid(self.a * attention)

        return attention
