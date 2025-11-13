import torch
import torch.nn as nn
from torch import Tensor

from model.GraphLearner import GraphLearner
from model.GatedGraphNeuralNetworks import GGNNLayer

class SGLCell(nn.Module):
    """
    Spatio-Graph Learning Cell (SGLC) with Graph Learner, the Gated Graph Neural Networks and the GRU module
    """
    def __init__(self, input_dim:int, num_nodes:int, hidden_dim_GL:int, hidden_dim_GGNN:int, graph_skip_conn:float=0.3, dropout:float=0, epsilon:float=None, num_heads:int=16, num_steps:int=5, use_GATv2:bool=False, device:str=None):
        """
        Use the Graph Learner, the Gated Graph Neural Networks and the GRU module to obtain new representations
        
        Args:
            input_dim (int):            Feature dimension of input nodes
            num_nodes (int):            Number of nodes in both input graph and hidden state
            
            hidden_dim_GL (int):        Hidden dimension for Graph Learner module
            hidden_dim_GGNN (int):      Hidden dimension for the Gated Graph Neural Networks module
            
            graph_skip_conn (float):    Skip connection weight for adjacency matrix updates
            
            dropout (float):            Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):            Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            num_heads (int):            Number of heads for multi-head attention in the Graph Learner module
            num_steps (int):            Number of propagation steps in the Gated Graph Neural Networks module
            use_GATv2 (bool):           Use GATV2 instead of GAT for the multi-head attention in the Gated Graph Neural Networks module
            
            device (str):               Device to place the model on
        """
        super(SGLCell, self).__init__()
        self._num_nodes = num_nodes
        self._hidden_dim_GGNN = hidden_dim_GGNN
        self.graph_skip_conn = graph_skip_conn

        self.graph_learner = GraphLearner(
            input_size=input_dim,
            hidden_size=hidden_dim_GL,
            num_nodes=num_nodes,
            dropout=dropout,
            epsilon=epsilon,
            num_heads=num_heads,
            use_GATv2=use_GATv2,
            device=device
        )
        
        self.ggnn = GGNNLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim_GGNN,
            num_nodes=num_nodes,
            num_steps=num_steps,
            output_dim=input_dim,
            device=device
        )
        
        self.gru = nn.GRUCell(
            input_size= num_nodes*input_dim,
            hidden_size= num_nodes*hidden_dim_GGNN,
            device=device
        )

    def forward(self, inputs:Tensor, supports:Tensor, state:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the new representation of the parameters by:
        1. Learn new adjacency matrix via Graph Learner with skip connections
        2. Update node features via Gated Graph Neural Networks with sigmoid activation
        3. Update hidden state via GRUCell
        
        Args:
            inputs (Tensor):                Matrix of node with size (batch_size, num_nodes*input_dim)
            supports (Tensor):              Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            state (Tensor):                 Hidden state matrix with size (batch_size, num_nodes*hidden_dim_GGNN)
        Returns:
            tuple(Tensor, Tensor, Tensor):  A tuple containing:
                - Update matrix of node with same size
                - Update adjacency matrix with same size
                - Update hidden state matrix with same size
        """
        # print()
        # print("ACTUAL {} -\tREAL {}".format("(batch_size, num_nodes*input_size)", inputs.shape))
        # print("ACTUAL {} -\tREAL {}".format("(batch_size, num_nodes, num_nodes)", supports.shape))
        raw_adj = self.graph_learner.forward(inputs, supports)        
        adj = torch.softmax(raw_adj, dim=-1)
        supports = self.graph_skip_conn * supports + (1 - self.graph_skip_conn) * adj
        
        inputs = torch.sigmoid(self.ggnn.forward(inputs, supports, state))
        
        state= self.gru(inputs, state)

        return inputs, supports, state

    def hidden_state_empty(self, batch_size:int) -> Tensor:
        """
        Create an uninitialized hidden state tensor
            :param batch_size (int):   The size of the batch dimension
            :return Tensor:            Hidden state tensor with size (batch_size, num_nodes * hidden_dim_GGNN)
        """
        return torch.empty([batch_size, self._num_nodes * self._hidden_dim_GGNN])
