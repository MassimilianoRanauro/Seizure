import torch
import torch.nn as nn
from torch import Tensor

class Propogator(nn.Module):
    """
    Gated Propagator for GGNN using GRU-style gating mechanism
    """
    def __init__(self, state_dim:int, device:str=None):
        """
        Propagates the input through the adjacency matrix using both incoming and outgoing edges, controlled by learned reset and update gates.
        
        Args:
            state_dim (int):    Dimension of node features used in the `forward` method
            device (str):       Device to place the model on
        """
        super(Propogator, self).__init__()
        self.state_dim = state_dim

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        ).to(device=device)
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Sigmoid()
        ).to(device=device)
        self.transform = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),
            nn.Tanh()
        ).to(device=device)

    def forward(self, x:Tensor, supports:Tensor) -> Tensor:
        """
        Compute the new representation of the input matrix by:
        1. Aggregate features from incoming (A·x) and outgoing (A^T·x) neighbors
        2. Concatenate with current features: [A·x || A^T·x || x]
        3. Compute reset (r) and update (z) gates
        4. Compute candidate state: h_hat = tanh(W·[A·x || A^T·x || r⊙x])
        5. Update: output = (1-z)⊙x + z⊙h_hat
        
        Args:
            x (Tensor):         Node features matrix with size (batch_size, num_nodes, state_dim)
            supports (Tensor):  Adjacency matrix with size (batch_size, num_nodes, num_nodes)

        Returns:
            Tensor:             New representation of the node features matrix (batch_size, num_nodes, state_dim)
        """
        a_in =  torch.matmul(supports, x)
        a_out = torch.matmul(supports.transpose(1, 2), x)   # transposing to obtain a new (batch_size, num_nodes, num_nodes) matrix

        a = torch.cat((a_in, a_out, x), dim=2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        
        joined_input = torch.cat((a_in, a_out, r * x), dim=2)
        
        h_hat = self.transform(joined_input)

        output = (1 - z) * x + z * h_hat

        return output


class GGNNLayer(nn.Module):
    """
    Gated Graph Neural Networks (GGNN) Layer for learning the feature/node matrix
    """
    def __init__(self, input_dim:int, hidden_dim:int, num_nodes:int, output_dim:int, num_steps:int, device:str=None):
        """
        Use iterative propagation with the GRU mechanism to learn a new representation of the feature/node matrix.
        Args:
            input_dim (int):    Dimension of input node features
            hidden_dim (int):   Dimension of hidden state features
            num_nodes (int):    Number of nodes in both input graph and hidden state
            output_dim (int):   Dimension chosen for the output of the new feature/node matrix
            num_steps (int):    Number of propagation iterations
            device (str):       Device to place the model on
        """
        super(GGNNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim= hidden_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_steps = num_steps

        self.propagator = Propogator(input_dim + hidden_dim, device=device)
        self.fc = nn.Linear(input_dim + hidden_dim, output_dim, device=device)

    def forward(self, inputs:Tensor, supports:Tensor, state:Tensor) -> Tensor:
        """
        Compute the new representation of the feature/node matrix via iterative propagation:
        1. Concatenate input features and hidden state
        2. Apply propagator num_steps times
        3. Apply the fully-connected network
        
        Args:
            inputs (Tensor):        Node features matrix with size (batch_size, num_nodes*input_dim)
            supports (Tensor):      Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            state (Tensor):         Hidden state matrix with size (batch_size, num_nodes*hidden_dim)

        Returns:
            Tensor:                 New representation of the feature/node matrix with size (batch_size, num_nodes*output_dim)
        """
        batch_size = inputs.shape[0]
        inputs = inputs.reshape(batch_size, self.num_nodes, self.input_dim)
        state = state.reshape(batch_size, self.num_nodes, self.hidden_dim)

        # (batch_size, num_nodes, input_dim + hidden_dim)
        x = torch.cat([inputs, state], dim=2)  

        for _ in range(self.num_steps):
            x = self.propagator.forward(x, supports)

        # (batch_size, num_nodes, output_dim)
        x:Tensor = self.fc(x)  

        x = x.reshape(batch_size, self.num_nodes*self.output_dim)
        return x
