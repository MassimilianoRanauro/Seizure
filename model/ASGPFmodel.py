import torch
import torch.nn as nn
from torch import Tensor

from model.SGLCell import SGLCell
import model.utils as utils

import os

class SGLCEncoder(nn.Module):
    """
    Spatio-Graph Learning Cell (SGLC) Encoder using multiple SGLCell layers
    """
    def __init__(self, num_cells:int, input_dim:int, num_nodes:int, hidden_dim_GL:int, hidden_dim_GGNN:int, graph_skip_conn:float=0.3, dropout:float=0, epsilon:float=None, num_heads:int=16, num_steps:int=5, use_GATv2:bool=False, device:str=None):
        """
        Use a stack of SGLCell to learn from the data. Each SGLCell use the GL, the GGNN and the GRUCell module
        
        Args:
            num_cells (int):            Number of the SGLCell layers in the stack
            input_dim (int):            Feature dimension of input nodes
            num_nodes (int):            Number of nodes in both input graph and hidden state
            
            hidden_dim_GL (int):        Hidden dimension for the Graph Learner module
            hidden_dim_GGNN (int):      Hidden dimension for the Gated Graph Neural Networks module
            
            graph_skip_conn (float):    Skip connection weight for adjacency matrix updates
            
            dropout (float):            Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):            Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            num_heads (int):            Number of heads for multi-head attention in the Graph Learner module
            num_steps (int):            Number of propagation steps in the Gated Graph Neural Networks module
            use_GATv2 (bool):           Use GATV2 instead of GAT for the multi-head attention in the Gated Graph Neural Networks module
            
            device (str):               Device to place the model on
        """
        super(SGLCEncoder, self).__init__()

        encoding_cells = list()
        for _ in range(num_cells):
            encoding_cells.append(
                SGLCell(
                    input_dim=input_dim,
                    num_nodes=num_nodes,
                    
                    hidden_dim_GL=hidden_dim_GL,
                    hidden_dim_GGNN=hidden_dim_GGNN,
                    graph_skip_conn=graph_skip_conn,
                    
                    dropout=dropout,
                    epsilon=epsilon,
                    num_heads=num_heads,
                    num_steps=num_steps,
                    use_GATv2=use_GATv2,
                    
                    device=device
                )
            )
        self.encoding_cells = nn.ModuleList(encoding_cells)
    
    def forward(self, inputs: torch.Tensor, supports: torch.Tensor, initial_hidden_state: torch.Tensor):
        """
        Compute the new representation of the parameters using the stack of SGLCell modules
        
        Args:
            inputs (Tensor):                Matrix of node with size (sequential_length, batch_size, num_nodes, input_dim)
            supports (Tensor):              Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            initial_hidden_state (Tensor):  Hidden state matrix with size (num_cells, batch_size, num_nodes*hidden_dim)

        Returns:
            tuple(Tensor, Tensor, Tensor):  A tuple containing:
                - Update matrix of node with same size
                - Update adjacency matrix with same size
                - Update hidden state matrix with same size
        """
        seq_length = inputs.size(0)
        original_shape = inputs.shape
        
        # Reshape: (sequential_length, batch_size, num_nodes*input_dim)
        inputs = inputs.reshape(inputs.size(0), inputs.size(1), -1)
        
        # Initialize outputs to avoid modifying in place
        processed_inputs = []
        current_supports = supports.clone()
        current_hidden = initial_hidden_state

        # Process each timestep
        cell:SGLCell= None
        for t in range(seq_length):
            input_t = inputs[t]
            updated_states = []

            # Pass through each encoding cell
            for cell_idx,cell in enumerate(self.encoding_cells):
                hidden_state = current_hidden[cell_idx]
                
                input_t, current_supports, hidden_state = cell.forward(input_t, current_supports, hidden_state)
        
                updated_states.append(hidden_state)

            # Update hidden states and store output
            current_hidden = torch.stack(updated_states, dim=0)
            processed_inputs.append(input_t)

        # Reconstruct output tensor with original spatial dimensions
        new_input_representation = torch.stack(processed_inputs, dim=0)
        new_input_representation = new_input_representation.reshape(*original_shape)

        return new_input_representation, current_supports, current_hidden

    def hidden_state_empty(self, batch_size:int) -> Tensor:
        """
        Create an initialized hidden state tensor for all SGLCell layers
            :param batch_size (int):   The size of the batch dimension
            :return Tensor:            Hidden state tensor, calculated as stack of the hidden state for each SGLCell
        """
        init_states = []
        sglc_cell:SGLCell= None
        for sglc_cell in self.encoding_cells:
            init_states.append(sglc_cell.hidden_state_empty(batch_size))
            
        return torch.stack(init_states, dim=0)

class SGLCModel_classification(nn.Module):
    """
    Classification model using SGLC Encoder with fully connected output layer
    """
    def __init__(self, num_classes:int, num_cells:int, input_dim:int, num_nodes:int, hidden_dim_GL:int, hidden_dim_GGNN:int, graph_skip_conn:float=0.3, dropout:float=0, epsilon:float=None, num_heads:int=16, num_steps:int=5, use_GATv2:bool=False, device:str=None):
        """
        Use a stack of SGLCell to learn from the data. Each SGLCell use the GL, the GGNN and the GRUCell module
        
        Args:
            num_classes (int):          Number of output classes
        
            num_cells (int):            Number of the SGLCell layers in the encoder stack
            input_dim (int):            Feature dimension of input nodes
            num_nodes (int):            Number of nodes in both input graph and hidden state
            
            hidden_dim_GL (int):        Hidden dimension for Graph Learner module
            hidden_dim_GGNN (int):      Hidden dimension for Gated Graph Neural Networks module
            
            graph_skip_conn (float):    Skip connection weight for adjacency updates
            
            dropout (float):            Dropout probability applied in the attention layer of the Graph Learner module
            epsilon (float):            Threshold for deleting weak connections in the learned graph in the Graph Learner module. If None, no deleting is applied
            num_heads (int):            Number of heads for multi-head attention in the Graph Learner module
            num_steps (int):            Number of propagation steps in the Gated Graph Neural Networks module
            use_GATv2 (bool):           Use GATV2 instead of GAT for the multi-head attention in the Gated Graph Neural Networks module
            
            device (str):               Device to place the model on
        """
        super(SGLCModel_classification, self).__init__()

        # Store configuration for saving
        self.config = {
            'num_classes': num_classes,
            'num_cells': num_cells,
            'input_dim': input_dim,
            'num_nodes': num_nodes,
            'hidden_dim_GL': hidden_dim_GL,
            'hidden_dim_GGNN': hidden_dim_GGNN,
            'graph_skip_conn': graph_skip_conn,
            'dropout': dropout,
            'epsilon': epsilon,
            'num_heads': num_heads,
            'num_steps': num_steps,
            'use_GATv2': use_GATv2,
            'device': device
        }
        
        self.device= device
        self.encoder = SGLCEncoder(
            num_cells=num_cells,
            input_dim=input_dim,
            num_nodes=num_nodes,
            
            hidden_dim_GL=hidden_dim_GL,
            hidden_dim_GGNN=hidden_dim_GGNN,
            graph_skip_conn=graph_skip_conn,
            
            dropout=dropout,
            epsilon=epsilon,
            num_heads=num_heads,
            num_steps=num_steps,
            use_GATv2=use_GATv2,
            
            device=device
        )

        # self.fc = nn.Linear(input_dim, num_classes, device=device)
        self.fc= nn.Sequential(
            nn.Linear(num_nodes*input_dim, num_classes*5, device=device),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(num_classes*5, num_classes, device=device),
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_seq:Tensor, supports:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Use the SGLCEncoder to calculate the new representations of the feature/node matrix and the adjacency matrix using a hidden state initialize at all zeros.
        1. Processes the inputs through SGLC encoder
        2. Computes adjacency-weighted node aggregation:
           - adj_mean: mean of [adj || adj^T] along rows --> (batch_size, num_nodes)
           - weighted_features: last timestep features weighted by adj_mean
           - features_mean: mean over nodes --> (batch_size, input_dim)
        3. Applies FC layer and ReLU activation function over feature dimension
        
        Args:
            input_seq (Tensor):     Input features matrix with size shape (batch_size, sequential_length, num_nodes, input_dim)
            supports (Tensor):      Adjacency matrix with size (batch_size, num_nodes, num_nodes)
            
        Returns:
            tuple(Tensor, Tensor, Tensor): A tuple containing:
                - result: Class probability distribution with shape (batch_size, num_classes)
                - input_seq: Encoded feature sequence with shape (batch_size, seq_length, num_nodes, input_dim)
                - supports: Learned adjacency matrix with shape (batch_size, num_nodes, num_nodes)
        """
        init_hidden_state = torch.zeros_like( self.encoder.hidden_state_empty(batch_size=input_seq.size(0)) )
        init_hidden_state = init_hidden_state.to(device=self.device)

        # Transpose: (batch_size, seq_length, num_nodes, input_dim) --> (seq_length, batch_size, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        input_seq, supports, _ = self.encoder.forward(input_seq, supports, init_hidden_state)
        
        features_mean= input_seq[-1]
        features_mean= features_mean.reshape(features_mean.size(0), -1)
        result = self.fc(features_mean)

        # Transpose: (seq_length, batch_size, num_nodes, input_dim) --> (batch_size, seq_length, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)
        
        return result, input_seq, supports

    def save(self, filepath:str) -> None:
        """
        Save the model's state dictionary and configuration to a file.
            :param filepath (str): Path where the model will be saved (e.g., 'model.pth')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config
        }
        
        torch.save(save_dict, filepath)
    
    @staticmethod
    def load(filepath:str, device:str=None):
        """
        Load a saved model from a file.
        
        Args:
            filepath (str): Path to the saved model file
            device (str):   Device to place the model on. If None, uses the device from saved config
            
        Returns:
            SGLCModel_classification: Loaded model instance
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        
        config = checkpoint['config']
        
        # Override device if specified
        if device:
            config['device'] = device
        
        model = SGLCModel_classification(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model