"""Contains constant useful to the computation of EEG files"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS INFO
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

FREQUENCY_CHB_MIT= 256
"""Frequency of the data"""

VERY_SMALL_NUMBER = 1e-12
"""Value used to avoid zero division"""

INF = 1e20
"""Value used to avoid `float('Inf')`"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO FOR data.dataloader.SeizureDataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

TIME_STEP_SIZE= 1
"""Used in :data:`data.dataloader.SeizureDataset` \\
Duration of each time step in seconds for FFT analysis"""

MAX_SEQ_LEN= 12
"""Used in :data:`data.dataloader.SeizureDataset` \\
Total duration of the output EEG clip in seconds"""

USE_FFT= True
"""Used in :data:`data.dataloader.SeizureDataset` \\
Use the Fast Fourier Transform when obtain the slice from the file"""

TOP_K= None
"""Used in :data:`data.dataloader.SeizureDataset` \\
Maintain only the `top_k` higher value when compute the adjacency matrix"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# DAMPING FACTOR FOR LOSS FUNCTIONS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

DAMP_SMOOTH= 0.5
"""Damping factor for :func:`model.loss_functions.smoothness_loss_func`"""

DAMP_DEGREE= 0.01
"""Damping factor for :func:`model.loss_functions.degree_regularization_loss_func`"""

DAMP_SPARSITY= 0.3
"""Damping factor for :func:`model.loss_functions.sparsity_loss_func`"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO FOR model.ASGPFmodel.SGLCModel_classification
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

NUM_CLASSES= 2
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of output classes"""

NUM_CELLS= 2
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of the :data:`model.SGLCell.SGLCell` layers in the encoder stack"""

HIDDEN_DIM_GL= 128
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Hidden dimension for the :data:`model.GraphLearner.GraphLearner` module"""

HIDDEN_DIM_GNN= 256
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Hidden dimension in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

GRAPH_SKIP_CONN= 0.3
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Skip connection weight for adjacency updates"""

DROPOUT= 0.1
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Dropout probability applied in the attention layer for the :data:`model.GraphLearner.GraphLearner` module"""

EPSILON= None
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Threshold for deleting weak connections in the learned graph for the :data:`model.GraphLearner.GraphLearner` module. If None, no deleting is applied"""

NUM_HEADS= 8
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of heads for multi-head attention in the :data:`model.GraphLearner.GraphLearner` module"""

NUM_STEPS= 5
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Number of propagation steps in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

USE_GATv2= True
"""Used in :data:`model.ASGPFmodel.SGLCModel_classification` \\
Use GATV2 instead of GAT for the multi-head attention in the :data:`model.GatedGraphNeuralNetworks.GGNNLayer` module"""

USE_CUDA= True
"""Use cuda if available otherwise use cpu"""
