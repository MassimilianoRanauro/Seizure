"""Contains constant useful to the main functions instad of have a lot of parameters passed in line"""
import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FOLDERS AND FILE NAMES
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

METRICS_SAVE_FOLDER= "./metrics"
"""Folder where save the metric values"""

MODEL_SAVE_FOLDER= "./weights"
"""Folder where save the model weights"""

MODEL_NAME= "Model"
"""Name of the file where the model will be saved"""

MODEL_EXTENTION= "pth"
"""Extention of the model weights"""

METRICS_EXTENTION= "npz"
"""Extention of the metrics"""

MODEL_PARTIAL_PATH= os.path.join(MODEL_SAVE_FOLDER, MODEL_NAME)
"""Model path without extention"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# INFO DATASET INITIALIZATION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

PERCENTAGE_TEST_SPLIT= 0.2
"""How much of the dataset must be used only for test"""

BATCH_SIZE= 64
"""Batch size used during the training. The last batch can have different size"""

NUM_WORKERS= 0
"""How many subprocesses to use for data loading"""

RANDOM_STATE= 13
"""Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# OTHERS INFO
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

LEARNING_RATE= 1e-3
"""Learning rate of the model"""

BEST_K_MODELS= 0
"""Maximum number of model maintained as best model in performance"""

MAX_NUM_EPOCHS= 5
"""Maximum number of epochs after which the model will be save"""

PERCENTAGE_MARGIN= 0.05
"""Percentage of margin needed with respect to the metric considered to save the model"""
