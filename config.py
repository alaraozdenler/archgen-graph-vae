"""
Hyperparameter configuration for Relational GraphVAE
"""

# Model architecture
LATENT_DIM = 32
HIDDEN_DIM = 128
EDGE_EMB_DIM = 32

# Encoder
ENCODER_NUM_LAYERS = 2
ENCODER_AGGREGATOR = "mean"

# Decoder
EDGE_DECODER_HIDDEN_DIM = 128
NODE_DECODER_NUM_MP_LAYERS = 2

# Training
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
NUM_EPOCHS = 100
GRADIENT_CLIP_MAX_NORM = 1.0

# Loss weighting
BETA = 1.0  # KL divergence weight (use with annealing)
LAMBDA_EDGE_TYPE = 1.0
LAMBDA_EDGE_CONT = 1.0
LAMBDA_NODE_CAT = 1.0

# KL annealing
KL_ANNEAL_EPOCHS = 50  # Number of epochs to linearly increase beta from 0

# Data
DATA_PATH = "/Users/alaraozdenler/Desktop/Thesis/archgen_graphs.pkl"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Regularization
USE_DROPOUT = True
DROPOUT_RATE = 0.1
USE_BATCH_NORM = True
USE_LAYER_NORM = True

# Stability
LOGVAR_CLAMP_MIN = -10.0
LOGVAR_CLAMP_MAX = 10.0

# Sampling
EDGE_SAMPLING_TEMPERATURE = 1.0
NEGATIVE_SAMPLING_RATIO = 3  # For scalability: negatives per positive edge

# Device
USE_CUDA = False
USE_MPS = True
DEVICE = "cuda" if USE_CUDA else "mps" if USE_MPS else "cpu"

# Weight decay for optimizer
WEIGHT_DECAY = 0.0

# Model output dimensions (same as input for reconstruction)
OUT_NODE_CONT_DIM = 3  # pos_x, pos_y, angle
OUT_NODE_CAT_DIM = 4  # 4 node types
