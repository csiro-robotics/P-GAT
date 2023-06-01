import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #

_C = CN()

_C.MODEL = CN()

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
# The similarity score for entire graph
_C.DATASETS.SCORE = "labels_dummy.pt"
# The weight for the node pairs in the graph
_C.DATASETS.SWITCH = "weight_dummy.pt"
# List of the feature dataset names for training
_C.DATASETS.FEATURE_TRAIN = (["features_dummy.pt"])
# List of the pose dataset names for training
_C.DATASETS.POSE_TRAIN = (["poses_dummy.pt"])
# List of the mask dataset names for training
_C.DATASETS.MASK_TRAIN = (["masks_dummy.pt"])
# The node information
_C.DATASETS.NODEINFO_TRAIN = (["subgraph_nodes_train.pt"])
# The pairing information
_C.DATASETS.PAIR_TRAIN = "pair.json"
# The run information for training
_C.DATASETS.RUN_ID_TRAIN = (["subgraph_run_id_train.pt"])
# List of the dataset names for testing
_C.DATASETS.FEATURE_TEST = (["features_dummy.pt"])
# List of the pose dataset names for testing
_C.DATASETS.POSE_TEST = (["poses_dummy.pt"])
# List of the mask dataset names for testing
_C.DATASETS.MASK_TEST = (["masks_dummy.pt"])
# The run information for testing
_C.DATASETS.RUN_ID_TEST = (["subgraph_run_id_test.pt"])
# The node information
_C.DATASETS.NODEINFO_TEST = (["subgraph_nodes_test.pt"])

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCH = 40
_C.SOLVER.LR = 0.001
# Weight decay
_C.SOLVER.WD = 0.0
# Batch size
_C.SOLVER.BATCHSIZE = 256

# ---------------------------------------------------------------------------- #
# Attentional Graph options
# ---------------------------------------------------------------------------- #
_C.MODEL.ATTENTION_GRAPH = CN()
_C.MODEL.ATTENTION_GRAPH.POSE_DIM = 3
_C.MODEL.ATTENTION_GRAPH.FEATURE_DIM = 256
_C.MODEL.ATTENTION_GRAPH.KEYPOINT_HIDDEN_DIM = [32, 64, 128]
_C.MODEL.ATTENTION_GRAPH.NUM_HEADS = 1
_C.MODEL.ATTENTION_GRAPH.NUM_LAYERS = 1
_C.MODEL.ATTENTION_GRAPH.INCLUDE_POSE = True
_C.MODEL.ATTENTION_GRAPH.DROPOUT = 0.0

# ---------------------------------------------------------------------------- #
# Specific train option
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.SAVE_INTERVAL = 50
_C.TRAIN.WEIGHT = [0.9, 0.1]
_C.TRAIN.MODEL_PARAM = None
_C.TRAIN.HAS_VALIDATION = True
_C.TRAIN.POS_RATE = 0.1

# ---------------------------------------------------------------------------- #
# Specific test option
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.MODEL_PARAM = "param.pt"
_C.TEST.STRIDE = 10

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./"
_C.DATA_DIR = ["../data/model_train/"]
_C.SAVE_MODEL = False

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"