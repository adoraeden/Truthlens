"""Configuration for the from-scratch CIFAKE CNN experiment.

All hyperparameters and paths live here so the rest of the experiment stays
simple and easy to tweak. Nothing in this package touches the production
the original application code in ``TruthLens/src/``.
"""

# --- Image size ---
IMAGE_SIZE_SMOKE = 32   # quick smoke check on a small input
IMAGE_SIZE_MAIN = 64    # main experiment size

# --- Training hyperparameters (the training loop comes in a later stage) ---
BATCH_SIZE = 64
LEARNING_RATE = 0.0003  # 3e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

# --- Classification setup ---
NUM_CLASSES = 1          # one raw logit per image (binary)
POSITIVE_CLASS = "FAKE"  # 1 = AI-generated (experiment convention)
NEGATIVE_CLASS = "REAL"  # 0 = genuine

# --- Dataset split ---
VAL_FRACTION = 0.10      # 10% of the CIFAKE train directory used for validation
