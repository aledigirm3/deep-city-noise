import os

# --- Paths ---
BASE_DATASET_PATH = "../UrbanSound8K/UrbanSound8K/"
METADATA_PATH = os.path.join(BASE_DATASET_PATH, "metadata", "UrbanSound8K.csv")
AUDIO_BASE_PATH = os.path.join(BASE_DATASET_PATH, "audio")
CHECKPOINT_DIR = "checkpoints/"

# --- Feature Extraction Parameters ---
# Choose the type of feature: 'mel', 'stft', 'log_stft', 'mfcc'
FEATURE_TYPE = 'mfcc'

# --- Audio Parameters ---
SAMPLE_RATE = 22050
DURATION = 4  # s
N_FFT = 2048
HOP_LENGTH = 512

# Specific parameters for the features
N_MELS = 128  # Used for 'mel' and 'mfcc'
N_MFCC = 40   # Used only for 'mfcc'

# Dynamic calculation of the expected spectrogram length
EXPECTED_FRAMES = (SAMPLE_RATE * DURATION) // HOP_LENGTH + 1

# --- Training Parameters ---
NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VALIDATION_SPLIT_RATIO = 0.1
RANDOM_STATE = 42

# --- Callback Settings ---
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5