import torch
import random
import pandas as pd
from ssm import train, SSM, anbn_ssm, random_star_ab, random_and, random_tiptup, random_xor, random_anbn
from ssm import evaluate_and, evaluate_tiptup, evaluate_model_unpaired  # From existing evaluation code

# Constants for testing
NUM_TRAINING_SAMPLES = 1000  # Number of training samples
PRINT_EVERY = 100  # How often to print during training
LEARNING_RATE = 0.01  # Learning rate for the optimizer
K = 2  # Dimensionality of state space
S = 3  # Dimensionality of observation space
TRAINING_DATA = [random_anbn() for _ in range(NUM_TRAINING_SAMPLES)]  # TODO Generate training data 

# Train a new state-space model using training data
print("Training SSM model with random_anbn data...")
model = train(K, S, TRAINING_DATA, print_every=PRINT_EVERY, lr=LEARNING_RATE)

# Evaluate the trained model
print("\nEvaluating model with 'AND' logic...")
df_and = evaluate_and(model)
print("AND Evaluation Results:\n", df_and)

print("\nEvaluating model with 'TIPTUP' logic...")
df_tiptup = evaluate_tiptup(model)
print("TIPTUP Evaluation Results:\n", df_tiptup)

# Evaluate with custom data sets
print("\nCustom evaluation with different good and bad strings...")
good_strings = [
    [1, 1, 2, 2, 0],  # A valid anbn sequence
    [1, 2, 0],  # Simple valid sequence
]
bad_strings = [
    [1, 2, 2],  # Invalid sequence with extra b's
    [1, 1, 1, 2, 2, 0],  # Extra a's
]
df_custom = evaluate_model_unpaired(model, good_strings, bad_strings)
print("Custom Evaluation Results:\n", df_custom)
