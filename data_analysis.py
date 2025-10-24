import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
data = pd.read_csv('/Users/huteng/Desktop/ssm/output/model_evaluations/TSL.2.1.0/ptsl2/ptsl2_bs4_ep10_lr0.01.txt')

# Set up the plotting style
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Training Evaluation - ptsl2_bs4_ep10_lr0.01', fontsize=16, fontweight='bold')

# Plot 1: Loss over steps
axes[0, 0].plot(data['step'], data['mean_loss'], color='red', linewidth=1.5, alpha=0.8)
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Mean Loss')
axes[0, 0].set_title('Training Loss Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Grammatical vs Ungrammatical Scores
axes[0, 1].plot(data['step'], data['grammatical_scores'], label='Grammatical', color='green', linewidth=1.5, alpha=0.8)
axes[0, 1].plot(data['step'], data['ungrammatical_scores'], label='Ungrammatical', color='orange', linewidth=1.5, alpha=0.8)
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Scores')
axes[0, 1].set_title('Grammatical vs Ungrammatical Scores')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Grammatical-Ungrammatical Difference
axes[1, 0].plot(data['step'], data['grammatical_ungrammatical_diff'], color='blue', linewidth=1.5, alpha=0.8)
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Score Difference')
axes[1, 0].set_title('Grammatical - Ungrammatical Score Difference')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: T-statistic
axes[1, 1].plot(data['step'], data['grammatical_ungrammatical_diff_t'], color='purple', linewidth=1.5, alpha=0.8)
axes[1, 1].set_xlabel('Step')
axes[1, 1].set_ylabel('T-statistic')
axes[1, 1].set_title('T-statistic for Score Difference')
axes[1, 1].grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig('/Users/huteng/Desktop/ssm/model_evaluation_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a second figure for epoch-based analysis
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Model Training by Epoch - sl2_bs4_ep10_lr0.01', fontsize=16, fontweight='bold')

# Color code by epoch
epochs = data['epoch'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(epochs)))

# Plot by epoch - Loss
for i, epoch in enumerate(epochs):
    epoch_data = data[data['epoch'] == epoch]
    axes2[0, 0].scatter(epoch_data['step'], epoch_data['mean_loss'],
                       color=colors[i], label=f'Epoch {epoch}', alpha=0.7, s=20)
axes2[0, 0].set_xlabel('Step')
axes2[0, 0].set_ylabel('Mean Loss')
axes2[0, 0].set_title('Loss by Epoch')
axes2[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes2[0, 0].grid(True, alpha=0.3)

# Plot by epoch - Score difference
for i, epoch in enumerate(epochs):
    epoch_data = data[data['epoch'] == epoch]
    axes2[0, 1].scatter(epoch_data['step'], epoch_data['grammatical_ungrammatical_diff'],
                       color=colors[i], label=f'Epoch {epoch}', alpha=0.7, s=20)
axes2[0, 1].set_xlabel('Step')
axes2[0, 1].set_ylabel('Score Difference')
axes2[0, 1].set_title('Score Difference by Epoch')
axes2[0, 1].grid(True, alpha=0.3)

# Boxplot of loss by epoch
epoch_list = []
loss_list = []
for epoch in epochs:
    epoch_data = data[data['epoch'] == epoch]
    epoch_list.extend([f'Epoch {epoch}'] * len(epoch_data))
    loss_list.extend(epoch_data['mean_loss'].tolist())

axes2[1, 0].boxplot([data[data['epoch'] == epoch]['mean_loss'] for epoch in epochs],
                    labels=[f'Epoch {epoch}' for epoch in epochs])
axes2[1, 0].set_ylabel('Mean Loss')
axes2[1, 0].set_title('Loss Distribution by Epoch')
axes2[1, 0].grid(True, alpha=0.3)

# Boxplot of score difference by epoch
axes2[1, 1].boxplot([data[data['epoch'] == epoch]['grammatical_ungrammatical_diff'] for epoch in epochs],
                    labels=[f'Epoch {epoch}' for epoch in epochs])
axes2[1, 1].set_ylabel('Score Difference')
axes2[1, 1].set_title('Score Difference Distribution by Epoch')
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/huteng/Desktop/ssm/model_evaluation_by_epoch.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("=== Summary Statistics ===")
print(f"Total steps: {len(data)}")
print(f"Epochs: {data['epoch'].nunique()}")
print(f"Final loss: {data['mean_loss'].iloc[-1]:.4f}")
print(f"Final score difference: {data['grammatical_ungrammatical_diff'].iloc[-1]:.4f}")
print(f"Final t-statistic: {data['grammatical_ungrammatical_diff_t'].iloc[-1]:.4f}")
print(f"Best (lowest) loss: {data['mean_loss'].min():.4f} at step {data.loc[data['mean_loss'].idxmin(), 'step']}")
print(f"Best (highest) score difference: {data['grammatical_ungrammatical_diff'].max():.4f} at step {data.loc[data['grammatical_ungrammatical_diff'].idxmax(), 'step']}")
