import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import os

# Create plots directory if it doesn't exist
output_dir = r'c:\Users\USER\Documents\Claude\Projects\LLM\brain-tumor-3class\outputs\plots'
os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_csv(r'c:\Users\USER\Documents\Claude\Projects\LLM\brain-tumor-3class\outputs\predictions\predict_new_results.csv')

# Drop empty rows if any
df = df.dropna(subset=['prediction', 'true_label'])

classes = ['GLI', 'MEN', 'MET']

# 1. Confusion Matrix
cm = confusion_matrix(df['true_label'], df['prediction'], labels=classes)
cm_norm = CM_NORM = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax1, annot_kws={"size": 14, "weight": "bold"})
ax1.set_title('Confusion Matrix (Count)', fontsize=16, pad=20)
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('True', fontsize=12)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax2, annot_kws={"size": 14, "weight": "bold"})
ax2.set_title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('True', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'new_results_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance Metrics
report = classification_report(df['true_label'], df['prediction'], labels=classes, output_dict=True)

metrics_df = pd.DataFrame({
    'Class': classes,
    'Precision': [report[c]['precision'] for c in classes],
    'Recall': [report[c]['recall'] for c in classes],
    'F1-Score': [report[c]['f1-score'] for c in classes]
})

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.25
x = np.arange(len(classes))

ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#E74C3C')
ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#3498DB')
ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#2ECC71')

ax.set_ylabel('Score')
ax.set_title('Performance Metrics by Class', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim(0, 1.1)

# Add value labels
for i in range(len(classes)):
    ax.text(i - width, metrics_df['Precision'][i] + 0.02, f"{metrics_df['Precision'][i]:.3f}", ha='center', fontweight='bold')
    ax.text(i, metrics_df['Recall'][i] + 0.02, f"{metrics_df['Recall'][i]:.3f}", ha='center', fontweight='bold')
    ax.text(i + width, metrics_df['F1-Score'][i] + 0.02, f"{metrics_df['F1-Score'][i]:.3f}", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'new_results_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Confidence Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='confidence', hue='correct', element='step', palette={True: '#2ECC71', False: '#E74C3C'}, bins=20)
plt.title('Confidence Distribution by Prediction Correctness', fontsize=16, pad=20)
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'new_results_confidence_dist.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Visualization complete. Plots saved in outputs/plots/")
