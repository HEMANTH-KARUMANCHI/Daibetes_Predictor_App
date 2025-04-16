# visuals.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(df):
    """Plot histograms of each feature."""
    df.hist(bins=20, figsize=(15,10), color='#4C72B0')
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    """Plot correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_confusion_matrix(cm):
    """Plot a confusion matrix."""
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
