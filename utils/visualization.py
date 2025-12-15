"""
utils/visualization.py - Visualization Utilities
================================================
Creates charts and visual outputs
"""

import matplotlib.pyplot as plt
import numpy as np

from config import CLASS_NAMES


def create_probability_chart(
    probabilities: np.ndarray,
    predicted_class: int,
    model_name: str
) -> plt.Figure:
    """
    Create a bar chart showing prediction probabilities.
    
    Args:
        probabilities: Array of 4 probabilities
        predicted_class: Index of predicted class (0-3)
        model_name: Name of model for title
        
    Returns:
        Matplotlib figure
    """
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Colors for bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create bars
    bars = ax.bar(CLASS_NAMES, probabilities * 100, color=colors)
    
    # Highlight predicted class
    bars[predicted_class].set_color('#2ca02c')
    bars[predicted_class].set_edgecolor('black')
    bars[predicted_class].set_linewidth(2)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Labels
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title(f'{model_name} Predictions', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    return fig