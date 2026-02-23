import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_loss_plot(x, y, loss_title, x_label, filename):
    """
    Generates a minimalist, elegant loss function plot matching the Nixtla style.
    """
    # Set the overall seaborn style to white for a clean background
    sns.set_theme(style="white")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Define color palette based on the provided examples
    line_color = '#79AFAA'  # Muted teal/sage
    axis_color = '#A5A5B4'  # Muted purple-gray
    grid_color = '#E5E5E5'  # Light gray
    
    # Plot the loss function curve
    ax.plot(x, y, color=line_color, linewidth=2.5)
    
    # Add light gridlines behind the plot
    ax.grid(True, color=grid_color, linestyle='-', linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    
    # Draw thick, custom central axes
    ax.axhline(0, color=axis_color, linewidth=3, zorder=1)
    ax.axvline(0, color=axis_color, linewidth=3, zorder=1)
    
    # Remove all standard spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Remove tick marks and tick labels for a minimalist look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add LaTeX-formatted labels
    # Position the Loss title slightly to the right of the vertical axis
    ax.text(
        0.1, max(y), 
        loss_title, 
        fontsize=22, 
        va='center', 
        ha='left', 
        color='black'
    )
    
    # Position the X-axis label at the far right of the horizontal axis
    ax.text(
        max(x) + 0.05, -0.05 * max(y), 
        x_label, 
        fontsize=24, 
        va='top', 
        ha='left', 
        color='black'
    )
    
    # Adjust layout and save with a transparent background
    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight')
    plt.close()

# --- Generate the Specific Plots ---

# Define the range of the error/difference (centered around 0)
x = np.linspace(-2.5, 2.5, 500)

# 1. FFT Mean Absolute Error (V-shape)
# The error is the absolute difference in magnitudes
y_mae = np.abs(x)
generate_loss_plot(
    x=x, 
    y=y_mae, 
    loss_title=r'$FFTMAE(y, \hat{y})$', 
    x_label=r'$|F(\hat{y})_{\tau}|$', 
    filename='fft_mae_loss.png'
)

# 2. FFT Mean Squared Error (Parabola/U-shape)
# The error is the squared difference in magnitudes
y_mse = x**2
generate_loss_plot(
    x=x, 
    y=y_mse, 
    loss_title=r'$FFTMSE(y, \hat{y})$', 
    x_label=r'$|F(\hat{y})_{\tau}|$', 
    filename='fft_mse_loss.png'
)

# 3. FFT Root Mean Squared Error
# For a single point difference, RMSE scales linearly like MAE (V-shape), 
# but it's good to have it exported for the docs completeness.
y_rmse = np.sqrt(x**2) 
generate_loss_plot(
    x=x, 
    y=y_rmse, 
    loss_title=r'$FFTRMSE(y, \hat{y})$', 
    x_label=r'$|F(\hat{y})_{\tau}|$', 
    filename='fft_rmse_loss.png'
)

print("Loss function visualizations successfully generated!")