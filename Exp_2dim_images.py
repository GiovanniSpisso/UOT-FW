import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FW_2dim_p2 import PW_FW_dim2_p2


def plot_distributions(mu, nu, filename="plot.png", max_iter=10000, delta=0.001, eps=0.001):
    """
    Plot 4 subplots showing mu, nu, x_marg*mu, and y_marg*nu and save to file.
    
    Parameters:
    -----------
    mu, nu : ndarray
        Distribution arrays
    filename : str
        Filename to save the plot (relative to Plots/images/)
    max_iter : int
        Maximum iterations for the algorithm
    delta : float
        Tolerance to stop the gap
    eps : float
        Tolerance for calculating the descent direction
    """
    # Parameters to set
    M = 2 * (np.sum(mu) + np.sum(nu))  # upper bound for delimiting the generalized simplex
    
    pi, grad, x_marg, y_marg = PW_FW_dim2_p2(mu, nu, M,
                                             max_iter=max_iter, delta=delta, eps=eps)
    
    vmin = min(mu.min(), nu.min(), (x_marg*mu).min(), (y_marg*nu).min())
    vmax = max(mu.max(), nu.max(), (x_marg*mu).max(), (y_marg*nu).max())
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Plot mu
    im1 = axes[0, 0].imshow(mu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("mu")
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot nu
    im2 = axes[0, 1].imshow(nu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("nu")
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot x_marg*mu
    im3 = axes[1, 0].imshow(x_marg*mu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("x_marg")
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot y_marg*nu
    im4 = axes[1, 1].imshow(y_marg*nu, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("y_marg")
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "Plots/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {filepath}")

    plt.show()
    plt.close()


## EXP: from chessboard to rotated chessboard (64 * 64)
#df1 = pd.read_csv("DOTmark_1.0/Shapes/data64_1003.csv", header=None)
#df2 = pd.read_csv("DOTmark_1.0/Shapes/data64_1003.csv", header=None)
#
#mu = df1.values/100000
#nu = df2.values/100000
#nu = np.rot90(mu, k=1)
#
## Call the function
#plot_distributions(mu, nu, filename="chessboard_64.png")


# EXP: from photographer to jet (512 * 512)
df1 = pd.read_csv("DOTmark_1.0/data512_1001.csv", header=None)
df2 = pd.read_csv("DOTmark_1.0/data512_1002.csv", header=None)

mu = df1.values/100000
nu = df2.values/100000

# Call the function
plot_distributions(mu, nu, filename="photographer_jet_512.png")