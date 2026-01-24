import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Function to plot n-sigma ellipse
def plot_sigma_ellipse(mean, cov, ax, n_std=1.0, facecolor='none', edgecolor='red', **kwargs):
    """
    Plots an n-sigma ellipse for a given mean and covariance matrix.
    """
    # Calculate eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(cov)
    # Calculate angle of rotation
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    # Width and height are 2 * n_std * sqrt(val)
    width, height = 2 * n_std * np.sqrt(vals)
    ellip = Ellipse(xy=mean, width=width, height=height, angle=theta, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    ax.add_artist(ellip)
    return ellip

# Example usage:
mean = np.array([0, 1])
cov = np.array([[4.93827e-06, -4.93827e-05], [-4.93827e-05, 0.000987654]]) # Example covariance

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
ax.scatter(mean[0], mean[1], color='blue', marker='o')
plot_sigma_ellipse(mean, cov, ax, n_std=1.0) # Plots 1-sigma ellipse

# Example usage:
mean = np.array([1, 1])
cov = np.array([[0.000893827,0.000938272 ], [0.000938272, 0.000987654]]) # Example covariance

fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
ax.scatter(mean[0], mean[1], color='blue', marker='o')
plot_sigma_ellipse(mean, cov, ax, n_std=1.0) # Plots 1-sigma ellipse


ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'1-Sigma Ellipse for $\mu={mean}$, $\\Sigma={cov.tolist()}$')
ax.grid(True)
plt.show()

