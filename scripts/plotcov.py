import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import chi2
# -----------------------------------------------------------
# Convert quaternion (x,y,z,w) to rotation matrix
# -----------------------------------------------------------
def quat_to_rot(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
    ])
    return R

# ---- Parameters ----
mean = np.array([0.468088,0.328454,3.36584])
cov = np.array([
 [0.154353, 0.120599,  1.23585],
[0.120599, 0.101318, 0.982738],
 [1.23585, 0.982738,  10.0706],
])

quaternion = np.array([-0.00642275 , -0.00119429 , 0.0302575 , 0.706768])
rot = np.array([
    [0.912981, -0.058773,  0.403748],
[0.0267673,  0.996067,  0.084468],
 [0.407124, 0.0663104, -0.910963],
])
scales = np.sqrt(np.array([
0.0018799,
0.00399683,
   5.91561,
]))

# ---- Eigen-decomposition (shape of ellipsoid) ----
eigvals, eigvecs = np.linalg.eigh(cov)

print("eigvals")
print(eigvals)
print("eigvecs")
print(eigvecs)

# Sort eigenvalues largest→smallest
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# ---- Confidence level (probability mass inside ellipsoid) ----
alpha = 0.5  # 50% isosurface
k = chi2.ppf(alpha, df=3)  # chi-square quantile

# ---- Build ellipsoid ----
# Parametric grid
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

# Radii scale
radii = np.sqrt(eigvals * k)
xyz = np.stack([x, y, z], axis=0).reshape(3, -1)
ellipsoid = eigvecs @ np.diag(radii) @ xyz
X = ellipsoid[0, :].reshape(50, 50) + mean[0]
Y = ellipsoid[1, :].reshape(50, 50) + mean[1]
Z = ellipsoid[2, :].reshape(50, 50) + mean[2]

# ---- Plot ----
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(0,0,0, "*r")

ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                alpha=0.4, edgecolor='k', linewidth=0.3)


# -----------------------------------------------------------
# Build ellipsoid in local coordinates
# -----------------------------------------------------------

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)

x = scales[0] * np.outer(np.cos(u), np.sin(v))
y = scales[1] * np.outer(np.sin(u), np.sin(v))
z = scales[2] * np.outer(np.ones_like(u), np.cos(v))

points = np.stack([x, y, z], axis=0).reshape(3, -1)

# -----------------------------------------------------------
# Apply pose (rotation + translation)
# -----------------------------------------------------------

R = quat_to_rot(quaternion)
rotated = (R @ points).reshape(3, 50, 50)

X = rotated[0] + mean[0]
Y = rotated[1] + mean[1]
Z = rotated[2] + mean[2]

# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------


ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                alpha=0.5, edgecolor='r', linewidth=0.3)


ax.set_title("3D Multivariate Normal Distribution (Isosurface)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1,1,1])  # equal aspect ratio

plt.show()
