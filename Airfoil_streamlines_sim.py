"""
===============================================================================
SIMULATION 5: 2D AIRFOIL STREAMLINES & VELOCITY VECTOR FIELD
File: simulation_5_streamlines.py
Author: Paxton B.
Date: October 2026

Description:
  Computes and visualizes precise two-dimensional potential flow streamlines
  and velocity magnitude heatmaps around a cambered airfoil using potential
  flow theory (superposition with conformal Joukowsky mapping).
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# =============================================================================
# 1. PHYSICAL CONSTANTS & FLOWFIELD DOMAIN
# =============================================================================
V_INF: float = 30.0  # Free-stream velocity (m/s)
ALPHA_DEG: float = 7.0  # Angle of attack \alpha (degrees)
ALPHA_RAD: float = np.radians(ALPHA_DEG)

# Joukowsky Conformal Cylinder Parameters
R_CYL: float = 1.0  # Cylinder radius
X_CENTER: float = -0.08  # Thickness offset
Y_CENTER: float = 0.08  # Camber offset
C_VAL: float = 0.90  # Focus transformation constant

# Zero-lift angle calculation & Circulation strength (\Gamma)
BETA: float = np.arctan2(Y_CENTER, R_CYL - abs(X_CENTER))
GAMMA: float = 4.0 * np.pi * R_CYL * V_INF * np.sin(ALPHA_RAD + BETA)

# Computational Mesh Grid
X_MIN, X_MAX = -2.2, 2.2
Y_MIN, Y_MAX = -1.5, 1.5
NUM_GRID: int = 450

x_vec = np.linspace(X_MIN, X_MAX, NUM_GRID)
y_vec = np.linspace(Y_MIN, Y_MAX, NUM_GRID)
X, Y = np.meshgrid(x_vec, y_vec)


# =============================================================================
# 2. AIRFOIL SURFACE GEOMETRY GENERATION
# =============================================================================
def generate_airfoil_geometry(num_points: int = 500) -> tuple:
    """Generates precise 2D airfoil surface boundary points."""
    theta = np.linspace(0, 2 * np.pi, num_points)
    z_circle = (X_CENTER + R_CYL * np.cos(theta)) + 1j * (Y_CENTER + R_CYL * np.sin(theta))

    # Joukowsky Transformation: w = z + c^2 / z
    w_airfoil = z_circle + (C_VAL ** 2) / z_circle

    # Rotate by angle of attack
    w_rotated = w_airfoil * np.exp(-1j * ALPHA_RAD)

    return np.real(w_rotated), np.imag(w_rotated)


airfoil_x, airfoil_y = generate_airfoil_geometry()


# =============================================================================
# 3. STREAM FUNCTION (\Psi) & COMPLEX VELOCITY FIELD ENGINE
# =============================================================================
def compute_flowfield(X: np.ndarray, Y: np.ndarray, af_x: np.ndarray, af_y: np.ndarray) -> tuple:
    """Computes exact stream function and velocity field in physical z-plane."""
    Z_phys = (X + 1j * Y) * np.exp(1j * ALPHA_RAD)  # Un-rotate mesh by \alpha

    # Inverse Joukowsky transform: \zeta = (z + \sqrt{z^2 - 4c^2}) / 2
    # Choosing principal branch carefully
    sqrt_term = np.sqrt(Z_phys ** 2 - 4 * (C_VAL ** 2))
    zeta_1 = (Z_phys + sqrt_term) / 2.0
    zeta_2 = (Z_phys - sqrt_term) / 2.0

    # Select solution outside the mapped cylinder
    zeta = np.where(np.abs(zeta_1 - (X_CENTER + 1j * Y_CENTER)) >= R_CYL * 0.95, zeta_1, zeta_2)

    # Shift origin to cylinder center
    zeta_shift = zeta - (X_CENTER + 1j * Y_CENTER)
    r = np.abs(zeta_shift)
    theta = np.angle(zeta_shift)

    # Stream Function in cylinder plane: \psi = V_\infty r \sin(\theta)(1 - R^2/r^2) + (\Gamma / 2\pi) \ln(r/R)
    psi = V_INF * (r - (R_CYL ** 2) / r) * np.sin(theta) + (GAMMA / (2.0 * np.pi)) * np.log(r / R_CYL)

    # Complex velocity in cylinder plane: dW/d\zeta = V_\infty (1 - R^2 / \zeta_{shift}^2) + i \Gamma / (2\pi \zeta_{shift})
    dW_dzeta = V_INF * (1.0 - (R_CYL ** 2) / (zeta_shift ** 2)) + 1j * GAMMA / (2.0 * np.pi * zeta_shift)

    # Mapping derivative: dz/d\zeta = 1 - c^2 / \zeta^2
    dz_dzeta = 1.0 - (C_VAL ** 2) / (zeta ** 2)

    # Velocity in rotated physical plane: W(z) = dW/dz = (dW/d\zeta) / (dz/d\zeta)
    W_phys = dW_dzeta / dz_dzeta

    # Velocity components (u, v) rotated back into wind axis coordinates
    u = np.real(W_phys * np.exp(-1j * ALPHA_RAD))
    v = -np.imag(W_phys * np.exp(-1j * ALPHA_RAD))
    vel_mag = np.sqrt(u ** 2 + v ** 2)

    # Polygon Interior Masking
    airfoil_poly = Path(np.column_stack((af_x, af_y)))
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    interior_mask = airfoil_poly.contains_points(grid_points).reshape(X.shape)

    # Apply NaN Mask inside the airfoil body
    psi[interior_mask] = np.nan
    vel_mag[interior_mask] = np.nan
    u[interior_mask] = np.nan
    v[interior_mask] = np.nan

    return psi, vel_mag, u, v


psi_grid, vel_mag_grid, U_grid, V_grid = compute_flowfield(X, Y, airfoil_x, airfoil_y)

# =============================================================================
# 4. PUBLICATION-QUALITY PLOTTING ENGINE
# =============================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)

# --- 1. Normalized Velocity Magnitude Background Heatmap (|V| / V_\infty) ---
vel_ratio = vel_mag_grid / V_INF
heatmap_levels = np.linspace(0.2, 1.7, 120)

heatmap = ax.contourf(
    X, Y, vel_ratio,
    levels=heatmap_levels,
    cmap='coolwarm',
    extend='both'
)

# Colorbar Setup
cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.02, shrink=0.85)
cbar.set_label(r'Normalized Velocity Magnitude, $\|\mathbf{V}\| / V_{\infty}$', fontsize=10, fontweight='bold',
               labelpad=10)

# --- 2. Smooth Fluid Streamlines (\Psi Contour Lines) ---
streamline_levels = np.linspace(np.nanmin(psi_grid), np.nanmax(psi_grid), 50)
ax.contour(
    X, Y, psi_grid,
    levels=streamline_levels,
    colors='black',
    linewidths=0.6,
    alpha=0.6
)

# --- 3. Airfoil Solid Geometry ---
ax.fill(airfoil_x, airfoil_y, color='#1c2833', zorder=10, label='Airfoil Geometry')
ax.plot(airfoil_x, airfoil_y, color='black', linewidth=1.2, zorder=11)

# --- 4. Annotations ---
# Upper Surface Suction Peak
ax.annotate(
    r'Suction Peak' + '\n' + r'($\|\mathbf{V}\| > V_{\infty}$, Low Static Pressure)',
    xy=(0.0, 0.32),
    xytext=(-1.4, 1.1),
    arrowprops=dict(arrowstyle='->', color='#b22222', lw=1.4, connectionstyle='arc3,rad=-0.15'),
    fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#b22222', alpha=0.95),
    zorder=15
)

# Lower Surface Stagnation Region
ax.annotate(
    r'Stagnation Region' + '\n' + r'($\mathbf{V} \approx 0$, High Static Pressure)',
    xy=(-0.82, -0.22),
    xytext=(-2.0, -1.0),
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.4, connectionstyle='arc3,rad=0.15'),
    fontsize=8.5,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#1f77b4', alpha=0.95),
    zorder=15
)

# Axes Limits and Styling
ax.set_xlabel(r'Normalized Horizontal Coordinate ($x / c$)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel(r'Normalized Vertical Coordinate ($y / c$)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title(fr'2D Airfoil Potential Flowfield & Velocity Heatmap ($\alpha = {ALPHA_DEG:.1f}^\circ$)',
             fontsize=12, fontweight='bold', pad=12)

ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.grid(False)

plt.tight_layout()

# Export Figure
output_filename = "simulation_5_streamlines.pdf"
plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
print(f"[SUCCESS] High-resolution 2D streamline plot exported as: '{output_filename}'")

plt.show()
