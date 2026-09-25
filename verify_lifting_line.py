from pathlib import Path
import numpy as np
import af_2d

file_path = Path(r"c:/Users/Graham/Documents/VS_Code/aerodynamics-tools/airfoil-performance/s7055.csv")
alpha_deg = 1.0
alpha = np.deg2rad(alpha_deg)

b = 5.0
c_r = 0.2
lambda_t = 1.0
n_modes = 24

cl_0 = af_2d.get_C_l_0(str(file_path))
cd_0 = af_2d.get_C_d_0(str(file_path))
a0 = af_2d.get_C_l_vs_alpha(str(file_path))
alpha_l_0 = af_2d.get_alpha_l_0(str(file_path))
cd_alpha = af_2d.get_C_d_vs_alpha(str(file_path), alpha_min=0.0, alpha_max=5.0)

S = 0.5 * b * c_r * (1.0 + lambda_t)
AR = b**2 / S

theta = np.pi * np.arange(1, n_modes + 1) / (n_modes + 1)
y = -(b / 2.0) * np.cos(theta)
eta = 2.0 * y / b
chord = c_r * (1.0 - (1.0 - lambda_t) * np.abs(eta))

M = np.empty((n_modes, n_modes), dtype=float)
for j, th in enumerate(theta):
    for n in range(1, n_modes + 1):
        M[j, n - 1] = n * np.sin(n * th) / np.sin(th) + (4.0 * b / chord[j]) * np.sin(n * th)

rhs = a0 * (alpha - alpha_l_0) * np.ones(n_modes)
A = np.linalg.solve(M, rhs)
mode_numbers = np.arange(1, n_modes + 1)
spanwise_load = np.sin(np.outer(mode_numbers, theta)) @ A
cl_dist = 4.0 * b * spanwise_load / chord
CL_from_span = (1.0 / S) * np.trapezoid(chord * cl_dist, y)
CL_fourier = np.pi * AR * A[0]
CD_i = np.pi * AR * np.sum(mode_numbers * A**2)
CD_profile = cd_0 + cd_alpha * (alpha - alpha_l_0)
CD_total = CD_profile + CD_i
L_over_D = CL_from_span / CD_total

print(f"cl_0={cl_0:.4f}")
print(f"cd_0={cd_0:.4f}")
print(f"alpha_l_0_deg={np.rad2deg(alpha_l_0):.4f}")
print(f"a0={a0:.4f}")
print(f"AR={AR:.4f}")
print(f"CL_from_span={CL_from_span:.6f}")
print(f"CL_fourier={CL_fourier:.6f}")
print(f"CD_i={CD_i:.6f}")
print(f"CD_profile={CD_profile:.6f}")
print(f"CD_total={CD_total:.6f}")
print(f"L/D={L_over_D:.6f}")
print(f"A1={A[0]:.6f}")
print(np.round(A[:8], 6))
