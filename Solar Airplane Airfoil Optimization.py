"""
Solar UAV: 3-station airfoil optimization (root / mid / tip) on a rectangular wing.

- Objective : minimize power required (D * V)  ->  maximum endurance at fixed W, V
- Aero      : NonlinearLiftingLine ONLY (NeuralFoil sections + 3D vortex coupling).
              Used inside the optimizer, for the alpha sweep, and for the spanwise lift plot.
- Panel     : a shared chordwise interval [xs, xe] on the upper surface of every
              station must satisfy |curvature| <= kappa_max and have an arc length
              >= L_panel (same criterion as the curve-calculator notebook).
"""
import aerosandbox as asb
import aerosandbox.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.special import comb
from scipy.interpolate import CubicSpline


def f(v):
    """Any AeroSandbox/CasADi/numpy result -> plain Python float."""
    return float(onp.asarray(v).ravel()[0])


# ----------------------------------------------------------------------------
# Constants / assumptions
# ----------------------------------------------------------------------------
W, V       = 70, 13.5      # weight [N], airspeed [m/s]
n_w        = 8             # Kulfan weights per surface
L_panel    = 0.162         # solar panel length (arc length on upper surface) [m]
kappa_max  = 8.0           # max allowed curvature [1/m]  (0.08 1/cm)
arc_margin = 1.01          # small margin on panel arc length (polyline vs spline)
kappa_margin = 0.97        # small safety margin on curvature (grid/spline mismatch)
rear_excl  = 0.25          # no panel in the rear 25% of chord
M          = 40            # grid points across the panel interval
TE         = 0.002         # trailing-edge thickness (x/c)
x_spar     = 0.20          # chordwise location of the thickness constraint (x/c)
t_spar_min = 0.085         # minimum thickness at x_spar (fraction of chord)
t_max_c    = 0.14          # maximum allowed airfoil thickness (fraction of chord)
R_le_min   = 0.005         # min LE radius, fraction of chord
w0_min     = (2 * R_le_min) ** 0.5   # Kulfan: R_le = w0^2 / 2 on each surface
eta        = onp.array([0.0, 0.5, 1.0])   # station locations, fraction of semispan
station_names = ["Root", "Midspan", "Tip"]
LL_RES     = 3             # spanwise panels per wing section in the lifting line

# ----------------------------------------------------------------------------
# Optimization problem
# ----------------------------------------------------------------------------
opti  = asb.Opti()
span  = opti.variable(init_guess=2.5, lower_bound=2,    upper_bound=5)
chord = opti.variable(init_guess=0.5, lower_bound=0.25, upper_bound=0.6) # [m]
alpha = opti.variable(init_guess=1,   lower_bound=-1,   upper_bound=3)

# Shared panel interval (fractions of chord)
xs = opti.variable(init_guess=0.10, lower_bound=0.05, upper_bound=0.5)
xe = opti.variable(init_guess=0.70, lower_bound=0.2,  upper_bound=1 - rear_excl)
opti.subject_to((xe - xs)*chord >= 0.168)

init = asb.Airfoil("s7055").to_kulfan_airfoil(n_weights_per_side=n_w)


def cst(x, w, le, te_half):
    """Kulfan/CST surface y(x) for x/c in (0, 1)."""
    n = n_w - 1
    C = x**0.5 * (1 - x)
    S = sum(w[i] * comb(n, i) * x**i * (1 - x)**(n - i) for i in range(n_w))
    return C * S + le * x * (1 - x)**(n + 0.5) + x * te_half


# Panel grid has M points spanning [xs, xe], plus one "ghost" point beyond each end so the
# curvature stencil can be evaluated AT xs and xe (curvature peaks at the forward end).
xi_ext = onp.linspace(-1 / (M - 1), 1 + 1 / (M - 1), M + 2)
xg_ext = xs + (xe - xs) * xi_ext
h      = (xe - xs) / (M - 1)
x_chk  = onp.linspace(0.02, 0.98, 40)


def make_station(name):
    uw = opti.variable(init_guess=init.upper_weights,
                       lower_bound=init.upper_weights - 0.25,
                       upper_bound=init.upper_weights + 0.25)
    lw = opti.variable(init_guess=init.lower_weights,
                       lower_bound=init.lower_weights - 0.25,
                       upper_bound=init.lower_weights + 0.25)
    le = opti.variable(init_guess=init.leading_edge_weight,
                       lower_bound=-0.5, upper_bound=1.0)

    # --- solar panel criterion: curvature + arc length on [xs, xe] ---
    yu_e = cst(xg_ext, uw, le, TE / 2)                  # M+2 points (with ghost points)
    y1   = (yu_e[2:] - yu_e[:-2]) / (2 * h)             # M points: exactly xs ... xe
    y2   = (yu_e[:-2] - 2 * yu_e[1:-1] + yu_e[2:]) / h**2
    k_nd = y2 / (1 + y1**2) ** 1.5                      # per unit chord
    k_lim = kappa_margin * kappa_max * chord            # kappa_phys = k_nd / chord
    opti.subject_to([k_nd <= k_lim, k_nd >= -k_lim])
    yu   = yu_e[1:-1]                                   # the M points on the panel

    arc_nd = np.sum(np.sqrt(h**2 + (yu[1:] - yu[:-1])**2))
    opti.subject_to(arc_nd * chord >= arc_margin * L_panel)

    # --- thickness constraints ---
    t = cst(x_chk, uw, le, TE / 2) - cst(x_chk, lw, le, -TE / 2)
    opti.subject_to([t >= 0.005, np.max(t) <= t_max_c])   # no crossover, max thickness cap
    t_spar = cst(x_spar, uw, le, TE / 2) - cst(x_spar, lw, le, -TE / 2)
    opti.subject_to(t_spar >= t_spar_min)                 # min thickness at x_spar

    # --- leading-edge radius: keeps the nose round (R_le = w0^2/2 per surface) ---
    opti.subject_to([uw[0] >= w0_min, lw[0] <= -w0_min])

    return asb.KulfanAirfoil(name=name, upper_weights=uw, lower_weights=lw,
                             leading_edge_weight=le, TE_thickness=TE)


afs = [make_station(f"af_{i}") for i in range(len(eta))]

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[asb.WingXSec(xyz_le=[0, eta[i] * span / 2, 0], chord=chord, airfoil=afs[i])
           for i in range(len(eta))],
)
airplane = asb.Airplane(name="Solar UAV", wings=[wing], xyz_ref=[0.25 * chord, 0, 0])

# ---- Nonlinear lifting line, embedded in OUR Opti ----
#  * opti=opti        : the analysis adds its variables/equations to this problem
#  * run(solve=False) : do NOT solve inside run(); return the residuals instead,
#                       and constrain them to zero in the outer problem
#  * symmetry shortcut is not implemented for this analysis -> leave it False
ll = asb.NonlinearLiftingLine(
    airplane=airplane,
    op_point=asb.OperatingPoint(velocity=V, alpha=alpha),
    spanwise_resolution=LL_RES,
    run_symmetric_if_possible=False,
    opti=opti,
)
aero = ll.run(solve=False)
opti.subject_to(aero["residuals"] == 0)

# Initial guess for the circulation: roughly uniform, Gamma ~ L / (rho * V * b)
opti.set_initial(ll.vortex_strengths, W / (1.225 * V * 2.5) * onp.ones(ll.n_panels))

opti.subject_to(aero["L"] == W)
opti.subject_to(aero["CL"] <= 1.2)
opti.minimize(aero["D"] * V)

sol = opti.solve(max_iter=300)

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
span_v, chord_v, alpha_v = f(sol(span)), f(sol(chord)), f(sol(alpha))
xs_v, xe_v = f(sol(xs)), f(sol(xe))
L_v, D_v   = f(sol(aero["L"])), f(sol(aero["D"]))
CL_v, CD_v = f(sol(aero["CL"])), f(sol(aero["CD"]))
S_v        = span_v * chord_v

print("\n================ RESULTS ================")
print(f"L/D                : {L_v / D_v:.2f}")
print(f"Lift / Drag        : {L_v:.2f} N / {D_v:.3f} N")
print(f"CL / CD            : {CL_v:.3f} / {CD_v:.5f}")
print(f"CDi / CDp          : {f(sol(aero['CDi'])):.5f} / {f(sol(aero['CDp'])):.5f}")
print(f"Endurance CL^1.5/CD: {CL_v ** 1.5 / CD_v:.2f}")
print(f"Power required     : {D_v * V:.2f} W")
print(f"Alpha              : {alpha_v:.2f} deg")
print("--- Planform (rectangular) ---")
print(f"Span               : {span_v:.3f} m")
print(f"Chord              : {chord_v:.3f} m  ({chord_v * 100:.1f} cm)")
print(f"Area               : {S_v:.3f} m^2")
print(f"Aspect ratio       : {span_v / chord_v:.2f}")
print("--- Solar panel interval ---")
print(f"x/c from {xs_v:.3f} to {xe_v:.3f}  "
      f"({xs_v * chord_v * 100:.1f} to {xe_v * chord_v * 100:.1f} cm from LE)")
print(f"Panel strip planform area: {(xe_v - xs_v) * chord_v * span_v:.3f} m^2")


def check_panel(af_opt, chord_m, xs_frac, xe_frac):
    """Notebook-style spline check: max |curvature| and arc length in the panel."""
    coords = af_opt.coordinates * chord_m
    i_le = int(onp.argmin(coords[:, 0]))
    up = coords[: i_le + 1][::-1]                          # LE -> TE, upper surface
    seg = onp.linalg.norm(onp.diff(up, axis=0), axis=1)
    s = onp.concatenate(([0], onp.cumsum(seg)))
    cs = CubicSpline(s, up, axis=0)
    ss = onp.linspace(0, s[-1], 4000)
    p, d1, d2 = cs(ss), cs(ss, 1), cs(ss, 2)
    kappa = onp.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / onp.linalg.norm(d1, axis=1) ** 3
    mask = (p[:, 0] >= xs_frac * chord_m) & (p[:, 0] <= xe_frac * chord_m)
    arc = onp.sum(onp.linalg.norm(onp.diff(p[mask], axis=0), axis=1))
    return kappa[mask].max(), arc


af_opts = [sol(a) for a in afs]

print("--- Thickness per station ---")
_xt = onp.linspace(0.02, 0.98, 400)
for name, a in zip(station_names, af_opts):
    t_all = cst(_xt, a.upper_weights, a.leading_edge_weight, TE / 2) \
          - cst(_xt, a.lower_weights, a.leading_edge_weight, -TE / 2)
    t_sp = cst(x_spar, a.upper_weights, a.leading_edge_weight, TE / 2) \
         - cst(x_spar, a.lower_weights, a.leading_edge_weight, -TE / 2)
    print(f"{name:8s}: t/c at x/c={x_spar:.2f}: {f(t_sp):.4f} (min {t_spar_min}), "
          f"max t/c: {f(onp.max(t_all)):.4f} at x/c={_xt[int(onp.argmax(t_all))]:.2f}")

print("--- Leading-edge radius per station (fraction of chord) ---")
for name, a in zip(station_names, af_opts):
    print(f"{name:8s}: R_le/c = {f(a.LE_radius()):.4f}  (min {R_le_min})")

print("--- Panel check per station (spline method, like the notebook) ---")
for name, a in zip(station_names, af_opts):
    k, arc = check_panel(a, chord_v, xs_v, xe_v)
    print(f"{name:8s}: max curvature {k:.2f} 1/m (limit {kappa_max}), "
          f"panel arc length {arc * 100:.2f} cm (need {L_panel * 100:.1f})")

# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
# 1) Airfoil sections (scaled to physical size, cm)
fig1, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax, name, a, e in zip(axes, station_names, af_opts, eta):
    c = a.coordinates * chord_v * 100
    ax.plot(c[:, 0], c[:, 1], "k-", lw=1.8)
    i_le = int(onp.argmin(c[:, 0]))
    up = c[: i_le + 1]
    m = (up[:, 0] >= xs_v * chord_v * 100) & (up[:, 0] <= xe_v * chord_v * 100)
    ax.plot(up[m, 0], up[m, 1], color="crimson", lw=4, label="Solar panel region")
    ax.axvline((1 - rear_excl) * chord_v * 100, color="orange", ls="--", lw=1,
               label=f"Rear exclusion ({1 - rear_excl:.0%} chord)")
    ax.set_aspect("equal")
    ax.set_ylabel("y [cm]")
    ax.set_title(f"{name} (y = {e * span_v / 2:.2f} m)")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("x [cm]")
fig1.suptitle("Optimized airfoil sections", fontsize=14)
fig1.tight_layout()
fig1.savefig("airfoil_sections.png", dpi=150)

# 2) Top-down planform (leading edge at top, x chordwise pointing down)
fig2, ax = plt.subplots(figsize=(11, 4.5))
half = span_v / 2
for sign in (-1, 1):
    y0 = 0 if sign > 0 else -half
    ax.add_patch(Rectangle((y0, 0), half, chord_v, fc="lightsteelblue", ec="k", lw=1.5))
    ax.add_patch(Rectangle((y0, xs_v * chord_v), half, (xe_v - xs_v) * chord_v,
                           fc="crimson", alpha=0.6, ec="none",
                           label="Solar panel strip" if sign > 0 else None))
for e, name in zip(eta, station_names):
    for sign in (-1, 1):
        ax.axvline(sign * e * half, color="k", ls=":", lw=1)
    ax.text(e * half, -0.03 * chord_v, name, ha="center", va="bottom", fontsize=9)
ax.axhline(0.25 * chord_v, color="gray", ls="--", lw=1, label="Quarter chord")
ax.set_xlim(-half * 1.05, half * 1.05)
ax.set_ylim(chord_v * 1.15, -chord_v * 0.25)           # invert: LE at top
ax.set_aspect("equal")
ax.set_xlabel("Spanwise y [m]")
ax.set_ylabel("Chordwise x [m]")
ax.set_title(f"Planform: b = {span_v:.2f} m, c = {chord_v:.3f} m, "
             f"AR = {span_v / chord_v:.1f}, S = {S_v:.3f} m$^2$")
ax.legend(loc="lower right")
fig2.tight_layout()
fig2.savefig("planform_top_view.png", dpi=150)

# 3) Performance curves -- all from NonlinearLiftingLine
airplane_opt = sol(airplane)                       # airplane with numeric geometry
alphas = onp.linspace(-3, 10, 14)
CL_s, CD_s, L_s = [], [], []
for a_deg in alphas:                               # standalone solve at each alpha
    r = asb.NonlinearLiftingLine(
        airplane=airplane_opt,
        op_point=asb.OperatingPoint(velocity=V, alpha=float(a_deg)),
        spanwise_resolution=LL_RES,
        run_symmetric_if_possible=False,
    ).run()
    CL_s.append(f(r["CL"])); CD_s.append(f(r["CD"])); L_s.append(f(r["L"]))
CL_s, CD_s, L_s = map(onp.array, (CL_s, CD_s, L_s))
endurance = onp.where(CL_s > 0, onp.abs(CL_s) ** 1.5 / CD_s, onp.nan)   # CL^1.5 / CD

# Spanwise lift distribution straight from the lifting line (Kutta-Joukowski: l' = rho V Gamma)
rho = 1.225
gamma = onp.asarray(sol(ll.vortex_strengths)).ravel()
y_panel = onp.asarray(sol(ll.vortex_centers))[:, 1]
order = onp.argsort(y_panel)
y_sorted = y_panel[order]
lprime = rho * V * gamma[order]                                 # lift per unit span [N/m]
_trap = getattr(onp, "trapezoid", None) or onp.trapz   # NumPy 2 renamed trapz
L_ll = _trap(lprime, y_sorted)                              # integrated lift [N]

y_ell = onp.linspace(-span_v / 2, span_v / 2, 400)
l_ell = (4 * L_ll / (onp.pi * span_v)) * onp.sqrt(1 - (2 * y_ell / span_v) ** 2)

fig3, (axE, axL, axD) = plt.subplots(1, 3, figsize=(18, 5))

axE.plot(alphas, endurance, "b-o", ms=3)
axE.axvline(alpha_v, color="crimson", ls="--", label=f"Optimum alpha = {alpha_v:.2f} deg")
axE.plot([alpha_v], [CL_v ** 1.5 / CD_v], "r*", ms=12, label=f"Optimizer: {CL_v ** 1.5 / CD_v:.2f}")
axE.set_xlabel("Angle of attack [deg]")
axE.set_ylabel(r"Endurance factor $C_L^{1.5}/C_D$")
axE.set_title("Endurance factor vs alpha (lifting line)")
axE.grid(alpha=0.3); axE.legend()

axL.plot(alphas, L_s, "b-o", ms=3, label="Lift")
axL.axhline(W, color="gray", ls=":", label=f"Weight = {W} N")
axL.axvline(alpha_v, color="crimson", ls="--", label="Optimum alpha")
axL.set_xlabel("Angle of attack [deg]")
axL.set_ylabel("Lift force [N]")
axL.set_title(f"Lift vs alpha (V = {V} m/s)")
axL.grid(alpha=0.3); axL.legend()

axD.plot(y_sorted, lprime, "b-o", ms=3, label=f"Lifting line (integrated L = {L_ll:.1f} N)")
axD.plot(y_ell, l_ell, "k--", label="Ideal elliptical (same L, span)")
axD.set_xlabel("Spanwise y [m]")
axD.set_ylabel("Lift per unit span [N/m]")
axD.set_title("Spanwise lift distribution")
axD.grid(alpha=0.3); axD.legend()

fig3.tight_layout()
fig3.savefig("performance_curves.png", dpi=150)

plt.show()