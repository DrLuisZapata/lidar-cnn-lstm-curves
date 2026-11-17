# Redraws the result figures of the revised manuscript from the CSV tables in results/v5 and results/v6.
# Run from the repository root: python make_article_figures.py
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle
import os
R5, R6, OUT = "results/v5/", "results/v6/", "figures_article/"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 11.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})
BLUES = ["#9ecae1", "#4292c6", "#08519c"]
INK, MUTED = "#222222", "#666666"
def save(fig, n):
    fig.savefig(OUT + n + ".png", dpi=300, bbox_inches="tight"); fig.savefig(OUT + n + ".pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig: main comparison (PandaSet + KITTI) ----------
MOD = ["Constant trend", "Ridge", "MLP", "LSTM", "GRU", "CNN-LSTM", "CNN-LSTM + PIDL"]
sp = pd.read_csv(R5 + "table_statistics_pandaset.csv")
sp = sp[sp.Reference == "Persistence"].drop_duplicates(["Horizon (s)", "Comparison", "Output"])
sp["Model"] = sp.Comparison.str.replace(" vs Persistence", "", regex=False)
sk = pd.read_csv(R5 + "table_statistics_kitti.csv")
LIM = 160
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharey=True)
for r, (ds, T) in enumerate([("PandaSet (5-fold sequence-wise CV, 39 sequences)", sp), ("KITTI Road (zero-shot, 12 sequences)", sk)]):
    for c, (out, name) in enumerate([("heading rate", "Heading rate"), ("speed", "Speed")]):
        ax = axes[r, c]
        for k, h in enumerate([1.0, 2.0, 3.0]):
            s = T[(T["Horizon (s)"] == h) & (T.Output == out)].set_index("Model").reindex(MOD)
            y = np.arange(len(MOD)) + (k - 1) * 0.24
            x = s.change_pct.values; lo = s.ci_low.values; hi = s.ci_high.values
            xc = np.clip(x, -100, LIM); loc = np.clip(lo, -100, LIM); hic = np.clip(hi, -100, LIM)
            ax.errorbar(xc, y, xerr=[xc - loc, hic - xc], fmt=["o", "s", "D"][k], ms=5, color=BLUES[k],
                        ecolor=BLUES[k], elinewidth=1.4, capsize=2, label=f"{h:.0f} s horizon", zorder=3)
            for xi, yi in zip(x, y):
                if xi > LIM:
                    ax.annotate(f"+{xi:.0f}% →", (LIM, yi), xytext=(-4, 0), textcoords="offset points",
                                ha="right", va="center", fontsize=7.5, color=MUTED)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_xlim(-100, LIM)
        ax.set_yticks(np.arange(len(MOD)), MOD)
        ax.invert_yaxis()
        ax.set_title(f"{name} — {ds}", fontsize=9.5, loc="left")
        if r == 1: ax.set_xlabel("MSE change vs. persistence (%)  [negative = lower error]")
axes[0, 1].legend(loc="lower right", fontsize=8.5, frameon=True)
fig.tight_layout()
save(fig, "fig_main_comparison")

# ---------- Fig: ablation v5 and v6 ----------
VAR = ["Dynamics only", "Geometry only", "No preview", "No banking/grade",
       "Preview 10 m only", "Preview 20 m only", "Preview 30 m only", "Preview 40 m only"]
a5, a6 = pd.read_csv(R5 + "table_ablation.csv"), pd.read_csv(R6 + "table_ablation.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
COLS = {"Height-band geometry (39 seq.)": ("#0072B2", "o", a5), "Semantic road geometry (30 seq.)": ("#D55E00", "s", a6)}
for ax, out, name in zip(axes, ["heading rate", "speed"], ["Heading rate", "Speed"]):
    for k, (lab, (col, mk, A)) in enumerate(COLS.items()):
        s = A[A.Output == out].set_index("Variant").reindex(VAR)
        y = np.arange(len(VAR)) + (k - 0.5) * 0.3
        x = s["vs full (%)"].values; lo = s["vs full CI low"].values; hi = s["vs full CI high"].values
        xc = np.clip(x, -60, 120); loc = np.clip(lo, -60, 120); hic = np.clip(hi, -60, 120)
        ax.errorbar(xc, y, xerr=[xc - loc, hic - xc], fmt=mk, ms=5, color=col, elinewidth=1.4, capsize=2, label=lab, zorder=3)
        for xi, yi in zip(x, y):
            if xi > 120:
                ax.annotate(f"+{xi:.0f}% →", (120, yi), xytext=(-4, 0), textcoords="offset points", ha="right", va="center", fontsize=7.5, color=MUTED)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_xlim(-60, 120)
    ax.set_yticks(np.arange(len(VAR)), VAR); ax.invert_yaxis()
    ax.set_xlabel("MSE change vs. full-input CNN-LSTM (%)")
    ax.set_title(name, fontsize=11, loc="left")
axes[1].legend(loc="lower right", fontsize=8.5)
fig.tight_layout(); save(fig, "fig_ablation")

# ---------- Fig: geometry validation ----------
v5, v6 = pd.read_csv(R5 + "table_preview_validation.csv"), pd.read_csv(R6 + "table_preview_validation.csv")
src = [("PandaSet, height band", v5[v5.Dataset == "PandaSet"], "#0072B2", "o"),
       ("PandaSet, semantic road", v6[v6.Dataset == "PandaSet (semantic road)"], "#D55E00", "s"),
       ("KITTI, height band", v5[v5.Dataset == "KITTI"], "#009E73", "D")]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
for k, (lab, d, col, mk) in enumerate(src):
    d = d[d["Distance (m)"].isin(["20", "30", "40"])]
    x = np.arange(3) + (k - 1) * 0.22
    ax.bar(x, d["r shape"].astype(float).values, 0.2, color=col, label=lab, zorder=3)
ax.axhline(0.6, color=INK, ls="--", lw=0.9); ax.text(2.45, 0.62, "r = 0.6 criterion", ha="right", fontsize=8, color=MUTED)
ax.set_xticks(range(3), ["20 m", "30 m", "40 m"]); ax.set_ylim(-0.1, 1.0)
ax.set_ylabel("Pearson r (LiDAR vs. future path)"); ax.set_title("Road shape $y_d - y_{10}$", fontsize=9.5, loc="left")
ax.legend(fontsize=8, loc="upper left")
ax = axes[1]
vals = [float(d[d["Distance (m)"] == "curvature"]["r offset"].iloc[0]) for _, d, _, _ in src]
ax.bar(range(3), vals, 0.6, color=[s[2] for s in src], zorder=3)
ax.set_xticks(range(3), ["PandaSet\nheight", "PandaSet\nsemantic", "KITTI\nheight"], fontsize=8.5)
ax.axhline(0, color=INK, lw=0.9); ax.set_ylim(-0.3, 1.0)
ax.set_ylabel("Spearman ρ"); ax.set_title("Road curvature κ", fontsize=9.5, loc="left")
fig.tight_layout(); save(fig, "fig_geometry_validation")

# ---------- Fig: force diagram ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]; ax.set_aspect("equal"); ax.axis("off")
th = np.radians(14)
L = 6.0
road = Polygon([[-L, -L*np.tan(th)*0], [L, L*np.tan(th)], [L, -1.2], [-L, -1.2]], closed=True, fc="#e6e6e6", ec="#888888")
# road surface line from (-L,-L tan th) to (L, L tan th)
p0, p1 = np.array([-L, -L*np.tan(th)]), np.array([L, L*np.tan(th)])
ax.add_patch(Polygon([p0, p1, [L, -2.2], [-L, -2.2]], fc="#e8e8e8", ec="#888888", lw=1))
t = (p1 - p0) / np.linalg.norm(p1 - p0); n = np.array([-t[1], t[0]])
c = np.array([0.0, 0.0]) + n * 0.75
car = np.array([c - t*1.5 - n*0.55, c + t*1.5 - n*0.55, c + t*1.5 + n*0.55, c - t*1.5 + n*0.55])
ax.add_patch(Polygon(car, fc="#cfe2f3", ec="#0072B2", lw=1.4))
G = c
def arrow(start, vec, col, lab, off=(0, 0), ha="left"):
    ax.add_patch(FancyArrowPatch(start, start + vec, arrowstyle="-|>", mutation_scale=14, lw=1.8, color=col))
    ax.text(*(start + vec + np.array(off)), lab, color=col, fontsize=10.5, ha=ha, va="center")
arrow(G, np.array([0, -2.6]), "#333333", "$W = mg$", (0.15, -0.1))
arrow(G - n*0.55, n*2.9, "#0072B2", "$N$", (0.1, 0.2))
arrow(G - n*0.55, -t*2.4, "#D55E00", "$F_f = \\mu N$", (-0.1, -0.35), ha="right")
arrow(G, np.array([-2.9, 0]), "#009E73", "$a_n = v^2/\\rho$\n(toward center)", (-0.1, 0.55), ha="right")
ax.plot([-L+0.2, -L+2.2], [p0[1], p0[1]], color=MUTED, lw=0.8, ls="--")
from matplotlib.patches import Arc
ax.add_patch(Arc(p0, 3.6, 3.6, theta1=0, theta2=np.degrees(th), color=INK, lw=1))
ax.text(p0[0]+1.95, p0[1]+0.2, "$\\theta$", fontsize=11, color=INK)
ax.text(-L+0.1, 3.4, "Center of curvature ←", fontsize=9, color=MUTED)
ax.text(L-0.2, -1.9, "outer edge", fontsize=8.5, color=MUTED, ha="right"); ax.text(-L+0.2, -1.9, "inner edge", fontsize=8.5, color=MUTED)
ax.set_xlim(-L-0.3, L+0.3); ax.set_ylim(-3.6, 4.2)
ax.set_title("(a) Cross-section of a banked curve at the sliding limit", fontsize=10, loc="left")
ax = axes[1]; ax.set_aspect("equal"); ax.axis("off")
rho = 4.0
a = np.linspace(np.radians(-75), np.radians(10), 100)
for off, ls in [(-0.6, "-"), (0.6, "-")]:
    ax.plot((rho+off)*np.cos(a), (rho+off)*np.sin(a), color="#888888", lw=1.2, ls=ls)
ax.plot(rho*np.cos(a), rho*np.sin(a), color="#0072B2", lw=2, label="vehicle path")
ax.plot([0], [0], "k+", ms=8); ax.text(0.15, 0.15, "O", fontsize=10)
ang = np.radians(-35); P = rho*np.array([np.cos(ang), np.sin(ang)])
ax.plot([0, P[0]], [0, P[1]], color=MUTED, lw=0.9, ls="--"); ax.text(P[0]/2-1.35, P[1]/2-0.2, "$\\rho = 1/|\\kappa|$", fontsize=10)
ax.add_patch(Circle(P, 0.18, color="#0072B2"))
tg = np.array([-np.sin(ang), np.cos(ang)])
ax.add_patch(FancyArrowPatch(P, P + tg*1.6, arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#0072B2"))
ax.text(*(P + tg*1.6 + np.array([0.1, 0])), "$v$", color="#0072B2", fontsize=11)
ax.add_patch(FancyArrowPatch(P, P - P/rho*1.1, arrowstyle="-|>", mutation_scale=14, lw=1.8, color="#009E73"))
ax.text(*(P - P/rho*1.1 + np.array([0.05, 0.3])), "$a_n$", color="#009E73", fontsize=11)
# LiDAR preview points
for d, col in zip([10, 20, 30, 40], ["#E69F00"]*4):
    aa = ang + d/40*np.radians(38)
    Q = rho*np.array([np.cos(aa), np.sin(aa)]); ax.plot(*Q, "o", color="#E69F00", ms=5)
ax.text(rho*np.cos(np.radians(8))+0.2, rho*np.sin(np.radians(8))+0.1, "LiDAR preview\nat 10–40 m", fontsize=8.5, color="#a66f00")
ax.text(1.0, -4.9, "$\\dot\\psi = v/\\rho$  (heading rate)", fontsize=9.5, color=INK)
ax.set_xlim(-0.6, 6.2); ax.set_ylim(-5.2, 1.6)
ax.set_title("(b) Plan view: curvature, speed and heading rate", fontsize=10, loc="left")
fig.tight_layout(); save(fig, "fig_force_diagram")
print("ok")

# ---------- Fig: physics sensitivity ----------
P = pd.read_csv(R5 + "table_physics_sensitivity.csv")
LC = {0.0: ("#666666", "o", "no physics term ($\\lambda$ = 0)"), 0.1: ("#56B4E9", "s", "$\\lambda$ = 0.1"),
      1.0: ("#0072B2", "D", "$\\lambda$ = 1"), 10.0: ("#D55E00", "^", "$\\lambda$ = 10")}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
for lam, (col, mk, lab) in LC.items():
    s = P[P["lambda"] == lam].sort_values("mu")
    axes[0].plot(s.mu, s["MSE speed"], marker=mk, color=col, lw=1.6, ms=6, label=lab)
    axes[1].plot(s.mu, s["predicted above v_max (%)"], marker=mk, color=col, lw=1.6, ms=6, label=lab)
    if lam > 0:
        axes[2].plot(s.mu, s["train penalty active (%)"], marker=mk, color=col, lw=1.6, ms=6, label=lab)
s0 = P[P["lambda"] == 0].sort_values("mu")
axes[1].plot(s0.mu, s0["actual above v_max (%)"], ls="--", color=INK, lw=1.2, label="actual future speed")
axes[0].set_yscale("log"); axes[0].set_ylabel("Speed MSE (km/h)$^2$, log scale")
axes[1].set_ylabel("Speeds above $v_{max}$ (%)"); axes[2].set_ylabel("Training samples with active penalty (%)")
for ax, t in zip(axes, ["(a) Test accuracy", "(b) Constraint violations (test)", "(c) Penalty activation"]):
    ax.set_xlabel("Friction coefficient $\\mu$"); ax.set_xticks([0.1, 0.2, 0.3, 0.5, 0.7]); ax.invert_xaxis()
    ax.set_title(t, fontsize=10, loc="left")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
fig.tight_layout(); save(fig, "fig_physics_sensitivity")

# ---------- Fig: curve strata ----------
E = pd.read_csv(R5 + "table_curve_strata.csv")
EST = ["Straight (<1 deg/s)", "Gentle curve (1-3 deg/s)", "Curve (>=3 deg/s)"]
LBL = ["Straight\n< 1 deg/s\n(1664 samples, 38 seq.)", "Gentle curve\n1–3 deg/s\n(297 samples, 19 seq.)", "Curve\n≥ 3 deg/s\n(26 samples, 2 seq.)"]
SHOW = [("Ridge", "#0072B2", "o"), ("GRU", "#009E73", "s"), ("LSTM", "#56B4E9", "D"), ("CNN-LSTM + PIDL", "#D55E00", "^")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
for ax, out, name in zip(axes, ["heading rate", "speed"], ["Heading rate", "Speed"]):
    for j, (m, col, mk) in enumerate(SHOW):
        s = E[E.Model == m].set_index("Stratum").reindex(EST)
        x = np.arange(3) + (j - 1.5) * 0.16
        v, lo, hi = s[f"{out} vs persistence (%)"], s[f"{out} CI low"], s[f"{out} CI high"]
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt=mk, color=col, ms=6, capsize=2, elinewidth=1.3, label=m)
    ax.axhline(0, color=INK, lw=0.9); ax.set_xticks(range(3), LBL, fontsize=8.5)
    ax.set_ylabel("MSE change vs. persistence (%)"); ax.set_title(f"{name} (1 s horizon; negative = lower error)", fontsize=10, loc="left")
axes[0].legend(fontsize=8.5, loc="upper left")
fig.tight_layout(); save(fig, "fig_curve_strata")
print("ok2")
