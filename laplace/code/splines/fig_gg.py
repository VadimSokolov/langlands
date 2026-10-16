"""Figure fig:lapspline (figs/lap_spline.pdf) of Chapter 23 (ch_laplace).

The coauthor's script, kept unchanged as fig_gg.original.py, edited for the book on 2026-10-13 (decision 9 of
misc/lap-apply-brief.md): the book's palette and serif type; drawn 4.68 in wide, just under the text width of 4.7 in,
so that no label, tick, title or legend prints below 7 pt; legends placed off the data; omega for the penalty weight, as
in the chapter; the centre panel no longer clipped, so that the spike of the estimate at pi/2 shows; and, on the right,
the B spline formula of Theorem thm:lapspline(b) computed exactly instead of averaged over the Monte Carlo sample.
Centre panel: the angle law of 32a has an atom at pi/2 (the primes with a_p = 0), which a histogram would draw as one bar
about 14.5 high. Its mass is drawn as an arrow at pi/2 and the bars show the other angles, both as fractions of all the
primes used; this needs the traces a_p of 32a1, which gg_mple.py does not store, so they are recomputed here with PARI
exactly as gg_mple.py computes them, and the bars with the atom put back are checked against the histogram in gg_mple.json.
Right panel: the density of H_67(2) by partial fractions (line), by the B spline formula
    p(t) = (prod c_i) t^k/k! * integral of exp(-t x) M(x; c_0, ..., c_k) dx
with M the Curry Schoenberg B spline on the atoms c_i, by Gauss Legendre quadrature on every knot interval, where M is a
cubic, so to rounding error (circles), and the histogram of the 400000 samples of checks.py (Gamma times B spline variable).
Run on Hopper, never locally:  python3 fig_gg.py [png]   (with png it also writes a PNG preview at 200 dpi, for checking).
Reads gg_mple.json (gg_mple.py) and checks.json (checks.py); uses cypari2; writes figs/lap_spline.pdf.
"""
import io
import json
import math
import os
import sys

import cypari2
import matplotlib
import numpy as np
from scipy.interpolate import BSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cbook  # noqa: E402
from matplotlib.backends.backend_pdf import PdfFile, RendererPdf  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

INK, RUST, TEAL, GOLD, PLUM, SKY = "#1F3B63", "#9C3D1F", "#1B7F79", "#C99A2E", "#6B3F7A", "#8FB3DE"
BAR_GREY, ATOM_GREY = "0.82", "0.45"
FS_LABEL, FS_TICK, FS_TITLE, FS_LEG = 9, 8, 8.5, 7.5
plt.rcParams.update({"font.family": "serif", "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
                     "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                     "legend.fontsize": FS_LEG, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
PNG = "png" in sys.argv[1:]
WIDTH = 4.68  # just under the text width of the book, 4.7 in, so that the figure prints at a scale of at least 1

G = json.load(open("gg_mple.json"))
H = json.load(open("checks.json"))["hecke"]
NBIN = 90
edges = np.linspace(0, math.pi, NBIN + 1)
mid = (edges[1:] + edges[:-1]) / 2
w = edges[1] - edges[0]

# Frobenius angles of 32a1 at the odd primes below 10^6, as in gg_mple.py
pari = cypari2.Pari()
pari.allocatemem(10 ** 9)
ap = np.array([int(v) for v in pari("E=ellinit([0,0,0,-1,0]); vector(primepi(10^6)-1,k,ellap(E,prime(k+1)))")])
pr = np.array([int(v) for v in pari("vector(primepi(10^6)-1,k,prime(k+1))")])
th32 = np.arccos(np.clip(ap / (2 * np.sqrt(pr)), -1, 1))
n32 = len(th32)
atom = ap == 0
mass_atom = atom.sum() / n32
bars32 = np.histogram(th32[~atom], bins=NBIN, range=(0, math.pi))[0] / (n32 * w)
full32 = np.histogram(th32, bins=NBIN, range=(0, math.pi), density=True)[0]
print("32a1: %d primes from %d to %d, %d with a_p = 0 (mass %.6f); histogram against gg_mple.json: max diff %.2e"
      % (n32, pr[0], pr[-1], atom.sum(), mass_atom, np.max(np.abs(full32 - np.array(G["32a1"]["hist"])))), flush=True)
assert n32 == G["32a1"]["n"]


def gg_density(b, th):
    g = np.sqrt(2 / math.pi) * sum(b[j] * np.sin((j + 1) * th) for j in range(len(b)))
    return g * g


# B spline formula of Theorem thm:lapspline(b), without sampling
c = np.sort(np.array(H["atoms"], dtype=float))
k = len(c) - 1
N = BSpline.basis_element(c, extrapolate=False)  # degree k-1, partition of unity normalisation
xg, wg = np.polynomial.legendre.leggauss(40)
nodes = np.concatenate([0.5 * (c[i + 1] - c[i]) * xg + 0.5 * (c[i + 1] + c[i]) for i in range(k)])
wts = np.concatenate([0.5 * (c[i + 1] - c[i]) * wg for i in range(k)])
Mn = N(nodes) * k / (c[-1] - c[0])  # the Curry Schoenberg B spline M(x; c_0, ..., c_k), of integral 1


def pdf_bspline(t):
    t = np.atleast_1d(t)
    return np.prod(c) * t ** k / math.factorial(k) * (np.exp(-np.outer(t, nodes)) @ (wts * Mn))


def pdf_pf(t):
    t = np.atleast_1d(t)
    return np.prod(c) * sum(np.exp(-ci * t) / np.prod([cj - ci for cj in c if cj != ci]) for ci in c)


tf = np.linspace(0, 4, 401)
tc = np.linspace(0.2, 4, 20)
err = np.max(np.abs(pdf_bspline(tf) - pdf_pf(tf)))
print("H_67(2): atoms %s, k = %d, integral of M %.15f; B spline formula against partial fractions on [0, 4]: max diff %.2e"
      % (np.round(c, 6).tolist(), k, np.sum(wts * Mn), err), flush=True)

os.makedirs("figs", exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.3), layout="constrained")
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.04)
th = np.linspace(0, math.pi, 721)
st = 2 / math.pi * np.sin(th) ** 2
for i, (name, title) in enumerate((("11a1", "11a1"), ("32a1", "32a"))):
    a = ax[i]
    if name == "32a1":
        a.bar(mid, bars32, width=w, color=BAR_GREY, edgecolor="none")
        a.annotate("", xy=(math.pi / 2, 2.0), xytext=(math.pi / 2, 0), zorder=2.5,
                   arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=ATOM_GREY, lw=0.9,
                                   shrinkA=0, shrinkB=0))
        a.text(math.pi / 2 + 0.12, 1.95, "mass %.3f" % mass_atom, fontsize=FS_LEG, color="0.3", va="top", ha="left")
        a.set_ylim(0, 2.0)
    else:
        a.bar(mid, G[name]["hist"], width=w, color=BAR_GREY, edgecolor="none")
        a.set_ylim(0, 0.72)
    a.plot(th, st, color="k", lw=0.9, ls=(0, (3.5, 2)), zorder=3)
    for om, col in (("0.01", RUST), ("1.0", INK)):
        a.plot(th, gg_density(G[name][om]["b"], th), color=col, lw=1.1, zorder=4)
    a.set_xlim(0, math.pi)
    a.set_xticks([0, math.pi / 2, math.pi])
    a.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    a.set_xlabel(r"$\theta_p$")
    a.set_title(title)
hands = [Patch(color=BAR_GREY), Line2D([], [], color="k", lw=0.9, ls=(0, (3.5, 2))),
         Line2D([], [], color=RUST, lw=1.1), Line2D([], [], color=INK, lw=1.1)]
labs = ["Frobenius angles", r"$(2/\pi)\sin^2\theta$", r"Good and Gaskins, $\omega=0.01$", r"Good and Gaskins, $\omega=1$"]

a = ax[2]
hist = np.array(H["hist"])
hed = np.linspace(0, 4, len(hist) + 1)
a.stairs(hist, hed, fill=True, color=SKY, lw=0)
a.plot(tf, pdf_pf(tf), color=INK, lw=1.2)
a.plot(tc, pdf_bspline(tc), "o", ms=3.2, mfc="none", mec=RUST, mew=0.8)
a.set_xlim(0, 4)
a.set_ylim(0, None)
a.set_xticks([0, 1, 2, 3, 4])
a.set_xlabel(r"$t$")
a.set_title(r"density of $H_{67}(2)$")
hands2 = [Line2D([], [], color=INK, lw=1.2), Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=RUST, mew=0.8),
          Patch(color=SKY)]
labs2 = ["partial fractions", "B spline formula", "400000 samples"]

# legends between the titles and the axes, off the data: one shared by the two angle panels (kept out of the layout
# and centred over both once the layout is known), one for the right panel. The titles share one baseline, lifted so
# that GAP points of white separate the deepest title from the higher legend; legends and titles are measured with the
# PDF backend's own metrics, at 72 dpi (the Agg renderer hints glyphs to whole pixels and measures a little differently)
GAP = 3.0
kw = dict(loc="lower center", frameon=False, handlelength=1.8, handletextpad=0.5, columnspacing=1.0, labelspacing=0.3,
          borderpad=0.1, borderaxespad=0.3)
leg1 = ax[0].legend(hands, labs, ncol=2, bbox_to_anchor=(0.5, 1.0), **kw)
leg1.set_in_layout(False)
leg2 = ax[2].legend(hands2, labs2, bbox_to_anchor=(0.5, 1.0), **kw)
fig.set_dpi(72)
fig.canvas.draw()
w_in, h_in = fig.get_size_inches()
rp = RendererPdf(PdfFile(io.BytesIO()), 72, 72 * h_in, 72 * w_in)
rise = max(leg1.get_window_extent(rp).y1 - ax[0].get_window_extent(rp).y1,
           leg2.get_window_extent(rp).y1 - ax[2].get_window_extent(rp).y1)
ra = fig.canvas.get_renderer()
rise_agg = max(leg1.get_window_extent(ra).y1 - ax[0].get_window_extent().y1,
               leg2.get_window_extent(ra).y1 - ax[2].get_window_extent().y1)
dep = max(rp.get_text_width_height_descent(a.get_title(), a.title.get_fontproperties(),
                                           cbook.is_math_text(a.get_title()))[2] for a in ax)
pad = rise + dep + GAP
print("lap_spline: legend bands %.2f pt above the axes (Agg at 72 dpi %.2f pt), deepest title %.2f pt, pad %.2f pt"
      % (rise, rise_agg, dep, pad), flush=True)
for a in ax:
    a.set_title(a.get_title(), pad=pad)
fig.canvas.draw()
p0, p1 = ax[0].get_position(), ax[1].get_position()
leg1.set_bbox_to_anchor(((p0.x0 + p1.x1) / 2, 1.0),  # x in figure coordinates, y at the top of the axes
                        transform=blended_transform_factory(fig.transFigure, ax[0].transAxes))
fig.savefig("figs/lap_spline.pdf")
if PNG:
    fig.savefig("figs/lap_spline.png", dpi=200)
plt.close(fig)
print("wrote figs/lap_spline.pdf", flush=True)
