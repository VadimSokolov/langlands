"""Figures fig:lapfaces (figs/lap_faces.pdf) and fig:lapzeta (figs/lap_zeta.pdf) of Chapter 23 (ch_laplace).

The coauthor's script, kept unchanged as figs.original.py, edited for the book on 2026-10-13 (decision 9 of
misc/lap-apply-brief.md): the book's palette and serif type; drawn 4.68 in wide, just under the text width of 4.7 in,
so that no label, tick, title or legend prints below 7 pt; legends placed off the data; the chapter's notation
(Lambda_rho(n), g_alpha, Phi_rho, gamma(s, rho), Xi_zeta, V_a); the Cauchy superpositions of Theorem thm:lapcauchy drawn
over the densities they match; and the tail data read from zeta_tail.json (written by zeta_tail.py), since no script of
the coauthor writes the block H2, tailA of zeta_switch.json that the original read. The data and the curves are those of
the original.
Run on Hopper, never locally:  python3 figs.py [png]   (with png it also writes PNG previews at 200 dpi, for checking).
Reads switch23_a.json (switch23.py), switch23_b.json (switch23b.py), zeta_switch.json (zeta_switch.py: the entries
cauchy and H.pw) and zeta_tail.json (zeta_tail.py); writes figs/lap_faces.pdf and figs/lap_zeta.pdf.
"""
import io
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cbook  # noqa: E402
from matplotlib.backends.backend_pdf import PdfFile, RendererPdf  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK, RUST, TEAL, GOLD, PLUM = "#1F3B63", "#9C3D1F", "#1B7F79", "#C99A2E", "#6B3F7A"
AXIS_GREY, SUP_GREY = "0.6", "0.5"
FS_LABEL, FS_TICK, FS_TITLE, FS_LEG = 9, 8, 8.5, 7.5
plt.rcParams.update({"font.family": "serif", "font.size": FS_LABEL, "axes.labelsize": FS_LABEL,
                     "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                     "legend.fontsize": FS_LEG, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.03})
PNG = "png" in sys.argv[1:]
WIDTH = 4.68  # just under the text width of the book, 4.7 in, so that the figure prints at a scale of at least 1


def band(ax, title, handles, labels, ncol=1, right=False):
    """Title above the axes and the legend between the title and the axes, so that the legend covers no data."""
    leg = ax.legend(handles, labels, loc="lower right" if right else "lower center",
                    bbox_to_anchor=(1.0 if right else 0.5, 1.0), ncol=ncol, frameon=False, handlelength=1.8,
                    handletextpad=0.5, columnspacing=1.0, labelspacing=0.3, borderpad=0.1, borderaxespad=0.3)
    leg.band = True
    ax.set_title(title)


GAP = 3.0  # points of white between the lowest point of a title and the top of the legend band below it


def pdf_renderer(fig):
    """A renderer with the text metrics of the PDF backend, which lays out text at 72 dpi. The Agg renderer hints
    glyphs to whole pixels, so it measures a legend a point or two taller or shorter than the PDF prints it."""
    w, h = fig.get_size_inches()
    return RendererPdf(PdfFile(io.BytesIO()), 72, 72 * h, 72 * w)


def depth(text, r):
    """Depth of the ink of a text artist below its baseline, in points, as the PDF backend measures it."""
    s = text.get_text()
    return r.get_text_width_height_descent(s, text.get_fontproperties(), cbook.is_math_text(s))[2]


def save(fig, rows, name):
    """Lift the titles of each row above the legend bands of the row, then write the PDF (and a PNG preview).
    Legends and titles are measured with the PDF backend's own metrics, at 72 dpi, and the titles of a row share one
    baseline, placed so that GAP points of white separate the deepest title from the highest band."""
    fig.set_dpi(72)
    fig.canvas.draw()
    r = pdf_renderer(fig)
    for row in rows:
        rise, rise_agg = 0.0, 0.0
        for ax in row:
            leg = ax.get_legend()
            if leg is not None and getattr(leg, "band", False):
                rise = max(rise, leg.get_window_extent(r).y1 - ax.get_window_extent(r).y1)
                ra = fig.canvas.get_renderer()
                rise_agg = max(rise_agg, leg.get_window_extent(ra).y1 - ax.get_window_extent().y1)
        dep = max(depth(ax.title, r) for ax in row)
        pad = max(4.0, rise + dep + GAP)
        print("%s [%s]: legend band %.2f pt above the axes (Agg at 72 dpi %.2f pt), deepest title %.2f pt, pad %.2f pt"
              % (name, " | ".join(ax.get_title() for ax in row), rise, rise_agg, dep, pad), flush=True)
        for ax in row:
            ax.set_title(ax.get_title(), pad=pad)
    fig.savefig("figs/%s.pdf" % name)
    if PNG:
        fig.savefig("figs/%s.png" % name, dpi=200)
    plt.close(fig)
    print("wrote figs/%s.pdf" % name, flush=True)


def no_handle():
    return Line2D([], [], linestyle="none", marker="none")


os.makedirs("figs", exist_ok=True)
A = json.load(open("switch23_a.json"))
B = json.load(open("switch23_b.json"))
Z = json.load(open("zeta_switch.json"))
T = json.load(open("zeta_tail.json"))

# Figure fig:lapfaces: four faces of L(s, rho) = L(s, eta(z)eta(23z))
fig, ax = plt.subplots(2, 2, figsize=(WIDTH, 4.45), layout="constrained")
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.04, wspace=0.06, hspace=0.06)

# top left, Galois face: atoms Lambda_rho(n) n^-2 at log n, from Frobenius classes and as Gaussian means of g_2
a = ax[0, 0]
tab = B["bohr"]  # rows [n, recovered Lambda(n), Lambda(n) from Frobenius]
ns = np.array([r[0] for r in tab], dtype=float)
rec = np.array([r[1] for r in tab])
lam = np.array([r[2] for r in tab])
x = np.log(ns)
a.axhline(0, color=AXIS_GREY, lw=0.5, zorder=0)
a.vlines(x, 0, lam / ns ** 2, color=INK, lw=1.6)
a.plot(x, rec / ns ** 2, "o", ms=3.2, mfc="none", mec=RUST, mew=0.8)
a.set_xlim(0.5, 4.35)
a.set_xlabel(r"$\log n$")
a.set_ylabel(r"$\Lambda_\rho(n)\,n^{-2}$")
band(a, r"Galois face, $\alpha=2$",
     [Line2D([], [], color=INK, lw=1.6), Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=RUST, mew=0.8)],
     ["Frobenius classes", r"Gaussian means of $g_2$"])

# top right, transport face: the transported density g_1.5, true and with a_59 replaced by 0
a = ax[0, 1]
rs = np.array(A["rs"])
a.axhline(0, color=AXIS_GREY, lw=0.5, zorder=0)
a.plot(rs, A["gG"], color=INK, lw=1.1)
a.plot(rs, A["gF"], color=RUST, lw=0.7)
a.set_xlim(0, 40)
a.set_xlabel(r"$r$")
a.set_ylabel(r"$g_{1.5}(r)$")
band(a, r"transported density, $\alpha=1.5$",
     [Line2D([], [], color=INK, lw=1.1), Line2D([], [], color=RUST, lw=0.7)],
     [r"Frobenius $=$ $q$ expansion", r"$a_{59}$ replaced by $0$"])

# bottom left, automorphic face: Phi_rho by Bromwich inversion against 2 f(i e^u/sqrt 23) e^(u/2)
a = ax[1, 0]
us = np.array(B["us"])
a.plot(us, B["Phi_aut"], color=INK, lw=1.2)
a.plot(us[::4], np.array(B["Phi_inv"])[::4], "o", ms=3.2, mfc="none", mec=RUST, mew=0.8)
a.set_xlabel(r"$u$")
a.set_ylabel(r"$\Phi_\rho(u)$")
band(a, "automorphic face",
     [Line2D([], [], color=INK, lw=1.2), Line2D([], [], ls="none", marker="o", ms=3.2, mfc="none", mec=RUST, mew=0.8)],
     [r"$2f(ie^u/\sqrt{23})\,e^{u/2}$", "Bromwich inversion"])

# bottom right, zero face: Lambda(1/2+it, rho)/|gamma(1/2+it, rho)|, gamma(s, rho) = 23^(s/2) Gamma_C(s),
# Gamma_C(s) = 2 (2 pi)^(-s) Gamma(s), and |Gamma(1/2+it)|^2 = pi/cosh(pi t)
a = ax[1, 1]
ts = np.array(B["ts"])
m = ts <= 30
gam = 23 ** 0.25 * 2 * (2 * np.pi) ** -0.5 * np.sqrt(np.pi / np.cosh(np.pi * ts[m]))
a.axhline(0, color=AXIS_GREY, lw=0.5, zorder=0)
a.plot(ts[m], np.array(B["Lam"])[m] / gam, color=INK, lw=1.0)
a.set_xlim(0, 30)
a.set_xlabel(r"$t$")
band(a, "zero face", [Line2D([], [], color=INK, lw=1.0), no_handle()],
     [r"$\Lambda(1/2+it,\rho)\,/\,|\gamma(1/2+it,\rho)|$", r"$\gamma(s,\rho)=23^{s/2}\,\Gamma_{\mathbb{C}}(s)$"])
save(fig, [[ax[0, 0], ax[0, 1]], [ax[1, 0], ax[1, 1]]], "lap_faces")

# Figure fig:lapzeta: Cauchy smoothing of the zeros of zeta, and the law of the Thorin variable of zeta at alpha = 2
fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.5), layout="constrained")
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06, rect=(0, 0, 0.985, 1))  # room for the right title

# left: 2r V_a(r^2) of Xi_zeta at a = 0.5, 0.2, 0.05, with the Cauchy superpositions over the zeros on top
a = ax[0]
CURVES = (("0.5", INK), ("0.2", GOLD), ("0.05", RUST))
for key, col in CURVES:
    d = Z["cauchy"][key]
    a.plot(d["r"], d["V"], color=col, lw=1.3)
for key, col in CURVES:
    d = Z["cauchy"][key]
    a.plot(d["r"], d["C"], color=SUP_GREY, lw=0.6, ls=(0, (2.2, 1.6)))
a.set_yscale("log")
a.set_xlim(0, 50)
a.set_ylim(5e-4, 30)
a.set_xlabel(r"$r$")
a.set_ylabel(r"$2r\,V_a(r^2)$")
h = [Line2D([], [], color=col, lw=1.3) for _, col in CURVES]
lab = [r"$a=%s$" % key for key, _ in CURVES]
sup = Line2D([], [], color=SUP_GREY, lw=0.6, ls=(0, (2.2, 1.6)))
band(a, r"Thorin densities of $\Xi_\zeta$", [h[0], h[2], h[1], sup], [lab[0], lab[2], lab[1], "Cauchy superposition"], ncol=2)

# right: density of the Thorin variable H of zeta at alpha = 2 (Talbot), its tail A t^(-5/2), Post Widder k = 4 and 32
a = ax[1]
t = np.array(T["t"])
p = np.array(T["dens"])
keep = p > 1e-12
h = []
h += a.loglog(t[keep], p[keep], color=INK, lw=1.3)
tt = np.geomspace(2.0, 400.0, 60)
h += a.loglog(tt, T["A"] * tt ** -2.5, color=RUST, lw=1.0, ls="--")
pw = Z["H"]["pw"]
tp = [0.05, 0.1, 0.2, 0.4]
h += a.plot(tp, pw["4"], "v", ms=4, mfc="none", mec=TEAL, mew=0.9)
h += a.plot(tp, pw["32"], "^", ms=4, mfc="none", mec=PLUM, mew=0.9)
a.set_xlim(2e-3, 500)
a.set_ylim(1e-9, 20)
a.set_xticks([1e-2, 1e0, 1e2])
a.set_yticks([1e-9, 1e-6, 1e-3, 1e0])
a.set_xlabel(r"$t$")
a.set_ylabel(r"density $p(t)$ of $H$")
band(a, r"Thorin variable of $\zeta$, $\alpha=2$", h,
     ["Talbot", r"$At^{-5/2}$", r"Post Widder, $k=4$", r"Post Widder, $k=32$"], ncol=2, right=True)
save(fig, [[ax[0], ax[1]]], "lap_zeta")
