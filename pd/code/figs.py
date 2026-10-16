"""Figures fig:pdintegers (figs/pd_integers.pdf) and fig:pddirichlet (figs/pd_dirichlet.pdf) of Chapter 24 (ch_pd).

The coauthor's script, kept unchanged as figs.original.py, edited for the book on 2026-10-15 (decision 8 of
misc/pd-apply-brief.md, brief misc/pd-figs-brief.md): the book's palette and serif type, mathematics included; drawn
4.68 in wide, with no label, tick, title or legend set below 7.5 pt; every legend in a band between its panel's title
and the axes, off the data; the chapter's notation (varrho for Dickman's function, E for the standard exponential
variable, P for the Dirichlet process, Xi_sigma(Q)); pd_integers in two rows, the three colourings below at the full
width; Dickman's function computed here on [0, 20] from its delay equation, by Taylor series in 50 digit arithmetic
(the curve cdf_th of pd1.json is set to 0 below v = 1/3); the empirical distribution function of log P+/log n drawn
from the sample of pd_samples.npz at step 0.0005 in v (pd1.json keeps it at 96 points); integer ticks for omega(n);
the vertical axis of the size biased shares raised to 2, so that the last bin of the identity class shows; the floors
of the identity class and of the transpositions marked; one colour for one meaning within each panel. The data are
the coauthor's.

Run on Hopper, never locally:  python3 figs.py [png] [check]
Reads pd1.json and pd_samples.npz (written by pd1.py) and pd2.json (pd2.py); writes figs/pd_integers.pdf and
figs/pd_dirichlet.pdf. With png it also writes PNG previews at 150 dpi; with check it writes figs/pd_figs_check.json:
the data of every drawn artist, the Taylor series of Dickman's function and its values at the integers, and the
bounding boxes of every text, legend and axes with the points of every curve, marker and bar, measured with the PDF
backend's renderer at 72 dpi (in points), for check_pdfigs.py.
"""
import io
import json
import math
import os
import sys

import matplotlib
import mpmath
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cbook  # noqa: E402
from matplotlib.backends.backend_pdf import PdfFile, RendererPdf  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.stats import norm, poisson  # noqa: E402

INK, RUST, TEAL, GOLD, PLUM = "#1F3B63", "#9C3D1F", "#1B7F79", "#C99A2E", "#6B3F7A"
BAR_GREY, HAAR_GREY = "0.80", "0.68"
FS_LABEL, FS_TICK, FS_TITLE, FS_LEG = 9, 8, 8.5, 7.5
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": FS_LABEL,
                     "axes.labelsize": FS_LABEL, "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK,
                     "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
ARGS = sys.argv[1:]
PNG, CHECK = "png" in ARGS, "check" in ARGS
WIDTH = 4.68  # the width of the book's other figures, drawn for a text width of 4.7 in
GAP = 3.0  # points of white between the lowest point of a title and the top of the legend band below it
RECORD = {}


def band(ax, title, handles, labels, ncol=1, right=False, compact=False):
    """Title above the axes and the legend between the title and the axes, so that the legend covers no data;
    with right, the legend is aligned with the right end of the axes; with compact, shorter handles and columns."""
    leg = ax.legend(handles, labels, loc="lower right" if right else "lower center",
                    bbox_to_anchor=(1.0 if right else 0.5, 1.0), ncol=ncol, frameon=False,
                    handlelength=1.5 if compact else 1.8, handletextpad=0.4 if compact else 0.5,
                    columnspacing=0.8 if compact else 1.2, labelspacing=0.3, borderpad=0.1, borderaxespad=0.3)
    leg.band = True
    ax.set_title(title)


def pdf_renderer(fig):
    """A renderer with the text metrics of the PDF backend, which lays out text at 72 dpi."""
    w, h = fig.get_size_inches()
    return RendererPdf(PdfFile(io.BytesIO()), 72, 72 * h, 72 * w)


def depth(text, r):
    """Depth of the ink of a text artist below its baseline, in points, as the PDF backend measures it."""
    s = text.get_text()
    return r.get_text_width_height_descent(s, text.get_fontproperties(), cbook.is_math_text(s))[2]


def save(fig, rows, name):
    """Lift the titles of each row above the legend bands of the row, then write the PDF (and the PNG preview and the
    check record). Legends and titles are measured with the PDF backend's metrics, at 72 dpi, and the titles of a row
    share one baseline, placed so that GAP points of white separate the deepest title from the highest band."""
    fig.set_dpi(72)
    fig.canvas.draw()
    r = pdf_renderer(fig)
    for row in rows:
        rise = 0.0
        for ax in row:
            leg = ax.get_legend()
            if leg is not None and getattr(leg, "band", False):
                rise = max(rise, leg.get_window_extent(r).y1 - ax.get_window_extent(r).y1)
        dep = max(depth(ax.title, r) for ax in row)
        pad = max(4.0, rise + dep + GAP)
        print("%s [%s]: legend band %.2f pt above the axes, deepest title %.2f pt, pad %.2f pt"
              % (name, " | ".join(ax.get_title() for ax in row), rise, dep, pad), flush=True)
        for ax in row:
            ax.set_title(ax.get_title(), pad=pad)
    fig.savefig("figs/%s.pdf" % name)
    print("wrote figs/%s.pdf" % name, flush=True)
    if CHECK:
        RECORD[name] = record(fig)
    if PNG:
        fig.savefig("figs/%s.png" % name, dpi=150)
        print("wrote figs/%s.png" % name, flush=True)
    plt.close(fig)


def box(b):
    return [round(float(b.x0), 3), round(float(b.y0), 3), round(float(b.x1), 3), round(float(b.y1), 3)]


def drawn_ticklabels(axis):
    """The tick labels that are drawn (Axis._update_ticks returns the ticks inside the view, as get_tightbbox uses)."""
    out = []
    for t in axis._update_ticks():
        for lab in (t.label1, t.label2):
            if lab.get_visible() and lab.get_text():
                out.append(lab)
    return out


def sample(xy, step=0.5):
    """Points every `step` points along a polyline given in points."""
    out = []
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        if not np.all(np.isfinite([x0, y0, x1, y1])):
            continue
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0.0, 1.0, n + 1)
        out.append(np.column_stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]))
    return np.vstack(out) if out else np.zeros((0, 2))


def inside(pts, b, tol=0.5):
    return pts[(pts[:, 0] >= b.x0 - tol) & (pts[:, 0] <= b.x1 + tol) & (pts[:, 1] >= b.y0 - tol) & (pts[:, 1] <= b.y1 + tol)]


def record(fig):
    """What the figure draws, for check_pdfigs.py: the data of every artist with a gid (in data coordinates), and the
    layout in points at 72 dpi measured with the PDF backend's renderer after the PDF was written: every text drawn,
    every legend and axes, and the points of every curve, marker and bar inside its axes."""
    r = pdf_renderer(fig)
    drawn, texts, legends, axes, data = {}, [], [], [], []
    for i, ax in enumerate(fig.axes):
        abox = ax.get_window_extent(r)
        axes.append({"axes": i, "title": ax.get_title(), "bbox": box(abox),
                     "xlim": [float(v) for v in ax.get_xlim()], "ylim": [float(v) for v in ax.get_ylim()]})
        tl = [("title", ax.title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]
        tl += [("xtick", t) for t in drawn_ticklabels(ax.xaxis)] + [("ytick", t) for t in drawn_ticklabels(ax.yaxis)]
        tl += [("offset", ax.xaxis.offsetText), ("offset", ax.yaxis.offsetText)]
        tl += [("annotation", t) for t in ax.texts]
        leg = ax.get_legend()
        if leg is not None:
            tl += [("legend", t) for t in leg.get_texts()]
            legends.append({"axes": i, "bbox": box(leg.get_window_extent(r))})
        for kind, t in tl:
            if t.get_visible() and t.get_text().strip():
                texts.append({"axes": i, "kind": kind, "text": t.get_text(), "bbox": box(t.get_window_extent(r)),
                              "size": float(t.get_fontsize())})
        for ln in ax.lines:
            xy = ln.get_transform().transform(ln.get_xydata())
            if ln.get_gid():
                drawn[ln.get_gid()] = {"x": ln.get_xydata()[:, 0].tolist(), "y": ln.get_xydata()[:, 1].tolist()}
            if ln.get_linestyle() not in ("None", "none", "", " "):
                pts = inside(sample(xy), abox)
                data.append({"axes": i, "gid": ln.get_gid(), "kind": "line", "lw": float(ln.get_linewidth()),
                             "pts": np.unique(np.round(pts, 1), axis=0).tolist()})
            if ln.get_marker() not in ("None", "none", "", " ", None):
                hw = 0.5 * float(ln.get_markersize()) + 0.5 * float(ln.get_markeredgewidth())
                for x, y in inside(xy, abox):
                    data.append({"axes": i, "gid": ln.get_gid(), "kind": "box",
                                 "bbox": [round(x - hw, 2), round(y - hw, 2), round(x + hw, 2), round(y + hw, 2)]})
        bars = {}
        for p in ax.patches:
            if p.get_gid():
                bars.setdefault(p.get_gid(), []).append([float(p.get_x() + p.get_width() / 2), float(p.get_height())])
            b = p.get_window_extent(r)
            data.append({"axes": i, "gid": p.get_gid(), "kind": "box", "bbox": box(b)})
        for g, v in bars.items():
            drawn[g] = {"x": [a for a, _ in v], "height": [h for _, h in v]}
        for c in ax.collections:
            segs = c.get_segments() if hasattr(c, "get_segments") else []
            if c.get_gid():
                drawn[c.get_gid()] = {"segments": [np.asarray(s).tolist() for s in segs]}
            for s in segs:
                xy = c.get_transform().transform(np.asarray(s))
                pts = inside(sample(xy), abox)
                lw = c.get_linewidth()
                data.append({"axes": i, "gid": c.get_gid(), "kind": "line", "lw": float(np.max(lw)),
                             "pts": np.unique(np.round(pts, 1), axis=0).tolist()})
    w, h = fig.get_size_inches()
    return {"figure_size_in": [float(w), float(h)], "drawn": drawn,
            "layout": {"texts": texts, "legends": legends, "axes": axes, "data": data}}


def dickman_series(umax=20, terms=120, dps=50):
    """Dickman's function varrho on [0, umax] from varrho = 1 on [0, 1] and the delay equation
    u varrho'(u) = -varrho(u - 1), as Taylor series at the midpoints of the intervals [k, k + 1]: with
    varrho(k + 1/2 + z) = sum_j a_j z^j and b_j the coefficients of the interval before,
    (k + 1/2)(j + 1) a_(j+1) + j a_j = -b_j, and a_0 is fixed by continuity at u = k. The piece of varrho on
    [k, k + 1] continues analytically to |z| < 3/2 (its nearest singularity is at u = k - 1), so on |z| <= 1/2 the
    terms fall like 3^-j and 120 terms leave about 1e-57. The arithmetic is mpmath's at 50 digits: varrho(k) is a
    difference of larger numbers, and a forward recursion in double precision keeps varrho only to an absolute 1e-16
    or so, while varrho(20) is near 1e-29. Returns the coefficients, one list per interval [k, k + 1]."""
    mpmath.mp.dps = dps
    half = mpmath.mpf(1) / 2
    b = [mpmath.mpf(1)] + [mpmath.mpf(0)] * terms
    coef = [b]
    for k in range(1, umax):
        c = k + half
        a = [mpmath.mpf(0)] * (terms + 1)
        for j in range(terms):
            a[j + 1] = -(b[j] + j * a[j]) / (c * (j + 1))
        a[0] = mpmath.polyval(b[::-1], half) - mpmath.polyval(a[::-1], -half)
        coef.append(a)
        b = a
    return coef


def varrho(u, coef):
    """varrho at the points u, 0 <= u <= len(coef), from the series of dickman_series, as floats."""
    half = mpmath.mpf(1) / 2
    rev = [c[::-1] for c in coef]
    out = np.empty(len(u))
    for i, x in enumerate(np.asarray(u, dtype=float)):
        k = min(int(math.floor(x)), len(coef) - 1)
        out[i] = float(mpmath.polyval(rev[k], mpmath.mpf(x) - k - half))
    return out


os.makedirs("figs", exist_ok=True)
A = json.load(open("pd1.json"))
B = json.load(open("pd2.json"))
S = np.load("pd_samples.npz")
X = A["X"]
LL = A["loglog"]  # log log x at x = 10^7
COEF = dickman_series()
if CHECK:
    RECORD["varrho_at_integers"] = {str(k): float(v) for k, v in zip(range(1, 21), varrho(np.arange(1, 21), COEF))}
    RECORD["varrho_series"] = [[mpmath.nstr(c, 50) for c in a] for a in COEF]

# Figure fig:pdintegers: top left, Dickman's law; top right, omega(n); bottom, three colourings by Frobenius class
fig, axd = plt.subplot_mosaic([["L", "C"], ["R", "R"]], figsize=(WIDTH, 4.1), layout="constrained",
                              height_ratios=[1.12, 1.0])
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.04, wspace=0.08, hspace=0.06)

# top left: the empirical distribution function of log P+/log n at step 0.0005 in v, against varrho(1/v) for
# 0.05 <= v <= 1 (u = 1/v in [1, 20]), drawn from (0, 0), since varrho(1/v) <= varrho(20) = 2.46e-29 for v <= 0.05
a = axd["L"]
fm = np.sort(S["frac_max"])
ve = np.linspace(0.0, 1.0, 2001)
a.plot(ve, np.searchsorted(fm, ve, side="right") / len(fm), color=INK, lw=1.2, gid="cdf_emp")
vd = np.linspace(0.05, 1.0, 1901)
a.plot(np.r_[0.0, vd], np.r_[0.0, varrho(1.0 / vd, COEF)], color=RUST, lw=1.0, ls=(0, (3.5, 2.0)), gid="dickman")
a.set_xlim(0, 1.0)
a.set_ylim(-0.02, 1.03)
a.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
a.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
a.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
a.set_xlabel(r"$v$")
a.set_ylabel(r"$\mathbb{P}(\log P^+/\log n\leq v)$")
band(a, "largest prime factor",
     [Line2D([], [], color=INK, lw=1.2), Line2D([], [], color=RUST, lw=1.0, ls=(0, (3.5, 2.0)))],
     [r"sample, $n\leq10^7$", r"Dickman $\varrho(1/v)$"])

# top right: omega(n) for the sample, the Poisson law of mean log log x + B and the normal law of Erdos and Kac
a = axd["C"]
om = S["omega"]
ks = np.arange(0, 9)
mu = LL + B["B_total"]
a.bar(ks, [float(np.mean(om == k)) for k in ks], width=0.8, color=BAR_GREY, gid="omega_hist")
xx = np.linspace(-0.5, 8.5, 181)
a.plot(xx, norm.pdf(xx, LL, math.sqrt(LL)), color=RUST, lw=1.0, gid="normal")
a.plot(ks, poisson.pmf(ks, mu), "o", ms=3.4, color=INK, mec=INK, gid="poisson")
a.set_xlim(-0.6, 8.6)
a.set_xticks(ks)
a.set_xlabel(r"$k$")
a.set_ylabel(r"$\mathbb{P}(\omega(n)=k)$")
band(a, "number of prime factors",
     [Patch(facecolor=BAR_GREY), Line2D([], [], ls="none", marker="o", ms=3.4, color=INK),
      Line2D([], [], color=RUST, lw=1.0)],
     [r"$\omega(n)$, sample", r"Poisson, mean $\log\log x+B$", "normal, Erdős and Kac"])

# bottom: the class of the largest prime factor, of a size biased prime factor and of the distinct prime factors
# (pooled), against Haar measure, with the predictions of Theorem thm:pdmertens(ii) and (iii) as short bars
a = axd["R"]
x = np.arange(3)
w = 0.19
a.bar(x - 1.5 * w, [1 / 6, 1 / 2, 1 / 3], w, color=HAAR_GREY, gid="haar")
a.bar(x - 0.5 * w, A["cls_max"], w, color=INK, gid="cls_max")
a.bar(x + 0.5 * w, A["cls_sb"], w, color=TEAL, gid="cls_sb")
a.bar(x + 1.5 * w, A["share_count"], w, color=RUST, gid="share_count")
a.hlines(B["pred_sb"], x + 0.5 * w - 0.6 * w, x + 0.5 * w + 0.6 * w, color="k", lw=1.5, gid="pred_sb")
a.hlines(B["pred_count"], x + 1.5 * w - 0.6 * w, x + 1.5 * w + 0.6 * w, color="k", lw=1.5, gid="pred_count")
a.set_xticks(x)
a.set_xticklabels(["identity", "transposition", "three cycle"])
a.tick_params(axis="x", length=0)
a.set_xlim(-0.55, 2.55)
a.set_ylim(0, 0.55)
a.set_ylabel("share")
band(a, r"Frobenius classes of $x^3-x-1$",
     [Patch(facecolor=HAAR_GREY), Patch(facecolor=INK), Patch(facecolor=TEAL), Patch(facecolor=RUST),
      Line2D([], [], color="k", lw=1.5)],
     ["Haar measure", "largest factor", "size biased factor", "distinct factors", "predictions of (ii) and (iii)"],
     ncol=3)
save(fig, [[axd["L"], axd["C"]], [axd["R"]]], "pd_integers")

# Figure fig:pddirichlet: left, the Laplace transforms of Theorem thm:pddirichlet(ii) for Q = tr^2 on S_3;
# right, the size biased share by Frobenius class of the chosen prime
fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.6), layout="constrained")
fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.08)

a = ax[0]
lam = np.array(B["lam"])
a.plot(lam, B["th"], color=INK, lw=1.3, gid="limit")
a.plot(lam, B["exact101"], color=TEAL, lw=1.3, ls=(0, (3.5, 2.0)), gid="sigma101")
a.plot(lam[::2], np.array(B["mc"])[::2], "o", ms=3.4, mfc="none", mec=RUST, mew=0.9, gid="mc")
a.set_xlim(-0.15, 6.15)
a.set_xlabel(r"$\lambda$")
a.set_ylabel("Laplace transform")
band(a, r"Frobenius functional, $Q=\mathrm{tr}^2$",
     [Line2D([], [], color=INK, lw=1.3), Line2D([], [], ls="none", marker="o", ms=3.4, mfc="none", mec=RUST, mew=0.9),
      Line2D([], [], color=TEAL, lw=1.3, ls=(0, (3.5, 2.0)))],
     ["limit", r"$E\,(4P(\{1\})+P(C_3))$", r"$\Xi_\sigma(Q)$, $\sigma=1.01$"])

a = ax[1]
mids = (np.linspace(0, 1, 11)[1:] + np.linspace(0, 1, 11)[:-1]) / 2
CLASSES = (("0", "identity", GOLD, "o"), ("1", "transposition", PLUM, "s"), ("2", "three cycle", TEAL, "^"))
a.axhline(1.0, color=INK, lw=0.9, ls=(0, (3.5, 2.0)), gid="uniform")
for key, lab, col, mk in CLASSES:
    a.plot(mids, A["sb_hist"][key], color=col, lw=1.0, marker=mk, ms=3.2, mec=col, mfc=col, gid="sb_hist_" + key)
# floors at x = 10^7: the smallest primes of the identity class and of the transpositions are 59 and 5, so a size
# biased factor in these classes has share at least log 59/log x and log 5/log x; each floor is a dotted line in the
# colour of its class with its value above it (the two at different heights, so that the values keep apart)
f_id, f_tr = math.log(59) / math.log(X), math.log(5) / math.log(X)
for f, top, col, g in ((f_tr, 1.52, PLUM, "floor_transposition"), (f_id, 1.74, GOLD, "floor_identity")):
    a.plot([f, f], [0.0, top], color=col, lw=0.9, ls=(0, (1.0, 1.5)), gid=g)
    a.text(f, top + 0.04, "%.2f" % f, color=col, fontsize=FS_LEG, ha="center", va="bottom")
a.set_xlim(0, 1)
a.set_ylim(0, 2.0)
a.set_yticks([0, 0.5, 1.0, 1.5])  # no tick label at the top, where the legend band begins
a.set_xlabel(r"share $e_p\log p/\log n$")
a.set_ylabel("density")
band(a, r"size biased share, $x=10^7$",
     [Line2D([], [], color=col, lw=1.0, marker=mk, ms=3.2, mec=col, mfc=col) for _, _, col, mk in CLASSES]
     + [Line2D([], [], color=INK, lw=0.9, ls=(0, (3.5, 2.0)))],
     [lab for _, lab, _, _ in CLASSES] + ["uniform limit"], ncol=2, right=True, compact=True)
save(fig, [[ax[0], ax[1]]], "pd_dirichlet")

if CHECK:
    json.dump(RECORD, open("figs/pd_figs_check.json", "w"))
    print("wrote figs/pd_figs_check.json", flush=True)
