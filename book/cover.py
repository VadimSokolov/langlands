import numpy as np, math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

pages = int(sys.argv[1]) if len(sys.argv) > 1 else 180
bleed, trimw, trimh = 0.125, 7.0, 10.0  # the 7 x 10 in KDP trim; width 14.25 + spine, height 10.25
spine = pages * 0.002252
W = 2 * bleed + 2 * trimw + spine; H = 2 * bleed + trimh
BG = "#0B1026"
cmap = LinearSegmentedColormap.from_list("lang", ["#2B5BA8", "#1B9E9A", "#E0B13A", "#E3632D", "#B8325A"])

def tree_segments(cx, cy, R, depth=7, q=2):
    # radial drawing of a (q+1) regular tree truncated at given depth
    segs, cols, widths = [], [], []
    def rec(x, y, ang0, ang1, d, parent_ang):
        if d == depth: return
        k = q + 1 if d == 0 else q
        span = (ang1 - ang0) / k
        r1 = R * (1 - 0.82 ** (d + 1)) / (1 - 0.82 ** depth)
        for i in range(k):
            a = ang0 + span * (i + 0.5)
            nx_, ny_ = cx + r1 * math.cos(a), cy + r1 * math.sin(a)
            segs.append([(x, y), (nx_, ny_)]); cols.append(d / depth); widths.append(2.4 * 0.78 ** d)
            rec(nx_, ny_, ang0 + span * i, ang0 + span * (i + 1), d + 1, a)
    rec(cx, cy, 0, 2 * math.pi, 0, 0)
    return segs, cols, widths

fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), W, H, color=BG))
# subtle starfield of normalized Frobenius traces mapped to angles
rng = np.random.default_rng(11)
theta = np.arccos(np.clip(rng.uniform(-1, 1, 6000), -1, 1))
xs = rng.uniform(0, W, 2500); ys = rng.uniform(0, H, 2500)
ax.scatter(xs, ys, s=rng.uniform(0.05, 0.6, 2500), color="white", alpha=0.25, lw=0)

fx0 = bleed + trimw + spine  # left edge of front cover
cx, cy = fx0 + trimw / 2, bleed + 4.6
# glow
for r, a in [(3.5, 0.05), (2.85, 0.06), (2.2, 0.07), (1.55, 0.08)]:
    ax.add_patch(plt.Circle((cx, cy), r, color="#2B5BA8", alpha=a, lw=0))
segs, cols, widths = tree_segments(cx, cy, 3.0, depth=8, q=2)
lc = LineCollection(segs, colors=cmap(np.array(cols)), linewidths=widths, alpha=0.95, capstyle="round")
ax.add_collection(lc)
ends = np.array([s[1] for s in segs]); dd = np.array(cols)
ax.scatter(ends[:, 0], ends[:, 1], s=6 * 0.7 ** (dd * 8), c=cmap(dd), lw=0, zorder=3)
ax.scatter([cx], [cy], s=60, color="#F4E3B1", zorder=4, lw=0)
# Kesten McKay curves rising from the bottom of the front cover
xg = np.linspace(-2, 2, 400)
for q, c in zip([2, 3, 5, 11], ["#E3632D", "#E0B13A", "#1B9E9A", "#7FB2F0"]):
    d = (q + 1) / ((math.sqrt(q) + 1 / math.sqrt(q)) ** 2 - xg ** 2) * np.sqrt(np.clip(4 - xg ** 2, 0, None)) / (2 * np.pi)
    ax.plot(cx + xg * 1.4, bleed + 1.0 + d * 2.4, color=c, lw=1.2, alpha=0.9)
sc = np.sqrt(np.clip(4 - xg ** 2, 0, None)) / (2 * np.pi)
ax.plot(cx + xg * 1.4, bleed + 1.0 + sc * 2.4, color="white", lw=1.0, ls=(0, (2, 2)), alpha=0.8)

serif = {"family": "serif"}
ax.text(cx, bleed + trimh - 0.9, "ON LANGLANDS", ha="center", va="center", color="#F4E3B1", fontsize=38, fontdict=serif)
ax.text(cx, bleed + trimh - 1.55, "CONJECTURES", ha="center", va="center", color="#F4E3B1", fontsize=38, fontdict=serif)
ax.text(cx, bleed + trimh - 2.2, "Probabilistic Methods", ha="center", va="center", color="#E0B13A", fontsize=21, style="italic", fontdict=serif)
ax.text(cx, bleed + 0.5, "NICK POLSON   ·   VADIM SOKOLOV", ha="center", va="center", color="white", fontsize=14, fontdict=serif)

# spine
sx = bleed + trimw + spine / 2
ax.text(sx, H / 2 + 0.6, "ON LANGLANDS CONJECTURES   Probabilistic Methods", rotation=-90, ha="center", va="center", color="#F4E3B1", fontsize=13, fontdict=serif)
ax.text(sx, bleed + 1.3, "POLSON  SOKOLOV", rotation=-90, ha="center", va="center", color="white", fontsize=10, fontdict=serif)

# back cover
bx = bleed + 0.6; bw = trimw - 1.2
blurb = ("The Langlands conjectures predict that the Frobenius classes attached to an automorphic "
         "representation behave like Haar random elements of a compact group. This book studies "
         "that probabilistic shadow: what it proves, what it tests, and where it stops.\n\n"
         "Part by part it develops the Weyl criterion and the Sato Tate law; Hecke operators as "
         "random walks on Bruhat Tits trees, Ihara zeta functions, Ramanujan graphs and hitting "
         "times on supersingular isogeny graphs; infinitely divisible laws, generalized gamma "
         "convolutions and Thorin measures for L functions, and in the same language squarefree "
         "integers, Dirichlet's beta function, quadratic reciprocity and Fermat's Last Theorem; "
         "inverse Laplace transforms that pass between the faces of one L function; "
         "random integers and the Poisson Dirichlet law; Dirichlet and Hecke characters and "
         "complex multiplication; Maass forms and Brownian motion on the modular surface; "
         "crossing probabilities in critical percolation and the L function of an elliptic curve; "
         "an extreme value representation of the "
         "L functions of eta products; and Bayesian identification of Sato Tate groups.\n\n"
         "New results include an exact limit for the Kemeny constants of isogeny graphs, an "
         "equivalence between the Riemann hypothesis for Ramanujan's Delta and a generalized gamma "
         "convolution, and the identity L(s,f) = E[Q(s, 2 pi M)] expressing L functions as "
         "expectations. Large scale computations accompany every chapter, and each chapter ends "
         "with open computational problems that together form a research programme.")
blurb = blurb.replace("l/(l-1)", "ℓ/(ℓ−1)").replace("2 pi M", "2πM")
# no line break inside these names (textwrap does not break at a no-break space)
for pair in ("L functions", "L function", "Sato Tate", "Bruhat Tits"):
    blurb = blurb.replace(pair, pair.replace(" ", " "))
import textwrap
y = bleed + trimh - 0.75
for para in blurb.split("\n\n"):
    for line in textwrap.wrap(para, 68):
        ax.text(bx, y, line, ha="left", va="top", color="#E8E8EE", fontsize=11, fontdict=serif); y -= 0.235
    y -= 0.2
ax.text(bx, bleed + 2.55, "Nick Polson is Robert Law Jr. Professor of Econometrics and Statistics at the\nBooth School of Business, University of Chicago. Vadim Sokolov is on the\nfaculty of George Mason University.", ha="left", va="top", color="#B9BED0", fontsize=9.5, fontdict=serif, linespacing=1.5)
# no box for the barcode: KDP prints its own, on a white background of its own, over whatever is there (lower right of the back cover)
fig.savefig("cover_wrap.pdf"); fig.savefig("cover_wrap.png", dpi=110)
# front only
fig2 = plt.figure(figsize=(trimw, trimh)); a2 = fig2.add_axes([0, 0, 1, 1]); a2.axis("off")
img = plt.imread("cover_wrap.png"); h, w = img.shape[:2]
x0 = int((fx0) / W * w); x1 = int((fx0 + trimw) / W * w); y0 = int(bleed / H * h); y1 = int((bleed + trimh) / H * h)
a2.imshow(img[h - y1:h - y0, x0:x1]); fig2.savefig("cover_front.png", dpi=110)
print(W, H, spine)
