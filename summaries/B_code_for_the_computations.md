# Appendix B. Code for the Computations

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), appendix B. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23419248. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

Which code lies behind the numbers and figures of the book, and how is it organised? This appendix prints the core routines in condensed, cumulative Python listings that call PARI/GP through cypari2, with NumPy, SciPy and mpmath, and states in comments the values they return: Frobenius traces of elliptic curves by Legendre symbols, checked against PARI's ellap, and point counts on genus two curves; a PARI/GP function that builds supersingular isogeny graphs of prime degree from a root of a Hilbert class polynomial and the classical modular polynomials, whose adjacency matrices for primes congruent to 1 mod 12 are the Brandt matrices; Kemeny constants by the spectral formula, and hitting times both by direct solution and through the fundamental matrix; a Monte Carlo and quadrature evaluation of the extreme value representation of L(s, Delta) with the regularised incomplete gamma function; the Hadamard product over the zeros of L(s, Delta) behind its generalized gamma convolution, with a correction for the zeros above the cutoff; and clipped log Bayes factors between Sato Tate groups. A last section lists the stage scripts and JSON result files with the chapters that use them, the folders that hold the code of the later chapters, the archive of the scripts in a layout with relative paths, and the scripts that draw the figures, with notes on the order in which the scripts run and the files they read and write, on which of them reproduce the stored results when rerun and which do not run as they stand, on the two chapters that came without scripts, on the few printed numbers that come from checks made for the book, and on the scripts of Chapters 25 and 26, which have not been rerun for it. The appendix states no theorem.

## From the literature

The routines rely on PARI/GP (The PARI Group 2023) and on the fundamental matrix of a finite Markov chain (Kemeny and Snell 1960); the appendix presents code and the values it returns, not results.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

code, source code, Python, PARI/GP, cypari2, NumPy, SciPy, mpmath, reproducibility, scripts, JSON results, Frobenius traces, point counting, Legendre symbols, ellap, genus two curves, supersingular isogeny graph, Brandt matrix, Hilbert class polynomial, classical modular polynomial, Kemeny constant, hitting times, fundamental matrix, extreme value representation, incomplete gamma function, Monte Carlo, Hadamard product, zeros of L(s, Delta), lfunlambda, generalized gamma convolution, GGC, Bayes factor, Sato Tate group
