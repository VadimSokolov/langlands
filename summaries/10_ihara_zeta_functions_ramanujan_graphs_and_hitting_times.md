# Chapter 10. Ihara Zeta Functions, Ramanujan Graphs and Hitting Times

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 10. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23503453. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

What does the graph Riemann hypothesis say about random walk? The chapter studies the Ihara zeta function of a finite regular graph through the Ihara Bass determinant formula and the Hashimoto nonbacktracking matrix, and proves three forms of the graph Riemann hypothesis equivalent: nontrivial poles on the critical circle, the Ramanujan eigenvalue bound, and return probabilities approaching equilibrium exponentially, no slower than the Kesten rate of the tree (the third form a reformulation). Supersingular isogeny graphs are quotients of the Bruhat Tits tree whose adjacency matrices, Brandt matrices, carry the Hecke eigenvalues of weight two cusp forms of prime level N, so they are Ramanujan. The main theorem: as N grows through primes congruent to 1 mod 12, the Kemeny constant of the supersingular ℓ isogeny graph (the mean hitting time of a uniformly random target) divided by the number of vertices tends to the expected number of visits to the root of simple random walk on the (ℓ+1) regular tree: 2 for ℓ = 2 and 3/2 for ℓ = 3, by Serre's vertical Sato Tate theorem and the Ramanujan bound. A trace series, hitting time bounds, data at 207 levels and isogeny cryptography complete the chapter.

## From the literature

- The Ihara zeta function (Ihara 1966), extended to all graphs (Sunada 1986), and its determinant formula (Hashimoto 1989; Bass 1992).
- Ramanujan graphs (Lubotzky, Phillips and Sarnak 1988; Margulis 1988; Morgenstern 1994); the Alon Boppana bound (Nilli 1991); the Ramanujan bound (Eichler 1954; Deligne 1974).
- Supersingular isogeny graphs, Brandt matrices and their spectra (Deuring 1941; Eichler 1954; Jacquet and Langlands 1970; Mestre 1986; Gross 1987; Pizer 1990).
- Random regular graphs (Friedman 2008; Bordenave 2020; Marcus, Spielman and Srivastava 2015, 2018; Huang, McKenzie and Yau, preprint 2024).
- Kemeny's constant (Kemeny and Snell 1960); cover times of random regular graphs (Cooper and Frieze 2005).
- Isogeny based cryptography (Charles, Goren and Lauter 2009; Jao and De Feo 2011; Delfs and Galbraith 2016; Wesolowski 2022; Castryck and Decru 2023).

## New results

The Kemeny limit is new as far as the authors know; of the three forms only the mixing form, a reformulation, is the book's.

<!-- results:begin -->
- [Three equivalent forms of the graph Riemann hypothesis, Ihara poles, Ramanujan bound and mixing at the Kesten rate](../RESULTS.md#graph_riemann_hypothesis_three_equivalent_forms) (Proposition 10.3, page 148; proved): For a connected regular graph the Riemann hypothesis for its Ihara zeta function is equivalent both to the Ramanujan eigenvalue bound and to return probabilities approaching equilibrium at an exponential rate no slower than the rate at which those of the regular tree decay to zero, with the exact rate set by the largest absolute value of a nontrivial eigenvalue; the first equivalence is standard, and the mixing form is the book's reformulation. New in this book.
- [Limit of the normalised Kemeny constants of supersingular isogeny graphs](../RESULTS.md#kemeny_constant_supersingular_isogeny_graphs_limit) (Theorem 10.6, page 151; proved): As the characteristic grows through primes congruent to 1 modulo 12, with the prime isogeny degree fixed, the mean time for simple random walk on a supersingular isogeny graph to hit a uniformly random curve is asymptotic to the number of vertices times the expected number of visits of simple random walk on the regular tree of the same degree to its starting point, so about twice the number of vertices for 2 isogenies. New in this book.
<!-- results:end -->

## Key terms

Ihara zeta function, Ihara Bass formula, nonbacktracking matrix, graph Riemann hypothesis, Ramanujan graph, Alon Boppana bound, Kesten rate, supersingular isogeny graph, Pizer graph, Brandt matrix, Kemeny constant, mean hitting time, Green function, vertical Sato Tate, Eichler Selberg trace formula, isogeny based cryptography, CGL hash
