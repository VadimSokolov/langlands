# Chapter 19. Signed Thorin Measures

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 19. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23445874. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

By the Thorin criterion, the generalized Riemann hypothesis (GRH) for a self dual automorphic L function holds exactly when its Thorin measure, with atoms at the squared ordinates of the zeros, is positive. The chapter computes, with no hypothesis on the zeros, a smoothed signed version of that measure: the Stieltjes inversion, at distance epsilon above the cut, of the logarithmic derivative of the clock Λ(c + √s)/Λ(c). Zeros on the critical line give positive Lorentzians; a pair of zeros off the line gives a zero mass term with a negative lobe, which vanishes as epsilon tends to zero and can be masked by a nearby zero on the line, so one width excludes nothing by itself; positivity at every width, with no real zeros, is the Herglotz form of the criterion, which contains GRH and is not evidence for it. Uncertified computations for zeta, the real character modulo 12 and the Ramanujan Delta function find the signed density positive on finite windows and match zero counts by the argument principle with sign changes on the critical line. An opening section explains where Scholze's work enters the Langlands programme and why it does not enter these computations.

## From the literature

- Continuation and functional equation of the standard L functions of GL(n) (Godement and Jacquet 1972), after Hecke (1918, 1936, 1937) and Tate (1967); the conductor (Jacquet, Piatetski Shapiro and Shalika 1981).
- Galois representations for torsion classes (Scholze 2015); a new proof (Scholze 2013) of the local Langlands correspondence of Harris and Taylor (2001) and Henniart (2000); potential automorphy over CM fields (Allen, Calegari, Caraiani, Gee, Helm, Le Hung, Newton, Scholze, Taylor and Thorne 2023); L parameters (Fargues and Scholze 2026).
- Stieltjes inversion (Stieltjes 1894; Akhiezer 1965); Pólya's kernel (Pólya 1926); theta functions with positive real zeros (Louboutin and Munsch 2013).
- Zero verifications (Backlund 1914; Rumely 1993; Platt and Trudgian 2021; Hiary, Ireland and Kyi 2027) and Turing's method (Turing 1953; Booker 2006).
- From the authors' papers: the Thorin theory of xi that the criterion extends (Polson 2018); positivity of the Thorin density of xi forces every extremal zero onto the critical line (Polson 2026).

## New results

Its one proposition is an elementary consequence of the product over the zeros.

<!-- results:begin -->
- [Negative lobe of the signed Thorin density](../RESULTS.md#signed_thorin_density_negative_lobe) (Proposition 19.1, page 325; proved): In the signed Thorin density obtained by Stieltjes inversion of the logarithmic derivative of an automorphic clock, zeros on the critical line give positive Lorentzians, a pair of zeros off the line contributes a term of total mass zero, negative near its centre when the smoothing width is below twice the ordinate times the distance of the zero from the line and tending to zero with the width, and with no real zeros positivity at every width is equivalent to all zeros lying on the line. New in this book.
- [Characterising positive Pólya kernels of self dual L functions](../RESULTS.md#positive_polya_kernels_characterisation) (Open Problem 19.1, page 329; open): Asks which self dual L functions with root number one have a positive Pólya kernel, and whether at fixed degree positivity is decided by finitely many coefficients and the conductor. New in this book.
- [Masking of the negative lobe of a signed Thorin density](../RESULTS.md#negative_lobe_masking) (Open Problem 19.2, page 329; open): Asks under which conditions on the nearby zeros positivity of the signed Thorin density at a single smoothing width rules out a zero off the critical line, given that a zero on the line can mask the negative lobe. New in this book.
<!-- results:end -->

## Key terms

signed Thorin density, Herglotz function, Pick function, Nevanlinna function, generalized gamma convolution, GGC, Turing's method, Pólya kernel, chi_12, Ramanujan Delta function, potential automorphy, local Langlands correspondence
