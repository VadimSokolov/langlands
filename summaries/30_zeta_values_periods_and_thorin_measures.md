# Chapter 30. Zeta Values, Periods and Thorin Measures

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 30. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23904867. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

What probabilistic object is the number zeta(3)? The earlier chapters treat an L function as a function of s and ask for its Thorin measure; this one looks at the values themselves. The odd zeta values turn out to be the odd cumulants of the Gumbel law, whose Thorin measure is the counting measure on the positive integers, and the values L(n, chi) are signed combinations of cumulants of log Gamma laws.

The main construction starts from Apery's continued fraction. Its Jacobi matrix, with cubic entries, is diagonally dominant and positive definite, so it has a spectral measure on the half line from 4, which is a Thorin measure, and zeta(3)/6 is its first inverse moment. The generalized gamma convolution it defines has mean zeta(3)/6 and is, to six digits, a single Gamma variable of shape 0.99992 and rate 4.99106. Apery's denominators are the orthonormal polynomials of the measure at zero, and his linear forms are the coefficients of 1/x in those polynomials, which gives the errors of Gauss quadrature. The moment problem is determinate. The arithmetic of the denominators is not explained by any of this, and the chapter says so.

The fraction for zeta(2) has plus signs. It comes from an indefinite matrix, with a probability measure on both sides of the origin, so no Thorin measure arises in this way. The Beukers variable, whose mean is twice zeta(3), is shown not to be a generalized gamma convolution, although the derivative checks at interior points all pass; the obstruction sits at the two ends of its support.

Perturbation theory gives the small eigenvalues and weights of the matrix. For large level the couplings exceed the gaps, and a Bohr Sommerfeld and WKB heuristic, supported by computation to level 3000, gives slightly different constants, which are stated as an open problem. Nothing here proves anything about irrationality.

## From the literature

Irrationality of zeta(3) is due to Apery (1979), with the integral of Beukers (1979). Infinitely many odd zeta values are irrational by Rivoal and by Ball and Rivoal; Zudilin showed that one of zeta(5), zeta(7), zeta(9), zeta(11) is. The 2 adic value zeta_2(5) and the independence of 1, zeta(2), L(2, chi_-3) are due to Calegari, Dimitrov and Tang. The motivic setting is due to Deligne and Goncharov and to Brown. The theory of Thorin measures is that of Thorin and Bondesson.

## New results

<!-- results:begin -->
- [Apery's continued fraction as the first inverse moment of a Thorin measure](../RESULTS.md#apery_fraction_inverse_moment_of_thorin_measure) (Theorem 30.4, page 553; proved): The spectral measure of the positive Jacobi matrix with diagonal 34n^3+51n^2+27n+5 and off diagonal n^3 is a Thorin measure of mass one on the half line from 4, whose first inverse moment is zeta(3)/6, so the generalized gamma convolution it defines has mean zeta(3)/6. New in this book.
- [Apery's sequences as the orthogonal polynomials of the Thorin measure at zero](../RESULTS.md#apery_sequences_as_orthogonal_polynomials_at_zero) (Theorem 30.7, page 556; proved): The orthonormal polynomials of the Apery measure at zero are, up to sign, Apery's denominators, its second kind polynomials are his numerators over 6, his linear forms are the coefficients of 1/x and give the errors of Gauss quadrature, and the moment problem is determinate. New in this book.
- [The Beukers variable is not a generalized gamma convolution](../RESULTS.md#beukers_variable_is_not_a_ggc) (Theorem 30.9, page 558; proved): The variable G over 1 minus exp(-G) with G of law Gamma(2), whose mean is twice zeta(3), is not a generalized gamma convolution with or without drift. New in this book.
- [Eigenvalues and weights of the Apery matrix at large level](../RESULTS.md#apery_matrix_large_level_asymptotics_open_problem) (Open Problem 30.1, page 560; open): Bound the errors of the perturbative formulas for the Apery matrix at small level and prove the behaviour found numerically at large level, namely eigenvalues asymptotic to C k^3 with C=33.960748 and log w_k over k tending to -12.16024. New in this book.
- [The arithmetic of Apery's denominators as a property of the measure](../RESULTS.md#arithmetic_of_apery_denominators_from_the_measure_open_problem) (Open Problem 30.2, page 560; open): Find the property of the Apery measure, or of its Jacobi matrix, that is equivalent to twice the cube of lcm(1,...,n) times the second kind polynomial at zero being an integer. New in this book.
- [Laguerre and Legendre expansions of the Beukers function and the Apery measure](../RESULTS.md#laguerre_and_legendre_expansions_of_beukers_variable_open_problem) (Open Problem 30.3, page 560; open): Expand g over 1 minus exp(-g) in the Laguerre polynomials of the Gamma(2) weight and in the Legendre system of Beukers on the cube, relate the coefficients to the Apery measure, and decide whether the Laguerre coefficients are Apery like. New in this book.
- [Hyperbolic complete monotonicity of the Apery generalized gamma convolution](../RESULTS.md#hyperbolic_complete_monotonicity_of_apery_ggc_open_problem) (Open Problem 30.4, page 560; open): Decide whether the generalized gamma convolution of the Apery measure is hyperbolically completely monotone, and whether it admits a van Dantzig pair. New in this book.
- [The Jacobi construction for other zeta recurrences and linear forms](../RESULTS.md#jacobi_construction_for_other_zeta_recurrences_open_problem) (Open Problem 30.5, page 561; open): Repeat the Jacobi construction for Zudilin's recurrence for zeta(4) and for linear forms in odd zeta values, decide which matrices are positive definite and find their Thorin measures. New in this book.
- [Positive Jacobi matrices for Dirichlet L values without a closed form](../RESULTS.md#positive_jacobi_matrices_for_dirichlet_values_open_problem) (Open Problem 30.6, page 561; open): For L(3,chi_12) and L(2,chi_-3) find a three term recurrence with polynomial coefficients and a positive Jacobi matrix whose first inverse moment is the value. New in this book.
<!-- results:end -->

## Key terms

zeta values, Apery's theorem, Apery's continued fraction, Jacobi matrix, spectral measure, orthogonal polynomials, Gauss quadrature, determinate moment problem, Thorin measure, generalized gamma convolution, Gumbel law, cumulants, log Gamma law, Hurwitz zeta function, Dirichlet L values, Beukers integral, Legendre control variates, perturbation theory, WKB, Bohr Sommerfeld, mixed Tate motives, periods
