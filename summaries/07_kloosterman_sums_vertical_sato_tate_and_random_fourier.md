# Chapter 7. Kloosterman Sums: Vertical Sato Tate and Random Fourier Series

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 7. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23503453. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

How are Kloosterman sums distributed, and what do their partial sums look like? The chapter reads Kloosterman sums Kl(a; p) as Frobenius traces on the Kloosterman sheaf and presents Weil's bound, Katz's vertical Sato Tate theorem (as the prime grows, the Kloosterman angles become Sato Tate distributed, since the monodromy group is SL(2)) and joint equidistribution of additive shifts. Through the Bruggeman Kuznetsov formula and Selberg's 3/16 theorem it links Kloosterman sums with Maass forms and reads the Linnik Selberg conjecture as the absence of drift; Katz's horizontal conjecture remains open. It then treats the Kloosterman paths of Kowalski and Sawin, whose finite dimensional distributions converge to a random Fourier series with independent Sato Tate coefficients. It proves this convergence from the joint equidistribution by completion and a mean square tail bound, and proves, as its own formulations of facts implicit in or easily derived from Kowalski and Sawin's work, exact second moments of the paths at every odd prime, the covariance and a reflection symmetry of the limit (a Sato Tate Brownian motion), a subgaussian tail bound, and a variance deficit, 1/π² for the imaginary part at the midpoint, when the parameter multiplies the inverse.

## From the literature

- Kloosterman's bound of order p^(3/4) (Kloosterman 1926).
- The Weil bound |Kl(a, b; p)| ≤ 2√p for ab prime to p (Weil 1948), with the elementary method of Stepanov (1969) in Bombieri's form (1974).
- Katz's vertical Sato Tate theorem and the monodromy of Kloosterman sheaves (Katz 1988), through Deligne's equidistribution theorem (Deligne 1980).
- The Bruggeman Kuznetsov formula (Bruggeman 1978; Kuznetsov 1980) and Kuznetsov's bound x^(1/6+ε) for sums of Kloosterman sums (Kuznetsov 1980).
- Selberg's 3/16 theorem (Selberg 1965) and the bound 975/4096 (Kim and Sarnak 2003).
- Kloosterman paths, the completion formula, the joint equidistribution of shifted sums, finite dimensional convergence to the random Fourier series and its doubly exponential tails (Kowalski and Sawin 2016).
- Maxima of partial sums (Lamzouri 2020; Autissier, Bonolis and Lamzouri 2021) and paths for prime power moduli (Ricotta and Royer 2018).

## New results

<!-- results:begin -->
- [Exact second moments of Kloosterman paths at every odd prime](../RESULTS.md#kloosterman_paths_exact_second_moments) (Proposition 7.13, page 101; proved): Exact formulas, valid at every odd prime, for the mean and the second moments of a normalised partial Kloosterman sum averaged over its parameter, in both natural parametrisations, which the book proves as its own formulation of facts implicit in or easily derived from the work of Kowalski and Sawin. New in this book.
- [Variance deficit of Kloosterman paths when the parameter multiplies the inverse](../RESULTS.md#kloosterman_paths_variance_deficit) (Corollary 7.14, page 102; proved): When the parameter of a partial Kloosterman sum multiplies the inverse of the summation variable, the limiting variances of its real and imaginary parts fall below those of the Kloosterman random Fourier series by explicit trigonometric terms, by exactly one over pi squared for the imaginary part at the midpoint, which the book proves as its own formulation of facts implicit in or easily derived from the work of Kowalski and Sawin. New in this book.
- [Covariance and reflection symmetry of the Kloosterman random Fourier series](../RESULTS.md#kloosterman_random_fourier_series_covariance_and_symmetry) (Proposition 7.15, page 103; proved): The random Fourier series with independent Sato Tate coefficients that Kowalski and Sawin found as the limit of Kloosterman paths has the covariance of complex Brownian motion on the first half of the unit interval and an exact reflection symmetry that determines the second half, which leads the book to call it a Sato Tate Brownian motion, and the book proves this as its own formulation of facts implicit in or easily derived from their work. New in this book.
- [Unboundedness and subgaussian tail of the Kloosterman random Fourier series](../RESULTS.md#kloosterman_random_fourier_series_subgaussian_tail) (Proposition 7.16, page 105; proved): The imaginary part of the Kloosterman random Fourier series at the midpoint is unbounded in both directions yet obeys a Gaussian tail bound, and every square summable real combination of independent Sato Tate variables is subgaussian with variance proxy equal to its variance; the book proves this as its own formulation of facts implicit in or easily derived from the work of Kowalski and Sawin, and notes that the tail bound is far weaker than their doubly exponential estimates. New in this book.
<!-- results:end -->

## Key terms

Kloosterman sum, Kloosterman angle, Kloosterman sheaf, Weil bound, Katz equidistribution theorem, vertical Sato Tate, horizontal Sato Tate, Katz horizontal conjecture, semicircle law, Goursat Kolchin Ribet criterion, Bruggeman Kuznetsov formula, Selberg 3/16 theorem, Selberg eigenvalue conjecture, Linnik Selberg conjecture, Maass forms, Kloosterman paths, partial Kloosterman sums, completion method, random Fourier series, Sato Tate Brownian motion, variance deficit, Birch sums, exponential sums over finite fields
