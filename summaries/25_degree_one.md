# Chapter 25. Degree One

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 25. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23445874. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

What do the book's probabilistic readings of reciprocity give in degree one, where reciprocity is a theorem of class field theory? Dirichlet's theorem on primes in arithmetic progressions becomes a Dirichlet process limit for the residue classes of the prime factors of a zeta distributed random integer, with Mertens constants for progressions from L(1, χ) and Landau's count of integers avoiding a class as the atom at zero; Hecke's equidistribution of the angles of Gaussian primes becomes a Dirichlet process on the circle. For an elliptic curve with complex multiplication, Deuring's theorem gives a Frobenius angle law half uniform and half an atom, and the chapter proves that base change to the CM field squares the Rankin Selberg Euler product exactly, so the Gamma shape of the Rankin Selberg subordinator over that field is 2, while a curve without complex multiplication by that field keeps shape 1: an unconditional test for complex multiplication. For a primitive real Dirichlet character with L(1/2, χ) nonzero, the generalized Riemann hypothesis (GRH) for L(s, χ) is shown equivalent to a generalized gamma convolution representation with Thorin measure on the squared ordinates of the zeros, a reformulation and not a proof, and a shifted Thorin representation holds unconditionally at every basepoint from 1 on.

## From the literature

- The Kronecker Weber theorem and Artin's reciprocity law (Artin 1927; account in Neukirch 1999), Hecke's Grössencharaktere (Hecke 1920) and Tate's thesis (Tate 1950).
- Mertens' theorem for arithmetic progressions (Williams 1974) and the Dirichlet process (Ferguson 1973).
- Landau's theorem on sums of two squares (Landau 1908) and Wirsing's mean value theorem (Wirsing 1961).
- Equidistribution of the angles of Gaussian primes (Hecke 1920), with its discrepancy under GRH (Kubilius 1955).
- The L function of an elliptic curve with complex multiplication is a Hecke L function (Deuring 1953).
- The primes with a_p = 0 have density zero on a curve without complex multiplication (Serre 1981); Faltings' isogeny theorem (Faltings 1983).

## New results

The catalogued theorems carry the book's Gamma shape and Thorin results to degree one; none proves a case of GRH.

<!-- results:begin -->
- [Complex multiplication doubles the Gamma shape of the Rankin Selberg subordinator](../RESULTS.md#complex_multiplication_doubles_the_gamma_shape) (Theorem 25.7, page 475; proved): Over its CM field the Rankin Selberg Euler product of an elliptic curve with complex multiplication is exactly the square of its Euler product over the rationals, so the limiting Gamma shape of the associated subordinator doubles from 1 to 2, while a curve without complex multiplication by that field keeps shape 1, which gives an unconditional test for complex multiplication. New in this book.
- [Thorin form of the generalized Riemann hypothesis for a real Dirichlet character](../RESULTS.md#thorin_form_of_grh_for_real_dirichlet_characters) (Theorem 25.8, page 477; proved): For a real primitive Dirichlet character whose L function does not vanish at the centre, the generalized Riemann hypothesis is equivalent to a generalized gamma convolution representation of the completed L function with Thorin measure on the squared heights of the zeros, and a shifted representation holds unconditionally; this reformulates the hypothesis and does not prove it. New in this book.
- [Limit law for the discrepancy of the Hecke angles of Gaussian primes](../RESULTS.md#hecke_angle_discrepancy_limit_law) (Open Problem 25.1, page 480; open): Asks whether the normalised discrepancy of the Hecke angles of Gaussian primes has a limit law, the Kolmogorov law that a Brownian bridge predicts or a different law forced by arithmetic, and the same for the other imaginary quadratic fields of class number one. New in this book.
- [Euler Kronecker constants of Hecke characters and the rate of the Gaussian prime Dirichlet process](../RESULTS.md#euler_kronecker_constants_of_hecke_characters) (Open Problem 25.2, page 480; open): Asks for the constants that govern how fast the Dirichlet process of Gaussian prime angles is approached, the sign pattern of the deficits of arcs, and an analogue for arcs of the theorem on atoms at zero. New in this book.
- [An effective Gamma shape test for complex multiplication](../RESULTS.md#effective_gamma_shape_test_for_complex_multiplication) (Open Problem 25.3, page 480; open): Asks for an explicit bound on the number of primes after which the empirical Gamma shape provably decides whether an elliptic curve has complex multiplication by a given field. New in this book.
<!-- results:end -->

## Key terms

GL(1), degree one, class field theory, Kronecker Weber, Artin reciprocity, Dirichlet characters, Hecke characters, Grössencharaktere, Gaussian primes, Hecke angles, Dirichlet process, zeta distributed random integer, Mertens constants for progressions, Landau Ramanujan constant, Wirsing's theorem, complex multiplication, CM elliptic curves, Deuring's theorem, Rankin Selberg subordinator, Gamma shape test, generalized gamma convolution, GGC, Thorin measure, generalized Riemann hypothesis, GRH
