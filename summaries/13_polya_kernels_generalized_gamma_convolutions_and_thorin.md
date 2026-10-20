# Chapter 13. Pólya Kernels, Generalized Gamma Convolutions and Thorin Measures

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 13. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23503453. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

Pólya studied Riemann's Xi function as the Fourier transform of a positive kernel and asked which kernels have transforms with only real zeros. This chapter carries his programme from ζ to the L function of the discriminant modular form Δ, in the language of generalized gamma convolutions (GGC) and Thorin measures. Suitably weighted, Δ on the imaginary axis in logarithmic coordinates is the density of a symmetric random variable W, the Pólya variable, whose moment generating function is Λ(6+z, Δ)/Λ(6, Δ); so the Riemann hypothesis for Δ is equivalent to the characteristic function of W having only real zeros. Hadamard factorisation gives, unconditionally, a product over the zeros and a variance identity for W, a trace formula in miniature. The main theorem shows that the Riemann hypothesis for Δ holds if and only if Λ(6, Δ)/Λ(6+√s, Δ) is the Laplace transform of a GGC, whose Thorin measure then puts unit mass at each squared ordinate; the same holds for every self dual cuspidal automorphic representation of GL(n), n ≥ 2, with nonzero central value. These are reformulations, not proofs; the chapter poses as open the construction of that GGC without reference to the zeros, Hilbert's eighth problem for GL(2).

## From the literature

- The Fourier integral of Riemann's Xi function and real zero theorems (Pólya 1926, 1927).
- Heat flow and the de Bruijn Newman constant (de Bruijn 1950; Newman 1976), its nonnegativity (Rodgers and Tao 2020), also for the extended Selberg class (Dobner 2021).
- Pólya frequency functions (Schoenberg 1951; Karlin 1968) and the Laguerre Pólya class (Levin 1964).
- Generalized gamma convolutions (Thorin 1977; Bondesson 1992).
- Bessel hitting times and the Riemann ξ function (Biane, Pitman and Yor 2001).
- A positive proportion of zeros on the critical line for level one Hecke eigenforms (Hafner 1983, after Selberg 1942).
- From the first author: the Thorin representation of the reciprocal Xi function (Polson 2017) and the equivalence of real zeros and the Thorin property for real entire functions of order at most one (Polson 2018).

## New results

The Notes call these theorems elementary consequences of Hecke's theory and Hadamard's factorisation.

<!-- results:begin -->
- [Pólya kernel of the discriminant form Delta and its characteristic function](../RESULTS.md#polya_kernel_of_the_discriminant_form) (Theorem 13.6, page 204; proved): The discriminant modular form on the imaginary axis, read in logarithmic coordinates and weighted, is the density of a symmetric random variable whose moment generating function is the normalised completed L function of the form, so that the Riemann hypothesis for this form is equivalent to the characteristic function of an explicit probability law having only real zeros. New in this book.
- [Hadamard product and variance identity for the Pólya variable of Delta](../RESULTS.md#hadamard_product_and_variance_identity_for_delta) (Theorem 13.7, page 205; proved): Unconditionally, the normalised completed L function of the discriminant form is a product over its zeros with no exponential factor, and the variance of its Pólya variable equals twice the sum of the inverse squares of the zero ordinates, a trace formula in miniature that the book checks against computed zeros. New in this book.
- [Riemann hypothesis for Delta as a generalized gamma convolution property](../RESULTS.md#riemann_hypothesis_for_delta_as_ggc_property) (Theorem 13.8, page 206; proved): Proves that the Riemann hypothesis for the L function of the discriminant form is equivalent to a reciprocal of its completed L function, in the square root variable, being the Laplace transform of a generalized gamma convolution, whose Thorin measure is then the sum of unit masses at the squared zero ordinates; it is a reformulation and proves no case of the Riemann hypothesis. New in this book.
- [Thorin criterion for the generalized Riemann hypothesis of self dual automorphic L functions](../RESULTS.md#grh_for_self_dual_l_functions_as_ggc_property) (Theorem 13.10, page 208; proved): For a self dual cuspidal automorphic representation of GL(n), n at least 2, with nonvanishing central value, the generalized Riemann hypothesis is equivalent to a reciprocal of the completed L function, in the square root variable, being the Laplace transform of a generalized gamma convolution, whose Thorin measure then sits on the squared zero ordinates; a reformulation that proves no case of the generalized Riemann hypothesis. New in this book.
<!-- results:end -->

## Key terms

Pólya kernel, Polya kernel, Pólya variable, Riemann Xi function, generalized gamma convolution, GGC, Thorin measure, complete monotonicity, Hadamard factorisation, Laguerre Pólya class, Pólya frequency function, de Bruijn Newman constant, Ramanujan Delta function, discriminant modular form, Riemann hypothesis for Delta, generalized Riemann hypothesis, GRH, self dual automorphic representation, Hilbert's eighth problem
