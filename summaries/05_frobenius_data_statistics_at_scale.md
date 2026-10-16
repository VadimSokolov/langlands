# Chapter 5. Frobenius Data: Statistics at Scale

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 5. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23419248. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

How fast do the normalised Frobenius traces of an elliptic curve approach the Sato Tate law, what correction of size log x encodes the rank, and how often does a trace take a fixed value? The chapter computes traces by point counting for a curve without complex multiplication, a CM curve, Ramanujan's tau function, two hundred random curves and the Cremona curves 11a1, 37a1, 389a1 and 5077a1 of ranks 0 to 3. It relates the Kolmogorov discrepancy to Weyl sums, recalls that a square root rate would imply the Riemann hypothesis for every symmetric power L function, and compares the scaled discrepancy across curves with the Kolmogorov law of the Brownian bridge. Under the Riemann hypothesis for the symmetric powers, the explicit formula damps each fixed Chebyshev frequency below the scale of independent sampling, and the chapter conjectures that the scaled discrepancy tends to zero in logarithmic density. Under the Riemann hypothesis for L(s, E), the sums of a_p log p / p, divided by log x, tend in logarithmic mean to one half minus the analytic rank, a rank bias from the central zero and the prime squares. For the Lang Trotter conjecture it derives the heuristic, computes the constant of a Serre curve and counts supersingular primes. The data test the explicit formula, not the Birch and Swinnerton Dyer conjecture.

## From the literature

- Point counting (Schoof 1985; Harvey 2014).
- The Erdős Turán inequality (Erdős and Turán 1948; Niederreiter 1991 for Sato Tate).
- The discrepancy conjecture and the Riemann hypothesis (Akiyama and Tanigawa 1999; Mazur 2008); effective Sato Tate (V. K. Murty 1985; Rouse and Thorner 2017; Thorner 2021).
- Limiting distributions (Kolmogorov 1933; Rubinstein and Sarnak 1994).
- Partial Euler products and the rank (Birch and Swinnerton Dyer 1965; Goldfeld 1982; Kuo and Murty 2005; Conrad 2005; Kim and Murty 2023).
- The Lang Trotter conjecture (Lang and Trotter 1976), with the open image theorem (Serre 1972), supersingular primes (Elkies 1987) and averages over families (Fouvry and Murty 1996; David and Pappalardi 1999).

## New results

One conjecture, the logarithmic density form of the discrepancy at the square root scale, is the authors' own.

<!-- results:begin -->
- [Sato Tate discrepancy vanishes at the square root scale in logarithmic density](../RESULTS.md#sato_tate_discrepancy_logarithmic_density_conjecture) (Conjecture 5.3, page 67; open): For an elliptic curve over the rationals without complex multiplication, the Kolmogorov distance between the empirical law of its first n normalised Frobenius traces and the Sato Tate law is conjectured to be, in logarithmic density, of smaller order than the inverse square root of n that independent sampling gives; the same conjecture records the rate conjectured by Akiyama and Tanigawa. New in this book.
<!-- results:end -->

## Key terms

Frobenius traces, point counting, Schoof algorithm, Kolmogorov discrepancy, Sato Tate discrepancy, effective Sato Tate, Erdős Turán inequality, Akiyama Tanigawa conjecture, Brownian bridge, explicit formula, logarithmic density, rank bias, Birch and Swinnerton Dyer, Lang Trotter conjecture, supersingular primes, Serre curve
