# Chapter 30. Bayesian Identification of Sato Tate Groups

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 30. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23445874. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

How can one tell from Frobenius traces which Sato Tate group an abelian variety has? The chapter recasts the moment matching of Fité, Kedlaya, Rotger and Sutherland as Bayesian model selection: the working model treats the normalised traces as an exchangeable sample from the image of Haar measure under the trace, and equidistribution takes the place of the strong law of large numbers. Its consistency theorem, new in this book, is deterministic: for a sequence equidistributed for a true law, atoms allowed, with finite divergence from one candidate and a log likelihood ratio continuous off the atoms, the clipped log Bayes factor of two candidates is asymptotically linear, and its slope tends, as the clipping level grows, to the difference of the Kullback Leibler divergences of the truth from the two candidates. For elliptic curves over the rationals this gives explicit rates for detecting complex multiplication. The chapter also treats Wald's sequential probability ratio test, moments against the likelihood, the trace laws of USp(4), SU(2)×SU(2) and diagonal SU(2), three genus 2 curves (one outside the candidate list), misspecification, and nonparametric and arithmetic priors. The theorems assume equidistribution and prove nothing new about the Sato Tate conjecture.

## From the literature

- The 52 Sato Tate groups of abelian surfaces (Fité, Kedlaya, Rotger and Sutherland 2012) and the moment method in genus 2 (Kedlaya and Sutherland 2009).
- Sato Tate for elliptic curves without complex multiplication (Barnet Lamb, Geraghty, Harris and Taylor 2011) and joint Sato Tate (Harris 2009).
- Exchangeability (de Finetti 1937; Hewitt and Savage 1955), consistency of maximum likelihood (Wald 1949), the Kullback Leibler divergence (Kullback and Leibler 1951).
- The sequential probability ratio test (Wald 1947; Wald and Wolfowitz 1948).
- Supersingular primes (Elkies 1987); hyperelliptic Jacobians without complex multiplication (Zarhin 2000).
- Misspecification (Berk 1966; Kleijn and van der Vaart 2006).
- Dirichlet processes (Ferguson 1973), Pólya trees (Lavine 1992), canonical moments (Dette and Studden 1997), empirical Bayes (Robbins 1956), endomorphism rings (Costa, Mascot, Sijsling and Voight 2019).

## New results

Both results are the authors' own by the chapter's Notes.

<!-- results:begin -->
- [Consistency of Bayes factors under equidistribution](../RESULTS.md#consistency_of_bayes_factors_under_equidistribution) (Theorem 30.1, page 555; proved): For any deterministic sequence, such as normalised Frobenius traces, that is equidistributed for a true law with finitely many atoms and finite Kullback Leibler divergence from one candidate law, and with a log likelihood ratio continuous off the atoms, the clipped log Bayes factor between two candidate laws is asymptotically linear in the number of observations, and as the clipping level grows its slope tends to the difference of the divergences of the true law from the two candidates, with no independence assumed. New in this book.
- [Bayes factor rates for detecting complex multiplication of elliptic curves](../RESULTS.md#bayes_factor_rates_for_complex_multiplication_of_elliptic_curves) (Corollary 30.3, page 557; proved): For an elliptic curve over the rationals, the clipped Bayes factor of a complex multiplication model against a Sato Tate model with a small atom at zero, computed from the normalised Frobenius traces at the good primes, has an explicit limiting rate per prime, as the clipping level grows, that is negative without complex multiplication and positive with it. New in this book.
<!-- results:end -->

## Key terms

Sato Tate group, Sato Tate distribution, abelian surface, genus 2 curve, Frobenius traces, L polynomial, Bayes factor, Bayesian model selection, posterior consistency, Kullback Leibler divergence, relative entropy, equidistribution, exchangeability, working model, misspecification, sequential probability ratio test, SPRT, complex multiplication, supersingular primes, arcsine law, semicircle law, USp(4), Weyl integration formula, moment method, Galois endomorphism type, Dirichlet process, Pólya tree, empirical Bayes
