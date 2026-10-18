# Chapter 32. A Computational Programme

<!-- cite:begin -->
From *On Langlands Conjectures: Probabilistic Methods* by Nick Polson and Vadim Sokolov (2026), chapter 32. Cite as Polson and Sokolov (2026), *On Langlands Conjectures: Probabilistic Methods*, edition 1, https://doi.org/10.5281/zenodo.23445874. Book page: https://vsokolov.org/books/langlands/. Source: https://github.com/VadimSokolov/langlands.
<!-- cite:end -->

How should the computations suggested by a probabilistic reading of the Langlands programme be organised, scaled and reported? This chapter gathers the open computational problems posed at the ends of the earlier chapters into one programme, the agenda of the second author's computational work. It sets standards for experiments on deterministic arithmetic sequences: certified computation where possible (point counts, Hecke matrices, isogeny graphs, zeros with rigorous error bounds), statistical tests whose statistic, null law and reference population are fixed in advance, breadth across families rather than depth when the conjectured limit is a law, and tables that can be regenerated from a script and a seed. The problems fall into six workstreams: Frobenius data at scale (point counting in average polynomial time, a Sato Tate atlas, Lang Trotter constants, Koblitz's conjecture, Chebotarev races); families and random matrices (one level densities, moments, spacings of zeros); buildings, graphs and hitting times (Plancherel rates, Kemeny constants, critical groups of isogeny graphs); L functions as laws (Thorin measures, the de Bruijn Newman constant, complete monotonicity and Pólya kernel tests); inference (classifiers for Sato Tate groups, Haar priors); and Kloosterman sums, Cohen Lenstra statistics and log correlated fields. A table gives each problem a scale and a priority, and a first year plan orders six steps, later ones reusing the infrastructure of earlier ones. The chapter poses tasks and proves nothing.

## From the literature

The programme builds on point counting in average polynomial time (Harvey 2014; Harvey and Sutherland 2014), the LMFDB and PARI/GP (The PARI Group 2023), and compares data with results and predictions such as the moment conjectures of Keating and Snaith (2000), the random Fourier series limit of Kloosterman paths (Kowalski and Sawin 2016), the maxima predicted by Fyodorov, Hiary and Keating (2012) and the Haar model for the sandpile groups of supersingular isogeny graphs (Munier and Shnidman 2023). Its Notes recall the machine computations of Birch and Swinnerton Dyer (1965) behind their conjecture.

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

computational programme, computational number theory, experimental mathematics, reproducibility, certified computation, interval arithmetic, ball arithmetic, point counting, average polynomial time, Frobenius traces, Sato Tate atlas, Lang Trotter constants, Koblitz conjecture, Chebotarev races, one level density, Hecke eigenvalues, isogeny graphs, Kemeny constant, Thorin measure, de Bruijn Newman constant, complete monotonicity, Pólya kernel, sequential classifier, Sato Tate groups, Kloosterman sums, Cohen Lenstra heuristics, log correlated fields, LMFDB, PARI/GP, Sage, Magma, smalljac
