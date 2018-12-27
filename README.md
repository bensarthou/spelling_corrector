# Typo corrector using HMM


Typo corrector using Hidden Markov Model, for course TC4 of master AIC at Université Paris-Sud.

Autors : N. Cadart, B. Sarthou

Date : December 2018


## Exploration and ameliorations of HMM algorithms

- [x] 1st order HMM
- [x] 2nd order HMM
- [x] Expand model to noisy characters insertions
- [x] Expand model to noisy characters deletions
- [x] Unsupervised training (EM) for HMM1

- [x] Initialization of matrices with simple uniform probablility distribution (~ Laplace smoothing)
- [x] More complex smoothing for proba estimations for HMM2

- [x] Acceleration of Viterbi algorithm with numpy broadcasting
- [x] Change floating point numbers precision in case of large states or observations sets
- [x] Use log-probabilities


"Extended Viterbi algorithm for second order hidden Markov process"
