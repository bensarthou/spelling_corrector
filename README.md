# Typo corrector using HMM

## Exploration and ameliorations of HMM algorithms

- [x] 2nd order HMM
- [x] Expand model to noisy characters insertions
- [x] Expand model to noisy characters deletions
- [ ] Unsupervised HMM training (EM)

- [x] Initialization of matrices with simple uniform probablility distribution (~ Laplace smoothing)
- [x] More complex smoothing for HMM2
- [ ] Better smoothing for proba estimations?

- [x] Acceleration of Viterbi algorithm with numpy broadcasting
- [x] Change floating point numbers precision in case of large states or observations sets
- [x] Use log-probabilities

[HOHMM implementation](https://github.com/jacobkrantz/Simple-HOHMM/blob/master/)

"Extended Viterbi algorithm for second order hidden Markov process"


## Noisy insertion of characters

State   Observation
```
a    -> a
c    -> d
b    -> a
.    -> b
```
Idea: Add an "empty state"


## Omission of characters

State   Observation
```
a    -> a
c    -> d
b    -> .
b    -> b

a    -> a
c    -> d
bb   -> b
```

Idea: Changing state space from S (of size N) to state space SxS (of size N^2)

For the database, randomly select 2 successives letters (of same word if possible)
and concatenate the states (making the first observation disappear)
