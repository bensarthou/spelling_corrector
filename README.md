# Typo corrector using HMM

## Exploration of HMM algorithms

- forward-backward
- 2nd order HMM
- smoothing for proba estimations
- 2nd order observations
- add initial states for state 0 and 1
- correct init proba estimation
- add unsupervised HMM training

[HOHMM implementation](https://github.com/jacobkrantz/Simple-HOHMM/blob/master/)



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
