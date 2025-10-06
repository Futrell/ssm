# TSL function

    projection function
todo: make a separate file to run the model with the processed data

1. write a loop to go through all the files in the data folder; mlregtest
2. run the code on realistic corpora
3. pTSL implementation

Plan:
MLRegTest
Even language with similar properties, entropy
Learn different formal classes --- email Shunjie
Venue: Language--some linguistics venues

June - July: Contact RAs, debug code, and run simulations
July - August: Grant submission (NSF), write the paper
August - Sept:  write the paper

Fall semester: submit something

Dataset:

Dev
A: Adverserial (matched)
R: Random
S: Short
L: Long

k = 2 for 2.1.0

* testing pairs, only keep those that have the matching length of grammatical and ungrammatical strings.
* what's the role of context window---cannot compute a nonlinear function
  * it's not really a thing, you can only look at one thing at a time, so no XOR
* Complexity hierarchy circuit
  * what amount of time
* A. Gradient learning is good; Horning paper---Soloman's induction
* B. SSM is good
* Avcu paper
* Everything is learnable depends on what cost
* RQs:
  * is gradient-based learning viable, can it really generalize on hard-coded models
  * can we learn in SSM, PFAs?
  * if we actually learn a SSM, can it actually biases towards the SL or SP languages; simpler SL patterns
  * having more parameters make you learn better?
    * two different parameterizations e.g. PFA characterized by one T, vs. PFA characterized by two Ts.
  * Cognition: repeated patterns + dimensionality reduction (Rabuseau)



chmod +x run_evals.sh

./run_evals.sh
