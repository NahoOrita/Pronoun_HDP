Author: Naho Orita (naho@umd.edu)

This is a python implementation of a non-parametric author topic model for simulation of pronoun category learning ([Orita et al. 2013](http://mindmodeling.org/cogsci2013/papers/0569/paper0569)). This code comes with no guarantees. You will need to have the NLTK package installed.

### How to use

`> python pronoun_HDP.py iterations alpha gamma beta`

* iterations: a number of iterations (say 2000?)	
* alpha, gamma, beta: hyperparameters (described in Orita et al. 2013)

### Data

*input.py* contains input data for this simulation.
(50 tokens: 25 reflexive pronouns and 25 non-reflexive pronouns)

