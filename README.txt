@author: Naho Orita (naho@umd.edu)

This is a python implementation of a non-parametric author topic
model using Hierarchical Dirichlet Process for simulation of pronoun
category learning (Orita et al. 2013). 
(paper here: http://mindmodeling.org/cogsci2013/papers/0569/paper0569)

*** This code comes with no guarantees. ***
You will need to have the NLTK package installed.

How to use:
> python pronoun_HDP.py iterations alpha gamma beta

:iterations: a number of iterations (say 2000?)	
:alpha, gamma, beta: hyperparameters (described in Orita et al. 2013)

Data:
'input.py' contains input data for this simulation.
(50 tokens: 25 reflexive pronouns and 25 non-reflexive pronouns)
Put 'input.py' and 'pronoun_HDP.py' in a same directory.

References:
[1] Teh et al. 2006. Hierarchical Dirichlet processes. 
[2] Rosen-Zvi et al. 2004. The author-topic model for authors and
documents.
