"""
@author: Naho Orita (naho@umd.edu)

This is a python implementation of a non-parametric author topic
model for the simulation of pronoun category learning (Orita et
al. 2013).
"""

import sys
import time
from math import log, factorial
from random import random, randint

from nltk.probability import FreqDist, ConditionalFreqDist

# input data
from input import observing_pronouns, discourse_priors


def lgammln(xx):
    """
    Returns the gamma function of xx.
    Gamma(z) = Integral(0,infinity) of t^(z-1)exp(-t) dt.
    (Adapted from: Numerical Recipies in C.)
    Usage: lgammln(xx)
    Copied from stats.py by strang@nmr.mgh.harvard.edu
    """

    coeff = [76.18009173, -86.50532033, 24.01409822, -1.231739516,
             0.120858003e-2, -0.536382e-5]
    x = xx - 1.0
    tmp = x + 5.5
    tmp = tmp - (x + 0.5) * log(tmp)
    ser = 1.0
    for j in range(len(coeff)):
        x = x + 1
        ser = ser + coeff[j] / x
    return -tmp + log(2.50662827465 * ser)


def dict_sample(d):
    """
    Sample a key from a dictionary using the values as probabilities
    (unnormalized).
    Copied from lda.py Jordan Boyd-Graber
    """
    cutoff = random()
    normalizer = float(sum(d.values()))

    current = 0
    for i in d:
        assert(d[i] > 0)
        current += float(d[i]) / normalizer
        if current >= cutoff:
            return i
    assert False, "Didn't choose anything: %f %f" % (cutoff, current)


class State:
    """
    Store the assignments of data.
    """

    def __init__(self, alpha, beta, gamma, observing_pronouns, discourse_priors):
        """
        Create a new state for the model.
        
        :param alpha: controls new dishes in restaurants at the second
        level.
        :param beta: vocabulary Dirichlet hyperparameter.
        :param gamma: controls new dishes in a franchise restaurant at
        the first level.
        """
        
        # hyperparameters 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # vocabulary (10 types of pronouns)
        self.vocab = range(1,11)
        # V: vocab size
        self.V = 10
        # N: the number of data points
        self.N = len(observing_pronouns)
        # K: topic size (initially 50)
        self.K = self.N
        # J: the number of syntactic positions 
        # defined by [+-local] and [+-c-command] 
        self.J = 4
        
        # state for observing pronouns
        self.word_state = observing_pronouns
        
        # initial state for latent topics (i.e., pronoun categories)
        # (different topics are initially assigned to 50 data points)
        self.topic_state = [topic for topic in range(self.K)]
        
        # initial state for antecedents' syntactic positions
        # (randomly assign 0-3)
        self.antecedent_state = [randint(0, self.J-1) for a in
                                  range(self.N)]
        
        # priors from discourse
	self.discourse_priors = discourse_priors
        
        # a list of active topic indices
        self.used_topics = [topic for topic in range(self.K)]
        
        # a list of inactive topic indices (20 added)
        self.unused_topics = [topic for topic in range(self.K, 70)]
        
        # V_ki: the number of pronoun i in topic k
        self.V_ki = ConditionalFreqDist(zip(self.topic_state,
                                             self.word_state))
        
        # N_jk: the number of topic k in syntactic position j
        self.N_jk = ConditionalFreqDist(zip(self.antecedent_state,
                                             self.topic_state))
        
        # M_k: the number of topic k across all syntactic positions
        all_dishes = []
        for j in range(self.J):
            all_dishes.extend(self.N_jk[j].keys())
        self.M_k = FreqDist(all_dishes)

        
    def report(self):
        """
        Report of topic-word distribution.
        """
        
        readable_vocab = {1: 'me         ', 
                          2: 'us         ', 
                          3: 'you        ',
                          4: 'him        ',
                          5: 'them       ', 
                          6: 'myself     ', 
                          7: 'yourself   ',
                          8: 'himself    ',
                          9: 'themselves ',
                          10: 'ourselves '}

        # the freq of each pronoun in the input data
        n_token = {1: 4, 2: 1, 3: 10, 4: 7, 5: 3, 6: 4, 7: 10,
                   8: 7, 9: 3, 10: 1}

        print "\n Topic-word distribution \n"
        
	for topic in self.used_topics:
            N = self.V_ki[topic].N()
            print "==========\t Topic: #%d \t==========\n" % (topic)
                
            for word_ind in self.V_ki[topic].keys():
                print("%s\t%g" % (readable_vocab[word_ind],
                                  float(self.V_ki[topic][word_ind])/N))

        
    def change_topic(self, index, new_j, new_k):
        """
        Change the topic of a single token.
        
        :param index: index of the token
        :param new_k: a new topic (-1 for unassigned) 
        :param new_j: a new syntactic position of the antecedent
        (-1 for unassigned)
        """

        # unassign topic k and antecedent position j
	if new_k == -1:
            self.topic_state[index] = -1
            self.antecedent_state[index] = -1
            
        # assign new topic k and antecedent position j
        else:
            self.topic_state[index] = new_k            
            self.antecedent_state[index] = new_j
        
            
    def remove_topic(self, topic):
        """
        Remove the topic currently not-used.
        Put it in unused_topics list.
        """
        self.used_topics.remove(topic)
        self.unused_topics.insert(0, topic)
        self.M_k[topic] = 0
        self.K = len(self.used_topics)

        
    def add_topic(self, topic):
        """
        Add a new topic to used_topics list.
        """
        new_topic = self.unused_topics.pop(0)
        assert new_topic == topic
        self.used_topics.append(new_topic)
        self.M_k[new_topic] = 1
        self.K = len(self.used_topics)
        

    def old_dish_lhood(self, topic, word):
        """
        Compute likelihood term when the used topic is assigned
        (i.e., the old dish at franchise level)
        """
        lhood = (self.V_ki[topic][word] + self.beta)
        lhood /= (self.V_ki[topic].N() + self.V * self.beta)
        
        return lhood

        
    def probability(self, term, index):        
        """
        Return the probability of a particular term taking on a
	topic assignment, conditioned on all other assignments:
	p(x,z|-) = p(w|-) * p(z|-) * p(Discourse) 
                -- p(w|-): likelihood
                -- p(z|-): prior given by Chinese restaurant franchise
                -- p(Discourse): prior estimated in the experiment
        """

        for j in range(self.J):

            # Return probs for choosing an used topic.
            for topic in self.used_topics:
                
                # compute lhood first
                prob = self.old_dish_lhood(topic, term)
                
                # old table & old dish
                if topic in self.N_jk[j]:
                    prob *= (self.N_jk[j][topic] / (self.N_jk[j].N() + self.alpha))
                    prob *= self.discourse_priors[j][index]
                    yield ((j, topic), prob)
                                
                # If topic k is not used in j but used in other j. 
                # (new table & old dish)
                else:
                    prob *= (self.alpha / (self.N_jk[j].N() + self.alpha))
                    prob *= (self.M_k[topic] / (self.M_k.N() + self.gamma))
                    prob *= self.discourse_priors[j][index]
                    yield ((j, topic), prob)

            # Return probs for choosing a new topic.
            new_topic = self.unused_topics[0]
            prob = 1.0/self.V # lhood for a new topic
            prob *= self.alpha / (self.N_jk[j].N() + self.alpha)
            prob *= self.gamma / (self.M_k.N() + self.gamma)
            prob *= self.discourse_priors[j][index]
            yield ((j, new_topic), prob)


    def model_lhood(self):
        """
        Compute the log likelihood of the model.
        """
        log_lhood = 0
        log_lhood += self.table_log_lhood()
        log_lhood += self.dish_log_lhood()
        log_lhood += self.word_log_lhood()
        
        return log_lhood

    
    def table_log_lhood(self):
        """
        Compute prior at the second level restaurants.
        """
        log_lhood = 0
        for j in xrange(self.J):
            used_tables = [k for k in self.N_jk[j] if self.N_jk[j][k] > 0]
            log_lhood += log(self.alpha) * len(used_tables)
            for k in used_tables:
                log_lhood += lgammln(self.N_jk[j][k])
                    
            denominator = 1
            for t in range(1, len(self.N_jk[j])+1):
                denominator *= (t + self.alpha -1)
            log_lhood -= log(denominator)
            
        return log_lhood

    
    def dish_log_lhood(self):
        """
        Compute prior at the topic level (dishes at the franchise level). 
        """
        # N for the denominator prod
        for j in xrange(self.J):
            used_tables = [k for k in self.N_jk[j] if self.N_jk[j][k] > 0] 
        N = len(used_tables)

        # the numerator
        log_lhood = log(self.gamma) * self.K
        for k in self.used_topics:
            log_lhood += lgammln(self.M_k[k])
            
        denominator = 1
        for m in range(1, N+1):
            denominator *= (m + self.gamma -1)

        return log_lhood - log(denominator)

    
    def word_log_lhood(self):
        """
        compute lhood for words p(w|-).
        """

        log_lhood = lgammln(self.V * self.beta)
        log_lhood -= lgammln(self.beta) * self.V
        log_lhood = log_lhood * self.K
        for k in range(self.K):
            for i in range(self.V):
                log_lhood += lgammln(self.V_ki[k][i] + self.beta)
            log_lhood -= lgammln(self.V_ki[k].N() + (self.V * self.beta))
        
        return log_lhood
            
    
class Sampler:
    """
    A class for collapsed Gibbs sampling.
    """

    def __init__(self, state):
        """
        Create a new sampler.
        """
        print 'Sampler.__init__ called.'

        self.state = state
        self.topic_state = state.topic_state
        self.word_state = state.word_state
        self.antecedent_state = state.antecedent_state
        self.used_topics = state.used_topics
	self.V_ki = state.V_ki
	self.N_jk = state.N_jk
        

    def sample_word(self, index):
        """
        Sample the topic assignment of a single token.
        :param index: The index of the token.
        :return: The new topic associated with the token.
        """
        
	old_topic = self.topic_state[index]
	old_ante = self.antecedent_state[index]
        term = self.word_state[index]

        # Change topic and antecedent state
	state.change_topic(index, -1, -1)
        
        # Decrement count
	self.V_ki[old_topic][term] -= 1
	self.N_jk[old_ante][old_topic] -= 1
        
        if self.V_ki[old_topic].N() == 0:
            # Remove a topic currently not-used
            state.remove_topic(old_topic)

        # Compute the conditional probs
        new_probs = {}        
        for (new_ante, new_topic), prob in state.probability(term,
                                                             index): 
            if prob > 0:
                new_probs[(new_ante, new_topic)] = prob

        # Sample new topic and ante position
	new_ante, new_topic = dict_sample(new_probs)

        # Update state
        state.change_topic(index, new_ante, new_topic)
        
        # Increment count
        if new_topic in self.used_topics:
            self.V_ki[new_topic][term] += 1
            self.N_jk[new_ante][new_topic] += 1
        else:
            state.add_topic(new_topic)
            state.V_ki[new_topic][term] = 1
            state.N_jk[new_ante][new_topic] = 1


    def run_sampler(self, iterations):
        """
        Run a number of Gibbs sampling iterations.
        :param iterations: The number of iterations to run the sampler.
        :return: A list of likelihoods obtained during inference.
        """

        lhoods = []
        
        for iter in xrange(1, iterations+1):
            processing_time = time.time()
            for ii in xrange(self.state.N): self.sample_word(ii)
            processing_time = time.time() - processing_time

            lhood = self.state.model_lhood()
            lhoods.append(lhood)
            
            print("iteration %d finished in %d seconds with log-likelihood %g"
                  % (iter, processing_time, lhood))

        print("\n Successfully sampled %d pronouns: %d topics learned.\n"
              % (self.state.N, self.state.K))
        
    
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("USAGE: python HDP_AuthorTopic.py iterations alpha gamma beta")

    elif len(sys.argv) == 2:
        iterations = int(sys.argv[1])        
        print "Hyperparameter values unspecified:"
        print "Set alpha=%f, gamma=%f, beta=%f" %  (1.0, 0.001, 0.01)
        alpha = 1.0
        gamma = 0.001
        beta = 0.01
    else:
        iterations = int(sys.argv[1])
        alpha = float(sys.argv[2])
        gamma = float(sys.argv[3])
        beta = float(sys.argv[4])

            
# Create sate and sampler instances
state = State(alpha, beta, gamma, observing_pronouns, discourse_priors)
sampler = Sampler(state)

# Run sampler
sampler.run_sampler(iterations)

# Show topic-word distribution
state.report()

