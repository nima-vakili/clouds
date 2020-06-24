"""
Testing how Python handles multiple inheritence
"""
from clouds import Probability, PriorMeans

class Prior(Probability):

    def __init__(self):
        print 'Prior: __init__ called'
        
    def log_prob(self):
        print 'Prior: log_prob called'
        return 0.

class SpecialPrior(Prior):

    def __init__(self):
        super(SpecialPrior, self).__init__()
        print 'SpecialPrior: __init__ called'

    def log_prob(self):
        print 'SpecialPrior: log_prob called'
        return 0.

class Likelihood(Probability):

    def log_prob(self):
        print 'Likelihood: log_prob called'
        return 0.

class Posterior(Prior):

    @property
    def _Prior(self):
        return self.__class__.__bases__[-1]

    def __init__(self, likelihood):

        self._Prior.__init__(self)

        self._likelihood = likelihood

    def log_prob(self):

        return self._Prior.log_prob(self) + self._likelihood.log_prob()

class SpecialPosterior(Posterior, SpecialPrior):

    pass

L = Likelihood()
p = Posterior(L)

print p.log_prob()

q = SpecialPosterior(L)
print '\nCalling SpecialPosterior'
print q.log_prob()
