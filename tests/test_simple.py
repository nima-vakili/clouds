from clouds import Parameters, Probability

class TestProbability(Probability):

    def log_prob(self):
        return - 0.5 * np.sum((self.params.mu-self.params.mu_0)**2)

K = 50
N = 100
M = 35

params = Parameters(K,N,M)
prob = TestProbability()

## this shouldn't work yet because the parameters haven't been set

try:
    print prob.log_prob()
except Exception, msg:
    print 'WARNING: prob.log_prob() failed due to the following reason:'
    print msg

## set the parameters globally in the Probability class

Probability.set_params(params)

## now it should work

print prob.log_prob()

