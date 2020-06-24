import numpy as np

from copy import deepcopy

def print_rates(rex_history, beta=None):

    rates = []
    pairs = rex_history.keys()
    pairs.sort(lambda a, b: cmp(a[0],b[0]))

    out = '{0}<->{1}: {2:.1f}%'
    
    for pair in pairs:
        rates.append(np.mean(rex_history[pair]))
        a = beta[pair[0]] if beta is not None else pair[0]
        b = beta[pair[1]] if beta is not None else pair[1]
        print out.format(a, b, rates[-1]*100)

    return rates

class ReplicaExchangeMC(object):

    def __init__(self, models, samplers):

        self.models = models
        self.samplers = samplers       
        self.prob_prior = []
        self.prob_posterior = []
        self.stop  = False

    def move_parallel(self, x):
        for k, T in enumerate(self.samplers):
            x[k] = T.run(x[k])[-1]

    def propose_swap(self, x, y):
        return y, x
            
    def run(self, x, n_samples, verbose=False, return_mu=False, return_tau=False, return_rotations=False):

        self.state = x
        self.stop  = False
        
        posterior = deepcopy(self.models[-1])
        
        pairs = [[(i,i+1) for i in range(0,len(self.samplers),2) if i+1 < len(self.samplers)],
                 [(i,i+1) for i in range(1,len(self.samplers),2) if i+1 < len(self.samplers)]]
     
        self.n_acc = n_acc = {(i,j): [] for p in pairs for i,j in p}

        rotations = []
        self.means = means = []
        precisions = []

        for r in range(int(n_samples)-1):

            self.move_parallel(x)

            direction = r % 2     
            for i, j in pairs[direction]:

                x_j, x_i = self.propose_swap(x[i], x[j]) 
                
                E_ii = -self.models[i].log_prob(x[i])    
                E_jj = -self.models[j].log_prob(x[j])    

                E_ij = -self.models[j].log_prob(x_i)
                E_ji = -self.models[i].log_prob(x_j)
    
                dE = E_ji + E_ij - E_ii - E_jj
                accept = np.log(np.random.random()) < -dE
                        
                if accept:
                    x[i], x[j] = x_j, x_i
                n_acc[(i,j)].append(int(accept))
            
            if return_rotations: rotations.append(x[-1][-1].copy())
            if return_mu: means.append(x[-1][0].copy())
            if return_tau: precisions.append(x[-1][2])

                

            self.state = x

            posterior.params.beta = 0.
            self.prob_prior.append([posterior.log_prob(state) for state in x])
            posterior.params.beta = 1.
            self.prob_posterior.append([posterior.log_prob(state) for state in x])

            if verbose and not r % verbose:
                print 'replica iteration:', r

            if self.stop: break

        if return_rotations&return_mu&return_tau:
            return x, n_acc, means, precisions, rotations
        else:
            return x, n_acc
        



